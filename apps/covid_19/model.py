import os
import itertools

from autumn import constants
from autumn.constants import Compartment
from autumn.demography.social_mixing import get_total_contact_rates_by_age
from summer.model import StratifiedModel
from autumn.db import Database, find_population_by_agegroup
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit import get_model_times_from_inputs, schema_builder as sb
from autumn.tool_kit.utils import normalise_sequence

from . import preprocess
from . import outputs

from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
    element_wise_list_division,
)
from autumn.constants import Compartment
from autumn.summer_related.parameter_adjustments import (
    adjust_upstream_stratified_parameter,
    split_prop_into_two_subprops,
)


# Database locations
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
input_database = Database(database_name=INPUT_DB_PATH)

# Define agegroup strata
AGEGROUP_MAX = 80  # years
AGEGROUP_STEP = 5  # years


class Clinical:
    NON_SYMPT = "non_sympt"
    SYMPT_NON_HOSPITAL = "sympt_non_hospital"
    SYMPT_ISOLATE = "sympt_isolate"
    HOSPITAL_NON_ICU = "hospital_non_icu"
    ICU = "icu"
    ALL = [
        NON_SYMPT,
        SYMPT_NON_HOSPITAL,
        SYMPT_ISOLATE,
        HOSPITAL_NON_ICU,
        ICU,
    ]


validate_params = sb.build_validator(
    # Country info
    country=str,
    iso3=str,
    # Running time.
    times=sb.Dict(start_time=float, end_time=float, time_step=float),
    # Compartment construction
    compartment_periods=sb.DictGeneric(str, float),
    compartment_periods_calculated=dict,
    # Age stratified params
    hospital_props=sb.List(float),
    hospital_inflate=bool,
    infection_fatality_props=sb.List(float),
    # Clinical status stratified params
    symptomatic_props=sb.List(float),
    icu_prop=float,
    icu_mortality_prop=float,
    prop_isolated_among_symptomatic=float,
    # Youth reduced susceiptibility adjustment.
    young_reduced_susceptibility=float,
    reduced_susceptibility_agegroups=sb.List(float),
    # Mixing matrix
    mixing=sb.DictGeneric(str, sb.Dict(times=list, values=sb.List(float))),
    npi_effectiveness=sb.DictGeneric(str, float),
    # Importation of disease from outside of region.
    importation=sb.Dict(
        active=bool,
        times=sb.List(float),
        cases=sb.List(float),
        self_isolation_effect=float,
        enforced_isolation_effect=float,
    ),
    # Other stuff
    contact_rate=float,
    non_sympt_infect_multiplier=float,
    hospital_non_icu_infect_multiplier=float,
    icu_infect_multiplier=float,
    infectious_seed=int,
    # Death rates.
    infect_death=float,
    universal_death_rate=float,
)


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run the TB model for Covid-19.
    Returns the final model with all parameters and stratifications.
    """
    # Update parameters stored in dictionaries that need to be modified during calibration
    # FIXME: This needs to be generic, goes outside of build_model.
    # params = update_dict_params_for_calibration(params)

    validate_params(params)

    # Build mixing matrix.
    # FIXME: unit tests for build_static
    # FIXME: unit tests for build_dynamic
    country = params["country"]
    npi_effectiveness_params = params["npi_effectiveness"]
    static_mixing_matrix = preprocess.mixing_matrix.build_static(country, None)
    dynamic_mixing_matrix = None
    dynamic_mixing_params = params["mixing"]
    if dynamic_mixing_params:
        dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
            country, dynamic_mixing_params, npi_effectiveness_params
        )

    # FIXME: how consistently is this used?
    # Adjust infection for relative all-cause mortality compared to China,
    # using a single constant: infection-rate multiplier.
    ifr_multiplier = params.get("ifr_multiplier")
    hospital_inflate = params["hospital_inflate"]
    hospital_props = params["hospital_props"]
    infection_fatality_props = params["infection_fatality_props"]
    if ifr_multiplier:
        infection_fatality_props = [p * ifr_multiplier for p in infection_fatality_props]
    if ifr_multiplier and hospital_inflate:
        hospital_props = [p * ifr_multiplier for p in hospital_props]

    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        Compartment.RECOVERED,
    ]
    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.EARLY_INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }

    # Calculate compartment periods
    # FIXME: Tests meeee!
    base_compartment_periods = params["compartment_periods"]
    compartment_periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_compartment_periods, compartment_periods_calc
    )

    # Get progression rates from sojourn times, distinguishing within_presymptomatic in order to split this parameter later
    time_within_compartment_params = {}
    for compartment in compartment_periods:
        param_key = f"within_{compartment}"
        time_within_compartment_params[param_key] = 1.0 / compartment_periods[compartment]

    # Get initial population: distribute infectious seed across infectious compartments
    infectious_seed = params["infectious_seed"]
    total_infectious_times = sum([compartment_periods[c] for c in is_infectious])
    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_infectious_times for c in is_infectious
    }

    # Set integration times
    times = params["times"]
    start_time = times["start_time"]
    end_time = times["end_time"]
    time_stemp = times["time_step"]
    integration_times = get_model_times_from_inputs(start_time, end_time, time_stemp)

    # Add flows compartments
    is_importation_active = params["importation"]["active"]
    flows = preprocess.flows.get_flows(is_importation_active)

    # Get the agegroup strata breakpoints.
    agegroup_strata = list(range(0, AGEGROUP_MAX, AGEGROUP_STEP))

    # Calculate the country population size by age-group, using UN data
    country_iso3 = params["iso3"]
    total_pops, _ = find_population_by_agegroup(input_database, agegroup_strata, country_iso3)
    total_pops = [int(1000.0 * total_pops[agebreak][-1]) for agebreak in list(total_pops.keys())]
    starting_pop = sum(total_pops)

    model_params = {
        "import_secondary_rate": "import_secondary_rate",  # This might be important for some reason.
        "infect_death": params["infect_death"],
        "contact_rate": params["contact_rate"],
        **time_within_compartment_params,
    }

    # Define model
    model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_params,
        flows,
        birth_approach="no_birth",
        starting_population=starting_pop,
        infectious_compartment=[c for c in is_infectious if is_infectious[c]],
    )
    if dynamic_mixing_matrix:
        model.find_dynamic_mixing_matrix = dynamic_mixing_matrix
        model.dynamic_mixing_matrix = True

    # Set time-variant importation rate
    if is_importation_active:
        import_params = params["importation"]
        param_name = "import_secondary_rate"
        self_isolation_effect = import_params["self_isolation_effect"]
        enforced_isolation_effect = import_params["enforced_isolation_effect"]
        import_times = import_params["times"]
        import_cases = import_params["cases"]
        # FIXME: This is a little much
        model.adaptation_functions[
            "import_secondary_rate"
        ] = preprocess.importation.get_importation_rate_func(
            country,
            import_times,
            import_cases,
            self_isolation_effect,
            enforced_isolation_effect,
            params["contact_rate"],
            starting_pop,
        )

    # Stratify model by age.
    # Coerce age breakpoint numbers into strings - all strata are represented as strings.
    agegroup_strata = [str(s) for s in agegroup_strata]
    # Create parameter adjustment request for age stratifications
    youth_agegroups = params["reduced_susceptibility_agegroups"]
    youth_reduced_susceptibility = params["young_reduced_susceptibility"]
    adjust_requests = {
        # No change, required for further stratification by clinical status.
        "within_presymptomatic": {s: 1 for s in agegroup_strata},
        "infect_death": {s: 1 for s in agegroup_strata},
        "within_late": {s: 1 for s in agegroup_strata},
        # Adjust susceptibility for children
        "contact_rate": {
            str(agegroup): youth_reduced_susceptibility for agegroup in youth_agegroups
        },
    }
    if is_importation_active:
        adjust_requests["import_secondary_rate"] = get_total_contact_rates_by_age(
            static_mixing_matrix, direction="horizontal"
        )

    # Distribute starting population over agegroups
    requested_props = {
        agegroup: prop for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }

    # We use "agegroup" instead of "age", to avoid triggering automatic demography features.
    model.stratify(
        "agegroup",
        agegroup_strata,
        compartment_types_to_stratify=[],  # Apply to all compartments
        requested_proportions=requested_props,
        mixing_matrix=static_mixing_matrix,
        adjustment_requests=adjust_requests,
    )

    # Stratify infectious compartment by clinical status

    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)
    """

    # Define stratification
    import pdb
    from pprint import pprint

    strata_to_implement = Clinical.ALL
    compartments_to_split = [Compartment.EARLY_INFECTIOUS, Compartment.LATE_INFECTIOUS]
    #  Get the raw proportions of persons dying and progressing to symptomatic,
    # hospital if symptomatic, and to ICU if hospitalised
    #  adjusting to required number of age groups.
    # FIXME: WHY DO WE DO THIS?!?!?!?!??!?!?!?!?!??!
    infection_fatality_props = params["infection_fatality_props"]
    symptomatic_props = params["symptomatic_props"]
    hospital_props = params["hospital_props"]
    icu_prop = params["icu_prop"]
    adjusted_infection_fatality_props = repeat_list_elements_average_last_two(
        infection_fatality_props
    )
    raw_sympt = repeat_list_elements(2, symptomatic_props)
    raw_hospital = repeat_list_elements_average_last_two(hospital_props)

    # Find the absolute progression proportions from the requested splits
    abs_props = split_prop_into_two_subprops([1.0] * 16, "", raw_sympt, "sympt")
    abs_props.update(
        split_prop_into_two_subprops(abs_props["sympt"], "sympt", raw_hospital, "hospital")
    )
    abs_props.update(
        split_prop_into_two_subprops(abs_props["hospital"], "hospital", [icu_prop] * 16, "icu")
    )

    # Calculate the absolute proportion of all patients who should eventually reach hospital death or ICU death.
    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths\
    icu_mortality_prop = params["icu_mortality_prop"]
    hospital_death, icu_death = [], []
    for i_agegroup, agegroup in enumerate(agegroup_strata):

        # If IFR for age group is greater than absolute proportion hospitalised, increased hospitalised proportion
        if adjusted_infection_fatality_props[i_agegroup] > abs_props["hospital"][i_agegroup]:
            abs_props["hospital"][i_agegroup] = adjusted_infection_fatality_props[i_agegroup]

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if any
        target_icu_abs_mort = abs_props["icu"][i_agegroup] * icu_mortality_prop
        left_over_mort = adjusted_infection_fatality_props[i_agegroup] - target_icu_abs_mort

        # If some IFR will be left over for the hospitalised
        if left_over_mort > 0.0:
            hospital_death.append(left_over_mort)
            icu_death.append(target_icu_abs_mort)

        # Otherwise if all IFR taken up by ICU
        else:
            hospital_death.append(0.0)
            icu_death.append(adjusted_infection_fatality_props[i_agegroup])

    abs_death_props = {"hospital_death": hospital_death, "icu_death": icu_death}

    # Find the absolute proportion dying in hospital and in ICU
    abs_props.update(abs_death_props)

    # CFR for non-ICU hospitalised patients
    rel_props = {
        "hospital_death": element_wise_list_division(
            abs_props["hospital_death"], abs_props["hospital_non_icu"]
        ),
        "icu_death": element_wise_list_division(abs_props["icu_death"], abs_props["icu"]),
    }

    # Progression rates into the infectious compartment(s)
    fixed_prop_strata = ["non_sympt", "hospital_non_icu", "icu"]
    stratification_adjustments = adjust_upstream_stratified_parameter(
        "within_presymptomatic",
        fixed_prop_strata,
        "agegroup",
        agegroup_strata,
        [abs_props[stratum] for stratum in fixed_prop_strata],
    )

    # Set isolation rates as absolute proportions
    # Set the absolute proportions of new cases isolated and not isolated, and indicate to the model where they should be found.
    # Apply the isolated proportion to the symptomatic non-hospitalised group
    prop_isolated_among_symptomatic = params["prop_isolated_among_symptomatic"]
    for i_age, agegroup in enumerate(agegroup_strata):
        prop_isolated = (
            abs_props["sympt"][i_age] * prop_isolated_among_symptomatic
            - abs_props["hospital"][i_age]
        ) / abs_props["sympt_non_hospital"][i_age]
        stratification_adjustments["within_presymptomaticXagegroup_" + agegroup][
            "sympt_isolate"
        ] = (abs_props["sympt_non_hospital"][i_age] * prop_isolated)
        stratification_adjustments["within_presymptomaticXagegroup_" + agegroup][
            "sympt_non_hospital"
        ] = abs_props["sympt_non_hospital"][i_age] * (1.0 - prop_isolated)

    # Calculate death rates and progression rates for hospitalised and ICU patients
    progression_death_rates = {}
    for stratum in ("hospital", "icu"):
        death_rates, complements = find_rates_and_complements_from_ifr(
            rel_props[stratum + "_death"],
            1,
            [time_within_compartment_params["within_" + stratum + "_late"]] * 16,
        )
        progression_death_rates[stratum + "_infect_death"] = death_rates
        progression_death_rates[stratum + "_within_late"] = complements

    # Death and non-death progression between infectious compartments towards the recovered compartment
    for param in ("within_late", "infect_death"):
        stratification_adjustments.update(
            adjust_upstream_stratified_parameter(
                param,
                strata_to_implement[3:],
                "agegroup",
                agegroup_strata,
                [
                    progression_death_rates["hospital_" + param],
                    progression_death_rates["icu_" + param],
                ],
                overwrite=True,
            )
        )

    # Over-write rate of progression for early compartments for hospital and ICU
    stratification_adjustments.update(
        {
            "within_infectious": {
                "hospital_non_icuW": time_within_compartment_params["within_hospital_early"],
                "icuW": time_within_compartment_params["within_icu_early"],
            },
        }
    )

    # Sort out all infectiousness adjustments for entire model here
    # Make adjustment for hospitalisation and ICU admission
    strata_infectiousness = {}
    for stratum in strata_to_implement:
        # FIXME: We shouldn't do global param lookup like this
        if stratum + "_infect_multiplier" in params:
            strata_infectiousness[stratum] = params[stratum + "_infect_multiplier"]

    # Make adjustment for isolation/quarantine
    model.individual_infectiousness_adjustments = [
        [[Compartment.LATE_INFECTIOUS, "clinical_sympt_isolate"], 0.2]
    ]

    # Stratify the model using the SUMMER stratification function
    model.stratify(
        "clinical",
        strata_to_implement,
        compartments_to_split,
        infectiousness_adjustments=strata_infectiousness,
        requested_proportions={
            stratum: 1.0 / len(strata_to_implement) for stratum in strata_to_implement
        },
        adjustment_requests=stratification_adjustments,
    )

    # Track compartment output connections.
    stratum_names = list(set(["X".join(x.split("X")[1:]) for x in model.compartment_names]))
    incidence_connections = outputs.get_incidence_connections(stratum_names)
    progress_connections = outputs.get_progress_connections(stratum_names)
    model.output_connections = {
        **incidence_connections,
        **progress_connections,
    }

    # Add notifications to derived_outputs
    model.derived_output_functions["notifications"] = outputs.calculate_notifications_covid
    model.derived_output_functions["incidence_icu"] = outputs.calculate_incidence_icu_covid
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)

    return model


def update_dict_params_for_calibration(params):
    """
    Update some specific parameters that are stored in a dictionary but are updated during calibration.
    For example, we may want to update params['default']['compartment_periods']['incubation'] using the parameter
    ['default']['compartment_periods_incubation']
    :param params: dict
        contains the model parameters
    :return: the updated dictionary
    """

    if "n_imported_cases_final" in params:
        params["data"]["n_imported_cases"][-1] = params["n_imported_cases_final"]

    for location in ["school", "work", "home", "other_locations"]:
        if "npi_effectiveness_" + location in params:
            params["npi_effectiveness"][location] = params["npi_effectiveness_" + location]

    for comp_type in [
        "incubation",
        "infectious",
        "late",
        "hospital_early",
        "hospital_late",
        "icu_early",
        "icu_late",
    ]:
        if "compartment_periods_" + comp_type in params:
            params["compartment_periods"][comp_type] = params["compartment_periods_" + comp_type]

    return params
