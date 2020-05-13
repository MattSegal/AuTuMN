import os

from autumn import constants
from autumn.constants import Compartment
from autumn.demography.social_mixing import get_total_contact_rates_by_age
from summer.model import StratifiedModel
from autumn.db import Database, find_population_by_agegroup
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit import get_model_times_from_inputs, schema_builder as sb
from autumn.tool_kit.utils import normalise_sequence

from . import preprocess
from .stratification import stratify_by_clinical

# Database locations
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
input_database = Database(database_name=INPUT_DB_PATH)

# Define agegroup strata
AGEGROUP_MAX = 80  # years
AGEGROUP_STEP = 5  # years


# class Clinical:
#     NON_SYMPT = "non_sympt"
#     SYMPT_NON_HOSPITAL = "sympt_non_hospital"
#     SYMPT_ISOLATE = "sympt_isolate"
#     HOSPITAL_NON_ICU = "hospital_non_icu"
#     ICU = "icu"
#     ALL = [
#         NON_SYMPT,
#         SYMPT_NON_HOSPITAL,
#         SYMPT_ISOLATE,
#         HOSPITAL_NON_ICU,
#         ICU,
#     ]


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
    symptomatic_props=sb.List(float),
    hospital_props=sb.List(float),
    hospital_inflate=bool,
    icu_prop=float,
    infection_fatality_props=sb.List(float),
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
    icu_mortality_prop=float,
    non_sympt_infect_multiplier=float,
    hospital_non_icu_infect_multiplier=float,
    icu_infect_multiplier=float,
    prop_isolated_among_symptomatic=float,
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
    params = update_dict_params_for_calibration(params)

    validate_params(params)

    # Build mixing matrix.
    # FIXME: unit tests for build_static
    # FIXME: unit tests for build_dynamic
    country = params["country"]
    dynamic_mixing_params = params["mixing"]
    npi_effectiveness_params = params["npi_effectiveness"]
    static_mixing_matrix = preprocess.mixing_matrix.build_static(country, None)
    dynamic_mixing_matrix = None
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
    # TODO: FIX THIS
    model, model_parameters = stratify_by_clinical(model, model_parameters, compartments)

    # Define output connections to collate
    output_connections = {
        # Track flow from presymptomatic cases to infectious cases.
        "incidence": {
            "origin": Compartment.PRESYMPTOMATIC,
            "to": Compartment.EARLY_INFECTIOUS,
            "origin_condition": "",
            "to_condition": "",
        }
    }

    import itertools
    from datetime import date
    from summer.model.utils.string import find_name_components

    import pdb

    pdb.set_trace()

    def create_fully_stratified_incidence_covid(
        requested_stratifications, strata_dict, model_params
    ):
        """
        Create derived outputs for fully disaggregated incidence
        """

        all_tags_by_stratification = []
        for stratification in requested_stratifications:
            this_stratification_tags = []
            for stratum in strata_dict[stratification]:
                this_stratification_tags.append(stratification + "_" + stratum)
            all_tags_by_stratification.append(this_stratification_tags)

        all_tag_lists = list(itertools.product(*all_tags_by_stratification))

        for tag_list in all_tag_lists:
            stratum_name = "X".join(tag_list)
            out_connections["incidenceX" + stratum_name] = {
                "origin": Compartment.PRESYMPTOMATIC,
                "to": Compartment.EARLY_INFECTIOUS,
                "origin_condition": "",
                "to_condition": stratum_name,
            }

        return out_connections

    # Add fully stratified incidence to output_connections
    output_connections.update(
        create_fully_stratified_incidence_covid(
            model_parameters["stratify_by"],
            model_parameters["all_stratifications"],
            model_parameters,
        )
    )

    def create_fully_stratified_progress_covid(
        requested_stratifications, strata_dict, model_params
    ):
        """
        Create derived outputs for fully disaggregated incidence
        """
        out_connections = {}
        origin_compartment = (
            Compartment.EARLY_INFECTIOUS
            if model_params["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS] < 2
            else Compartment.EARLY_INFECTIOUS
            + "_"
            + str(model_params["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS])
        )
        to_compartment = (
            Compartment.LATE_INFECTIOUS
            if model_params["n_compartment_repeats"][Compartment.LATE_INFECTIOUS] < 2
            else Compartment.LATE_INFECTIOUS + "_1"
        )

        all_tags_by_stratification = []
        for stratification in requested_stratifications:
            this_stratification_tags = []
            for stratum in strata_dict[stratification]:
                this_stratification_tags.append(stratification + "_" + stratum)
            all_tags_by_stratification.append(this_stratification_tags)

        all_tag_lists = list(itertools.product(*all_tags_by_stratification))

        for tag_list in all_tag_lists:
            stratum_name = "X".join(tag_list)
            out_connections["progressX" + stratum_name] = {
                "origin": origin_compartment,
                "to": to_compartment,
                "origin_condition": "",
                "to_condition": stratum_name,
            }

        return out_connections

    output_connections.update(
        create_fully_stratified_progress_covid(
            model_parameters["stratify_by"],
            model_parameters["all_stratifications"],
            model_parameters,
        )
    )
    model.output_connections = output_connections

    def calculate_notifications_covid(model, time):
        """
        Returns the number of notifications for a given time.
        The fully stratified incidence outputs must be available before calling this function
        """
        notifications = 0.0
        this_time_index = model.times.index(time)
        for key, value in model.derived_outputs.items():
            if "progressX" in key and any(
                [stratum in key for stratum in model.all_stratifications["clinical"][2:]]
            ):
                notifications += value[this_time_index]
        return notifications

    def calculate_incidence_icu_covid(model, time):
        this_time_index = model.times.index(time)
        incidence_icu = 0.0
        for key, value in model.derived_outputs.items():
            if "incidence" in find_name_components(key) and "clinical_icu" in find_name_components(
                key
            ):
                incidence_icu += value[this_time_index]
        return incidence_icu

    def find_date_from_year_start(times, incidence):
        """
        Messy patch to shift dates over such that zero represents the start of the year and the number of cases are
        approximately correct for Australia at 22nd March
        """
        year, month, day = 2020, 3, 22
        cases = 1098.0
        data_days_from_year_start = (date(year, month, day) - date(year, 1, 1)).days
        model_days_reach_target = next(
            i_inc[0] for i_inc in enumerate(incidence) if i_inc[1] > cases
        )
        print(f"Integer date at which target reached is: {model_days_reach_target}")
        days_to_add = data_days_from_year_start - model_days_reach_target
        return [int(i_time) + days_to_add for i_time in times]

    # Add notifications to derived_outputs
    model.derived_output_functions["notifications"] = calculate_notifications_covid
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)
    model.derived_output_functions["incidence_icu"] = calculate_incidence_icu_covid

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
