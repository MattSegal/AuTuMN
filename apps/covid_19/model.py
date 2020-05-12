import os
from typing import Dict, List

from autumn import constants
from autumn.constants import Compartment
from autumn.disease_categories.emerging_infections.flows import (
    add_infection_flows,
    add_transition_flows,
    add_recovery_flows,
    add_sequential_compartment_flows,
    add_infection_death_flows,
)
from autumn.demography.social_mixing import (
    load_specific_prem_sheet,
    update_mixing_with_multipliers,
    get_total_contact_rates_by_age,
)
from summer.model import StratifiedModel
from autumn.db import Database, find_population_by_agegroup
from autumn.demography.ageing import add_agegroup_breaks
from autumn.demography.population import get_population_size
from autumn.summer_related.parameter_adjustments import split_multiple_parameters
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit import schema_builder as sb
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.utils import (
    find_relative_date_from_string_or_tuple,
    normalise_sequence,
    convert_list_contents_to_int,
)

from . import preprocess
from .stratification import stratify_by_clinical
from .outputs import (
    find_incidence_outputs,
    create_fully_stratified_incidence_covid,
    create_fully_stratified_progress_covid,
    calculate_notifications_covid,
    calculate_incidence_icu_covid,
)
from .importation import set_tv_importation_rate

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
    times={"start_time": float, "end_time": float, "time_step": float,},
    # Compartment construction
    compartment_periods=Dict[str, float],
    compartment_periods_calculated=dict,
    # Age stratified params
    symptomatic_props=List[float],
    hospital_props=List[float],
    hospital_inflate=bool,
    icu_prop=float,
    infection_fatality_props=List[float],
    # Youth reduced susceiptibility adjustment.
    young_reduced_susceptibility=float,
    reduced_susceptibility_agegroups=List[str],  # Why a string?
    # Mixing matrix
    mixing=Dict[str, {"times": list, "values": List[float]}],  # date or float
    npi_effectiveness=Dict[str, float],
    # Importation of disease from outside of region.
    importation={
        "active": bool,
        "times": List[float],
        "cases": List[float],
        "self_isolation_effect": float,
        "enforced_isolation_effect": float,
    }
    # Other stuff
    contact_rate=float,
    non_sympt_infect_multiplier=float,
    hospital_non_icu_infect_multiplier=float,
    icu_infect_multiplier=float,
    prop_isolated_among_symptomatic=float,
    infectious_seed=int,
    ifr_multipliers=List[float],  # FIXME: Not always used
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
    country = params["country"]

    # Build mixing matrix.
    # FIXME: unit tests for build_static
    # FIXME: unit tests for build_dynamic
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
    time_stemp = times["time_stemp"]
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
        import_params = params["import"]
        param_name = "import_secondary_rate"
        self_isolation_effect = import_params["self_isolation_effect"]
        enforced_isolation_effect = import_params["enforced_isolation_effect"]
        import_times = import_params["times"]
        import_cases = import_params["cases"]            
        model.parameters[param_name] = param_name
        model.adaptation_functions[param_name] = preprocess.importation.get_importation_rate_func(country, import_times, import_cases, self_isolation_effect, enforced_isolation_effect, params["contact_rate"], starting_pop)

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
            str(agegroup): youth_reduced_susceptibility
            for agegroup in youth_agegroups
        },
    }
    if is_importation_active:
        adjust_requests["import_secondary_rate"] = get_total_contact_rates_by_age(static_mixing_matrix, direction="horizontal")

    # Distribute starting population over agegroups
    entry_props = {agegroup: prop for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))}

    # We use "agegroup" instead of "age", to avoid triggering automatic demography features.
    model.stratify(
        "agegroup",  
        agegroup_strata,
        compartment_types_to_stratify=[],  # Apply to all compartments
        entry_proportions=entry_props,  
        mixing_matrix=static_mixing_matrix,
        adjustment_requests=adjust_requests,
    )

    # Stratify infectious compartment by clinical status
    model, model_parameters = stratify_by_clinical(model, model_parameters, compartments)

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

    # Add fully stratified incidence to output_connections
    output_connections.update(
        create_fully_stratified_incidence_covid(
            model_parameters["stratify_by"],
            model_parameters["all_stratifications"],
            model_parameters,
        )
    )
    output_connections.update(
        create_fully_stratified_progress_covid(
            model_parameters["stratify_by"],
            model_parameters["all_stratifications"],
            model_parameters,
        )
    )
    model.output_connections = output_connections

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
