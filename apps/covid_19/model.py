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
from autumn.db import Database
from autumn.demography.ageing import add_agegroup_breaks
from autumn.demography.population import get_population_size
from autumn.summer_related.parameter_adjustments import split_multiple_parameters
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit import schema_builder as sb
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.utils import find_relative_date_from_string_or_tuple, normalise_sequence, convert_list_contents_to_int

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
from .matrices import build_covid_matrices, apply_npi_effectiveness

# Database locations
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
input_database = Database(database_name=INPUT_DB_PATH)

validate_params = sb.build_validator(
    # Country info
    country=str,
    iso3=str,
    # Running time.
    times={
        'start_time': float,
        'end_time': float,
        'time_step': float,
    },
    # Compartment construction
    n_compartment_repeats=Dict[str, int],
    compartment_periods=Dict[str, float],
    prop_exposed_presympt=float,
    prop_infectious_early=float,
    # Strata stuff
    stratify_by=List[str],
    clinical_strata=List[str],
    agegroup_breaks=List[int],
    # Age stratified params
    symptomatic_props=List[float],
    hospital_props=List[float],
    hospital_inflate=bool,
    icu_prop=float,
    infection_fatality_props=List[float],
    young_reduced_susceptibility=float,
    reduced_susceptibility_agegroups=List[str],  # Why a string?
    # Mixing matrix
    mixing=Dict[str, {
        'times': list, # date or float
        'values': List[float]
    }],
    # Other stuff
    implement_importation=bool,
    contact_rate=float,
    non_sympt_infect_multiplier=float,
    hospital_non_icu_infect_multiplier=float,
    icu_infect_multiplier=float,
    prop_isolated_among_symptomatic=float,
    self_isolation_effect=float,
    enforced_isolation_effect=float,
    infectious_seed=int,
    ifr_multipliers=List[float],  # FIXME: Not always used
    young_reduced_susceptibility=float,
    prop_isolated_among_symptomatic=float,
    npi_effectiveness=Dict[str,float],
    data=Dict[str, List[float]],
    # Death rates.
    infect_death=float,
    universal_death_rate=float,
)


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run the TB model for Covid-19.
    Returns the final model with all parameters and stratifications.
    """
    validate_params(params)
    country = params['country']

 
    default = params["default"]

    # Adjust infection for relative all-cause mortality compared to China, if process being applied
    if "ifr_multiplier" in default:
        default["infection_fatality_props"] = [
            i_prop * default["ifr_multiplier"] for i_prop in default["infection_fatality_props"]
        ]
        if default["hospital_inflate"]:
            default["hospital_props"] = [
                i_prop * default["ifr_multiplier"] for i_prop in default["hospital_props"]
            ]

    # Calculate presymptomatic period from exposed period and relative proportion of that period spent infectious
    if "prop_exposed_presympt" in default:
        default["compartment_periods"][Compartment.EXPOSED] = default["compartment_periods"][
            "incubation"
        ] * (1.0 - default["prop_exposed_presympt"])
        default["compartment_periods"][Compartment.PRESYMPTOMATIC] = (
            default["compartment_periods"]["incubation"] * default["prop_exposed_presympt"]
        )

    # Calculate early infectious period from total infectious period and proportion of that period spent isolated
    if "prop_infectious_early" in default:
        default["compartment_periods"][Compartment.EARLY_INFECTIOUS] = (
            default["compartment_periods"]["total_infectious"] * default["prop_infectious_early"]
        )
        default["compartment_periods"][Compartment.LATE_INFECTIOUS] = default[
            "compartment_periods"
        ]["total_infectious"] * (1.0 - default["prop_infectious_early"])
    # =============== END FIXME


    params = add_agegroup_breaks(params)

    # Update parameters stored in dictionaries that need to be modified during calibration
    params = update_dict_params_for_calibration(params)

    model_parameters = params

    # Get population size (by age if age-stratified)
    total_pops, model_parameters = get_population_size(model_parameters, input_database)

    # Replace with Victorian populations
    # total_pops = load_population('31010DO001_201906.XLS', 'Table_6')
    # total_pops = \
    #     [
    #         int(pop) for pop in
    #         total_pops.loc[
    #             (i_pop for i_pop in total_pops.index if 'Persons' in i_pop),
    #             'Victoria'
    #         ]
    #     ]

    # Define compartments with repeats as needed
    all_compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        Compartment.RECOVERED,
    ]
    final_compartments, replicated_compartments = [], []
    for compartment in all_compartments:
        if params["n_compartment_repeats"][compartment] == 1:
            final_compartments.append(compartment)
        else:
            replicated_compartments.append(compartment)
    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.EARLY_INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    for compartment in params["compartment_periods"]:
        model_parameters["within_" + compartment] = 1.0 / params["compartment_periods"][compartment]

    # Multiply the progression rates by the number of compartments to keep the average time in exposed the same
    for compartment in is_infectious:
        model_parameters["within_" + compartment] *= float(
            model_parameters["n_compartment_repeats"][compartment]
        )
    for state in ["hospital_early", "icu_early"]:
        model_parameters["within_" + state] *= float(
            model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS]
        )
    for state in ["hospital_late", "icu_late"]:
        model_parameters["within_" + state] *= float(
            model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
        )

    # Distribute infectious seed across infectious compartments
    total_infectious_times = sum(
        [model_parameters["compartment_periods"][comp] for comp in is_infectious]
    )
    init_pop = {
        comp: model_parameters["infectious_seed"]
        * model_parameters["compartment_periods"][comp]
        / total_infectious_times
        for comp in is_infectious
    }

    # Set integration times 
    integration_times = get_model_times_from_inputs(**params['times'])

    # Add flows through replicated compartments
    flows = []
    for compartment in is_infectious:
        flows = add_sequential_compartment_flows(
            flows, model_parameters["n_compartment_repeats"][compartment], compartment
        )

    # Add other flows between compartment types
    flows = add_infection_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.EXPOSED]
    )
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.EXPOSED],
        model_parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC],
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        "within_exposed",
    )

    # Distinguish to_infectious parameter, so that it can be split later
    model_parameters["to_infectious"] = model_parameters["within_presympt"]
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC],
        model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS],
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        "to_infectious",
    )
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS],
        model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS],
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        "within_" + Compartment.EARLY_INFECTIOUS,
    )
    flows = add_recovery_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
    )
    flows = add_infection_death_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
    )

    # Add importation flows if requested
    if model_parameters["implement_importation"]:
        flows = add_transition_flows(
            flows,
            1,
            model_parameters["n_compartment_repeats"][Compartment.EXPOSED],
            Compartment.SUSCEPTIBLE,
            Compartment.EXPOSED,
            "import_secondary_rate",
        )

    # Get mixing matrix
    mixing_matrix = load_specific_prem_sheet("all_locations", model_parameters["country"])
    mixing_matrix_multipliers = model_parameters.get("mixing_matrix_multipliers")
    if mixing_matrix_multipliers is not None:
        mixing_matrix = update_mixing_with_multipliers(mixing_matrix, mixing_matrix_multipliers)

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        final_compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="no_birth",
        starting_population=sum(total_pops),
        infectious_compartment=[i_comp for i_comp in is_infectious if is_infectious[i_comp]],
    )

    # set time-variant importation rate
    if model_parameters["implement_importation"]:
        _covid_model = set_tv_importation_rate(
            _covid_model, params["data"]["times_imported_cases"], params["data"]["n_imported_cases"]
        )

    # Stratify model by age
    if "agegroup" in model_parameters["stratify_by"]:
        age_strata = model_parameters["all_stratifications"]["agegroup"]
        adjust_requests = split_multiple_parameters(
            ("to_infectious", "infect_death", "within_late"), age_strata
        )  # Split unchanged parameters for later adjustment

        if model_parameters["implement_importation"]:
            adjust_requests.update(
                {
                    "import_secondary_rate": get_total_contact_rates_by_age(
                        mixing_matrix, direction="horizontal"
                    )
                }
            )

        # Adjust susceptibility for children
        adjust_requests.update(
            {
                "contact_rate": {
                    key: value
                    for key, value in zip(
                        model_parameters["reduced_susceptibility_agegroups"],
                        [model_parameters["young_reduced_susceptibility"]]
                        * len(model_parameters["reduced_susceptibility_agegroups"]),
                    )
                }
            }
        )

        _covid_model.stratify(
            "agegroup",  # Don't use the string age, to avoid triggering automatic demography
            convert_list_contents_to_int(age_strata),
            [],  # Apply to all compartments
            {
                i_break: prop for i_break, prop in zip(age_strata, normalise_sequence(total_pops))
            },  # Distribute starting population
            mixing_matrix=mixing_matrix,
            adjustment_requests=adjust_requests,
            verbose=False,
        )

    # Stratify infectious compartment by clinical status
    if "clinical" in model_parameters["stratify_by"] and model_parameters["clinical_strata"]:
        _covid_model, model_parameters = stratify_by_clinical(
            _covid_model, model_parameters, final_compartments
        )

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
    _covid_model.output_connections = output_connections

    # Add notifications to derived_outputs
    _covid_model.derived_output_functions["notifications"] = calculate_notifications_covid
    _covid_model.death_output_categories = list_all_strata_for_mortality(
        _covid_model.compartment_names
    )
    _covid_model.derived_output_functions["incidence_icu"] = calculate_incidence_icu_covid

    # Do mixing matrix stuff
    mixing_instructions = model_parameters.get("mixing")
    if mixing_instructions:
        if "npi_effectiveness" in model_parameters:
            mixing_instructions = apply_npi_effectiveness(
                mixing_instructions, model_parameters.get("npi_effectiveness")
            )
        _covid_model.find_dynamic_mixing_matrix = build_covid_matrices(
            model_parameters["country"], mixing_instructions
        )
        _covid_model.dynamic_mixing_matrix = True

    return _covid_model




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


def get_mixing_lists_from_dict(working_dict):
    return [i_key for i_key in working_dict.keys()], [i_key for i_key in working_dict.values()]


def revise_mixing_data_for_dicts(parameters):
    list_of_possible_keys = ["home", "other_locations", "school", "work"]
    for age_index in range(16):
        list_of_possible_keys.append("age_" + str(age_index))
    for mixing_key in list_of_possible_keys:
        if mixing_key in parameters:
            (
                parameters[mixing_key + "_times"],
                parameters[mixing_key + "_values"],
            ) = get_mixing_lists_from_dict(parameters[mixing_key])
    return parameters


def revise_dates_if_ymd(mixing_params):
    """
    Find any mixing times parameters that were submitted as a three element list of year, month day - and revise to an
    integer representing the number of days from the reference time.
    """
    for key in (k for k in mixing_params if k.endswith("_times")):
        for i_time, time in enumerate(mixing_params[key]):
            if isinstance(time, (list, str)):
                mixing_params[key][i_time] = find_relative_date_from_string_or_tuple(time)
