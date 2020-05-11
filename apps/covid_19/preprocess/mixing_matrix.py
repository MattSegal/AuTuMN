from datetime import date

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2010, 12, 31)


def build_mixing_matrix(mixing_params: dict):
    """
    Build a time-varing mixing matrix
    """
    mixing = {}
    for strata_key in mixing_params.keys():
        mixing_data = mixing_params[strata_key]
        mixing[strata_key] = {
            "values": mixing_data["values"],
            "times": [process_time(t) for t in mixing_data["times"]],
        }


def process_time(t):
    if type(t) is float:
        return t
    else:
        t_delta = t - BASE_DATE
        return t_delta.days


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

# Get mixing matrix
mixing_matrix = load_specific_prem_sheet("all_locations", model_parameters["country"])
mixing_matrix_multipliers = model_parameters.get("mixing_matrix_multipliers")
if mixing_matrix_multipliers is not None:
    mixing_matrix = update_mixing_with_multipliers(mixing_matrix, mixing_matrix_multipliers)
