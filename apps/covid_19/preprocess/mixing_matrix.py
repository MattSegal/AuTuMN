from datetime import date
from typing import Callable

import numpy as np

from autumn.curve import scale_up_function

from autumn.demography.social_mixing import load_specific_prem_sheet

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2010, 12, 31)

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]


def build_static_mixing_matrix(country: str, multipliers: np.ndarray) -> np.ndarray:
    """
    Get a non-time-varying mixing matrix.
    multipliers is a matrix with the ages-specific multipliers.
    Returns the updated mixing-matrix
    """
    mixing_matrix = load_specific_prem_sheet("all_locations", country)
    if multipliers:
        # Update the mixing matrix using some age-specific multipliers
        assert mixing_matrix.shape == multipliers.shape
        return np.multiply(mixing_matrix, multipliers)
    else:
        return mixing_matrix


def build_dynamic_mixing_matrix(
    country: str, mixing_params: dict, npi_effectiveness_params: dict
) -> Callable[float, dict]:
    """
    Build a time-varing mixing matrix
    """
    # Preprocess mixing instructions for all included locations
    mixing = {}
    for location_key in mixing_params.keys():
        mixing_data = mixing_params[location_key]
        mixing[location_key] = {
            "values": mixing_data["values"],
            "times": [
                (t if type(t) is float else (t - BASE_DATE).days) for t in mixing_data["times"]
            ],
        }

    # Adjust the mixing parameters according by scaling them according to NPI effectiveness
    for location_key, adjustment_val in npi_effectiveness_params.items():
        mixing[location_key]["values"] = [
            1 - (1 - val) * adjustment_val for val in mixing[location_key]["values"]
        ]

    # Load all location-specific mixing info.
    matrix_components = {}
    for sheet_type in ["all_locations", "home", "other_locations", "school", "work"]:
        matrix_components[sheet_type] = load_specific_prem_sheet(sheet_type, country)

    def mixing_matrix_function(time: float):
        mixing_matrix = matrix_components["all_locations"]
        # Make adjustments by location
        for location in [
            loc
            for loc in ["home", "other_locations", "school", "work"]
            if loc + "_times" in mixing_params
        ]:
            location_adjustment = scale_up_function(
                mixing_params[location + "_times"], mixing_params[location + "_values"], method=4
            )
            mixing_matrix = np.add(
                mixing_matrix, (location_adjustment(time) - 1.0) * matrix_components[location],
            )

        # Make adjustments by age
        affected_age_indices = [
            age_index
            for age_index in range(16)
            if "age_" + str(age_index) + "_times" in mixing_params
        ]
        complement_indices = [
            age_index for age_index in range(16) if age_index not in affected_age_indices
        ]

        for age_index_affected in affected_age_indices:
            age_adjustment = scale_up_function(
                mixing_params["age_" + str(age_index_affected) + "_times"],
                mixing_params["age_" + str(age_index_affected) + "_values"],
                method=4,
            )
            for age_index_not_affected in complement_indices:
                mixing_matrix[age_index_affected, age_index_not_affected] *= age_adjustment(time)
                mixing_matrix[age_index_not_affected, age_index_affected] *= age_adjustment(time)

            # FIXME: patch for elderly cocooning in Victoria assuming
            for age_index_affected_bis in affected_age_indices:
                mixing_matrix[age_index_affected, age_index_affected_bis] *= (
                    1.0 - (1.0 - age_adjustment(time)) / 2.0
                )

        return mixing_matrix

    return mixing_matrix_function
