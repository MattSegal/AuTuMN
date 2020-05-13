from typing import List, Callable
from autumn.curve import scale_up_function


def get_importation_rate_func(
    country: str,
    importation_times: List[float],
    importation_n_cases: List[float],
    self_isolation_effect: float,
    enforced_isolation_effect: float,
    contact_rate: float,
    starting_population: float,
) -> Callable[[float], float]:
    """
    Returns a time varying function of importation secondary infection rate.
    See also: flows.
    """

    # scale-up curve for importation numbers
    get_importation_amount = scale_up_function(importation_times, importation_n_cases)

    # time-variant infectiousness of imported cases
    assert country == "victoria", "VIC only. Hard-coded Victorian values."
    mystery_times = [75.0, 77.0, 88.0, 90.0]
    mystery_vals = [
        1.0,
        1.0 - self_isolation_effect,
        1.0 - self_isolation_effect,
        1.0 - enforced_isolation_effect,
    ]
    tv_imported_infectiousness = scale_up_function(mystery_times, mystery_vals, method=4,)

    def recruitment_rate(t):
        return (
            get_importation_amount(t)
            * tv_imported_infectiousness(t)
            * contact_rate
            / model.starting_population
        )

    return recruitment_rate
