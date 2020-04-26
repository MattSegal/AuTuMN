from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements,
    normalise_sequence,
    convert_list_contents_to_int,
    repeat_list_elements_average_last_two
)
from autumn.summer_related.parameter_adjustments import split_multiple_parameters
from autumn.constants import Compartment
from autumn.summer_related.parameter_adjustments import (
    adjust_upstream_stratified_parameter,
    split_prop_into_two_subprops,
)


def stratify_by_age(model_to_stratify, age_strata, mixing_matrix, total_pops):
    """
    Stratify model by age
    Note that because the string passed is 'agegroup' rather than 'age', the standard automatic SUMMER demography is not
    triggered.
    Split the parameters to_infectious, infect_death and within_late without adjusting values, to allow values to be
    adjusted later.
    """
    model_to_stratify.stratify(
        "agegroup",  # Don't use the string age, to avoid triggering automatic demography
        convert_list_contents_to_int(age_strata),
        [],  # Apply to all compartments
        {i_break: prop for
         i_break, prop in zip(age_strata, normalise_sequence(total_pops))},  # Distribute starting population
        mixing_matrix=mixing_matrix,
        adjustment_requests=
        split_multiple_parameters(
            ("to_infectious", "infect_death", "within_late"),
            age_strata),  # Split parameters for later adjustment, but retain original value
        verbose=False,
    )
    return model_to_stratify


def stratify_by_clinical(_covid_model, model_parameters, compartments):
    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)
    """

    # DEFINE STRATIFICATION

    strata_to_implement = \
        model_parameters["clinical_strata"]
    model_parameters["all_stratifications"]["clinical"] = \
        strata_to_implement
    compartments_to_split = [
        comp
        for comp in compartments
        if comp.startswith(Compartment.EARLY_INFECTIOUS) or comp.startswith(Compartment.LATE_INFECTIOUS)
    ]

    # UNADJUSTED PARAMETERS

    # Repeat all the 5-year age-specific IFRs and clinical proportions, with adjustment for data length as needed
    model_parameters.update({
        "adjusted_infection_fatality_props":
            repeat_list_elements_average_last_two(model_parameters["infection_fatality_props"]),
        "raw_sympt":
            repeat_list_elements(2, model_parameters["symptomatic_props"]),
        "raw_hospital":
            repeat_list_elements_average_last_two(model_parameters["hospital_props"]),
        "raw_icu":
            repeat_list_elements_average_last_two(model_parameters["icu_props"]),
    })

    # ABSOLUTE PROGRESSION PROPORTIONS

    # Find the absolute progression proportions from the requested splits
    abs_props = split_prop_into_two_subprops([1.0] * 16, "", model_parameters["raw_sympt"], "sympt")
    abs_props.update(
        split_prop_into_two_subprops(abs_props["sympt"], "sympt", model_parameters["raw_hospital"], "hospital")
    )
    abs_props.update(
        split_prop_into_two_subprops(abs_props["hospital"], "hospital", model_parameters["raw_icu"], "icu")
    )

    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths
    abs_props["icu_death"], abs_props["hospital_death"] = [], []
    for i_agegroup in range(len(model_parameters["all_stratifications"]["agegroup"])):

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if positive
        target_icu_abs_mort = \
            abs_props["icu"][i_agegroup] * \
            model_parameters["icu_mortality_prop"]
        left_over_mort = \
            model_parameters["adjusted_infection_fatality_props"][i_agegroup] - \
            target_icu_abs_mort

        # If some IFR will be left over for the hospitalised
        if left_over_mort > 0.:
            abs_props["icu_death"].append(target_icu_abs_mort)
            abs_props["hospital_death"].append(left_over_mort)

        # Otherwise if all IFR taken up by ICU
        else:
            abs_props["icu_death"].append(model_parameters["adjusted_infection_fatality_props"][i_agegroup])
            abs_props["hospital_death"].append(0.0)

    # RELATIVE PROPORTIONS PROGRESSING

    # CFR for non-ICU hospitalised patients
    rel_props = {
        "hospital_death": [
            death_prop / total_prop
            for death_prop, total_prop in zip(
                abs_props["hospital_death"], abs_props["hospital_non_icu"]
            )
        ],
        "icu_death": [
            death_prop / total_prop
            for death_prop, total_prop in zip(abs_props["icu_death"], abs_props["icu"])
        ],
    }

    # RATES OUT OF LATE DISEASE

    # Calculate death rates and progression rates for hospitalised and ICU patients
    progression_death_rates = {}
    (
        progression_death_rates["hospital_infect_death"],
        progression_death_rates["hospital_within_late"],
    ) = find_rates_and_complements_from_ifr(
        rel_props["hospital_death"],
        model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS],
        [model_parameters["within_hospital_late"]] * 16,
    )
    (
        progression_death_rates["icu_infect_death"],
        progression_death_rates["icu_within_late"],
    ) = find_rates_and_complements_from_ifr(
        rel_props["icu_death"],
        model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS],
        [model_parameters["within_icu_late"]] * 16,
    )

    # RATES INTO EARLY DISEASE

    # Progression rates into the infectious compartment(s)
    fixed_prop_strata = ["non_sympt", "hospital_non_icu", "icu"]
    stratification_adjustments = adjust_upstream_stratified_parameter(
        "to_infectious",
        fixed_prop_strata,
        "agegroup",
        model_parameters["all_stratifications"]["agegroup"],
        [abs_props[stratum] for stratum in fixed_prop_strata],
    )

    # Define isolated proportion, which will be moved to inputs later in some way
    prop_isolated = lambda time: model_parameters["prop_isolated_among_symptomatic"]

    # Apply the isolated proportion to the symptomatic non-hospitalised group
    for i_age, agegroup in enumerate(model_parameters["all_stratifications"]["agegroup"]):
        isolated_name = "abs_prop_isolatedX" + agegroup
        not_isolated_name = "abs_prop_not_isolatedX" + agegroup
        _covid_model.time_variants[isolated_name] = lambda time: abs_props["sympt_non_hospital"][
            i_age
        ] * prop_isolated(time)
        _covid_model.time_variants[not_isolated_name] = lambda time: abs_props[
            "sympt_non_hospital"
        ][i_age] * (1.0 - prop_isolated(time))
        stratification_adjustments["to_infectiousXagegroup_" + agegroup][
            "sympt_isolate"
        ] = isolated_name
        stratification_adjustments["to_infectiousXagegroup_" + agegroup][
            "sympt_non_hospital"
        ] = not_isolated_name

    # Death and non-death progression between infectious compartments towards the recovered compartment
    for param in ("within_late", "infect_death"):
        stratification_adjustments.update(
            adjust_upstream_stratified_parameter(
                param,
                strata_to_implement[3:],
                "agegroup",
                model_parameters["all_stratifications"]["agegroup"],
                [progression_death_rates["hospital_" + param], progression_death_rates["icu_" + param]],
                overwrite=True,
            )
        )

    # INFECTIOUSNESS

    strata_infectiousness = {}
    for stratum in strata_to_implement:
        if stratum + "_infect_multiplier" in model_parameters:
            strata_infectiousness[stratum] = model_parameters[stratum + "_infect_multiplier"]
    stratification_adjustments.update(
        {"within_infectious":
             {"hospital_non_icuW":
                  model_parameters['within_hospital_early'],
              "icuW":
                  model_parameters['within_icu_early']},
         }
    )
    _covid_model.individual_infectiousness_adjustments = \
        [[[Compartment.LATE_INFECTIOUS, "clinical_sympt_isolate"], 0.2]]

    # STRATIFICATION

    # Stratify the model using the SUMMER stratification function
    _covid_model.stratify(
        "clinical",
        strata_to_implement,
        compartments_to_split,
        infectiousness_adjustments=strata_infectiousness,
        requested_proportions={
            stratum: 1.0 / len(strata_to_implement) for stratum in strata_to_implement
        },
        adjustment_requests=stratification_adjustments,
        verbose=False,
    )
    return _covid_model, model_parameters

