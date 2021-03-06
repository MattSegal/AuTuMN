import numpy as np
import math
from scipy import stats


def find_decent_starting_point(prior_dict):
        """
        Find an MCMC parameter initial value, using the mean of the specified prior
        :param prior_dict:
        :return: float (starting point)
        """
        if prior_dict["distribution"] == "uniform":
            x = np.mean(prior_dict["distri_params"])
        elif prior_dict["distribution"] == "beta":
            a = prior_dict["distri_params"][0]
            b = prior_dict["distri_params"][1]
            x = a / (a + b)
        elif prior_dict["distribution"] == "gamma":
            shape = prior_dict["distri_params"][0]
            scale = prior_dict["distri_params"][1]
            x = shape * scale
        else:
            raise_error_unsupported_prior(prior_dict["distribution"])

        return x


def calculate_prior(prior_dict, x, log=True):
    """
    Calculate the log-prior value given the distribution details and the evaluation point
    :param prior_dict: distribution details
    :param x: evaluation point
    :param log: boolean
        Whether to return the log-PDF of the PDF
    :return: log-PDF(x) or PDF(x)
    """
    if prior_dict["distribution"] == "uniform":
        if log:
            y = math.log(
                1.0 / (prior_dict["distri_params"][1] - prior_dict["distri_params"][0])
            )
        else:
            y = 1.0 / (prior_dict["distri_params"][1] - prior_dict["distri_params"][0])
    elif prior_dict["distribution"] == "lognormal":
        mu = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        if log:
            y = stats.lognorm.logpdf(x=x, s=sd, scale=math.exp(mu))  # see documentation of stats.lognorm for scale
        else:
            y = stats.lognorm.pdf(x=x, s=sd, scale=math.exp(mu))
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        if log:
            y = stats.beta.logpdf(x, a, b)
        else:
            y = stats.beta.pdf(x, a, b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        if log:
            y = stats.gamma.logpdf(x, shape, 0., scale)
        else:
            y = stats.gamma.pdf(x, shape, 0., scale)
    else:
        raise_error_unsupported_prior()
    return float(y)


def raise_error_unsupported_prior(distribution):
    raise ValueError(distribution + "distribution not supported in autumn_mcmc at the moment")
