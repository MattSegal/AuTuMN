from autumn.demography.social_mixing import load_all_prem_types
from math import tanh, sqrt
import numpy as np
from scipy.linalg import eig

matrices = load_all_prem_types('china')


def build_age_specific_profile_function(b, c, sigma_rel):
    def the_function(a):
        return (1. - sigma_rel) / 2. * tanh(b*(a-c)) + (1. + sigma_rel)/2.
    return the_function


def build_ngm(b, c, sigma_rel, etha):
    age_profile = build_age_specific_profile_function(b, c, sigma_rel)
    C = matrices['all_locations']
    K = np.zeros((16, 16))
    for i in range(16):
        a = i * 5. + 2.5 if i < 15 else 80.
        sigma_a = age_profile(a)
        for j in range(16):
            a_ = j * 5 + 2.5 if j < 15 else 80.
            beta_a_ = age_profile(a_)
            K[i, j] = etha * sigma_a * beta_a_ * C[i, j]
    return K


def get_r0_and_prop_infections(matrix):
    spectrum = eig(matrix)
    r_0 = max([abs(lamnda) for lamnda in spectrum[0]])
    i_max = list(spectrum[0]).index(r_0)
    eigen_vector = spectrum[1][:, i_max]
    eigen_vector = [z.real for z in list(eigen_vector)]
    s = sum(eigen_vector)
    prop_infection = [x / s for x in eigen_vector]
    return r_0, prop_infection


# K = build_ngm(b=0.05, c=40, sigma_rel=sqrt(.1), etha=.49)  # Michael SOM params
K = build_ngm(b=0.3, c=30, sigma_rel=.1, etha=.3)  # Emma's params

r_0, props = get_r0_and_prop_infections(K)
prop_kids = sum(props[0:4])

print("R0: " + str(r_0))
print("Proportion of infections in <20 yo: " + str(prop_kids))

