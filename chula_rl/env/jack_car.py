import numba
import numpy as np
from scipy.stats import poisson

N_CARS = 20

# renting and returning propabilities
P_RETURN_A = poisson.pmf(np.arange(0, N_CARS + 1), 3)
P_RENT_A = poisson.pmf(np.arange(0, N_CARS + 1), 3)
P_RETURN_B = poisson.pmf(np.arange(0, N_CARS + 1), 2)
P_RENT_B = poisson.pmf(np.arange(0, N_CARS + 1), 4)


def step_rent(n_car, p):
    pp = np.zeros(N_CARS + 1)
    r = 0.
    for i_rent in range(0, N_CARS + 1):
        _p = p[i_rent]
        # you rent out so many cars
        i_rent = min(n_car, i_rent)
        r += _p * 10. * i_rent
        pp[n_car - i_rent] += _p
    return pp, r


@numba.jit(nopython=True)
def step_return(p_car, p):
    pp = np.zeros(N_CARS + 1)
    for n_car in range(0, N_CARS + 1):
        for i_ret in range(0, N_CARS + 1):
            nn_car = min(n_car + i_ret, N_CARS)
            pp[nn_car] += p_car[n_car] * p[i_ret]
    return pp


def step_rent_return(n_car, p_rent, p_return):
    # each location is independent
    p, r = step_rent(n_car, p_rent)
    p = step_return(p, p_return)
    return p, r


def step_env(s, a):
    """step a on state s, returning the next state distribution and the expected reward"""
    # step
    n_a, n_b = s
    n_a -= a
    n_b += a
    r_move = -2 * abs(a)
    # env
    p_a, r_a = step_rent_return(n_a, P_RENT_A, P_RETURN_A)
    p_b, r_b = step_rent_return(n_b, P_RENT_B, P_RETURN_B)
    p = np.outer(p_a, p_b)
    r = r_move + r_a + r_b
    return p, r


def valid_a(s):
    """you can move at most -5 to 5 cars per night"""
    n_a, n_b = s
    a = []
    for i in range(-5, 6):
        if 0 <= n_a - i <= N_CARS and 0 <= n_b + i <= N_CARS: a.append(i)
    return a
