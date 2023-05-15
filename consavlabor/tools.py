from numba import njit
from numba import prange
import numba as nb
import numpy as np
import random

@nb.njit
def box_muller(mu=0, sigma=1):
    """Generate random number from normal distribution using Box-Muller transform."""
    u1 = random.random()
    u2 = random.random()
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z0 * sigma + mu

@nb.njit
def generate_zeta(par):
    """Generate shock zeta."""
    # epsilon = box_muller(0, par.sigma_eps/2)  # normal shock with half the original standard deviation
    epsilon = np.random.normal(0, par.sigma_eps)
    eta = 0
    if random.random() < par.gamma/2:  # large shock with half the original probability
        eta = par.pi/2  # large shock with half the original magnitude
    return epsilon + eta



