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


def equilogspace(x_min,x_max,n):
    """ like np.linspace. but (close to) equidistant in logs

    Args:

        x_min (double): maximum value
        x_max (double): minimum value
        n (int): number of points
    
    Returns:

        y (list): grid with unequal spacing

    """

    pivot = np.abs(x_min) + 0.25
    y = np.geomspace(x_min + pivot, x_max + pivot, n) - pivot
    y[0] = x_min  # make sure *exactly* equal to x_min
    return y