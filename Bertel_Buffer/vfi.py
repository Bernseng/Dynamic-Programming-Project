from numba import njit, prange
import numpy as np

from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D

import utility

@njit
def obj_bellman(c, a, z, l, v_plus, par):
    """ evaluate bellman equation """

    # a. end-of-period assets
    a_plus = a + (1 - par.tau) * z * l - c

    # b. continuation value
    w = 0
    for ishock in range(par.Nshocks):

        # i. shocks
        z_plus = par.z[ishock]
        z_w = par.z_w[ishock]

        # ii. weight
        weight = z_w

        # iii. interpolate
        w += weight * par.beta * linear_interp.interp_2d(par.grid_a, par.grid_z, v_plus, a_plus, z_plus)

    # c. total value
    value_of_choice = (c**(1 - par.rho)) / (1 - par.rho) - par.varphi * (l**(1 + par.nu)) / (1 + par.nu) + w

    return -value_of_choice  # we are minimizing

# b. solve bellman equation
@njit(parallel=True)
def solve_bellman(t, sol, par):
    """solve bellman equation using vfi"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    v = sol.v[t]

    # loop over outer states
    for ia in prange(par.Na):  # in parallel

        # a. assets
        a = par.grid_a[ia]

        # b. loop over productivity
        for iz in range(par.Nz):

            # i. productivity
            z = par.grid_z[iz]

            # ii. optimal choice
            c_low = np.fmin(a / 2, 1e-8)
            c_high = a
            l = par.optimal_l(a, z, par)  # you'll need to define this function to find the optimal labor supply (l) given assets (a) and productivity (z)
            c[ia, iz] = golden_section_search(obj_bellman, c_low, c_high, args=(a, z, l, sol.v[t + 1], par), tol=par.tol)

            # iii. optimal value
            v[ia, iz] = -obj_bellman(c[ia, iz], a, z, l, sol.v[t + 1], par)
