import numpy as np
from numba import njit, prange
from consav import linear_interp
from scipy.optimize import minimize_scalar, minimize

import utility

@njit
def obj_bellman(c_l, m, interp_w, par):
    """ evaluate bellman equation """

    c, l = c_l

    # a. end-of-period assets
    a = m - c - par.w * l

    # b. continuation value
    w = linear_interp.interp_1d(par.grid_a, interp_w, a)

    # c. total value
    value_of_choice = utility.func(c, l, par) + w

    return -value_of_choice

@njit(parallel=True)
def solve_bellman(t, sol, par):
    """solve bellman equation using nvfi"""

    v = sol.v[t]
    c = sol.c[t]
    l = sol.l[t]

    # loop over outer states
    for ip in prange(par.Np):

        # loop over cash-on-hand
        for im in range(par.Nm):
            
            # a. cash-on-hand
            m = par.grid_m[im]

            # b. optimal choice
            bounds = [(1e-8, m), (1e-8, 1)]  # bounds for c and l
            res = minimize(obj_bellman, [m / 2, 0.5], args=(m, sol.w[ip], par), bounds=bounds, tol=par.tol)

            c[ip, im] = res.x[0]
            l[ip, im] = res.x[1]

            # c. optimal value
            v[ip, im] = -obj_bellman(res.x, m, sol.w[ip], par)
