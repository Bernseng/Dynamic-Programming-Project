import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

@njit(parallel=True)
def solve_bellman(t, sol, par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    l = sol.l[t]

    for ip in prange(par.Np): # in parallel
        
        # a. temporary container (local to each thread)
        m_temp = np.zeros((par.Na+1, par.Nl))
        c_temp = np.zeros((par.Na+1, par.Nl))
        l_temp = np.zeros((par.Na+1, par.Nl))

        # b. invert Euler equation
        for il in range(par.Nl):
            for ia in range(par.Na):
                c_temp[ia+1, il], l_temp[ia+1, il] = utility.inv_marg_func(sol.q[ip, ia, il], par)
                m_temp[ia+1, il] = par.grid_a[ia] + c_temp[ia+1, il]

        # b. re-interpolate consumption and labor to common grid
        for im in range(par.Nm):
            for il in range(par.Nl):
                c[ip, im, il] = linear_interp.interp_1d(m_temp[:, il], c_temp[:, il], par.grid_m[im])
                l[ip, im, il] = linear_interp.interp_1d(m_temp[:, il], l_temp[:, il], par.grid_m[im])
