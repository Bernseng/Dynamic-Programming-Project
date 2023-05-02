from numba import njit, prange
import numpy as np

# local modules
import utility

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack (helps numba optimize)
    v = sol.v[t]
    c = sol.c[t]
    l = sol.l[t]

    # loop over states
    for ip in prange(par.Np): # in parallel
        for im in range(par.Nm):
            
            # a. states
            # _p = par.grid_p[ip]
            m = par.grid_m[im]
            ell = np.zeros_like(m) # added zero grid for labor supply

            # b. optimal choice (consume everything)
            c[ip,im] = m
            l[ip,im] = ell # added labor solution in last period
            # c. optimal value
            v[ip,im] = utility.func(c[ip,im],l[ip,im],par)