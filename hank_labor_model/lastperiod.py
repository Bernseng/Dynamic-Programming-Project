from numba import njit
from consav.grids import nonlinspace_jit
import utility
import numpy as np

@njit
def last_period(par,sol):

    m = sol.m
    c = sol.c
    l = sol.l
    inv_v = sol.inv_v

    # last period consume all       
    m[-1,:] = nonlinspace_jit(0,par.a_max,par.Na+1,par.m_phi)
    # c[-1,:] = sol.m[-1,:]
    c[-1,:] = m[-1,:]
    # l[-1,:] = utility.foc_l(sol.c[-1,:],par)
    l[-1,:] = np.zeros_like(sol.c)
    for i in range(1, par.Na+1):
        c_now = c[-1, i]
        l_now = l[-1, i]
        inv_v[-1,i] = 1.0/utility.utility(par, c_now, l_now)
        # inv_v[-1,1:] = 1.0/utility.utility(par,sol.c[-1,1:],sol.l[-1,1:])

    inv_v[-1,0] = 0

    print(sol.m.shape, sol.c.shape, sol.l.shape, sol.inv_v.shape)