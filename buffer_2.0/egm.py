import numpy as np
from numba import njit, prange, jit

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

# @njit(parallel=True)
# def solve_bellman(t,sol,par):
#     """solve the bellman equation using the endogenous grid method"""

#     # unpack (helps numba optimize)
#     c = sol.c[t]

#     for ip in prange(par.Np): # in parallel
        
#         # a. temporary container (local to each thread)
#         m_temp = np.zeros(par.Na+1) # m_temp[0] = 0
#         c_temp = np.zeros(par.Na+1) # c_temp[0] = 0

#         # b. invert Euler equation
#         for ia in range(par.Na):
#             c_temp[ia+1] = utility.inv_marg_func(sol.q[ip,ia],par)
#             m_temp[ia+1] = par.grid_a[ia] + c_temp[ia+1]
        
#         # b. re-interpolate consumption to common grid
#         if par.do_simple_w: # use an explicit loop
#             for im in range(par.Nm):
#                 c[ip,im] = linear_interp.interp_1d(m_temp,c_temp,par.grid_m[im])
#         else: # use a vectorized call (assumming par.grid_m is monotone)
#             linear_interp.interp_1d_vec_mon_noprep(m_temp,c_temp,par.grid_m,c[ip,:])


@njit(parallel=True)
def solve_bellman(t, sol, par):
    """solve the bellman equation using the endogenous grid method with endogenous labor supply"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    l = sol.l[t]

    for ip in prange(par.Np):  # in parallel

        # a. permanent income
        # p = par.grid_p[ip]

        # b. temporary container (local to each thread)
        m_temp = np.zeros(par.Na + 1)  # m_temp[0] = 0
        c_temp = np.zeros(par.Na + 1)  # c_temp[0] = 0
        l_temp = np.zeros(par.Na + 1)  # l_temp[0] = 0

        # c. invert Euler equation
        for ia in range(par.Na):
            c_temp[ia + 1] = utility.inv_marg_func_c(sol.q[ip, ia], par)
            fac = ((par.w 
                    # * p
                    ) / par.varphi) ** (1.0 / par.nu)

            l_temp[ia + 1] = fac * (c_temp[ia + 1]) ** (-par.rho / par.nu)
            m_temp[ia + 1] = (par.grid_a[ia] + c_temp[ia + 1] - par.w * 
                              #p * 
                              l_temp[ia + 1])

        # d. re-interpolate consumption and labor supply to common grid
        if par.do_simple_w:  # use an explicit loop
            for im in range(par.Nm):
                c[ip, im] = linear_interp.interp_1d(m_temp, c_temp, par.grid_m[im])
                l[ip, im] = linear_interp.interp_1d(m_temp, l_temp, par.grid_m[im])
        else:  # use a vectorized call (assuming par.grid_m is monotone)
            linear_interp.interp_1d_vec_mon_noprep(m_temp, c_temp, par.grid_m, c[ip, :])
            linear_interp.interp_1d_vec_mon_noprep(m_temp, l_temp, par.grid_m, l[ip, :])
