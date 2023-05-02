import numpy as np
from numba import njit, prange, jit

 # consav
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D

import utility

# a. define objective function
@njit
def obj_bellman(c, l, p, m, v_plus, par):
    """ evaluate bellman equation """

    # a. end-of-period assets
    a = m-c
    
    # b. continuation value
    w = 0
    for ishock in range(par.Nshocks):
            
        # i. shocks
        psi = par.psi[ishock]
        psi_w = par.psi_w[ishock]
        xi = par.xi[ishock]
        xi_w = par.xi_w[ishock]

        # ii. next-period states
        p_plus = p*psi
        y_plus = p_plus*xi*par.w*l
        m_plus = par.R*a + y_plus
        
        # iii. weight
        weight = psi_w*xi_w
        
        # iv. interpolate
        w += weight*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,v_plus,p_plus,m_plus)
    
    # c. total value
    value_of_choice = utility.func(c,l,par) + w

    return -value_of_choice # we are minimizing

# b. solve bellman equation        
@jit(parallel=True)
def solve_bellman(t, sol, par):
    """solve bellman equation using vfi with FOCs for consumption and labor"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    l = sol.l[t]
    v = sol.v[t]

    # loop over outer states (permanent income)
    for ip in prange(par.Np):

        # a. permanent income
        p = par.grid_p[ip]

        # b. loop over cash-on-hand
        for im in range(par.Nm):

            # i. cash-on-hand
            m = par.grid_m[im]

            # ii. use FOCs to find optimal consumption and labor supply
            fac = ((par.w*p)/par.varphi)**(1.0/par.nu)
            c_endo = (par.beta*sol.v[t+1,ip,:])**(-1.0/par.rho)
            # c_endo = np.power(par.beta*sol.v[t+1,ip,:],(-1.0/par.rho)) # c_endo can also be calculated using the 'np.power' operator instead
            l_endo = fac*(c_endo)**(-par.rho/par.nu)

            # iii. interpolate consumption and labor supply
            a_endo = m-c_endo+par.w*p*l_endo
            a_exo = (1+par.R)*par.grid_m

            c[ip, im] = np.interp(a_exo,a_endo,c_endo)
            l[ip, im] = np.interp(a_exo,a_endo,l_endo)

            # iv. optimal value
            v[ip,im]=(c[ip,im]**(1-par.rho))/(1-par.rho)-par.varphi*(l[ip,im]**(1+par.nu))/(1+par.nu)+par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,sol.v[t+1],p,a_exo)
