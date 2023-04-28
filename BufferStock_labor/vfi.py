<<<<<<< HEAD
import numpy as np
from numba import njit, prange

# consav
=======
from numba import njit, prange
import numpy as np

>>>>>>> work
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D

import utility

<<<<<<< HEAD
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
        y_plus = p_plus*xi
        m_plus = par.R*a + y_plus
        
        # iii. weight
        weight = psi_w*xi_w
        
        # iv. interpolate
        w += weight*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,v_plus,p_plus,m_plus)
    
    # c. total value
    value_of_choice = utility.func(c, l, par) + w

    return -value_of_choice # we are minimizing

# b. solve bellman equation        
=======
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
>>>>>>> work
@njit(parallel=True)
def solve_bellman(t, sol, par):
    """solve bellman equation using vfi"""

    # unpack (helps numba optimize)
    c = sol.c[t]
<<<<<<< HEAD
    l = sol.l[t]
    v = sol.v[t]

    # loop over outer states
    for ip in prange(par.Np): # in parallel

        # a. permanent income
        p = par.grid_p[ip]

        # d. loop over cash-on-hand
        for im in range(par.Nm):
            
            # a. cash-on-hand
            m = par.grid_m[im]

            # b. optimal choice
            for il in range(par.Nl):
                l_val = par.grid_l[il]
                c_low = np.fmin(m/2, 1e-8)
                c_high = m
                c[ip, im, il], l[ip, im, il] = golden_section_search.optimizer(obj_bellman, c_low, c_high, l_val, args=(p, m, sol.v[t+1], par), tol=par.tol)

                # note: the above finds the minimum of obj_bellman in range [c_low,c_high] with a tolerance of par.tol
                # and arguments (except for c) as specified 

                # c. optimal value
                v[ip, im, il] = -obj_bellman(c[ip, im, il], l[ip, im, il], p, m, sol.v[t+1], par)
=======
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
>>>>>>> work
