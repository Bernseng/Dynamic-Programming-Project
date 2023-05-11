import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d
import utility
import quantecon as qe


##################
# solution - vfi #
##################

@nb.njit
def value_of_choice(c,par,i_z,m,vbeg_plus):
    """ value of choice for use in vfi """

    # a. utility
    utility = c[0]**(1-par.sigma)/(1-par.sigma)

    # b. end-of-period assets
    a = m - c[0]

    # c. continuation value     
    vbeg_plus_interp = interp_1d(par.a_grid,vbeg_plus[i_z,:],a)

    # d. total value
    value = utility + par.beta*vbeg_plus_interp
    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,vbeg_plus,c_plus,vbeg,c,a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):

            # i. cash-on-hand and maximum consumption
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            c_max = m - par.b*par.w

            # ii. initial consumption and bounds
            c_guess = np.zeros((1,1))
            bounds = np.zeros((1,2))

            c_guess[0] = c_plus[i_z,i_a_lag]
            bounds[0,0] = 1e-8 
            bounds[0,1] = c_max

            # iii. optimize
            results = qe.optimize.nelder_mead(value_of_choice,
                c_guess, 
                bounds=bounds,
                args=(par,i_z,m,vbeg_plus))

            # iv. save
            c[i_z,i_a_lag] = results.x[0]
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
            v[i_z,i_a_lag] = results.fun # convert to maximum

    # b. expectation step
    vbeg[:,:] = par.z_trans@v