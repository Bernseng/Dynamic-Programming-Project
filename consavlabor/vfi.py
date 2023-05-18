import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d
from utility import func
import quantecon as qe


##################
# solution - vfi #
##################

@nb.njit
def value_of_choice(par,c,l,i_z,m,vbeg_plus):
    """ value of choice for use in vfi """

    # a. utility
    utility = func(par,c[0],l[0])

    # b. end-of-period assets
    # a = m - c[0]
    a = m + par.w*l[0] - c[0]

    # c. continuation value     
    vbeg_plus_interp = interp_1d(par.a_grid,vbeg_plus[i_z,:],a)

    # d. total value
    value = utility + par.beta*vbeg_plus_interp

    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,vbeg_plus,c_plus,vbeg,c,l,a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_z in nb.prange(par.Nz):

        # prepare
        z = par.z_grid[i_z]
        w = (1-par.tau)*par.w*z

        for i_a_lag in nb.prange(par.Na):

            # i. cash-on-hand and maximum consumption
            # m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            m = (1+par.r)*par.a_grid[i_a_lag] + w*l[i_z,i_a_lag]

            c_max = m

            # ii. initial consumption and bounds
            # c_guess = np.zeros((1,1))
            # l_guess = np.zeros((1,1))
            # bounds = np.zeros((1,2))

            # c_guess[0] = c_plus[i_z,i_a_lag]
            # l_guess[0] = 0.0
            # bounds[0,0] = 1e-8 
            # bounds[0,1] = c_max
            # ii. initial consumption and labor and bounds
            c_guess = np.zeros(2)
            l_guess = np.zeros(2)
            bounds = np.zeros((2,2))

            c_guess[0] = c_plus[i_z,i_a_lag]
            l_guess[0] = par.w*par.z_grid[i_a_lag] # Initial guess for labor
            bounds[0,0] = 1e-8 
            bounds[0,1] = c_max
            bounds[1,0] = 0.0 # Assuming labor can't be negative
            bounds[1,1] = 1.0 # Assuming labor can't exceed 1


            # iii. optimize
            # results = qe.optimize.nelder_mead(value_of_choice,
            #     c_guess, 
            #     bounds=bounds,
            #     args=(par,i_z,m,vbeg_plus))

            # iii. optimize
            results = qe.optimize.nelder_mead(value_of_choice,
                [c_guess, l_guess],
                bounds=bounds,
                args=(par,i_z,m,vbeg_plus))


            # iv. save
            # c[i_z,i_a_lag] = results.x[0]
            # a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
            # v[i_z,i_a_lag] = results.fun # convert to maximum

            # iv. save
            c[i_z,i_a_lag] = results.x[0]
            l[i_z,i_a_lag] = results.x[1]
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
            v[i_z,i_a_lag] = -results.fun # convert to maximum


    # b. expectation step
    vbeg[:,:] = par.z_trans@v