import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d_vec
import utility
from tools import generate_zeta

##################
# solution - egm #
##################    


@njit(parallel=True)
def solve_hh_backwards_egm(par,vbeg_plus,v_a,c_plus,ell,c,l,a,u):
    """ solve backwards with v_plus from previous iteration """

    for i_z in nb.prange(par.Nz):
        
        # prepare
        z = par.z_grid[i_z]
        w = (1-par.tau)*par.w*z

        # generate shocks
        # zeta = generate_zeta(par)

        # a. post-decision marginal value of cash
        q_vec = np.zeros(par.Na)
        for i_z_plus in range(par.Nz):
            q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_z_plus,:]**(-par.sigma)
        
        # b. implied consumption function
        fac = (w/par.varphi)**(1.0/par.nu)
        c_vec = (par.beta*q_vec)**(-1.0/par.sigma) #FOC c
        l_vec = fac*(c_vec)**(-par.sigma/par.nu) #FOC l

        m_endo = par.a_grid+c_vec - w*l_vec
        m_exo = (1+par.r)*par.a_grid
        # m_exo = (1+par.rh+zeta)*par.a_grid

        # interpolate
        interp_1d_vec(m_endo,c_vec,m_exo,c[i_z,:])
        interp_1d_vec(m_endo,l_vec,m_exo,l[i_z,:])

        # calculating savings
        # a[i_z,:] = (1+par.rh+zeta)*par.a_grid + w*l_vec - c[i_z,:]
        a[i_z,:] = m_exo + w*l_vec - c[i_z,:]


        # c. interpolate from (m,c) to (a_lag,c)
        for i_a_lag in range(par.Na):
         
            # If borrowing constraint is violated
            if a[i_z,i_a_lag] < 0.0:

                # Set to borrowing constraint
                a[i_z,i_a_lag] = 0.0 
                
                # Solve FOC for ell
                ell = l_vec[i_a_lag] 
                
                it = 0
                while True:

                    ci = (1.0+par.r)*par.a_grid[i_a_lag] + w*ell
                    error = ell - fac*ci**(-par.sigma/par.nu)
                    if np.abs(error) < par.tol_l:
                        break
                    else:
                        derror = 1.0 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1.0)*w
                        ell = ell - error/derror

                    it += 1
                    if it > par.max_iter_l: 
                        raise ValueError('too many iterations')

                    # Save
                    c[i_z,i_a_lag] = ci
                    l[i_z,i_a_lag] = ell

        # b. expectation steps
    v_a = (1.0+par.r)*c[:]**(-par.sigma)
    vbeg_plus = par.z_trans@v_a

    #Calculating utility
    u[i_z, :] = utility.func(c[i_z,:], l[i_z,:], par)