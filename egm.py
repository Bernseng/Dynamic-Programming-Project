import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d_vec
import utility

##################
# solution - egm #
##################    

@njit(parallel=True)
def solve_hh_backwards_egm(par,vbeg_plus,vbeg,c,l,a,u,mpc):
    """ solve backwards with v_plus from previous iteration """

    for i_z in nb.prange(par.Nz):
        
        # prepare
        z = par.z_grid[i_z]
        w = (1-par.tau)*par.w*z
        
        # consumption and labor function
        fac = (w/par.varphi)**(1.0/par.nu)
        c_vec = (par.beta*vbeg_plus[i_z,:])**(-1.0/par.sigma) #FOC c
        l_vec = fac*(c_vec)**(-par.sigma/par.nu) #FOC l

        m_endo = par.a_grid+c_vec - w*l_vec
        m_exo = (1+par.r)*par.a_grid

        # interpolate
        interp_1d_vec(m_endo,c_vec,m_exo,c[i_z,:])
        interp_1d_vec(m_endo,l_vec,m_exo,l[i_z,:])

        c_plus = np.empty_like(m_exo)
        c_minus = np.empty_like(m_exo)
        
        interp_1d_vec(m_endo,c_vec,m_exo+par.delta_m,c_plus)
        interp_1d_vec(m_endo,c_vec,m_exo,c_minus)

        mpc[i_z,:] = (c_plus-c_minus)/(par.delta_m)

        # calculating savings
        a[i_z,:] = m_exo + w*l[i_z,:] - c[i_z,:]

        # borrowing contraint
        for i_a_lag in range(par.Na):
         
            # If borrowing constraint is violated
            if a[i_z,i_a_lag] < 0.0:

                # Set to borrowing constraint
                a[i_z,i_a_lag] = 0.0 
                
                # Solve FOC for ell
                ell = l[i_z,i_a_lag] 
                
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

    # expectation steps
    v_a = (1.0+par.r)*c[:]**(-par.sigma)
    vbeg[:] = par.z_trans@v_a

    # calculating utility
    u[i_z, :] = utility.func(c[i_z,:], l[i_z,:], par)


@njit(parallel=True)
def solve_hh_backwards_egm_exo(par,vbeg_plus,vbeg,c,l,a,u,mpc):
    """ solve backwards with v_plus from previous iteration """

    for i_z in nb.prange(par.Nz):
        
        # prepare
        z = par.z_grid[i_z]
        w = (1-par.tau)*par.w*z
        income = l*w
        
        # consumption function
        c_vec = (par.beta*vbeg_plus[i_z])**(-1.0/par.sigma) #FOC c

        m_endo = par.a_grid+c_vec
        m_exo = (1+par.r)*par.a_grid + income

        # interpolate
        interp_1d_vec(m_endo,par.a_grid,m_exo,a[i_z])

        c_plus = np.empty_like(m_exo)
        c_minus = np.empty_like(m_exo)
        interp_1d_vec(m_endo, c_vec, m_exo + par.delta_m, c_plus)
        interp_1d_vec(m_endo, c_vec, m_exo, c_minus)
        mpc[i_z, :] = (c_plus - c_minus) / (par.delta_m)

        # calculating savings
        a[i_z,:] = np.fmax(a[i_z,:],0.0)
        c[i_z] = m_exo - a[i_z]

    # expectation steps
    v_a = (1.0+par.r)*c[:]**(-par.sigma)
    vbeg[:] = par.z_trans@v_a

    # calculating utility
    u[:] = utility.func(c,l,par)