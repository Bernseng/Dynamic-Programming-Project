import numpy as np
import numba as nb

from consav.grids import equilogspace
from consav.quadrature import log_normal_gauss_hermite
from consav.markov import choice, find_ergodic, log_rouwenhorst
from consav.linear_interp import interp_1d_vec
from EconModel import EconModelClass, jit

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,ell,l,u,chi):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # Loop over fixed states
    for i_fix in nb.prange(par.Nfix):
        
        zeta = par.zeta_grid[i_fix]
        varphi = par.varphi_grid[i_fix]
            
        # a. solve step
        for i_z in range(par.Nz):
        
            # prepare
            z = par.z_grid[i_z]
            wt = (1.0 - par.tau_l)*w*z 
            rt=(1.0 - par.tau_a)*r
            fac = ((wt*zeta)/varphi)**(1.0/par.nu) 
            
            # use FOCs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1.0/par.sigma) #FOC c
            ell_endo = fac*(c_endo)**(-par.sigma/par.nu) #FOC l

            # interpolation 
            m_endo = c_endo + par.a_grid - wt*ell_endo - chi
            m_exo = (1 + rt)*par.a_grid
            
            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])

            #Calculating savings
            a[i_fix,i_z,:] = m_exo - c[i_fix,i_z,:] + wt*ell[i_fix,i_z,:] + chi
            l[i_fix,i_z,:] = ell[i_fix,i_z,:]*z

            # Refinement of borrowing constraint
            for i_a in range(par.Na):
                
                # If borrowing constraint is violated
                if a[i_fix,i_z,i_a] < 0.0:

                    # Set to borrowing constraint
                    a[i_fix,i_z,i_a] = 0.0 
                    
                    # Solve FOC for ell
                    elli = ell[i_fix,i_z,i_a]

                    it = 0
                    while True:

                        ci = (1.0+rt)*par.a_grid[i_a] + wt*elli + chi
                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < par.toll_ell:
                            break
                        else:
                            derror = 1.0 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu - 1.0)*wt
                            elli = elli - error/derror

                        it += 1
                        if it > par.max_iter_ell: raise ValueError('too many iterations')

                        # Save
                        c[i_fix,i_z,i_a] = ci
                        ell[i_fix,i_z,i_a] = elli
                        l[i_fix,i_z,i_a] = z*elli
                    
        # b. expectation step
        v_a = (1.0+rt)*c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

        #Calculating utility
        u[i_fix,:,:] = c[i_fix]**(1-par.sigma)/(1-par.sigma) - par.varphi_grid[i_fix]*ell[i_fix]**(1+par.nu) / (1+par.nu)