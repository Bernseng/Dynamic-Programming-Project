import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d_vec
import utility


##################
# solution - egm #
##################    

@njit(parallel=True)
def solve_hh_backwards_egm(par,vbeg_plus,c_plus,ell,c,l,a,u):
    """ solve backwards with v_plus from previous iteration """

    for i_z in nb.prange(par.Nz):

        # prepare
        # z = par.z_grid[i_z]

        # a. post-decision marginal value of cash
        q_vec = np.zeros(par.Na)
        for i_z_plus in range(par.Nz):
            q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_z_plus,:]**(-par.sigma)
        
        # print(f'q_vec = {q_vec}')
        # b. implied consumption function
        fac = (par.w/par.varphi)**(1.0/par.nu)
        c_vec = (par.beta*q_vec)**(-1.0/par.sigma) #FOC c
        l_vec = fac*(c_vec)**(-par.sigma/par.nu) #FOC l

        m_endo = par.a_grid+c_vec - par.w*l_vec 
        m_exo = (1 + par.r)*par.a_grid

        # interpolate
        interp_1d_vec(m_endo,c_vec,m_exo,c[i_z,:])
        interp_1d_vec(m_endo,l_vec,m_exo,l[i_z,:])

        # print(f'c = {c[i_z]}')
        # calculating savings
        a[i_z,:] = m_exo - c[i_z,:] + par.w*l_vec
        # [i_z,:]
        # l[i_z,:] = l_vec[i_z,:]*par.z_grid


        # c. interpolate from (m,c) to (a_lag,c)
        for i_a_lag in range(par.Na):
        
            # m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
                
                # if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                #     c[i_z,i_a_lag] = m - par.b*par.w
                #     a[i_z,i_a_lag] = par.b*par.w
                # else: # unconstrained
                #     c[i_z,i_a_lag] = interp_1d(m_vec,c_vec,m) 
                #     a[i_z,i_a_lag] = m-c[i_z,i_a_lag] 
            

            # If borrowing constraint is violated
            if a[i_z,i_a_lag] < 0.0:

                # Set to borrowing constraint
                a[i_z,i_a_lag] = 0.0 
                
                # Solve FOC for ell
                ell = l_vec[i_a_lag] # removed ,i_z

                it = 0
                while True:

                    ci = (1.0+par.r)*par.a_grid[i_a_lag] + par.w*ell
                    error = ell - fac*ci**(-par.sigma/par.nu)
                    if np.abs(error) < par.toll_l:
                        break
                    else:
                        derror = 1.0 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu - 1.0)*par.w
                        ell = ell - error/derror

                    it += 1
                    if it > par.max_iter_l: 
                        raise ValueError('too many iterations')

                    # Save
                    c[i_z,i_a_lag] = ci
                    l[i_z,i_a_lag] = ell
                    # l_vec[i_z,i_a_lag] = ell
                    # l[i_z,i_a_lag] = par.z_grid*ell

        # b. expectation steps
    v_a = (1.0+par.r)*c[:]**(-par.sigma)
    # v_a_matrix = np.repeat(v_a[np.newaxis, :], par.Nz, axis=0)
    vbeg_plus = par.z_trans @ v_a
    # vbeg_plus = par.z_trans@v_a

    #Calculating utility
    u[i_z, :] = utility.func(c[i_z, :], l[i_z, :], par)