import time

import numpy as np
import numba as nb

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic, choice
from consav.quadrature import log_normal_gauss_hermite
from consav.linear_interp import binary_search, interp_1d
from consav.misc import elapsed

class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """
        self.name = 'ConSavModel_with_Labor'
        pass

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient

        # income
        par.w = 1.0 # wage level
        
        par.rho_zt = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock
        par.Nzt = 5 # number of grid points for zt
        
        par.sigma_xi = 0.10 # std. of transitory shock
        par.Nxi = 2 # number of grid points for xi

        # saving
        par.r = 0.02 # interest rate
        par.b = -0.10 # borrowing constraint relative to wage

        # grid
        par.a_max = 100.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # labor supply
        par.nu = 2.0 # inverse Frisch elasticity of labor supply
        par.tau_l = 0.0 # labor income tax
        par.tau_a = 0.0 # capital income tax

        # tolerance 
        par.tol_solve = 1e-8 # tolerance when solving
        par.max_iter_solve = 10_000 # maximum number of iterations


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        
        # a. transition matrix
        
        # persistent
        _out = log_rouwenhorst(par.rho_zt,par.sigma_psi,par.Nzt)
        par.zt_grid,par.zt_trans,par.zt_ergodic,par.zt_trans_cumsum,par.zt_ergodic_cumsum = _out
        
        # transitory
        if par.sigma_xi > 0 and par.Nxi > 1:
            par.xi_grid,par.xi_weights = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)
            par.xi_trans = np.broadcast_to(par.xi_weights,(par.Nxi,par.Nxi))
        else:
            par.xi_grid = np.ones(1)
            par.xi_weights = np.ones(1)
            par.xi_trans = np.ones((1,1))

        # combined
        par.Nz = par.Nxi*par.Nzt
        par.z_grid = np.repeat(par.xi_grid,par.Nzt)*np.tile(par.zt_grid,par.Nxi)
        par.z_trans = np.kron(par.xi_trans,par.zt_trans)
        par.z_trans_cumsum = np.cumsum(par.z_trans,axis=1)
        par.z_ergodic = find_ergodic(par.z_trans)
        par.z_ergodic_cumsum = np.cumsum(par.z_ergodic)
        par.z_trans_T = par.z_trans.T

        # b. asset grid
        assert par.b <= 0.0, f'{par.b = :.1f} > 0, should be negative'
        b_min = -par.z_grid.min()/par.r
        if par.b < b_min:
            print(f'parameter changed: {par.b = :.1f} -> {b_min = :.1f}') 
            par.b = b_min + 1e-8

        par.a_grid = par.w*equilogspace(par.b,par.a_max,par.Na)

        # c. solution arrays
        sol.c = np.zeros((par.Nz,par.Na))
        sol.ell = np.zeros((par.Nz, par.Na)) # labor supply array
        sol.a = np.zeros((par.Nz,par.Na))
        sol.vbeg = np.zeros((par.Nz,par.Na))

        # hist
        sol.pol_indices = np.zeros((par.Nz,par.Na),dtype=np.int_)
        sol.pol_weights = np.zeros((par.Nz,par.Na))

    def solve(self,do_print=True,algo='vfi'):
        """ solve model using value function iteration or egm """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # time loop
            it = 0
            while True:
                
                t0_it = time.time()

                # a. next-period value function
                if it == 0: # guess on consuming everything

                    m_plus = (1+par.r)*par.a_grid[np.newaxis,:] + par.w*par.z_grid[:,np.newaxis]*h_plus
                    c_plus_max = m_plus - par.w*par.b
                    c_plus = 0.99*c_plus_max # arbitary factor
                    v_plus = c_plus**(1-par.sigma)/(1-par.sigma)
                    vbeg_plus = par.z_trans@v_plus
                    
                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()
                    h_plus = sol.h.copy()

                # b. solve this period
                if algo == 'vfi':
                    solve_hh_backwards_vfi(par, vbeg_plus, c_plus, h_plus, sol.vbeg, sol.c, sol.h, sol.a)  
                    max_abs_diff = np.max(np.abs(sol.vbeg-vbeg_plus))
                elif algo == 'egm':
                    solve_hh_backwards_egm(par,c_plus,sol.c,sol.h,sol.a)
                    max_abs_diff = np.max(np.abs(sol.c-c_plus))
                else:
                    raise NotImplementedError

                # c. check convergence
                converged = max_abs_diff < par.tol_solve
                
                # d. break
                if do_print and (converged or it < 10 or it%100 == 0):
                    print(f'iteration {it:4d} solved in {elapsed(t0_it):10s}',end='')              
                    print(f' [max abs. diff. {max_abs_diff:5.2e}]')

                if converged: break

                it += 1
                if it > par.max_iter_solve: raise ValueError('too many iterations in solve()')
        
        if do_print: print(f'model solved in {elapsed(t0)}')              

##################
# solution - vfi #
##################

@nb.njit
def value_of_choice(c,h,par,i_z,m,vbeg_plus):
    """ value of choice for use in vfi with endogenous labor supply """

    # a. utility
    utility = c**(1-par.sigma)/(1-par.sigma) - h **(1+par.nu)/(1+par.nu)

    # b. end-of-period assets
    a = m - c + par.w * par.z_grid[i_z] * h # include labor income

    # c. continuation value     
    vbeg_plus_interp = interp_1d(par.a_grid,vbeg_plus[i_z,:],a)

    # d. total value
    value = utility + par.beta*vbeg_plus_interp
    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,vbeg_plus,c_plus,h_plus,vbeg,c,h,a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):

            # i. cash-on-hand and maximum consumption
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z] * h
            c_max = m - par.b*par.w

            # ii. initial consumption, labor and bounds
            ch_guess = np.zeros((2,))
            ch_bounds = np.zeros((2,2))

            ch_guess[0] = c_plus[i_z,i_a_lag]
            ch_guess[1] = h_plus[i_z,i_a_lag]
            ch_bounds[0,0] = 1e-8
            ch_bounds[0,1] = c_max
            ch_bounds[1,0] = 1e-8
            ch_bounds[1,1] = 1.0

            # iii. optimize
            results = qe.optimize.nelder_mead(value_of_choice,
                ch_guess, 
                bounds=ch_bounds,
                args=(par,i_z,m,vbeg_plus))

            # iv. save
            c[i_z,i_a_lag] = results.x[0]
            h[i_z, i_a_lag] = results.x[1]
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag] + par.w * par.z_grid[i_z] * h[i_z, i_a_lag]
            v[i_z,i_a_lag] = results.fun # convert to maximum

    # b. expectation step
    vbeg[:,:] = par.z_trans@v

##################
# solution - egm #
##################

@nb.njit(parallel=True)
def solve_hh_backwards_egm(par,c_plus,c,a):
    """ solve backwards with c_plus from previous iteration """

    for i_z in nb.prange(par.Nz):

        # a. post-decision marginal value of cash
        q_vec = np.zeros(par.Na)
        for i_z_plus in range(par.Nz):
            q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_z_plus,:]**(-par.sigma)
        
        # b. implied consumption function
        c_vec = (par.beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
        m_vec = par.a_grid+c_vec

        # c. interpolate from (m,c) to (a_lag,c)
        for i_a_lag in range(par.Na):
            
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            
            if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                c[i_z,i_a_lag] = m - par.b*par.w
                a[i_z,i_a_lag] = par.b*par.w
            else: # unconstrained
                c[i_z,i_a_lag] = interp_1d(m_vec,c_vec,m) 
                a[i_z,i_a_lag] = m-c[i_z,i_a_lag] 
