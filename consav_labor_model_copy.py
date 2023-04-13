import time

import numba as nb
import numpy as np
import quantecon as qe
from consav.grids import equilogspace
from consav.linear_interp import binary_search, interp_1d
from consav.markov import choice, find_ergodic, log_rouwenhorst
from consav.misc import elapsed
from consav.quadrature import log_normal_gauss_hermite
from EconModel import EconModelClass, jit


class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

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

        # labor supply
        par.nu = 2.0  # inverse Frisch elasticity of labor supply
        par.tau_l = 0.0  # labor income tax
        par.tau_a = 0.0  # capital income tax

        # saving
        par.r = 0.02 # interest rate
        par.b = -0.10 # borrowing constraint relative to wage

        # grid
        par.a_max = 100.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # simulation
        par.simT = 500 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # tolerances
        par.max_iter_solve = 10_000 # maximum number of iterations
        par.tol_solve = 1e-8 # tolerance when solving

    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim
        
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
        sol.c = np.zeros((par.Nz, par.Na))
        sol.a = np.zeros((par.Nz, par.Na))
        sol.ell = np.zeros((par.Nz, par.Na))  # labor supply
        sol.vbeg = np.zeros((par.Nz, par.Na))

        # hist
        sol.pol_indices = np.zeros((par.Nz,par.Na),dtype=np.int_)
        sol.pol_weights = np.zeros((par.Nz,par.Na))


    def solve(self,do_print=True,algo='vfi'):
        """ solve model using value function iteration or egm """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # define the utility function
            def utility(c, ell):
                return (c**(1-par.sigma) - 1)/(1-par.sigma) - par.nu*(ell**(1+par.tau_l))/(1+par.tau_l)

            # time loop
            it = 0
            while True:

                t0_it = time.time()

                # a. next-period value function
                if it == 0: # guess on consuming everything
                    
                    m_plus = (1+par.r)*par.a_grid[np.newaxis,:] + par.w*par.z_grid[:,np.newaxis]
                    c_plus_max = m_plus - par.w*par.b
                    c_plus = 0.99*c_plus_max # arbitary factor
                    ell_plus = np.ones_like(c_plus)  # initial guess for labor supply
                    v_plus = utility(c_plus, ell_plus)
                    vbeg_plus = par.z_trans@v_plus

                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()
                    ell_plus = sol.ell.copy()

                # b. solve this period
                if algo == 'vfi':
                    solve_hh_backwards_vfi(par,vbeg_plus,c_plus,ell_plus,sol.vbeg,sol.c,sol.ell,sol.a)  
                    max_abs_diff = np.max(np.abs(sol.vbeg-vbeg_plus))
                elif algo == 'egm':
                    solve_hh_backwards_egm(par,c_plus,ell_plus,sol.c,sol.ell,sol.a)
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

    def prepare_simulate(self,algo='mc',do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim

        if algo == 'mc':

            sim.a_ini[:] = 0.0
            sim.p_z_ini[:] = np.random.uniform(size=(par.simN,))
            sim.p_z[:,:] = np.random.uniform(size=(par.simT,par.simN))

        elif algo == 'hist':

            sim.Dbeg[0,:,0] = par.z_ergodic
            sim.Dbeg_[:,0] = par.z_ergodic

        else:
            
            raise NotImplementedError

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')

    def simulate(self,algo='mc',do_print=True):
        """ simulate model """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            # prepare
            if algo == 'hist': find_i_and_w(par,sol)

            # time loop
            for t in range(par.simT):
                
                if algo == 'mc':
                    simulate_forwards_mc(t,par,sim,sol)
                elif algo == 'hist':
                    sim.D[t] = par.z_trans.T@sim.Dbeg[t]
                    if t == par.simT-1: continue
                    simulate_hh_forwards_choice(par,sol,sim.D[t],sim.Dbeg[t+1])
                else:
                    raise NotImplementedError

        if do_print: print(f'model simulated in {elapsed(t0)} secs')
            
    def simulate_hist_alt(self,do_print=True):
        """ simulate model """

        t0 = time.time()


        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            Dbeg = sim.Dbeg_
            D = sim.D_

            # a. prepare
            find_i_and_w(par,sol)

            # b. iterate
            it = 0 
            while True:

                Dbeg_old = Dbeg.copy()
                simulate_hh_forwards_stochastic(par,Dbeg,D)
                simulate_hh_forwards_choice(par,sol,D,Dbeg)

                max_abs_diff = np.max(np.abs(Dbeg-Dbeg_old))
                if max_abs_diff < par.tol_simulate: 
                    Dbeg[:,:] = Dbeg_old
                    break

                it += 1
                if it > par.max_iter_simulate: raise ValueError('too many iterations in simulate()')

        if do_print: 
            print(f'model simulated in {elapsed(t0)} [{it} iterations]')



##################
# solution - vfi #
##################

@nb.njit
def value_of_choice(c_ell, par, i_z, m, vbeg_plus):
    """ value of choice for use in vfi """

    c, ell = c_ell

    # a. utility
    utility = (c**(1-par.sigma))/(1-par.sigma) - par.nu*(ell**(1+par.tau_l))/(1+par.tau_l)

    # b. end-of-period assets
    a = m - c - ell * par.w * par.z_grid[i_z]

    # c. continuation value     
    vbeg_plus_interp = interp_1d(par.a_grid, vbeg_plus[i_z, :], a)

    # d. total value
    value = utility + par.beta * vbeg_plus_interp
    return value

@nb.njit(parallel=True)
def solve_hh_backwards_vfi(par, vbeg_plus, c_plus, ell_plus, vbeg, c, ell, a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):

            # i. cash-on-hand and maximum consumption
            m = (1 + par.r) * par.a_grid[i_a_lag] + par.w * par.z_grid[i_z]
            c_max = m - par.b * par.w

            # ii. initial consumption and labor supply, and bounds
            c_ell_guess = np.zeros(2)
            bounds = np.zeros((2, 2))

            c_ell_guess[0] = c_plus[i_z, i_a_lag]
            c_ell_guess[1] = ell_plus[i_z, i_a_lag]
            bounds[0, 0] = 1e-8
            bounds[0, 1] = c_max
            bounds[1, 0] = 1e-8
            bounds[1, 1] = 1

            # iii. optimize
            results = qe.optimize.nelder_mead(value_of_choice,
                                              c_ell_guess,
                                              bounds=bounds,
                                              args=(par, i_z, m, vbeg_plus))

            # iv. save
            c[i_z, i_a_lag] = results.x[0]
            ell[i_z, i_a_lag] = results.x[1]
            a[i_z, i_a_lag] = m - c[i_z, i_a_lag] - ell[i_z, i_a_lag] * par.w * par.z_grid[i_z]
            v[i_z, i_a_lag] = results.fun  # convert to maximum

    # b. expectation step
    vbeg[:, :] = par.z_trans @ v



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

