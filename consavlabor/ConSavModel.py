import time
import numpy as np
import numba as nb
import random

# set seed
random.seed(0)

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.markov import log_rouwenhorst, find_ergodic, choice
from consav.grids import equilogspace
from consav.quadrature import log_normal_gauss_hermite
from consav.linear_interp import binary_search, interp_1d, interp_1d_vec
from consav.misc import elapsed

import egm
from sim import simulate_forwards_mc

class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient consumption
        par.nu = 2.0 # CRRA coefficient labor
        par.varphi = 1.0 # disutility of labor
        
        # income
        par.w = 1.0 # wage level
        par.rho_zt = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock
        par.Nzt = 5 # number of grid points for zt
        par.sigma_xi = 0.10 # std. of transitory shock
        par.Nxi = 2 # number of grid points for xi
        par.delta_m = 0.01 # change in m to calculate mpc
        
        # taxes
        par.tau = 0.28
        
        # saving
        par.r = 0.015 # interest rate
        par.b = 0.0 # minimum for a

        # grid
        par.a_max = 100.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # simulation
        par.simT = 500 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # tolerances
        par.max_iter_solve = 10_000 # maximum number of iterations
        par.max_iter_simulate = 10_000 # maximum number of iterations
        par.tol_solve = 1e-8 # tolerance when solving
        par.tol_simulate = 1e-8 # tolerance when simulating
        par.tol_l = 1e-8 # tolerance for l
        par.max_iter_l = 30

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

        par.a_grid = par.w*equilogspace(par.b,par.a_max,par.Na)

        # c. solution arrays
        sol.c = np.zeros((par.Nz,par.Na))
        sol.l = np.zeros((par.Nz,par.Na))
        sol.ell = 1.0 # exogenous labor supply
        sol.a = np.zeros((par.Nz,par.Na))
        sol.u = np.zeros((par.Nz,par.Na))
        sol.vbeg = np.zeros((par.Nz,par.Na))
        sol.mpc = np.zeros((par.Nz,par.Na))

        # d. simulation arrays
        sim.a_ini = np.zeros((par.simN,))
        sim.p_z_ini = np.zeros((par.simN,))
        sim.c = np.zeros((par.simT,par.simN))
        sim.l = np.zeros((par.simT,par.simN))
        sim.ell = 1.0*np.ones_like(sim.l)
        sim.a = np.zeros((par.simT,par.simN))
        sim.p_z = np.zeros((par.simT,par.simN))
        sim.i_z = np.zeros((par.simT,par.simN),dtype=np.int_)
        sim.mpc = np.zeros((par.simT, par.simN))

    def solve(self,do_print=True,algo='egm'):
        """ solve model using egm """

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
                    
                    ell = par.w*par.z_grid
                    m_plus = (1+par.r)*par.a_grid[np.newaxis,:] + (1-par.tau)*ell[:,np.newaxis]
                    c_plus = m_plus
                    v_plus = (1+par.r)*c_plus**(-par.sigma)
                    vbeg_plus = par.z_trans@v_plus

                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()

                # b. solve this period
                if algo == 'egm':
                    egm.solve_hh_backwards_egm(par,vbeg_plus,sol.vbeg,sol.c,sol.l,sol.a,sol.u,sol.mpc)
                    max_abs_diff = np.max(np.abs(sol.vbeg-vbeg_plus))
                elif algo == 'egm_exo':
                    egm.solve_hh_backwards_egm_exo(par,vbeg_plus,sol.vbeg,sol.c,sol.ell,sol.a,sol.u,sol.mpc)
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

    def prepare_simulate(self,do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim

        sim.a_ini[:] = 0.0
        sim.p_z_ini[:] = np.random.uniform(size=(par.simN,))
        sim.p_z[:,:] = np.random.uniform(size=(par.simT,par.simN))

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')

    def simulate(self,algo,do_print=True):
        """ simulate model """
        
        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            # prepare
            sim.a_ini[:] = 0.0
            sim.p_z_ini[:] = np.random.uniform(size=(par.simN,))
            sim.p_z[:,:] = np.random.uniform(size=(par.simT,par.simN))

            # time loop
            for t in range(par.simT):
                simulate_forwards_mc(t,par,sim,sol,algo)

        if do_print: print(f'model simulated in {elapsed(t0)} secs')
            