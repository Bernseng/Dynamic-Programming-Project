# -*- coding: utf-8 -*-
"""BufferStockModel

Solves the Deaton-Carroll buffer-stock consumption model with either:

A. vfi: standard value function iteration
B. nvfi: nested value function iteration
C. egm: endogenous grid point method (also in C++)

"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav.grids import nonlinspace # grids
from consav.quadrature import create_PT_shocks # income shocks
from consav.misc import elapsed

# local modules
import utility
import last_period
<<<<<<< HEAD
import BufferStock_labor.post_decision as post_decision
import vfi
import nvfi
import BufferStock_labor.egm as egm
import simulate
import figs
=======
import post_decision
import vfi
import nvfi
import egm
import simulate
import figs
import tools
>>>>>>> work

############
# 2. model #
############

class BufferStockModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. list not-floats for safe type inference
<<<<<<< HEAD
        self.not_floats = ['T','Npsi','Nxi','Nm','Np','Na','Nl','do_print','do_simple_w','simT','simN','sim_seed','cppthreads','Nshocks']
=======
        self.not_floats = ['T','Npsi','Nz','Nm','Np','Na','Nl','do_print','do_simple_w','simT','simN','sim_seed','cppthreads','Nshocks']
>>>>>>> work

        # e. cpp
        self.cpp_filename = 'cppfuncs/egm.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        """ set baseline parameters """   

        par = self.par

        # a. solution method
<<<<<<< HEAD
        par.solmethod = 'nvfi'
=======
        par.solmethod = 'vfi'
>>>>>>> work
        
        # b. horizon
        par.T = 5
        
        # c. preferences
        par.beta = 0.96
        par.rho = 2.0 # if par.rho = 2 the type is incorrectly inferred as int (error rasied)
<<<<<<< HEAD
        par.chi = 1.0
        par.gamma = 2.0

        # d. returns and income
        par.R = 1.03
        par.sigma_psi = 0.1
        par.Npsi = 6
        par.sigma_xi = 0.1
        par.Nxi = 6
        par.pi = 0.1
        par.mu = 0.5
=======
        par.varphi = 1.0
        par.nu = 2.0

        # d. returns and income
        par.R = 1.03
        par.sigma_z = 0.1
        par.Nz = 6
        # par.sigma_z = 0.1
        # par.Nz = 6
        par.pi = 0.1
        par.mu = 0.5
        par.tau = 0.01
>>>>>>> work
        
        # e. grids (number of points)
        par.Nm = 600
        par.Np = 400
        par.Na = 800
<<<<<<< HEAD
        par.Nl = 100
=======
>>>>>>> work

        # f. misc
        par.tol = 1e-8
        par.do_print = True
        par.do_simple_w = False
        par.cppthreads = 1

        # g. simulation
        par.simT = par.T
        par.simN = 1000
        par.sim_seed = 1998
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. states (unequally spaced vectors of length Nm)
        par.grid_m = nonlinspace(1e-6, 20, par.Nm, 1.1)
<<<<<<< HEAD
        par.grid_p = nonlinspace(1e-4, 10, par.Np, 1.1)
        
        # b. post-decision states (unequally spaced vector of length Na)
        par.grid_a = nonlinspace(1e-6, 20, par.Na, 1.1)

        # c. labor grid
        par.grid_l = nonlinspace(1e-6, 1, par.Nl, 1.1)
        
        
        # d. shocks (qudrature nodes and weights using GaussHermite)
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

=======
        par.grid_z = nonlinspace(1e-4, 10, par.Np, 1.1)
        
        # b. post-decision states (unequally spaced vector of length Na)
        par.grid_a = nonlinspace(1e-6, 20, par.Na, 1.1)        
        
        # d. shocks (qudrature nodes and weights using GaussHermite)
        # shocks = create_PT_shocks(
        #     par.sigma_z,par.Nz,
        #     par.pi,par.mu)
        # par.z,par.z_w,par.Nshocks = shocks

        z,z_w = tools.GaussHermite_lognorm(par.sigma_z,par.Nz)
>>>>>>> work
        # e. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ print checksum """

        return np.mean(self.sol.c[0])

    #########
    # solve #
    #########

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        sol.c = np.nan*np.ones((par.T,par.Np,par.Nm))
        sol.l = np.nan*np.ones((par.T,par.Np,par.Nm))      
        sol.v = np.nan*np.zeros((par.T,par.Np,par.Nm))
        sol.w = np.nan*np.zeros((par.Np,par.Na))
        sol.q = np.nan*np.zeros((par.Np,par.Na))

    def solve(self):
        """ solve the model using solmethod """

        with jit(self) as model:  # can now call jitted functions

            par = model.par
            sol = model.sol

            # backwards induction
            for t in reversed(range(par.T)):

                t0 = time.time()

                # a. last period
                if t == par.T - 1:

                    last_period.solve(t, sol, par)

                # b. all other periods
                else:

                    # i. compute post-decision functions
                    t0_w = time.time()

                    compute_w, compute_q = False, False
                    if par.solmethod in ['nvfi']:
                        compute_w = True
                    elif par.solmethod in ['egm']:
                        compute_q = True

                    if compute_w or compute_q:

                        if par.do_simple_w:
                            post_decision.compute_wq_simple(t, sol, par, compute_w=compute_w, compute_q=compute_q)
                        else:
                            post_decision.compute_wq(t, sol, par, compute_w=compute_w, compute_q=compute_q)

                    t1_w = time.time()

                    # ii. solve bellman equation
                    if par.solmethod == 'vfi':
                        vfi.solve_bellman(t, sol, par)
                    elif par.solmethod == 'nvfi':
                        nvfi.solve_bellman(t, sol, par)
                    elif par.solmethod == 'egm':
                        # Include labor in the EGM method
                        egm.solve_bellman(t, sol, par)
                    else:
                        raise ValueError(f'unknown solution method, {par.solmethod}')

                # c. print
                if par.do_print:
                    msg = f' t = {t} solved in {elapsed(t0)}'
                    if t < par.T - 1:
                        msg += f' (w: {elapsed(t0_w, t1_w)})'
                    print(msg)
        

    def solve_cpp(self):
        """ solve the model using egm written in C++ """

        par = self.par
        sol = self.sol

        # a. solve by EGM
        t0 = time.time()
       
        if par.solmethod in ['egm']:
            self.cpp.solve(par,sol)
        else:
            raise ValueError(f'unknown cpp solution method, {par.solmethod}')            
        
        t1 = time.time()

        return t0,t1


    ############
    # simulate #
    ############
    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. allocate
<<<<<<< HEAD
        sim.p = np.nan * np.zeros((par.simT, par.simN))
=======
        # sim.p = np.nan * np.zeros((par.simT, par.simN))
>>>>>>> work
        sim.m = np.nan * np.zeros((par.simT, par.simN))
        sim.c = np.nan * np.zeros((par.simT, par.simN))
        sim.a = np.nan * np.zeros((par.simT, par.simN))
        sim.l = np.nan * np.zeros((par.simT, par.simN))  # Allocate memory for labor choice 'l'

        # b. draw random shocks
<<<<<<< HEAD
        sim.psi = np.ones((par.simT, par.simN))
        sim.xi = np.ones((par.simT, par.simN))
=======
        # sim.psi = np.ones((par.simT, par.simN))
        sim.z = np.ones((par.simT, par.simN))
>>>>>>> work
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. allocate
<<<<<<< HEAD
        sim.p = np.nan * np.zeros((par.simT, par.simN))
=======
        # sim.p = np.nan * np.zeros((par.simT, par.simN))
>>>>>>> work
        sim.m = np.nan * np.zeros((par.simT, par.simN))
        sim.c = np.nan * np.zeros((par.simT, par.simN))
        sim.a = np.nan * np.zeros((par.simT, par.simN))
        sim.l = np.nan * np.zeros((par.simT, par.simN))  # Allocate memory for labor choice 'l'

        # b. draw random shocks
<<<<<<< HEAD
        sim.psi = np.ones((par.simT, par.simN))
        sim.xi = np.ones((par.simT, par.simN))
=======
        # sim.psi = np.ones((par.simT, par.simN))
        sim.z = np.ones((par.simT, par.simN))
>>>>>>> work

    def simulate(self):
        """ simulate model """

        with jit(self) as model:  # can now call jitted functions

            par = model.par
            sol = model.sol
            sim = model.sim

            t0 = time.time()

            # a. allocate memory and draw random numbers
            I = np.random.choice(par.Nshocks,
                                 size=(par.T, par.simN),
<<<<<<< HEAD
                                 p=par.psi_w * par.xi_w)

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]
=======
                                 p=par.z_w)

            # sim.psi[:] = par.psi[I]
            sim.z[:] = par.z[I]
>>>>>>> work

            # b. simulate
            simulate.lifecycle(sim, sol, par)

        if par.do_print:
            print(f'model simulated in {elapsed(t0)}')

    ########
    # figs #
    ########

    def consumption_function(self, t=0):
        figs.consumption_function(self, t)

    def consumption_function_interact(self):
        figs.consumption_function_interact(self)

    def lifecycle(self):
        figs.lifecycle(self)
   