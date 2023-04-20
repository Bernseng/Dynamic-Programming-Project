
"""ConsumptionSavingModel
Solves the Deaton-Carroll buffer-stock consumption model with vfi or egm:
"""

##############
# 1. imports #
##############

import time
import numpy as np
import numba as nb
import quantecon as qe

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav.grids import nonlinspace # grids
from quadrature import create_PT_shocks
from consav.grids import equilogspace
from consav.linear_interp import binary_search, interp_1d
from consav.markov import choice, find_ergodic, log_rouwenhorst
from consav.misc import elapsed
from consav.quadrature import log_normal_gauss_hermite
from EconModel import EconModelClass, jit

# import simulate
import lastperiod
import utility
import egm

class ConSavingLaborModel(ModelClass):    

    def settings(self):
        """ fundamental settings """

        # namespaces
        self.namespaces = []

        # other attributes
        self.other_attrs = []

        # savefolder
        self.savefolder = 'saved'

        # for safe type inference
        self.not_floats = ['solmethod','T','TR','Nxi','Npsi','Nm','Na','simN','Nshocks','sim_seed']


    def setup(self):
        """ baseline parameters """

        par = self.par

        # horizon and life cycle
        par.Tmin = 25 # enter the model (start work life)
        par.T = 80 - par.Tmin # death
        # par.Tr = 65 - par.Tmin # retirement age (end-of-period), no retirement if TR = T
        # par.G = 1.02 # growth factor
        # par.L = np.ones(par.T-1)
        # par.L[0:par.Tr] = np.linspace(1,1/par.G,par.Tr) 
        # par.L[par.Tr-1] = 0.67 # drop in permanent income at retirement age
        # par.L[par.Tr-1:] = par.L[par.Tr-1:]/par.G # constant permanent income after retirement
    
        par.Nfix = 4 
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        #par.varphi_min = 0.9
        #par.varphi_max = 1.1
        #par.varphi = 1.0
        #par.zeta = 1.0 # fixed individual productivity component
        #par.zeta_min = 0.9
        #par.zeta_max = 1.1
        par.nu = 1.0
        par.phi = 0.9
        par.tau_l = 0.3
        par.tau_a= 0.1 
        par.chi_ss= 0.0 # Government Transfer

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of shock

        # c. production and investment
        par.alpha = 0.30 # cobb-douglas
        par.delta = 0.1 # depreciation rate
        par.Gamma = 1.0 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 100.0 # maximum point in grid for a

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.03
        par.w_ss_target = 1.0

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_ell = 30

        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.toll_ell = 1e-12

        # preferences
        par.rho = 2.0 # CRRA coeficient
        par.beta = 0.965 # subjective discount factor

        # returns and incomes
        par.R = 1.03 #return on assets
        par.sigma_psi = 0.1 # stdev of shocks to permanent income
        par.sigma_xi = 0.1 # stdev of shocks to permanent income
        par.Npsi = 5 #nodes for psi shock
        par.Nxi = 5 #nodes for xi shock
        par.mpc_eps = 0.00749 #bump to m for mpc calculation
        
        # grids
        par.Nm = 100 #nodes for m grid
        par.m_max = 10 #maximum cash-on-hand level
        par.m_phi = 1.1 # curvature parameter
        par.Na = 100 #nodes for a grid
        par.a_max = par.m_max+1.0
        par.a_phi = 1.1 # curvature parameter
        par.Np = 50 #nodes for p grid
        par.p_min = 1e-4 #minimum permanent income
        par.p_max = 3.0 #maximum permanent income
        
        # simulation
        par.sigma_m0 = 0.2 #std for initial draw of m
        par.mu_m0 = -0.2 #mean for initial draw of m
        par.mu_p0 = -0.2 #mean for initial draw of p
        par.sigma_p0 = 0.2 #std for initial draw of p
        par.simN = 10000 # number of persons in simulation
        par.sim_seed = 1998 # seed for simulation
        par.euler_cutoff = 0.02 # euler error cutoff
        
        # misc
        par.t = 0
        par.tol = 1e-8
        par.do_print = False
        par.do_print_period = False
        par.do_marg_u = False
        
    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        # a. post-decision states
        par.grid_a = np.ones((par.T,par.Na))
        par.a_min = np.zeros(par.T) # never any borriwng
        for t in range(par.T):
            par.grid_a[t,:] = nonlinspace(par.a_min[t]+1e-6,par.a_max,par.Na,par.a_phi)
        
        # b. states
        par.grid_m = np.ones((par.T,par.Nm))
        for t in range(par.T):
            par.grid_m[t,:] = nonlinspace(par.a_min[t]+1e-6,par.m_max,par.Nm,par.m_phi)    

        # c. solution arrays
        sol.m = np.zeros((par.Nz, par.Na))
        sol.c = np.zeros((par.Nz, par.Na))
        sol.a = np.zeros((par.Nz, par.Na))
        sol.l = np.zeros((par.Nz, par.Na))  # labor supply
        sol.inv_v = np.zeros((par.Nz, par.Na))

    
    def solve(self,do_print=True):
        """ gateway for solving the model """

        par = self.par
        # sol = self.sol

        # b. solve
        tic = time.time()

        # backwards induction
        for t in reversed(range(self.par.T)):
            self.par.t = t
            
            with jit(self) as model:
                par = model.par
                sol = model.sol
                
                c = np.zeros((1, par.Na+1))
                l = np.zeros((1, par.Na+1))
                m = np.zeros((1, par.Na+1))
                inv_v = np.zeros((1, par.Na+1))
                
                # last period
                if t == par.T-1:
                    lastperiod.last_period(par,sol)
                
                # other periods    
                else:
                    egm.egm(par,sol,t,m,c,inv_v) # solve by egm

                    # ii. add zero consumption
                    # sol.m[t,0] = par.a_min[t]
                    # sol.m[t,1:] = m
                    # sol.c[t,0] = 0
                    # sol.c[t,1:] = c
                    # sol.l[t,0] = 0
                    # sol.l[t,1:] = l
                    # sol.inv_v[t,0] = 0
                    # sol.inv_v[t,1:] = inv_v
            
        toc = time.time()

        if par.do_print:
            print(f'model solved in {toc-tic:.1f} secs')