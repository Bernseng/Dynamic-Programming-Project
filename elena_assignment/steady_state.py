import time
import numpy as np
from scipy import optimize


from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # varphi-grid
    par.varphi_grid = np.array([0.9,0.9,1.1,1.1])

    # a-grid
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # z-grid
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz) 

    # zeta-grid
    par.zeta_grid = np.array([0.9,1.1,0.9,1.1])

    #############################################
    # 2. transition matrix initial distribution #    
    #############################################
    
    # Define the initial distribution of the transition matrix. Ensure sum of all elements is 1
    for i_varphi in range(par.Nvarphi):
        ss.z_trans[i_varphi,:,:] = z_trans # extract transition probabilities from defined markov chain
        ss.Dz[i_varphi,:] = z_ergodic / par.Nfix #  Divide by number of fixed states to ensure D sums to 1
        ss.Dbeg[i_varphi,:,0] = ss.Dz[i_varphi,:] # ergodic at a_lag = 0.0
        ss.Dbeg[i_varphi,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #      
    ################################################

    y = ss.w*par.z_grid
    c = m = (1+ss.r*(1-par.tau_a))*par.a_grid[np.newaxis,:] + (1-par.tau_l)*y[:,np.newaxis]
    v_a = (1+ss.r*(1-par.tau_a))*c**(-par.sigma) 

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(x, model, do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    ss.G = 0.30
    ss.chi= par.chi_ss
    
    #Steady state Capital Labor ratio    
    ss.Kappa = x 
    
    ss.rK = par.alpha*par.Gamma*(ss.Kappa)**(par.alpha-1.0)
    
    ss.w = (1.0-par.alpha)*par.Gamma*(ss.Kappa)**par.alpha
    
    ss.r = ss.rB = ss.rK - par.delta 
    
    #Solving household problem
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    
    ss.B = (par.tau_a*ss.r*ss.A_hh + par.tau_l*ss.w*ss.L_hh - ss.G - ss.chi)/ss.rB
    
    ss.L = ss.L_hh
    
    ss.K = ss.Kappa*ss.L
    
    ss.Y = par.Gamma*(ss.K**par.alpha)*ss.L**(1-par.alpha)
        
    ss.I = ss.K - (1-par.delta)*ss.K
    
    ss.C = ss.Y - ss.I - ss.G 
    
    #Market Clearing
    ss.clearing_A = ss.A_hh-ss.K-ss.B
    
    print(f'x={x:8.4f}',f'clearing_A={ss.clearing_A:8.4f}',f'r={ss.r:8.4f}',f'w={ss.w:8.4f}',f'A_hh={ss.A_hh:8.4f}',f'K={ss.K:8.4f}',f'B={ss.B:8.4f}',f'L_hh={ss.L_hh:8.4f}')
    
    return ss.clearing_A # target to hit
    
    
def find_ss(model,do_print=False):
    """ find steady state """
    
    par = model.par
    ss = model.ss
    
    #Find steady state
    t0 = time.time()

    Kappa_min = (((1.0/par.beta)-1.0+par.delta)/(par.alpha*par.Gamma))**(1.0/(par.alpha-1.0)) + 1e-2
    Kappa_max = (par.delta/(par.alpha*par.Gamma))**(1.0/(par.alpha-1.0)) - 1e-2

    res = optimize.root_scalar(obj_ss,bracket=(Kappa_min,Kappa_max),method='brentq',args=(model,))

    # final evaluation
    obj_ss(res.root,model,do_print=False)

    #Print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' K_ss    = {ss.K:8.4f}')
        print(f' L_ss    = {ss.L:8.4f}')
        print(f' Y_ss   = {ss.Y:8.4f}')
        print(f' C_ss   = {ss.C:8.4f}')
        print(f' B_ss   = {ss.B:8.4f}')
        print(f' G_ss   = {ss.G:8.4f}')
        print(f' r_ss   = {ss.r:8.4f}')
        print(f' w_ss   = {ss.w:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
              
        
