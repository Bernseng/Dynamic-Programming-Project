# Import package and module
import numpy as np
import utility as util
import tools


def EGM(sol,t,par):
    sol = EGM_loop(sol,t,par) 
    # sol = EGM_vec(sol,t,par) 
    return sol

def EGM_loop(sol,t,par):
    
    for i_a,a in enumerate(par.grid_a[t,:]):
        # if t+1 <= par.Tr: # No pension in the next period
            # fac = par.G*par.L[t]*par.psi_vec
        w = par.w
        fac = par.G*1.0*w*par.psi_vec
        xi = par.xi_vec

        # Future m and c
        m_plus = (1/fac)*par.R*a+xi
        c_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c[t+1,:],m_plus)
        # l_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.l[t+1,:],m_plus)
        
        # else:
            # fac = par.G*par.L[t]
            # w = 1
            # xi = 1

            # # Futute m and c
            # m_plus = (1/fac)*par.R*a+xi
            # c_plus = tools.interp_linear_1d_scalar(sol.m[t+1,:],sol.c[t+1,:], m_plus)

        # Future expected marginal utility
        marg_uc_plus = util.marg_util_c(fac*c_plus,par)
        marg_ul_plus = util.marg_util_l(fac*c_plus,par)

        avg_marg_uc_plus = np.sum(w*marg_uc_plus)
        avg_marg_ul_plus = np.sum(w*marg_ul_plus)

        # Current c and m (i_a+1 as we save the first index point to handle the credit constraint region)
        sol.c[t,i_a+1]=util.inv_marg_util_c(par.beta*par.R*avg_marg_uc_plus,par)
        sol.l[t,i_a+1]=util.inv_marg_util_l(par.beta*par.R*avg_marg_ul_plus,par)
        sol.m[t,i_a+1]=a+sol.c[t,i_a+1]

    return sol

def EGM_vec (sol,t,par):

    if t+1 <= par.Tr: 
        fac = np.tile(par.G*par.L[t]*par.psi_vec, par.Na) 
        xi = np.tile(par.xi_vec,par.Na)
        a = np.repeat(par.grid_a[t],par.Nshocks) 

        w = np.tile(par.w,(par.Na,1))
        dim = par.Nshocks
    else:
        fac = par.G*par.L[t]*np.ones((par.Na))
        xi = np.ones((par.Na))
        a = par.grid_a[t,:]
            
        w = np.ones((par.Na,1))
        dim = 1

    # Future m and c
    m_plus = (1/fac)*par.R*a+xi
    c_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c[t+1,:], m_plus)

    # Future expected marginal utility
    marg_u_plus = util.marg_util(fac*c_plus,par)
    marg_u_plus = np.reshape(marg_u_plus,(par.Na,dim))
    avg_marg_u_plus = np.sum(w*marg_u_plus,1)

    # Current C and m (we save the first index point to handle the credit constraint region)
    sol.c[t,1:]= util.inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    sol.m[t,1:]=par.grid_a[t,:]+sol.c[t,1:]

    return sol
