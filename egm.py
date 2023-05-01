# Import package and module
import numpy as np
import utility as util
import tools


def EGM(sol,t,par):
    #sol = EGM_loop(sol,t,par) 
    sol = EGM_vec(sol,t,par) 
    return sol

def EGM_loop (sol,t,par):
    for i_a,a in enumerate(par.grid_a[t,:]):

        if t < par.Tr: # Working period
            fac = par.G*par.L[t]*par.psi_vec
            w = par.w
            xi = par.xi_vec
            inv_fac = 1/fac
            dim = par.Nshocks
        else: # Retirement period
            fac = par.G*par.L[t]
            w = 1
            xi = 1
            inv_fac = 1/fac
            dim = 1
            m_plus = inv_fac*par.R*a+xi
            c_plus = tools.interp_linear_1d_scalar(sol.m[t+1,:],sol.c[t+1,:], m_plus)
            l_plus = tools.interp_linear_1d_scalar(sol.m[t+1,:],sol.l[t+1,:], m_plus)

        # Future marginal utility
        marg_u_plus_c, marg_u_plus_l = util.marg_util(fac * c_plus, l_plus, par)
        marg_u_plus_c = np.reshape(marg_u_plus_c, (par.Na, dim))
        marg_u_plus_l = np.reshape(marg_u_plus_l, (par.Na, dim))
        avg_marg_u_plus_c = np.sum(w * marg_u_plus_c, 1)
        avg_marg_u_plus_l = np.sum(w * marg_u_plus_l, 1)

        # Current C, m, and l
        sol.c[t, 1:], sol.l[t, 1:] = util.marg_util_inv(par.beta * par.R * avg_marg_u_plus_c, par.beta * par.R * avg_marg_u_plus_l, par)
        sol.m[t, 1:] = par.grid_a[t, :] + sol.c[t, 1:]

    return sol

def EGM_vec (sol, t, par):
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

    inv_fac = 1/fac

    # Future m, c and l
    m_plus = inv_fac*par.R*a+xi
    c_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c[t+1,:], m_plus)
    l_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.l[t+1,:], m_plus)

    # Future marginal utility
    marg_u_plus_c, marg_u_plus_l = util.marg_util(fac * c_plus, l_plus, par)
    marg_u_plus_c = np.reshape(marg_u_plus_c, (par.Na, dim))
    marg_u_plus_l = np.reshape(marg_u_plus_l, (par.Na, dim))
    avg_marg_u_plus_c = np.sum(w * marg_u_plus_c, 1)
    avg_marg_u_plus_l = np.sum(w * marg_u_plus_l, 1)

    # Current C, m, and l
    sol.c[t, 1:], sol.l[t, 1:] = util.marg_util_inv(par.beta * par.R * avg_marg_u_plus_c, par.beta * par.R * avg_marg_u_plus_l, par)
    sol.m[t, 1:] = par.grid_a[t, :] + sol.c[t, 1:]

    return sol