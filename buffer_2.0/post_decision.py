import numpy as np
from numba import njit, prange, jit

# consav
# from consav import linear_interp, interp_1d_vec # for linear interpolation
from consav.linear_interp import interp_1d_vec

# local modules
import utility

@njit(parallel=True)
def compute_wq(t, sol, par, compute_w=False, compute_q=False):
    """ compute the post-decision functions w and/or q with endogenous labor supply"""

    # unpack (helps numba optimize)
    w = sol.w
    q = sol.q

    # loop over outermost post-decision state
    for ip in prange(par.Np):  # in parallel

        # b. allocate containers and initialize at zero
        m_plus = np.empty(par.Na)
        if compute_w:
            w[ip, :] = 0
            v_plus = np.empty(par.Na)
        if compute_q:
            q[ip, :] = 0
            c_plus = np.empty(par.Na)
            l_plus = np.empty(par.Na)

        # v. next-period cash-on-hand and interpolate
        for ia in range(par.Na):
            m_plus[ia] = par.R*par.grid_a[ia] - par.w * par.grid_l[ip]


        # vi. interpolate v_plus and c_plus
        # vi. interpolate v_plus, c_plus, and l_plus
        if compute_w:
            interp_1d_vec(par.grid_m, sol.v[t + 1][ip, :], m_plus, v_plus)

        if compute_q:
            interp_1d_vec(par.grid_m, sol.c[t + 1][ip, :], m_plus, c_plus)
            interp_1d_vec(par.grid_m, sol.l[t + 1][ip, :], m_plus, l_plus)


        # vii. accumulate all
        if compute_w:
            for ia in range(par.Na):
                w[ip, ia] += par.beta * v_plus[ia]
        if compute_q:
            for ia in range(par.Na):
                q[ip, ia] += par.R * par.beta * utility.marg_func_c(c_plus[ia], par)




@njit(parallel=True)
def compute_wq_simple(t,sol,par,compute_w=False,compute_q=False):
    """ compute the post-decision functions w and/or q """

    # this is a variant of Algorithm 3 in Druedahl (2019): A Guide to Solve Non-Convex Consumption-Saving Problems
    
    # note: same result as compute_wq, simpler code, but much slower
 
    # unpack (helps numba optimize)
    w = sol.w
    q = sol.q

    # loop over outermost post-decision state
    for ip in prange(par.Np): # in parallel

        for ia in range(par.Na):

            # initialize at zero
            if compute_w:
                w[ip,ia] = 0
            if compute_q:
                q[ip,ia] = 0

            for ishock in range(par.Nshocks):
            
                # i. shocks
                psi = par.psi[ishock]
                psi_w = par.psi_w[ishock]
                xi = par.xi[ishock]
                xi_w = par.xi_w[ishock]

                # ii. next-period states
                p_plus = par.grid_p[ip]*psi
                y_plus = p_plus*xi
                m_plus = par.R*par.grid_a[ia] + y_plus
            
                # iii. weights
                weight = psi_w*xi_w

                # iv. interpolate and accumulate
                if compute_w:
                    w[ip,ia] += weight*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,sol.v[t+1],p_plus,m_plus)
                if compute_q:
                    c_plus_temp = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t+1],p_plus,m_plus)
                    q[ip,ia] += weight*par.R*par.beta*utility.marg_func_c(c_plus_temp,par)