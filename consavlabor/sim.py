import numba as nb
from consav.markov import choice
from consav.linear_interp import interp_1d

############################
# simulation - monte carlo #
############################

# @nb.njit(parallel=True)
# def simulate_forwards_mc(t,par,sim,sol,algo):
#     """ monte carlo simulation of model. """
    
#     c = sim.c
#     l = sim.l
#     ell = sim.ell
#     a = sim.a
#     i_z = sim.i_z

#     for i in nb.prange(par.simN):

#         # a. lagged assets
#         if t == 0:
#             p_z_ini = sim.p_z_ini[i]
#             i_z_lag = choice(p_z_ini,par.z_ergodic_cumsum)
#             a_lag = sim.a_ini[i]
#         else:
#             i_z_lag = sim.i_z[t-1,i]
#             a_lag = sim.a[t-1,i]

#         # b. productivity
#         p_z = sim.p_z[t,i]
#         i_z_ = i_z[t,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

#         # c. consumption and labor supply
#         c[t,i] = interp_1d(par.a_grid,sol.c[i_z_,:],a_lag)
#         if algo == 'endo':
#             l[t,i] = interp_1d(par.a_grid,sol.l[i_z_,:],a_lag)
#         elif algo == 'exo':
#             l[t,i] = ell[t,i]

#         # d. end-of-period assets
#         m = (1+par.r)*a_lag + (1-par.tau)*par.w*par.z_grid[i_z_]*l[t,i]
#         a[t,i] = m-c[t,i]

@nb.njit(parallel=True)
def simulate_forwards_mc(t,par,sim,sol,algo):
    """ monte carlo simulation of model. """
    
    c = sim.c
    l = sim.l
    ell = sim.ell
    a = sim.a
    i_z = sim.i_z
    mpc = sim.mpc  # assume you've added this to your sim object

    for i in nb.prange(par.simN):

        # a. lagged assets
        if t == 0:
            p_z_ini = sim.p_z_ini[i]
            i_z_lag = choice(p_z_ini,par.z_ergodic_cumsum)
            a_lag = sim.a_ini[i]
        else:
            i_z_lag = sim.i_z[t-1,i]
            a_lag = sim.a[t-1,i]

        # b. productivity
        p_z = sim.p_z[t,i]
        i_z_ = i_z[t,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

        # c. consumption and labor supply
        c[t,i] = interp_1d(par.a_grid,sol.c[i_z_,:],a_lag)
        if algo == 'endo':
            l[t,i] = interp_1d(par.a_grid,sol.l[i_z_,:],a_lag)
        elif algo == 'exo':
            l[t,i] = ell[t,i]

        # d. end-of-period assets
        if t >= par.simT//2:
            w = 1.01*par.w*par.z_grid[i_z_]  # Increase wage by 1% halfway through simulation
        else:
            w = par.w*par.z_grid[i_z_]
        m = (1+par.r)*a_lag + (1-par.tau)*w*l[t,i]
        a[t,i] = m - c[t,i]

        # e. MPC calculation
        if t == par.simT//2:  # introduce income shock halfway through simulation
            m_shock = m * 1.01
            c_shock = interp_1d(par.a_grid, sol.c[i_z_, :], a_lag)
            a_shock = m_shock - c_shock
            mpc[t, i] = (c_shock - c[t,i]) / (0.01 * m)
        elif t > par.simT//2:
            mpc[t, i] = (c[t,i] - c[t-1,i]) / (0.01 * w)



