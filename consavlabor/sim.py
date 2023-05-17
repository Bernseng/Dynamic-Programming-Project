############################
# simulation - monte carlo #
############################

@nb.njit(parallel=True)
def simulate_forwards_mc(t,par,sim,sol):
    """ monte carlo simulation of model. """
    
    c = sim.c
    l = sim.l
    a = sim.a
    i_z = sim.i_z

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
        l[t,i] = interp_1d(par.a_grid,sol.l[i_z_,:],a_lag)

        # d. end-of-period assets
        m = (1+par.r)*a_lag + (1-par.tau)*par.w*par.z_grid[i_z_]*l[t,i]
        a[t,i] = m-c[t,i]

        # e. enforce borrowing constraint
        if a[t,i] < 0.0:
            a[t,i] = 0.0