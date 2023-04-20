from numba import njit, prange

# local modules
import utility

@njit(parallel=True)
def solve(t, sol, par):
    """ solve the problem in the last period """

    # unpack (helps numba optimize)
    v = sol.v[t]
    c = sol.c[t]
    l = sol.l[t]

    # loop over states
    for ip in prange(par.Np):  # in parallel
        for im in range(par.Nm):

            # a. states
            _p = par.grid_p[ip]
            m = par.grid_m[im]

            # b. optimal choice (consume everything, choose optimal labor)
            max_val = -1e10  # Initialize a low value to compare against
            optimal_c = 0
            optimal_l = 0

            for il in range(par.Nl):
                _l = par.grid_l[il]

                # i. consumption
                _c = m

                # ii. value
                _v = utility.func(_c, _l, par)

                # iii. update maximum value and optimal choice
                if _v > max_val:
                    max_val = _v
                    optimal_c = _c
                    optimal_l = _l

            # c. save optimal choice and value
            c[ip, im] = optimal_c
            l[ip, im] = optimal_l
            v[ip, im] = max_val
