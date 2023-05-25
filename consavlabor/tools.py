import copy
import numpy as np
import numba as nb

def calc_mpc(model, algo):
    """ Calculate marginal propensity to consume. """

    # save original solution
    original_sol = copy.deepcopy(model.sol)

    # Increase w by a small amount (1%)
    model.par.w *= 1.01

    # solve the model again
    model.solve(do_print=False, algo=algo)

    # Measure the change in consumption
    delta_c = model.sol.c - original_sol.c
    
    # Calculate the MPC: change in consumption / change in income
    mpc = delta_c / (0.01 * model.par.w)

    # restore the original model parameters and solution
    model.par.w /= 1.01
    model.sol = original_sol

    return mpc





