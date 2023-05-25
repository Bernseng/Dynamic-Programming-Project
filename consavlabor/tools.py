import copy
import numpy as np

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

# def calc_mpc(model, algo):
#     """ Calculate marginal propensity to consume. """

#     # save original solution
#     original_sol = copy.deepcopy(model.sol)

#     # Initialize an array to hold MPC for each z
#     mpc_z = np.zeros(model.par.Nz)

#     for i_z in range(model.par.Nz):
#         # Increase w by a small amount (1%)
#         model.par.w *= 1.01

#         # solve the model again
#         model.solve(do_print=False, algo=algo)

#         # Measure the change in consumption
#         delta_c = model.sol.c[i_z] - original_sol.c[i_z]
        
#         # Calculate the MPC: change in consumption / change in income
#         mpc_z[i_z] = delta_c / (0.01 * model.par.w)

#         # restore the original model parameters and solution
#         model.par.w /= 1.01
#         model.sol = original_sol

#     return mpc_z



