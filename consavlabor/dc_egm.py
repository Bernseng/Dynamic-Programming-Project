
import numba as nb
from numba import njit
import numpy as np
from consav.linear_interp import interp_1d_vec
import utility
from tools import generate_zeta

@njit(parallel=True)
def solve_hh_backwards_dc_egm(par,vbeg_plus,vbeg,c,a,u):
    """ solve backwards with v_plus from previous iteration """

    for i_z in nb.prange(par.Nz):
        
        # prepare
        z = par.z_grid[i_z]
        w = (1-par.tau)*par.w*z
        income = par.l_exo*w
        
        # b. implied consumption function
        c_vec = (par.beta*vbeg_plus[i_z,:])**(-1.0/par.sigma) #FOC c

        m_endo = par.a_grid+c_vec
        m_exo = (1+par.r)*par.a_grid + income
        # m_exo = (1+par.rh+zeta)*par.a_grid

        # interpolate
        interp_1d_vec(m_endo,par.a_grid,m_exo,a[i_z,:])

        # calculating savings
        a[i_z,:] = np.fmax(a[i_z,:],0.0)
        c[i_z,:] = m_exo - a[i_z,:]

        # b. expectation steps
    v_a = (1.0+par.r)*c[:]**(-par.sigma)
    vbeg[:] = par.z_trans@v_a

    #Calculating utility
    u[:] = utility.func_(c, par)
