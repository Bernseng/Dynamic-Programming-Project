import numpy as np
from numba import njit

@njit(fastmath=True)
def p_plus_func(p,psi,par,t):
    if t+1<par.Tr: # before retirement
        p_plus = p*psi*par.G*par.L[t]
        p_plus = np.fmax(p_plus,par.p_min) #lower bound
        p_plus = np.fmin(p_plus,par.p_max) #upper bound
    else: # after retirement
        p_plus = p*par.G*par.L[t] #no shocks after retirement
        p_plus = np.fmax(p_plus,par.p_min) #lower bound
        p_plus = np.fmin(p_plus,par.p_max) #upper bound
    return p_plus 

@njit(fastmath=True)
def m_plus_func(a,xi_plus,psi_plus,par,t):
    if t+1<par.Tr: # before retirement
        m_plus = par.R*a/(psi_plus*par.G*par.L[t]) + xi_plus
    else: #after retirement
        m_plus = par.R*a/(par.G*par.L[t]) + 1 #no shocks after retirement
    return m_plus

@njit(fastmath=True)
def y_plus_func(p_plus,xi_plus,par,t):
    if t+1<par.Tr: #after retirement
        y_plus = p_plus*xi_plus
    else: #before retirement
        y_plus = p_plus #no shocks after retirement
    return y_plus