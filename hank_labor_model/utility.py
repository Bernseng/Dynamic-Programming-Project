from numba import njit

@njit(fastmath=True)
def utility(par,c,l):
    return (c**(1.0-par.sigma) - 1.0)/(1.0-par.sigma) - par.nu*(l**(1.0+par.phi))/(1.0+par.phi)

@njit(fastmath=True)
def foc_l(c,par):
    return ((par.w*par.zeta)/par.varphi)**(1.0/par.nu)*(c)**(-par.sigma/par.nu) #FOC l

@njit(fastmath=True)
def marg_util(c,par):
    return c**(-par.rho) # marginal utility

@njit(fastmath=True)
def inv_marg_util(u,par):
    return u**(-1/par.rho) #inverse marginal utility for egm