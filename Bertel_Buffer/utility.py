from numba import njit

@njit
def func(c, l, par):
    return (c**(1 - par.rho)/(1 - par.rho))-par.varphi * (l**(1 + par.nu) / (1 + par.nu))

@njit
def marg_util_c(c, par):
    return c**(-par.rho)

@njit
def marg_util_l(c, par):
    return c**(-par.rho/par.nu)*(par.w*par.xi*par.psi/par.varphi)**(1/par.rho)

@njit
def marg_util_l(c, par):
    return c**(-par.rho/par.nu)*(par.w*par.xi*par.psi/par.varphi)**(1/par.rho)

@njit
def inv_marg_util_c(q, par):
    return q**(-1/par.rho)

@njit
def inv_marg_util_l(q, par):
    return (-q/par.varphi) ** (1/par.nu)

def marg_util(c,par):
    return c**(-par.rho)

def inv_marg_util(u,par):
    return u**(-1/par.rho)