from numba import njit

# @njit
# def func(c,par):
#     return c**(1-par.rho)/(1-par.rho)

# @njit
# def marg_func(c,par):
#     return c**(-par.rho)

# @njit
# def inv_marg_func(q,par):
#     return q**(-1/par.rho)

@njit
def func(c, l, par):
    return (c**(1 - par.rho)/(1 - par.rho))-par.varphi * (l**(1 + par.nu) / (1 + par.nu))

@njit
def marg_func_c(c, par):
    return c**(-par.rho)

@njit
def marg_func_l(c, par):
    return c**(-par.rho/par.nu)*(par.w*par.xi*par.psi/par.varphi)**(1/par.rho)

@njit
def inv_marg_func_c(q, par):
    return q**(-1/par.rho)

@njit
def inv_marg_func_l(q, par):
    return (q*(par.varphi/(par.w*par.xi*par.psi))**(1/par.rho))**(-par.nu/par.rho)
