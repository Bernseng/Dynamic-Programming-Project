from numba import njit

@njit
def func(c, l, par):
    return (c**(1 - par.rho) / (1 - par.rho)) - par.chi * (l**(1 + par.gamma) / (1 + par.gamma))

@njit
def marg_func_c(c, par):
    return c**(-par.rho)

@njit
def inv_marg_func_c(q, par):
    return q**(-1/par.rho)

@njit
def marg_func_l(l, par):
    return -par.chi * (l**par.gamma)

@njit
def inv_marg_func_l(q, par):
    return (-q / par.chi) ** (1 / par.gamma)