from numba import njit

@njit
def func(c, l, par):
<<<<<<< HEAD
    return (c**(1 - par.rho) / (1 - par.rho)) - par.chi * (l**(1 + par.gamma) / (1 + par.gamma))
=======
    return (c**(1 - par.rho)/(1 - par.rho))-par.varphi * (l**(1 + par.nu) / (1 + par.nu))
>>>>>>> work

@njit
def marg_func_c(c, par):
    return c**(-par.rho)

@njit
def inv_marg_func_c(q, par):
    return q**(-1/par.rho)

@njit
<<<<<<< HEAD
def marg_func_l(l, par):
    return -par.chi * (l**par.gamma)

@njit
def inv_marg_func_l(q, par):
    return (-q / par.chi) ** (1 / par.gamma)
=======
def marg_func_l(c, par):
    return c**(-par.rho/par.nu)*(par.w*par.xi*par.psi/par.varphi)**(1/par.rho)

@njit
def inv_marg_func_l(q, par):
    return (-q/par.varphi) ** (1/par.nu)
>>>>>>> work
