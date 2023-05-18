from numba import njit

@njit(parallel=True)
def func(c,l,par):
    return (c**(1-par.sigma)/(1-par.sigma))-par.varphi*(l**(1+par.nu)/(1+par.nu))

@njit(parallel=True)
def func_(c,par):
    return (c**(1-par.sigma)/(1-par.sigma))-par.varphi*(par.l_exo**(1+par.nu)/(1+par.nu))

def func_2(c,l,par):
    return (c**(1-par.sigma)/(1-par.sigma))-par.varphi*(l**(1+par.nu)/(1+par.nu))