from numba import njit

@njit(parallel=True)
def func(c,l,par):
    return (c**(1-par.sigma)/(1-par.sigma))-par.varphi*(l**(1+par.nu)/(1+par.nu))

# for 3d plot
def func_2(c,l,par):
    return (c**(1-par.sigma)/(1-par.sigma))-par.varphi*(l**(1+par.nu)/(1+par.nu))