# utility.py

def util(c, l, par):
    return ((c**(1-par.rho) - 1) / (1-par.rho)) - par.chi*(l**(par.eta)) / (par.eta)

def marg_util(c, l, par):
    # Note: This now returns a tuple (marginal utility w.r.t. c, marginal utility w.r.t. l)
    return c**(-par.rho), par.chi * l**(par.eta - 1)

def marg_util_inv(u_c, u_l, par):
    # Note: This now takes two arguments (marginal utility w.r.t. c and w.r.t. l) and returns a tuple (c, l)
    return u_c**(-1/par.rho), (u_l / par.chi)**(1 / (par.eta - 1))