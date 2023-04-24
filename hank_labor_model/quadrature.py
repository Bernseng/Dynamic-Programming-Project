# -*- coding: utf-8 -*-
""" quadrature
Module is copied from consav/quadrature and a typo has been corrected.
Functions for quadrature (mostly Gauss-Hermite).
"""

import math
import numpy as np

def gauss_hermite(n):
    """ gauss-hermite nodes
    Args:
        n (int): number of points
    Returns:
        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n
    """

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def normal_gauss_hermite(sigma,n=7,mu=None):
    """ normal gauss-hermite nodes
    Args:
        sigma (double): standard deviation
        n (int): number of points
        mu (double,optinal): mean (if None, then mean zero)
    Returns:
        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n
    """

    if sigma == 0.0 or n == 1:

        w = np.ones(n)/n
        
        if mu is None:
            x = np.zeros(n)
        else:
            x = np.zeros(n)+mu

        return x,w

    # a. GaussHermite
    x,w = gauss_hermite(n)
    x *= np.sqrt(2)*sigma 
    w /= np.sqrt(math.pi)

    # b. adjust mean
    if mu is None:
        x = x 
    else:
        x = x + mu

    return x,w

def log_normal_gauss_hermite(sigma,n=7,mu=None):
    """ log-normal gauss-hermite nodes
    Args:
        sigma (double): standard deviation
        n (int): number of points
        mu (double,optinal): mean (if None, the mean one)
    Returns:
        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n
    """

    if sigma == 0.0 or n == 1:

        w = np.ones(n)/n
        
        if mu is None:
            x = np.exp(np.zeros(n))
        else:
            x = np.exp(np.zeros(n)+ np.log(mu))
        
        return x,w

    # a. GaussHermite
    x,w = gauss_hermite(n)
    x *= np.sqrt(2)*sigma 
    w /= np.sqrt(math.pi)

    # b. adjust mean
    if mu is None:
        x = np.exp(x-0.5*sigma**2)
    else:
        x = np.exp(x+np.log(mu)-0.5*sigma**2)

    return x,w

def create_PT_shocks(sigma_psi,Npsi,sigma_xi,Nxi): 
    """ log-normal gauss-hermite nodes for permanent transitory model,
        including house price shocks
    Args:
        sigma_psi (double): standard deviation of permanent shock
        Npsi (int): number of points for permanent shock
        sigma_xi (double): standard deviation of transitory shock
        Nxi (int): number of points for transitory shock 
    Returns:
        psi (numpy.ndarray): nodes for permanent shock of lengthNpsi*Nxi*Nz+1 
        psi_w (numpy.ndarray): weights for permanent shock of length Npsi*Nxi*Nz+1 
        xi (numpy.ndarray): nodes for transitory shock of length Npsi*Nxi*Nz+1 
        xi_w (numpy.ndarray): weights for transitory shock of length Npsi*Nxi*Nz+1 
        Nshocks (int): number of nodes = Npsi*Nxi 
    """

    # a. gauss hermite
    psi, psi_w = log_normal_gauss_hermite(sigma_psi,Npsi)
    xi, xi_w = log_normal_gauss_hermite(sigma_xi,Nxi)
        
    # c. tensor product matrix: ‘ij’-indexing of output
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    return psi.ravel(),psi_w.ravel(),xi.ravel(),xi_w.ravel(), psi.size