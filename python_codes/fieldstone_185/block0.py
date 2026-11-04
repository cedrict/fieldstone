
import numpy as np

###############################################################################

def analytical_solution_stress(x,y,z,lambdaa,mu):

    e_xx=1
    e_yy=0
    e_zz=0
    e_xy=0
    e_xz=0
    e_yz=0

    sigma_xx=lambdaa+2*mu*e_xx
    sigma_yy=lambdaa+2*mu*e_yy
    sigma_zz=lambdaa+2*mu*e_zz
    sigma_xy=        2*mu*e_xy
    sigma_xz=        2*mu*e_xz
    sigma_yz=        2*mu*e_yz

    return sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz

###############################################################################

def analytical_solution_strain(x,y,z):

    e_xx=1
    e_yy=0
    e_zz=0
    e_xy=0
    e_xz=0
    e_yz=0

    return e_xx,e_yy,e_zz,e_xy,e_xz,e_yz

###############################################################################
 
def analytical_solution_displacement(x,y,z):

    ux=x
    uy=0.
    uz=0.

    return ux,uy,uz
 
###############################################################################
