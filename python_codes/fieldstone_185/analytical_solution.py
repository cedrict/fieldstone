import numpy as np

###############################################################################

def analytical_solution_stress(x,y,z,radius,Lx,Ly,Lz,E,nu,sigma_H,sigma_h,sigma_v):

    # implement formula of Han/Jaeger -> compute stress in cyl coords

    # convert to cartesian coordinates -> see section 3.3.6

    return sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz

###############################################################################

def analytical_solution_strain(x,y,z,radius,Lx,Ly,Lz,E,nu,sigma_H,sigma_h,sigma_v):

    ...

    return e_xx,e_yy,e_zz,e_xy,e_xz,e_yz

###############################################################################
 
def analytical_solution_displacement(x,y,z,radius,Lx,Ly,Lz,E,nu,sigma_H,sigma_h,sigma_v):

    ...

    return ux,uy,uz
 
###############################################################################
 
