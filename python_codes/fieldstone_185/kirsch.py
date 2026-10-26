import numpy as np

###############################################################################

def analytical_solution_stress(r,theta,radius,sigma_H):

    rr=r/radius

    c0=np.cos(theta)
    s0=np.sin(theta)
    c20=np.cos(2*theta)
    s20=np.sin(2*theta)

    sigma_rr=0.5*(1-1/rr**2)+0.5*(1+3/rr**4-4/rr**2)*c20
    sigma_tt=0.5*(1+1/rr**2)-0.5*(1+3/rr**4)*c20
    sigma_rt=-0.5*(1-3/rr**4+2/rr**2)*s20

    sigma_rr*=sigma_H
    sigma_tt*=sigma_H
    sigma_rt*=sigma_H

    sigma_xx=sigma_rr*c0**2+sigma_tt*s0**2-sigma_rt*s20
    sigma_yy=sigma_rr*s0**2+sigma_tt*c0**2+sigma_rt*s20
    sigma_zz=0
    sigma_xy=s0*c0*(sigma_rr-sigma_tt)+sigma_rt*c20
    sigma_xz=0
    sigma_yz=0

    return sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz

###############################################################################

def analytical_solution_strain(r,theta,radius,sigma_H,E,nu):

    rr=r/radius

    c0=np.cos(theta)
    s0=np.sin(theta)
    c20=np.cos(2*theta)
    s20=np.sin(2*theta)

    sigma_rr=0.5*(1-1/rr**2)+0.5*(1+3/rr**4-4/rr**2)*c20
    sigma_tt=0.5*(1+1/rr**2)-0.5*(1+3/rr**4)*c20
    sigma_rt=-0.5*(1-3/rr**4+2/rr**2)*s20

    sigma_rr*=sigma_H
    sigma_tt*=sigma_H
    sigma_rt*=sigma_H

    e_rr=(sigma_rr-nu*sigma_tt)/E
    e_tt=(sigma_tt-nu*sigma_rr)/E
    e_rt=(1+nu)/E*sigma_rt

    e_xx=e_rr*c0**2+e_tt*s0**2-e_rt*s20
    e_yy=e_rr*s0**2+e_tt*c0**2+e_rt*s20
    e_zz=0
    e_xy=s0*c0*(e_rr-e_tt)+e_rt*c20
    e_xz=0
    e_yz=0

    return e_xx,e_yy,e_zz,e_xy,e_xz,e_yz

###############################################################################
 
def analytical_solution_displacement(r,theta,radius,sigma_H,E,nu):

    rr=r/radius

    c0=np.cos(theta)
    s0=np.sin(theta)
    c20=np.cos(2*theta)
    s20=np.sin(2*theta)

    ur=(1-nu)+(1+nu)/rr**2 + ( (1+nu)-(1+nu)/rr**4+4/rr**2 )*c20

    ut=-( (1+nu)+(1+nu)/rr**4+2*(1-nu)/rr**2 ) * s20

    ur*=rr*sigma_H/2/E
    ut*=rr*sigma_H/2/E

    ux=ur*c0-ut*s0
    uy=ur*s0+ut*c0
    uz=0

    return ux,uy,uz
 
###############################################################################
