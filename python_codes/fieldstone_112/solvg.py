import numpy as np

def SolVgGravity(x,y):

    eta=-np.sin(x*x*y*y+x*y+5.)+1.+0.001 #epsilon;

    costerm=np.cos(x*x*y*y+x*y+5.)

    deta_dx=-y*(2.*x*y+1.)*costerm
    deta_dy=-x*(2.*x*y+1.)*costerm

    dpdx=2. * x  *y*y +y 
    dpdy=2. * x*x*y   +x 

    exx= 3.*x*x * y +2.*x +y +1.
    eyy=-3.*x*x * y -2.*x -y -1.

    exy=0.5*(x*x*x + x -3.*x*y*y -2.*y)
    eyx=0.5*(x*x*x + x -3.*x*y*y -2.*y)

    dexxdx= 6.*x*y+2.
    deyxdy=-3.*x*y-1.

    dexydx= 0.5*(3.*x*x +1. -3.*y*y)
    deyydy= -3.*x*x -1.

    gx =-dpdx + 2.*eta*dexxdx + 2.*deta_dx*exx + 2.*eta*deyxdy + 2.*deta_dy*eyx
    gy =-dpdy + 2.*eta*dexydx + 2.*deta_dx*exy + 2.*eta*deyydy + 2.*deta_dy*eyy

    return -gx,-gy
