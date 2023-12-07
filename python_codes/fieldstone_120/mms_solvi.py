import numpy as np
import math as math

###############################################################################

def eta(x,y,param):
    if (np.sqrt(x*x+y*y) < 0.2):
       return 1e3
    else:
       return 1e0

###############################################################################

def vrms():
    return 0 

###############################################################################

def bx(x,y,param):
    return 0 

def by(x,y,param):
    return 0 

###############################################################################

left_bc  ='analytical'
right_bc ='analytical'
bottom_bc='analytical'
top_bc   ='analytical'

pnormalise=True

###############################################################################

def solution(x,y):
    min_eta = 1.
    max_eta = 1.e3
    epsilon = 1.
    A=min_eta*(max_eta-min_eta)/(max_eta+min_eta)
    r_inclusion=0.2 
    r2_inclusion=r_inclusion*r_inclusion
    r2=x*x+y*y
    # phi, psi, dphi are complex
    z=x+y*1j
    if r2<r2_inclusion:
       phi=0+0.*1j
       dphi=0+0.*1j
       psi=-4*epsilon*(max_eta*min_eta/(min_eta+max_eta))*z
       visc=1e3
    else:
       phi=-2*epsilon*A*r2_inclusion/z
       dphi=-phi/z
       psi=-2*epsilon*(min_eta*z+A*r2_inclusion*r2_inclusion/(z*z*z))
       visc=1.

    v = (phi-z*np.conjugate(dphi)-np.conjugate(psi))/(2.*visc)
    vx=v.real
    vy=v.imag
    p=-2*epsilon*dphi.real

    return vx,vy,p

###############################################################################
