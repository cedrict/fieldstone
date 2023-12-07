import numpy as np

###############################################################################

def f(x):
    return x**2*(1-x)**4

def fp(x):
    return 2*x*(1-3*x)*(1-x)**3

def fpp(x):
    return 2*(1-10*x+15*x**2)*(1-x)**2

def fppp(x):
    return 24*(-1+5*x-5*x**2)*(1-x)

###############################################################################

def g(y):
    return y**3*(1-y)**2

def gp(y):
    return y**2*(3-5*y)*(1-y)

def gpp(y):
    return 2*y*(3-12*y+10*y**2)

def gppp(y):
    return 6-48*y+60*y**2

###############################################################################

def exx(x,y):
    return 1000*fp(x)*gp(y)

def exy(x,y):
    return 500*(f(x)*gpp(y)-fpp(x)*g(y))

def eyy(x,y):
    return -1000*fp(x)*gp(y) 

###############################################################################

def dexxdx(x,y):
    return 1000*fpp(x)*gp(y)

def dexydx(x,y):
    return 500*(fp(x)*gpp(y)-fppp(x)*g(y))

def dexydy(x,y):
    return 500*(f(x)*gppp(y)-fpp(x)*gp(y))

def deyydy(x,y):
    return -1000*fp(x)*gpp(y)

###############################################################################

def solution(x,y):
    return 1000*f(x)*gp(y),\
          -1000*fp(x)*g(y),\
          np.pi**2*(x*y**3*np.cos(2*np.pi*x**2*y) - x**2*y*np.sin(2*np.pi*x*y) )+1/8 

###############################################################################
eta_max=1e4
eta_min=1#e-4

def eta(x,y,dum):
    return eta_min+(eta_max-eta_min)*x**2*(1-x)*y**2*(1-y)*721/16

def deta_dx(x,y):
    return (eta_max-eta_min)*721/16*(2*x*(1-x)-x**2)*y**2*(1-y)

def deta_dy(x,y):
    return (eta_max-eta_min)*721/16*x**2*(1-x)*(2*y*(1-y)-y**2)

###############################################################################

def dpdx(x,y):
    return np.pi**2*(y**3*np.cos(2*np.pi*x**2*y) \
                    -4*np.pi*x**2*y**4*np.sin(2*np.pi*x**2*y) \
                    -2*x*y*np.sin(2*np.pi*x*y) \
                    -2*np.pi*x**2*y**2*np.cos(2*np.pi*x*y) )
    
def dpdy(x,y):
    return np.pi**2*(3*x*y**2*np.cos(2*np.pi*x**2*y) \
                     -2*np.pi*x**3*y**3*np.sin(2*np.pi*x**2*y) \
                     -x**2*np.sin(2*np.pi*x*y) \
                     -2*np.pi*x**3*y*np.cos(2*np.pi*x*y) )

###############################################################################

def bx(x,y,dum):
    return dpdx(x,y)\
           -2*eta(x,y,dum)*dexxdx(x,y)\
           -2*eta(x,y,dum)*dexydy(x,y)\
           -2*deta_dx(x,y)*exx(x,y)\
           -2*deta_dy(x,y)*exy(x,y)

def by(x,y,dum):
    return dpdy(x,y)\
           -2*eta(x,y,dum)*dexydx(x,y)\
           -2*eta(x,y,dum)*deyydy(x,y)\
           -2*deta_dx(x,y)*exy(x,y)\
           -2*deta_dy(x,y)*eyy(x,y)

###############################################################################

def vrms():
    return 1.4953325891041323968540981

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

pnormalise=True
