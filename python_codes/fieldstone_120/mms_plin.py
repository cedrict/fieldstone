# functions for the Donea & Huerta benchmark (dh)

aaa=3
bbb=5

###############################################################################

def eta(x,y,param):
    return 1.

###############################################################################

def solution(x,y):
    return   x**2*(1.-x)**2*(2*y-6*y**2+4*y**3),\
            -y**2*(1.-y)**2*(2*x-6*x**2+4*x**3),\
            aaa*(x-0.5)+bbb*(y-0.5) 

###############################################################################

def dpdx_th(x,y):
    return aaa
    
def dpdy_th(x,y):
    return bbb

###############################################################################

def exx_th(x,y):
    return 0  

def exy_th(x,y):
    return 0

def eyy_th(x,y):
    return 0 

###############################################################################

def bx(x,y,param):
    f=x**2*(1.-x)**2
    fpp=2-12*x+12*x**2
    gp=2*y-6*y**2+4*y**3
    gppp=-12+24*y
    return aaa-fpp*gp-f*gppp

def by(x,y,param):
    fp=2*x-6*x**2+4*x**3
    fppp=-12+24*x
    g=y**2*(1.-y)**2
    gpp=2-12*y+12*y**2 
    return bbb+fp*gpp+ fppp*g

###############################################################################

def vrms():
    return 0.00777615791

###############################################################################

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

pnormalise=True
