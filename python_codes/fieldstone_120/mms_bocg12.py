#mms taken from bocg12

def f(x):
    return x**2*(x-1)**2

def fp(x):
    return 2*x*(x-1)*(2*x-1)

def fpp(x):
    return 2*(6*x**2-6*x+1)

def fppp(x):
    return 24*x-12

def g(y):
    return y**2*(y-1)**2
    
def gp(y):
    return 2*y*(y-1)*(2*y-1)

def gpp(y):
    return 2*(6*y**2-6*y+1)

def gppp(y):
    return 24*y-12

###################################3

def solution(x,y):
    return f(x)*gp(y),\
          -fp(x)*g(y),\
           0.5*x**2-1/6 

###################################3

def dpdx_th(x,y):
    return x 
    
def dpdy_th(x,y):
    return 0 

###################################3

def exx_th(x,y):
    return fp(x)*gp(y) 

def exy_th(x,y):
    return 0.5*(f(x)*gpp(y)-fpp(x)*g(y))

def eyy_th(x,y):
    return -fp(x)*gp(y)

###################################3

def dexxdx(x,y):
    return fpp(x)*gp(y)

def dexydx(x,y):
    return 0.5*(fp(x)*gpp(y)-fppp(x)*g(y))

def dexydy(x,y):
    return 0.5*(f(x)*gppp(y)-fpp(x)*gp(y))

def deyydy(x,y):
    return -fp(x)*gpp(y)

###################################3

def bx(x,y):
    return dpdx_th(x,y)-2*dexxdx(x,y)-2*dexydy(x,y)

def by(x,y):
    return dpdy_th(x,y)-2*dexydx(x,y)-2*deyydy(x,y)

###################################3

def vrms_th():
    return 0.007776157913597390787927885951

def eta(x,y):
    return 1

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

pnormalise=True
