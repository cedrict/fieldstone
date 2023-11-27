# functions for the Volker John III benchmark (vj3)

def eta(x,y,dum):
    return 1.

def solution(x,y):
    return 200*x**2*(1-x)**2*y*(1-y)*(1-2*y),\
          -200*x*(1-x)*(1-2*x)*y**2*(1-y)**2,\
           10*( (x-1./2.)**3*y**2+(1-x)**3*(y-1./2.)**3 )

def dpdx_th(x,y):
    return 30*(x-1./2.)**2*y**2-30*(1-x)**2*(y-1./2.)**3
    
def dpdy_th(x,y):
    return 20*(x-1./2.)**3*y + 30*(1-x)**3*(y-1./2.)**2  

def exx_th(x,y):
    return -400*(1-x)*x*(2*x-1)*(y-1)*y*(2*y-1) 

def exy_th(x,y):
    return 100*(1-x)**2*x**2*(6*y**2-6*y+1)-100*(6*x**2-6*x+1)*(1-y)**2*y**2 

def eyy_th(x,y):
    return 400*(x-1)*x*(2*x-1)*(1-y)*y*(2*y-1) 

def dexxdx(x,y):
    return 400*(6*x**2-6*x+1)*y*(2*y**2-3*y+1)

def dexydx(x,y):
    return 100*(-2*x**2*(1-x)*(6*y**2-6*y+1) + 2*x*(1-x)**2*(6*y**2-6*y+1) -6*(2*x-1)*(1-y)**2*y**2)

def dexydy(x,y):
    return 200*(6*x**2-6*x+1)*(1-y)*y**2 + 100*(1-x)**2*x**2*(12*y-6) -200*(6*x**2-6*x+1)*(1-y)**2*y

def deyydy(x,y):
    return -400*x*(2*x**2-3*x+1)*(6*y**2-6*y+1)

def bx(x,y,dum):
    return dpdx_th(x,y)-2*dexxdx(x,y)-2*dexydy(x,y)

def by(x,y,dum):
    return dpdy_th(x,y)-2*dexydx(x,y)-2*deyydy(x,y)

def vrms_th():
    return 0.777615791

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

pnormalise=True
