# functions for the Donea & Huerta benchmark (dh)

def eta(x,y,param):
    return 1.

def solution(x,y):
    return   x**2*(1.-x)**2*(2*y-6*y**2+4*y**3),\
            -y**2*(1.-y)**2*(2*x-6*x**2+4*x**3),\
             x*(1-x)-1./6.

#def u_th(x,y):
#    return x**2*(1.-x)**2*(2*y-6*y**2+4*y**3)

#def v_th(x,y):
#    return -y**2*(1.-y)**2*(2*x-6*x**2+4*x**3)

#def p_th(x,y):
#    return x*(1-x)-1./6.

def dpdx_th(x,y):
    return 1.-2.*x
    
def dpdy_th(x,y):
    return 0.

def exx_th(x,y):
    return x**2*(2*x-2)*(4*y**3-6*y**2+2*y)+2*x*(-x+1)**2*(4*y**3-6*y**2+2*y)

def exy_th(x,y):
    return (x**2*(-x+1)**2*(12*y**2-12*y+2)-y**2*(-y+1)**2*(12*x**2-12*x+2))/2

def eyy_th(x,y):
    return -exx_th(x,y) 

def bx(x,y,param):
    return ((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
           (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
           (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
           1.-4.*y+12.*y*y-8.*y*y*y)

def by(x,y,param):
    return ((8.-48.*y+48.*y*y)*x*x*x+
           (-12.+72.*y-72.*y*y)*x*x+
           (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
           12.*y*y+24.*y*y*y-12.*y**4)

def vrms_th():
    return 0.00777615791

left_bc  ='no_slip'
right_bc ='no_slip'
bottom_bc='no_slip'
top_bc   ='no_slip'

pnormalise=True
