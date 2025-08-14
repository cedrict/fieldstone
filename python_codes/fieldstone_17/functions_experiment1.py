import numpy as np

beta=0

def bx(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(y*z+3*x**2*y**3*z) + mu * (2+6*x*y) \
        +(2+4*x+2*y+6*x**2*y) * mux \
        +(x+x**3+y+2*x*y**2 ) * muy \
        +(-3*z-10*x*y*z     ) * muz
    return val

def by(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(x*z+3*x**3*y**2*z) + mu * (2 +2*x**2 + 2*y**2) \
       +(x+x**3+y+2*x*y**2   ) * mux \
       +(2+2*x+4*y+4*x**2*y  ) * muy \
       +(-3*z-5*x**2*z       ) * muz 
    return val

def bz(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(x*y+x**3*y**3) + mu * (-10*y*z) \
       +(-3*z-10*x*y*z        ) * mux \
       +(-3*z-5*x**2*z        ) * muy \
       +(-4-6*x-6*y-10*x**2*y ) * muz 
    return val

###############################################################################

def viscosity(x,y,z,beta):
    val=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    return val

def uth(x,y,z):
    val=x+x*x+x*y+x*x*x*y
    return val

def vth(x,y,z):
    val=y+x*y+y*y+x*x*y*y
    return val

def wth(x,y,z):
    val=-2*z-3*x*z-3*y*z-5*x*x*y*z
    return val

def pth(x,y,z):
    val=x*y*z+x*x*x*y*y*y*z-5./32.
    return val

def exx_th(x,y,z):
    val=1.+2.*x+y+3.*x*x*y
    return val

def eyy_th(x,y,z):
    val=1.+x+2.*y+2.*x*x*y
    return val

def ezz_th(x,y,z):
    val=-2.-3.*x-3.*y-5.*x*x*y
    return val

def exy_th(x,y,z):
    val=(x+y+2*x*y*y+x*x*x)/2
    return val

def exz_th(x,y,z):
    val=(-3*z-10*x*y*z)/2
    return val

def eyz_th(x,y,z):
    val=(-3*z-5*x*x*z)/2
    return val
