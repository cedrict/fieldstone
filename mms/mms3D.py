#------------------------------------

def f(x):
    return x*(1-x)
def fp(x):
    return 1-2*x
def fpp(x):
    return -2
def fppp(x):
    return 0

def g(y):
    return y*(1-y)
def gp(y):
    return 1-2*y
def gpp(y):
    return -2
def gppp(y):
    return 0

def h(z):
    return z*(1-z)
def hp(z):
    return 1-2*z
def hpp(z):
    return -2
def hppp(z):
    return 0

#------------------------------------

def p_th(x,y,z):
    return (2*x-1)*(2*y-1)*(2*z-1)

def dpdx(x,y,z):
    return 2*(2*y-1)*(2*z-1)

def dpdy(x,y,z):
    return 2*(2*x-1)*(2*z-1)

def dpdz(x,y,z):
    return 2*(2*x-1)*(2*y-1)

#------------------------------------

def bx(x,y,z):
    return dpdx(x,y,z)- (fpp(x)*gp(y)*hp(z)  + f(x)*gppp(y)*hp(z)  +  f(x)*gp(y)*hppp(z) )

def by(x,y,z):
    return dpdy(x,y,z)- (fppp(x)*g(y)*hp(z) + fp(x)*gpp(y)*hp(z)  +  fp(x)*g(y)*hppp(z) )

def bz(x,y,z):
    return dpdz(x,y,z)- (-2*fppp(x)*gp(y)*h(z) -2*fp(x)*gppp(y)*h(z) -2*fp(x)*gp(y)*hpp(z))

#------------------------------------

def u_th(x,y,z):
    return f(x)*gp(y)*hp(z)

def v_th(x,y,z):
    return fp(x)*g(y)*hp(z)

def w_th(x,y,z):
    return -2*fp(x)*gp(y)*h(z)

#------------------------------------

def exx_th(x,y,z):
    return fp(x)*gp(y)*hp(z)

def eyy_th(x,y,z):
    return fp(x)*gp(y)*hp(z)

def ezz_th(x,y,z):
    return -2*fp(x)*gp(y)*hp(z)

def exy_th(x,y,z):
    return ((fpp(x)*g(y)+f(x)*gpp(y)) *hp(z) )/2

def exz_th(x,y,z):
    return (f(x)*gp(y)*hpp(z)-2*fpp(x)*gp(y)*h(z))/2 

def eyz_th(x,y,z):
    return (fp(x)*g(y)*hpp(z)-2*fp(x)*gpp(y)*h(z))/2

