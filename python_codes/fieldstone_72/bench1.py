import numpy as np

###############################################################################
# bx and by are the body force components and analytical solution

def a(x):
    return -2*x*x*(x-1)**2
def b(y):
    return y*(2*y-1)*(y-1)  
def c(x):
    return x*(2*x-1)*(x-1) 
def d(y):
    return 2*y*y*(y-1)**2

def ap(x): 
    return -4*x*(2*x**2-3*x+1)
def app(x):
    return -4*(6*x**2-6*x+1) 
def bp(y): 
    return 6*y**2-6*y+1 
def bpp(y):
    return 12*y-6 
def cp(x): 
    return 6*x**2-6*x+1 
def cpp(x):
    return 12*x-6  
def dp(y): 
    return 4*y*(2*y**2-3*y+1)  
def dpp(y):
    return 4*(6*y**2-6*y+1)  

def exx_th(x,y):
    return ap(x)*b(y)
def eyy_th(x,y):
    return c(x)*dp(y)
def exx_th(x,y):
    return 0.5*(a(x)*bp(y)+cp(x)*d(y))

def dpdx_th(x,y):
    return (1-2*x)*(1-2*y)
def dpdy_th(x,y):
    return -2*x*(1-x)
