import numpy as np

###############################################################################

def eta(x,y,param):
    return 1.

###############################################################################

def solution(x,y):
    return np.sin(4*np.pi*x)*np.cos(4*np.pi*y),\
          -np.cos(4*np.pi*x)*np.sin(4*np.pi*y),\
           np.pi*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)

###############################################################################

def dpdx_th(x,y):
    return -4*np.pi**2*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)
    
def dpdy_th(x,y):
    return -4*np.pi**2*np.cos(4*np.pi*x)*np.sin(4*np.pi*y)

###############################################################################

def exx_th(x,y):
    return 0  

def exy_th(x,y):
    return 0

def eyy_th(x,y):
    return 0 

###############################################################################

def bx(x,y,param):
    return 28*np.pi**2*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)

def by(x,y,param):
    return -36*np.pi**2*np.cos(4*np.pi*x)*np.sin(4*np.pi*y) 

###############################################################################

def vrms():
    return 1. 

###############################################################################

left_bc  ='analytical'
right_bc ='analytical'
bottom_bc='analytical'
top_bc   ='analytical'

pnormalise=True
