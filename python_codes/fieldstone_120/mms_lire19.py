import numpy as np

###############################################################################

def eta(x,y,param):
    return 1.

###############################################################################

def solution(x,y):
    return np.cos(y),\
           np.sin(x),\
           np.sin(x+y)-(2*np.sin(1.)-np.sin(2.))  

###############################################################################

def dpdx_th(x,y):
    return np.cos(x+y)
    
def dpdy_th(x,y):
    return np.cos(x+y)

###############################################################################

def exx_th(x,y):
    return 0  

def exy_th(x,y):
    return 0

def eyy_th(x,y):
    return 0 

###############################################################################

def bx(x,y,param):
    return np.cos(y)+np.cos(x+y) 

def by(x,y,param):
    return np.sin(x)+np.cos(x+y) 

###############################################################################

def vrms():
    return 1. 

###############################################################################

left_bc  ='analytical'
right_bc ='analytical'
bottom_bc='analytical'
top_bc   ='analytical'

pnormalise=True
