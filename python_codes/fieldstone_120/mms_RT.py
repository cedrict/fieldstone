import numpy as np
    
amplitude=0.01
llambda=0.5

def eta(x,y,param):
    if y>0.5+amplitude*np.cos(2*np.pi*x/llambda):
       return 1
    else:
       return param

def by(x,y,drho):
    if y>0.5+amplitude*np.cos(2*np.pi*x/llambda):
       return -(1+drho)
    else:
       return -1

def u_th(x,y):
    return 0

def v_th(x,y):
    return 0 

def p_th(x,y):
    return 0 

def dpdx_th(x,y):
    return 0 
    
def dpdy_th(x,y):
    return 0 

def exx_th(x,y):
    return 0 

def exy_th(x,y):
    return 0 

def eyy_th(x,y):
    return 0 

def dexxdx(x,y):
    return 0 

def dexydx(x,y):
    return 0 

def dexydy(x,y):
    return 0 

def deyydy(x,y):
    return 0 

def bx(x,y):
    return 0 


def vrms_th():
    return 0

left_bc  ='free_slip'
right_bc ='free_slip'
bottom_bc='free_slip'
top_bc   ='free_slip'

pnormalise=True
