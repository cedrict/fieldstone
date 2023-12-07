# 0.25x0.25 sinker cube in the middle of unit square 

def eta(x,y,etastar):
    if abs(x-0.5)<0.125 and abs(y-0.75)<0.125:
       return etastar
    else:
       return 1

def by(x,y,drho):
    if abs(x-0.5)<0.125 and abs(y-0.75)<0.125:
       return -drho
    else:
       return 0

def solution(x,y):
    return 0,0,0

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

def bx(x,y,drho):
    return 0 


def vrms():
    return 0

left_bc  ='free_slip'
right_bc ='free_slip'
bottom_bc='free_slip'
top_bc   ='free_slip'

pnormalise=True
