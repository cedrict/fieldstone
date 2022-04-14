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
    return dpdx_th(x,y)-2*dexxdx(x,y)-2*dexydy(x,y)

def by(x,y):
    return dpdy_th(x,y)-2*dexydx(x,y)-2*deyydy(x,y)

def vrms_th():
    return 0

def eta(x,y):
    return 1
