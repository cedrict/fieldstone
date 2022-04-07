#Re=1,10,100,1000

Re=1

lambdaa=Re/2-np.sqrt(Re**2/4+4*np.pi**2)

def u_th(x,y):
    return 1-np.exp(lambdaa*x)*np.cos(2*np.pi*y)

def v_th(x,y):
    return lambdaa/2/np.pi*np.exp(lambdaa*x)*np.sin(2*np.pi*y)

def p_th(x,y):
    return 0.5*np.exp(2*lambdaa*x)

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

def vrms(x,y):
    return 0
