# this is example 1 of Mu & Ye (2017) \cite{muye17}

def u_th(x,y):
    return 2*np.pi*np.sin(np.pi*x)*np.sin(np.pi*x)* np.cos(np.pi*y)*np.sin(np.pi*y)

def v_th(x,y):
    return -2*np.pi*np.sin(np.pi*x)*np.cos(np.pi*x)* np.sin(np.pi*y)*np.sin(np.pi*y)

def p_th(x,y):
    return np.cos(np.pi*x)*np.cos(np.pi*y)

def dpdx_th(x,y):
    return -np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    
def dpdy_th(x,y):
    return -np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)

def exx_th(x,y):
    return 4*np.pi**2*np.sin(np.pi*x)*np.cos(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*y)

def exy_th(x,y):
    return np.pi**2*( ((np.sin(np.pi*x))**2*(np.cos(np.pi*y))**2 - (np.cos(np.pi*x))**2*(np.sin(np.pi*y))**2 )

def eyy_th(x,y):
    return -4*np.pi**2*np.sin(np.pi*x)*np.cos(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*y) 

def dexxdx(x,y):
    return 4*np.pi**3*((np.cos(np.pi*x))**2-(np.sin(np.pi*x))**2)*np.cos(np.pi*y)*np.sin(np.pi*y) 

def dexydx(x,y):
    return  2*np.pi**3*np.sin(np.pi*x)*np.cos(np.pi*x)

def dexydy(x,y):
    return -2*np.pi**3*np.sin(np.pi*y)*np.cos(np.pi*y)

def deyydy(x,y):
    return -4*np.pi**3 *np.sin(np.pi*x)*np.cos(np.pi*x)*((np.cos(np.pi*y))**2- (np.sin(np.pi*y))**2 )

def bx(x,y):
    return dpdx_th(x,y)-2*dexxdx(x,y)-2*dexydy(x,y)

def by(x,y):
    return dpdy_th(x,y)-2*dexydx(x,y)-2*deyydy(x,y)


