import numpy as np


def bx(x,y,z,beta):
    val=np.pi**2*np.sin(np.pi*z)+np.pi**2*np.cos(np.pi*y)\
       + np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
    return -val

def by(x,y,z,beta):
    val=np.pi**2*np.sin(np.pi*x)+np.pi**2*np.cos(np.pi*z)\
       - np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
    return -val

def bz(x,y,z,beta):
    val=np.pi**2*np.cos(np.pi*x)+np.pi**2*np.sin(np.pi*y)\
       - np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
    return -val

def viscosity(x,y,z,beta):
    val=1
    return val

def uth(x,y,z):
    val=np.sin(np.pi*z)+np.cos(np.pi*y)
    return val

def vth(x,y,z):
    val=np.sin(np.pi*x)+np.cos(np.pi*z)
    return val

def wth(x,y,z):
    val=np.sin(np.pi*y)+np.cos(np.pi*x)
    return val

def pth(x,y,z):
    val=np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
    return val

def exx_th(x,y,z):
    val=0
    return val

def eyy_th(x,y,z):
    val=0
    return val

def ezz_th(x,y,z):
    val=0
    return val

def exy_th(x,y,z):
    val=np.pi/2*(np.cos(np.pi*x) - np.sin(np.pi*y) )
    return val

def exz_th(x,y,z):
    val=np.pi/2*(np.cos(np.pi*z) - np.sin(np.pi*x) )
    return val

def eyz_th(x,y,z):
    val=np.pi/2*(np.cos(np.pi*y) - np.sin(np.pi*z) )
    return val



