from numba import jit
import numpy as np

@jit(nopython=True)
def gx(x,y,g0):
    val= -x/np.sqrt(x*x+y*y)*g0
    return val

@jit(nopython=True)
def gy(x,y,g0):
    val= -y/np.sqrt(x*x+y*y)*g0
    return val
