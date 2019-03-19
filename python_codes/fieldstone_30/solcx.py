import math as math
import numpy as np

def SolCxSolution(x,y):
    MX=3
    MY=2
    u=MY*np.pi*np.sin(MX*np.pi*x)*np.cos(MY*np.pi*y)
    v=-MX*np.pi*np.cos(MX*np.pi*x)*np.sin(MY*np.pi*y)
    p=0.
    return u,v,p
