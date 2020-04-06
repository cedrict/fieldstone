import math as math
import numpy as np

def Solution(x,z):
    eta1=100000.0
    eta2=1.0
    alpha=45./180.*np.pi
    V0=1.
    h=np.sqrt(2.)#*np.sin(alpha+np.pi/4)
    V1=(x*np.sin(alpha)+z*np.cos(alpha))*2*V0*eta2/(eta1+eta2)/h;
    V2=(x*np.sin(alpha)+z*np.cos(alpha))*2*V0*eta1/(eta1+eta2)/h+(eta2-eta1)*V0/(eta1+eta2);

    if x*np.sin(alpha)+z*np.cos(alpha)<0.5*h-1e-6:
       u=V1*np.cos(alpha)
       v=-V1*np.sin(alpha)
       #u=0.
       #v=0.
    else:
       u=V2*np.cos(alpha)
       v=-V2*np.sin(alpha)
    p=0

    return u,v,p
