###############################################################################
from numba import jit
import numpy as np

Ggrav=6.67430e-11

##@jit(parallel=True,nopython=True)

@jit(nopython=True)
def compute_gravity_at_point(xM,zM,nel,nqel,zq,radq,thetaq,massq,nel_phi):
    dphi=2*np.pi/nel_phi
    gx=0.
    gy=0.
    gz=0.
    counterq=0
    for iel in range(0,nel):
           for kq in range (0,nqel):
               factq=radq[counterq]*np.sin(thetaq[counterq])
               for jel in range(0,nel_phi):
                   x_c=factq*np.cos(jel*dphi)
                   y_c=factq*np.sin(jel*dphi)
                   dist32=((xM-x_c)**2 + y_c**2 + (zM-zq[counterq])**2)**1.5
                   Kernel=Ggrav/dist32*massq[counterq]
                   gx-= Kernel*(xM-x_c)
                   gy-= Kernel*(  -y_c)
                   gz-= Kernel*(zM-zq[counterq] )
               #end for
               counterq+=1
           #end for
    #end for
    return gx/(2*np.pi)*dphi,gy/(2*np.pi)*dphi,gz/(2*np.pi)*dphi

###############################################################################
# analytical solution for g and U, hollow sphere, see Thieulot 2018
###############################################################################

def self_U(r,R1,R2,rho0):
    if r<=R1:
       val=2*np.pi*Ggrav*rho0*(R1**2-R2**2)
    elif r<=R2:
       val=4*np.pi/3*Ggrav*rho0*(r**2/2+R1**3/r)-2*np.pi*rho0*Ggrav*R2**2
    else:
       val=-4*np.pi/3*rho0*(R2**3-R1**3)*Ggrav/r
    return val

def self_g(r,R1,R2,rho0):
    if r<=R1:
       val=0
    elif r<=R2:
       val=4*np.pi/3*rho0*(r-R1**3/r**2)*Ggrav
    else:
       val=4*np.pi/3*rho0*(R2**3-R1**3)*Ggrav/r**2
    return val

###############################################################################
