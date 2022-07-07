import numpy as np

pi=3.14159265358979323846264338327950288
eps=1e-8

def u(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce*x/4/pi/mu/R*(z/R**2-(1-2*nu)/(R+z))
    return val

def v(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce*y/4/pi/mu/R*(z/R**2-(1-2*nu)/(R+z))
    return val

def w(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce/4/pi/mu/R*(2*(1-nu)+z**2/R**2)
    return val

def sigmaxx(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce/2/pi/mu/R**2*( 3*x**2*z/R**3-(1-2*nu)*( z/R-R/(R+z)+x**2*(2*R+z)/R/(R+z)**2 ) )
    return val

def sigmayy(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce/2/pi/mu/R**2*( 3*y**2*z/R**3-(1-2*nu)*( z/R-R/(R+z)+y**2*(2*R+z)/R/(R+z)**2 ) )
    return val

def sigmazz(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-3*Pforce*z**3/2/pi/R**5
    return val

def sigmaxy(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-Pforce/2/pi/R**2*( 3*x*y*z/R**3-(1-2*nu)*x*y*(2*R+z)/R/(R+z)**2  )
    return val

def sigmaxz(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-3*Pforce*x*z**2/2/pi/R**5
    return val

def sigmayz(x,y,z,Pforce,nu,mu):
    R=np.sqrt(x**2+y**2+z**2)
    if R<eps:
       val=0
    else:
       val=-3*Pforce*y*z**2/2/pi/R**5
    return val








