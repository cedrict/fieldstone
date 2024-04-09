import random
import numpy as np

rad1=4700e3 ; theta1=14.5/180*np.pi
rad2=6371e3 ; theta2=22/180*np.pi
rad3=6371e3 ; theta3=0

rad4=0.5*(rad1+rad2) ; theta4=0.5*(theta1+theta2)
rad5=0.5*(rad2+rad3) ; theta5=0.5*(theta2+theta3)
rad6=0.5*(rad3+rad1) ; theta6=0.5*(theta3+theta1)

x1=rad1*np.sin(theta1) ; z1=rad1*np.cos(theta1)
x2=rad2*np.sin(theta2) ; z2=rad2*np.cos(theta2)
x3=rad3*np.sin(theta3) ; z3=rad3*np.cos(theta3)
x4=rad4*np.sin(theta4) ; z4=rad4*np.cos(theta4)
x5=rad5*np.sin(theta5) ; z5=rad5*np.cos(theta5)
x6=rad6*np.sin(theta6) ; z6=rad6*np.cos(theta6)

#----------------------------------------------------------

npts=1000

r=np.zeros(npts,dtype=np.float64)   
s=np.zeros(npts,dtype=np.float64)   

rad=np.zeros(npts,dtype=np.float64)   
theta=np.zeros(npts,dtype=np.float64)   

x_DJ=np.zeros(npts,dtype=np.float64)   
z_DJ=np.zeros(npts,dtype=np.float64)   
x_P1=np.zeros(npts,dtype=np.float64)   
z_P1=np.zeros(npts,dtype=np.float64)   
x_P2=np.zeros(npts,dtype=np.float64)   
z_P2=np.zeros(npts,dtype=np.float64)   

counter=0
for i in range(0,10*npts):
    
    # compute random r,s coordinates

    rr=random.uniform(0,1)
    ss=random.uniform(0,1)

    if ss<=1-rr:

       r[counter]=rr
       s[counter]=ss

       N1=1-r[counter]-s[counter]
       N2=r[counter]
       N3=s[counter]

       rad[counter]=N1*rad1+N2*rad2+N3*rad3
       theta[counter]=N1*theta1+N2*theta2+N3*theta3

       x_DJ[counter]=rad[counter]*np.sin(theta[counter])
       z_DJ[counter]=rad[counter]*np.cos(theta[counter])

       x_P1[counter]=N1*x1+N2*x2+N3*x3
       z_P1[counter]=N1*z1+N2*z2+N3*z3

       N1= 1-3*rr-3*ss+2*rr**2+4*rr*ss+2*ss**2 
       N2= -rr+2*rr**2
       N3= -ss+2*ss**2
       N4= 4*rr-4*rr**2-4*rr*ss
       N5= 4*rr*ss 
       N6= 4*ss-4*rr*ss-4*ss**2

       x_P2[counter]=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6
       z_P2[counter]=N1*z1+N2*z2+N3*z3+N4*z4+N5*z5+N6*z6

       counter+=1

       if counter>npts-1:
          break

np.savetxt('polar.ascii',np.array([rad,theta]).T)
np.savetxt('xz_DJ.ascii',np.array([x_DJ,z_DJ]).T)
np.savetxt('xz_P1.ascii',np.array([x_P1,z_P1]).T)
np.savetxt('xz_P2.ascii',np.array([x_P2,z_P2]).T)

#----------------------------------------------------------
# now placing points on 2-3 edge only
#----------------------------------------------------------

npts=1000

r=np.zeros(npts,dtype=np.float64)   
s=np.zeros(npts,dtype=np.float64)   

rad=np.zeros(npts,dtype=np.float64)   
theta=np.zeros(npts,dtype=np.float64)   

x_DJ=np.zeros(npts,dtype=np.float64)   
z_DJ=np.zeros(npts,dtype=np.float64)   
x_P1=np.zeros(npts,dtype=np.float64)   
z_P1=np.zeros(npts,dtype=np.float64)   
x_P2=np.zeros(npts,dtype=np.float64)   
z_P2=np.zeros(npts,dtype=np.float64)   

for i in range(0,npts):

    rr=random.uniform(0,1)

    r[i]=rr
    s[i]=1-rr

    N1=1-r[i]-s[i]
    N2=r[i]
    N3=s[i]

    rad[i]=N1*rad1+N2*rad2+N3*rad3
    theta[i]=N1*theta1+N2*theta2+N3*theta3

    x_DJ[i]=rad[i]*np.sin(theta[i])
    z_DJ[i]=rad[i]*np.cos(theta[i])

    x_P1[i]=N1*x1+N2*x2+N3*x3
    z_P1[i]=N1*z1+N2*z2+N3*z3

    ss=s[i]
    N1= 1-3*rr-3*ss+2*rr**2+4*rr*ss+2*ss**2 
    N2= -rr+2*rr**2
    N3= -ss+2*ss**2
    N4= 4*rr-4*rr**2-4*rr*ss
    N5= 4*rr*ss 
    N6= 4*ss-4*rr*ss-4*ss**2

    x_P2[i]=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6
    z_P2[i]=N1*z1+N2*z2+N3*z3+N4*z4+N5*z5+N6*z6

np.savetxt('polar_boundary.ascii',np.array([rad,theta]).T)
np.savetxt('xz_DJ_boundary.ascii',np.array([x_DJ,z_DJ,np.sqrt(x_DJ**2+z_DJ**2)]).T)
np.savetxt('xz_P1_boundary.ascii',np.array([x_P1,z_P1,np.sqrt(x_P1**2+z_P1**2)]).T)
np.savetxt('xz_P2_boundary.ascii',np.array([x_P2,z_P2,np.sqrt(x_P2**2+z_P2**2)]).T)

#----------------------------------------------------------

