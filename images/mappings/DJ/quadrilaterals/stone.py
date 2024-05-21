import random
import numpy as np

rad1=4700e3 ; theta1=10/180*np.pi
rad2=4700e3 ; theta2=25/180*np.pi
rad3=6371e3 ; theta3=25/180*np.pi
rad4=6371e3 ; theta4=10/180*np.pi

rad5=0.5*(rad1+rad2) ; theta5=0.5*(theta1+theta2)
rad6=0.5*(rad2+rad3) ; theta6=0.5*(theta2+theta3)
rad7=0.5*(rad3+rad4) ; theta7=0.5*(theta3+theta4)
rad8=0.5*(rad4+rad1) ; theta8=0.5*(theta4+theta1)
rad9=0.5*(rad1+rad3) ; theta9=0.5*(theta1+theta3)

x1=rad1*np.sin(theta1) ; z1=rad1*np.cos(theta1)
x2=rad2*np.sin(theta2) ; z2=rad2*np.cos(theta2)
x3=rad3*np.sin(theta3) ; z3=rad3*np.cos(theta3)
x4=rad4*np.sin(theta4) ; z4=rad4*np.cos(theta4)
x5=rad5*np.sin(theta5) ; z5=rad5*np.cos(theta5)
x6=rad6*np.sin(theta6) ; z6=rad6*np.cos(theta6)
x7=rad7*np.sin(theta7) ; z7=rad7*np.cos(theta7)
x8=rad8*np.sin(theta8) ; z8=rad8*np.cos(theta8)
x9=rad9*np.sin(theta9) ; z9=rad9*np.cos(theta9)

#----------------------------------------------------------

npts=1000

r=np.zeros(npts,dtype=np.float64)   
s=np.zeros(npts,dtype=np.float64)   

rad=np.zeros(npts,dtype=np.float64)   
theta=np.zeros(npts,dtype=np.float64)   

x_DJ=np.zeros(npts,dtype=np.float64)   
z_DJ=np.zeros(npts,dtype=np.float64)   
x_Q1=np.zeros(npts,dtype=np.float64)   
z_Q1=np.zeros(npts,dtype=np.float64)   
x_Q2=np.zeros(npts,dtype=np.float64)   
z_Q2=np.zeros(npts,dtype=np.float64)   

for i in range(0,npts):

    rr=random.uniform(-1,1)
    ss=random.uniform(-1,1)

    r[i]=rr
    s[i]=ss

    N1=0.25*(1-rr)*(1-ss)
    N2=0.25*(1+rr)*(1-ss)
    N3=0.25*(1+rr)*(1+ss)
    N4=0.25*(1-rr)*(1+ss)

    rad[i]=N1*rad1+N2*rad2+N3*rad3+N4*rad4
    theta[i]=N1*theta1+N2*theta2+N3*theta3+N4*theta4

    x_DJ[i]=rad[i]*np.sin(theta[i])
    z_DJ[i]=rad[i]*np.cos(theta[i])

    x_Q1[i]=N1*x1+N2*x2+N3*x3+N4*x4
    z_Q1[i]=N1*z1+N2*z2+N3*z3+N4*z4

    N1= 0.5*rr*(rr-1.) * 0.5*ss*(ss-1.)
    N2= 0.5*rr*(rr+1.) * 0.5*ss*(ss-1.)
    N3= 0.5*rr*(rr+1.) * 0.5*ss*(ss+1.)
    N4= 0.5*rr*(rr-1.) * 0.5*ss*(ss+1.)
    N5=     (1.-rr**2) * 0.5*ss*(ss-1.)
    N6= 0.5*rr*(rr+1.) *     (1.-ss**2)
    N7=     (1.-rr**2) * 0.5*ss*(ss+1.)
    N8= 0.5*rr*(rr-1.) *     (1.-ss**2)
    N9=     (1.-rr**2) *     (1.-ss**2)

    x_Q2[i]=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6+N7*x7+N8*x8+N9*x9
    z_Q2[i]=N1*z1+N2*z2+N3*z3+N4*z4+N5*z5+N6*z6+N7*z7+N8*z8+N9*z9

np.savetxt('polar.ascii',np.array([rad,theta]).T)
np.savetxt('xz_DJ.ascii',np.array([x_DJ,z_DJ]).T)
np.savetxt('xz_Q1.ascii',np.array([x_Q1,z_Q1]).T)
np.savetxt('xz_Q2.ascii',np.array([x_Q2,z_Q2]).T)

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
x_Q1=np.zeros(npts,dtype=np.float64)   
z_Q1=np.zeros(npts,dtype=np.float64)   
x_Q2=np.zeros(npts,dtype=np.float64)   
z_Q2=np.zeros(npts,dtype=np.float64)   

for i in range(0,npts):

    rr=random.uniform(-1,1)
    ss=1

    r[i]=rr
    s[i]=ss

    N1=0.25*(1-rr)*(1-ss)
    N2=0.25*(1+rr)*(1-ss)
    N3=0.25*(1+rr)*(1+ss)
    N4=0.25*(1-rr)*(1+ss)

    rad[i]=N1*rad1+N2*rad2+N3*rad3+N4*rad4
    theta[i]=N1*theta1+N2*theta2+N3*theta3+N4*theta4

    x_DJ[i]=rad[i]*np.sin(theta[i])
    z_DJ[i]=rad[i]*np.cos(theta[i])

    x_Q1[i]=N1*x1+N2*x2+N3*x3+N4*x4
    z_Q1[i]=N1*z1+N2*z2+N3*z3+N4*z4

    N1= 0.5*rr*(rr-1.) * 0.5*ss*(ss-1.)
    N2= 0.5*rr*(rr+1.) * 0.5*ss*(ss-1.)
    N3= 0.5*rr*(rr+1.) * 0.5*ss*(ss+1.)
    N4= 0.5*rr*(rr-1.) * 0.5*ss*(ss+1.)
    N5=     (1.-rr**2) * 0.5*ss*(ss-1.)
    N6= 0.5*rr*(rr+1.) *     (1.-ss**2)
    N7=     (1.-rr**2) * 0.5*ss*(ss+1.)
    N8= 0.5*rr*(rr-1.) *     (1.-ss**2)
    N9=     (1.-rr**2) *     (1.-ss**2)

    x_Q2[i]=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6+N7*x7+N8*x8*+N9*x9
    z_Q2[i]=N1*z1+N2*z2+N3*z3+N4*z4+N5*z5+N6*z6+N7*z7+N8*z8*+N9*z9

np.savetxt('polar_boundary.ascii',np.array([rad,theta]).T)
np.savetxt('xz_DJ_boundary.ascii',np.array([x_DJ,z_DJ,np.sqrt(x_DJ**2+z_DJ**2)]).T)
np.savetxt('xz_Q1_boundary.ascii',np.array([x_Q1,z_Q1,np.sqrt(x_Q1**2+z_Q1**2)]).T)
np.savetxt('xz_Q2_boundary.ascii',np.array([x_Q2,z_Q2,np.sqrt(x_Q2**2+z_Q2**2)]).T)

#----------------------------------------------------------

