import numpy as np
import random

def Q3(r):
    return (-1    +r +9*r**2 - 9*r**3)/16,\
           (+9 -27*r -9*r**2 +27*r**3)/16,\
           (+9 +27*r -9*r**2 -27*r**3)/16,\
           (-1    -r +9*r**2 + 9*r**3)/16

theta1=23./180.*np.pi
theta2=52./180.*np.pi
R1=1.
R2=1.5

npts=10000
rr=np.zeros(npts,dtype=np.float64)   
ss=np.zeros(npts,dtype=np.float64)   
xx=np.zeros(npts,dtype=np.float64)   
yy=np.zeros(npts,dtype=np.float64)   

volume=False

###############################################
Q=1
m=4
theta=[theta1,theta1,theta2,theta2]
r=[R1,R2,R2,R1]
x=r*np.cos(theta)
y=r*np.sin(theta)
np.savetxt('xy_Q1.ascii',np.array([x,y]).T)

for i in range(0,npts):
    
    # compute random r,s coordinates

    if volume:
       rr[i]=random.uniform(-1.,+1)
       ss[i]=random.uniform(-1.,+1)
    else:
       rr[i]=-1.
       ss[i]=-1.+2./(npts-1)*i

    # compute basis function values at r,s

    N0=0.25*(1-rr[i])*(1-ss[i])
    N1=0.25*(1+rr[i])*(1-ss[i])
    N2=0.25*(1+rr[i])*(1+ss[i])
    N3=0.25*(1-rr[i])*(1+ss[i])

    # compute x,y coordinates
   
    xx[i]=N0*x[0]+N1*x[1]+N2*x[2]+N3*x[3]
    yy[i]=N0*y[0]+N1*y[1]+N2*y[2]+N3*y[3]

np.savetxt('rs1.ascii',np.array([rr,ss]).T)
np.savetxt('xy1.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

###############################################
Q=2
m=9
theta=[theta1,             #00
       theta1,             #01
       theta2,             #02
       theta2,             #03
       theta1,             #04
       (theta1+theta2)/2., #05
       theta2,             #06
       (theta1+theta2)/2., #07
       (theta1+theta2)/2.] #08

r=[R1,         #00
   R2,         #01
   R2,         #02
   R1,         #03
   (R1+R2)/2., #04
   R2,         #05
   (R1+R2)/2., #06
   R1,         #07
   (R1+R2)/2.] #08

x=r*np.cos(theta)
y=r*np.sin(theta)
np.savetxt('xy_Q2.ascii',np.array([x,y]).T)


for i in range(0,npts):
    
    # compute random r,s coordinates

    if volume:
       rr[i]=random.uniform(-1.,+1)
       ss[i]=random.uniform(-1.,+1)
    else:
       rr[i]=-1.
       ss[i]=-1.+2./(npts-1)*i

    # compute basis function values at r,s

    N0= 0.5*rr[i]*(rr[i]-1.) * 0.5*ss[i]*(ss[i]-1.)
    N1= 0.5*rr[i]*(rr[i]+1.) * 0.5*ss[i]*(ss[i]-1.)
    N2= 0.5*rr[i]*(rr[i]+1.) * 0.5*ss[i]*(ss[i]+1.)
    N3= 0.5*rr[i]*(rr[i]-1.) * 0.5*ss[i]*(ss[i]+1.)
    N4=     (1.-rr[i]**2) * 0.5*ss[i]*(ss[i]-1.)
    N5= 0.5*rr[i]*(rr[i]+1.) *     (1.-ss[i]**2)
    N6=     (1.-rr[i]**2) * 0.5*ss[i]*(ss[i]+1.)
    N7= 0.5*rr[i]*(rr[i]-1.) *     (1.-ss[i]**2)
    N8=     (1.-rr[i]**2) *     (1.-ss[i]**2)

    # compute x,y coordinates
   
    xx[i]=N0*x[0]+N1*x[1]+N2*x[2]+N3*x[3]+N4*x[4]+N5*x[5]+N6*x[6]+N7*x[7]+N8*x[8]
    yy[i]=N0*y[0]+N1*y[1]+N2*y[2]+N3*y[3]+N4*y[4]+N5*y[5]+N6*y[6]+N7*y[7]+N8*y[8]

np.savetxt('rs2.ascii',np.array([rr,ss]).T)
np.savetxt('xy2.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

###############################################
Q=3
m=16
theta=[theta1,                      #00
       theta1,                      #01
       theta1,                      #02
       theta1,                      #03
       theta1+(theta2-theta1)/3.,   #04
       theta1+(theta2-theta1)/3.,   #05
       theta1+(theta2-theta1)/3.,   #06
       theta1+(theta2-theta1)/3.,   #07
       theta1+(theta2-theta1)*2/3., #08
       theta1+(theta2-theta1)*2/3., #09
       theta1+(theta2-theta1)*2/3., #10
       theta1+(theta2-theta1)*2/3., #11
       theta2,                      #12
       theta2,                      #13
       theta2,                      #14
       theta2]                      #15

r=[R1,               #00
   R1+(R2-R1)/3.,    #01
   R1+(R2-R1)*2./3., #02
   R2,               #03
   R1,               #04
   R1+(R2-R1)/3.,    #05
   R1+(R2-R1)*2./3., #06
   R2,               #07
   R1,               #08
   R1+(R2-R1)/3.,    #09
   R1+(R2-R1)*2./3., #10
   R2,               #11
   R1,               #12
   R1+(R2-R1)/3.,    #13
   R1+(R2-R1)*2./3., #14
   R2]               #15

x=r*np.cos(theta)
y=r*np.sin(theta)
np.savetxt('xy_Q3.ascii',np.array([x,y]).T)

for i in range(0,npts):
    
    # compute random r,s coordinates

    if volume:
       rr[i]=random.uniform(-1.,+1)
       ss[i]=random.uniform(-1.,+1)
    else:
       rr[i]=-1.
       ss[i]=-1.+2./(npts-1)*i

    # compute basis function values at r,s

    N0r,N1r,N2r,N3r=Q3(rr[i])
    N0s,N1s,N2s,N3s=Q3(ss[i])
    N00=N0r*N0s
    N01=N1r*N0s
    N02=N2r*N0s
    N03=N3r*N0s
    N04=N0r*N1s
    N05=N1r*N1s
    N06=N2r*N1s
    N07=N3r*N1s
    N08=N0r*N2s
    N09=N1r*N2s
    N10=N2r*N2s
    N11=N3r*N2s
    N12=N0r*N3s
    N13=N1r*N3s
    N14=N2r*N3s
    N15=N3r*N3s

    # compute x,y coordinates
   
    xx[i]=N00*x[0]+N01*x[1]+N02*x[2]+N03*x[3]+N04*x[4]+N05*x[5]+N06*x[6]+N07*x[7]+N08*x[8]\
         +N09*x[9]+N10*x[10]+N11*x[11]+N12*x[12]+N13*x[13]+N14*x[14]+N15*x[15]
    yy[i]=N00*y[0]+N01*y[1]+N02*y[2]+N03*y[3]+N04*y[4]+N05*y[5]+N06*y[6]+N07*y[7]+N08*y[8]\
         +N09*y[9]+N10*y[10]+N11*y[11]+N12*y[12]+N13*y[13]+N14*y[14]+N15*y[15]

np.savetxt('rs3.ascii',np.array([rr,ss]).T)
np.savetxt('xy3.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
