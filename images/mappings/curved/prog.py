import numpy as np
import random

###############################################################################

def Q3(r):
    return (-1    +r +9*r**2 - 9*r**3)/16,\
           (+9 -27*r -9*r**2 +27*r**3)/16,\
           (+9 +27*r -9*r**2 -27*r**3)/16,\
           (-1    -r +9*r**2 + 9*r**3)/16

def Q4(r):
    return (    r -   r**2 -4*r**3 + 4*r**4)/6,\
           ( -8*r +16*r**2 +8*r**3 -16*r**4)/6,\
           (1     - 5*r**2         + 4*r**4)  ,\
           (  8*r +16*r**2 -8*r**3 -16*r**4)/6,\
           (   -r -   r**2 +4*r**3 + 4*r**4)/6

###############################################################################

def dNdr(r,s,space):
    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-s) 
       val[1]=+0.25*(1.-s) 
       val[2]=+0.25*(1.+s) 
       val[3]=-0.25*(1.+s) 
    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[0]= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       val[1]= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       val[2]= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       val[3]= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       val[4]=       (-2.*r) * 0.5*s*(s-1)
       val[5]= 0.5*(2.*r+1.) *   (1.-s**2)
       val[6]=       (-2.*r) * 0.5*s*(s+1)
       val[7]= 0.5*(2.*r-1.) *   (1.-s**2)
       val[8]=       (-2.*r) *   (1.-s**2)
    if space=='Q3':
       val=np.zeros(16,dtype=np.float64)
       dN1rdr=( +1 +18*r -27*r**2)/16
       dN2rdr=(-27 -18*r +81*r**2)/16
       dN3rdr=(+27 -18*r -81*r**2)/16
       dN4rdr=( -1 +18*r +27*r**2)/16
       N1s=(-1    +s +9*s**2 - 9*s**3)/16
       N2s=(+9 -27*s -9*s**2 +27*s**3)/16
       N3s=(+9 +27*s -9*s**2 -27*s**3)/16
       N4s=(-1    -s +9*s**2 + 9*s**3)/16
       val[ 0]=dN1rdr*N1s ; val[ 1]=dN2rdr*N1s ; val[ 2]=dN3rdr*N1s ; val[ 3]=dN4rdr*N1s
       val[ 4]=dN1rdr*N2s ; val[ 5]=dN2rdr*N2s ; val[ 6]=dN3rdr*N2s ; val[ 7]=dN4rdr*N2s
       val[ 8]=dN1rdr*N3s ; val[ 9]=dN2rdr*N3s ; val[10]=dN3rdr*N3s ; val[11]=dN4rdr*N3s
       val[12]=dN1rdr*N4s ; val[13]=dN2rdr*N4s ; val[14]=dN3rdr*N4s ; val[15]=dN4rdr*N4s
    if space=='Q4':
       val=np.zeros(25,dtype=np.float64)
       dN1dr=(    1 - 2*r -12*r**2 +16*r**3)/6
       dN2dr=(   -8 +32*r +24*r**2 -64*r**3)/6
       dN3dr=(      -10*r          +16*r**3)
       dN4dr=(  8   +32*r -24*r**2 -64*r**3)/6
       dN5dr=(   -1 - 2*r +12*r**2 +16*r**3)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4)
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       val[ 0]=dN1dr*N1s ; val[ 1]=dN2dr*N1s ; val[ 2]=dN3dr*N1s ; val[ 3]=dN4dr*N1s ; val[ 4]=dN5dr*N1s
       val[ 5]=dN1dr*N2s ; val[ 6]=dN2dr*N2s ; val[ 7]=dN3dr*N2s ; val[ 8]=dN4dr*N2s ; val[ 9]=dN5dr*N2s
       val[10]=dN1dr*N3s ; val[11]=dN2dr*N3s ; val[12]=dN3dr*N3s ; val[13]=dN4dr*N3s ; val[14]=dN5dr*N3s
       val[15]=dN1dr*N4s ; val[16]=dN2dr*N4s ; val[17]=dN3dr*N4s ; val[18]=dN4dr*N4s ; val[19]=dN5dr*N4s
       val[20]=dN1dr*N5s ; val[21]=dN2dr*N5s ; val[22]=dN3dr*N5s ; val[23]=dN4dr*N5s ; val[24]=dN5dr*N5s
    return val

def dNds(r,s,space):
    if space=='Q1':
       val=np.zeros(4,dtype=np.float64)
       val[0]=-0.25*(1.-r)
       val[1]=-0.25*(1.+r)
       val[2]=+0.25*(1.+r)
       val[3]=+0.25*(1.-r)
    if space=='Q2':
       val=np.zeros(9,dtype=np.float64)
       val[0]= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       val[1]= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       val[2]= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       val[3]= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       val[4]=    (1.-r**2) * 0.5*(2.*s-1.)
       val[5]= 0.5*r*(r+1.) *       (-2.*s)
       val[6]=    (1.-r**2) * 0.5*(2.*s+1.)
       val[7]= 0.5*r*(r-1.) *       (-2.*s)
       val[8]=    (1.-r**2) *       (-2.*s)
    if space=='Q3':
       val=np.zeros(16,dtype=np.float64)
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       val[ 0]=N1r*dN1sds ; val[ 1]=N2r*dN1sds ; val[ 2]=N3r*dN1sds ; val[ 3]=N4r*dN1sds
       val[ 4]=N1r*dN2sds ; val[ 5]=N2r*dN2sds ; val[ 6]=N3r*dN2sds ; val[ 7]=N4r*dN2sds
       val[ 8]=N1r*dN3sds ; val[ 9]=N2r*dN3sds ; val[10]=N3r*dN3sds ; val[11]=N4r*dN3sds
       val[12]=N1r*dN4sds ; val[13]=N2r*dN4sds ; val[14]=N3r*dN4sds ; val[15]=N4r*dN4sds
    if space=='Q4':
       val=np.zeros(25,dtype=np.float64)
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4)
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       dN1ds=(    1 - 2*s -12*s**2 +16*s**3)/6
       dN2ds=( -8*1 +32*s +24*s**2 -64*s**3)/6
       dN3ds=(      -10*s          +16*s**3)
       dN4ds=(  8   +32*s -24*s**2 -64*s**3)/6
       dN5ds=(   -1 - 2*s +12*s**2 +16*s**3)/6
       val[ 0]=N1r*dN1ds ; val[ 1]=N2r*dN1ds ; val[ 2]=N3r*dN1ds ; val[ 3]=N4r*dN1ds ; val[ 4]=N5r*dN1ds
       val[ 5]=N1r*dN2ds ; val[ 6]=N2r*dN2ds ; val[ 7]=N3r*dN2ds ; val[ 8]=N4r*dN2ds ; val[ 9]=N5r*dN2ds
       val[10]=N1r*dN3ds ; val[11]=N2r*dN3ds ; val[12]=N3r*dN3ds ; val[13]=N4r*dN3ds ; val[14]=N5r*dN3ds
       val[15]=N1r*dN4ds ; val[16]=N2r*dN4ds ; val[17]=N3r*dN4ds ; val[18]=N4r*dN4ds ; val[19]=N5r*dN4ds
       val[20]=N1r*dN5ds ; val[21]=N2r*dN5ds ; val[22]=N3r*dN5ds ; val[23]=N4r*dN5ds ; val[24]=N5r*dN5ds
    return val

###############################################################################

theta1=23./180.*np.pi
theta2=52./180.*np.pi
R1=1.
R2=1.5

area_th=0.5*(theta2-theta1)*(R2**2-R1**2)

print('area=',area_th)

npts=10000
rr=np.zeros(npts,dtype=np.float64)   
ss=np.zeros(npts,dtype=np.float64)   
xx=np.zeros(npts,dtype=np.float64)   
yy=np.zeros(npts,dtype=np.float64)   

volume=False

###############################################################################
print('**********Q1*********')
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

if volume:
   np.savetxt('xy1_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy1_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqperdim in range(3,6):

   if nqperdim==3:
      coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      weights=[5./9.,8./9.,5./9.]
   if nqperdim==4:
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]
   if nqperdim==5:
      qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
      qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
      qc5c=0.
      qw5a=(322.-13.*np.sqrt(70.))/900.
      qw5b=(322.+13.*np.sqrt(70.))/900.
      qw5c=128./225.
      coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
      weights=[qw5a,qw5b,qw5c,qw5b,qw5a]
    
   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   area=0
   for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'Q1')
        dNNNVds=dNds(rq,sq,'Q1')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

   print('nqperdim=',nqperdim,area,'rel. error',(area-area_th)/area_th)

###############################################################################
print('**********Q2*********')
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

#np.savetxt('rs2.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy2_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy2_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqperdim in range(3,6):

   if nqperdim==3:
      coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      weights=[5./9.,8./9.,5./9.]
   if nqperdim==4:
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]
   if nqperdim==5:
      qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
      qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
      qc5c=0.
      qw5a=(322.-13.*np.sqrt(70.))/900.
      qw5b=(322.+13.*np.sqrt(70.))/900.
      qw5c=128./225.
      coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
      weights=[qw5a,qw5b,qw5c,qw5b,qw5a]
    
   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   area=0
   for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'Q2')
        dNNNVds=dNds(rq,sq,'Q2')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

   print('nqperdim=',nqperdim,area,'rel. error',(area-area_th)/area_th)

###############################################################################
print('**********Q3*********')
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

#np.savetxt('rs3.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy3_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy3_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqperdim in range(3,6):

   if nqperdim==3:
      coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      weights=[5./9.,8./9.,5./9.]
   if nqperdim==4:
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]
   if nqperdim==5:
      qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
      qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
      qc5c=0.
      qw5a=(322.-13.*np.sqrt(70.))/900.
      qw5b=(322.+13.*np.sqrt(70.))/900.
      qw5c=128./225.
      coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
      weights=[qw5a,qw5b,qw5c,qw5b,qw5a]
    
   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   area=0
   for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'Q3')
        dNNNVds=dNds(rq,sq,'Q3')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

   print('nqperdim=',nqperdim,area,'rel. error',(area-area_th)/area_th)

###############################################################################
print('**********Q4*********')
Q=4
m=25
theta=[theta1,                      #00
       theta1,                      #01
       theta1,                      #02
       theta1,                      #03
       theta1,                      #04
       theta1+(theta2-theta1)/4.,   #05
       theta1+(theta2-theta1)/4.,   #06
       theta1+(theta2-theta1)/4.,   #07
       theta1+(theta2-theta1)/4.,   #08
       theta1+(theta2-theta1)/4.,   #09
       theta1+(theta2-theta1)/2.,   #10
       theta1+(theta2-theta1)/2.,   #11
       theta1+(theta2-theta1)/2.,   #12
       theta1+(theta2-theta1)/2.,   #13
       theta1+(theta2-theta1)/2.,   #14
       theta1+(theta2-theta1)*3/4., #15
       theta1+(theta2-theta1)*3/4., #16
       theta1+(theta2-theta1)*3/4., #17
       theta1+(theta2-theta1)*3/4., #18
       theta1+(theta2-theta1)*3/4., #19
       theta2,                      #20
       theta2,                      #21
       theta2,                      #22
       theta2,                      #23
       theta2]                      #24

r=[R1,               #00
   R1+(R2-R1)/4,     #01
   R1+(R2-R1)/2,     #02
   R1+(R2-R1)*3/4,   #03
   R2,               #04
   R1,               #00
   R1+(R2-R1)/4,     #01
   R1+(R2-R1)/2,     #02
   R1+(R2-R1)*3/4,   #03
   R2,               #04
   R1,               #00
   R1+(R2-R1)/4,     #01
   R1+(R2-R1)/2,     #02
   R1+(R2-R1)*3/4,   #03
   R2,               #04
   R1,               #00
   R1+(R2-R1)/4,     #01
   R1+(R2-R1)/2,     #02
   R1+(R2-R1)*3/4,   #03
   R2,               #04
   R1,               #00
   R1+(R2-R1)/4,     #01
   R1+(R2-R1)/2,     #02
   R1+(R2-R1)*3/4,   #03
   R2]               #04

x=r*np.cos(theta)
y=r*np.sin(theta)

np.savetxt('xy_Q4.ascii',np.array([x,y]).T)

for i in range(0,npts):
    
    # compute random r,s coordinates

    if volume:
       rr[i]=random.uniform(-1.,+1)
       ss[i]=random.uniform(-1.,+1)
    else:
       rr[i]=-1.
       ss[i]=-1.+2./(npts-1)*i

    # compute basis function values at r,s

    N0r,N1r,N2r,N3r,N4r=Q4(rr[i])
    N0s,N1s,N2s,N3s,N4s=Q4(ss[i])

    N00=N0r*N0s
    N01=N1r*N0s
    N02=N2r*N0s
    N03=N3r*N0s
    N04=N4r*N0s

    N05=N0r*N1s
    N06=N1r*N1s
    N07=N2r*N1s
    N08=N3r*N1s
    N09=N4r*N1s

    N10=N0r*N2s
    N11=N1r*N2s
    N12=N2r*N2s
    N13=N3r*N2s
    N14=N4r*N2s

    N15=N0r*N3s
    N16=N1r*N3s
    N17=N2r*N3s
    N18=N3r*N3s
    N19=N4r*N3s

    N20=N0r*N4s
    N21=N1r*N4s
    N22=N2r*N4s
    N23=N3r*N4s
    N24=N4r*N4s

    # compute x,y coordinates
   
    xx[i]=N00*x[0]+N01*x[1]+N02*x[2]+N03*x[3]+N04*x[4]+N05*x[5]+N06*x[6]+N07*x[7]+N08*x[8]\
         +N09*x[9]+N10*x[10]+N11*x[11]+N12*x[12]+N13*x[13]+N14*x[14]+N15*x[15]+N16*x[16]\
         +N17*x[17]+N17*x[18]+N17*x[19]+N20*x[20]+N21*x[21]+N22*x[22]+N23*x[23]+N24*x[24]
    yy[i]=N00*y[0]+N01*y[1]+N02*y[2]+N03*y[3]+N04*y[4]+N05*y[5]+N06*y[6]+N07*y[7]+N08*y[8]\
         +N09*y[9]+N10*y[10]+N11*y[11]+N12*y[12]+N13*y[13]+N14*y[14]+N15*y[15]+N16*y[16]\
         +N17*y[17]+N17*y[18]+N17*y[19]+N20*y[20]+N21*y[21]+N22*y[22]+N23*y[23]+N24*y[24]

#np.savetxt('rs4.ascii',np.array([rr,ss]).T)

if volume:
   np.savetxt('xy4_volume.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)
else:
   np.savetxt('xy4_line.ascii',np.array([xx,yy,np.sqrt(xx**2+yy**2)]).T)

#######################################

for nqperdim in range(3,6):

   if nqperdim==3:
      coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      weights=[5./9.,8./9.,5./9.]
   if nqperdim==4:
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]
   if nqperdim==5:
      qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
      qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
      qc5c=0.
      qw5a=(322.-13.*np.sqrt(70.))/900.
      qw5b=(322.+13.*np.sqrt(70.))/900.
      qw5c=128./225.
      coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
      weights=[qw5a,qw5b,qw5c,qw5b,qw5a]
    
   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   area=0
   for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        dNNNVdr=dNdr(rq,sq,'Q4')
        dNNNVds=dNds(rq,sq,'Q4')
        jcb=np.zeros((2,2),dtype=np.float64)
        jcb[0,0]=np.dot(dNNNVdr[:],x[:])
        jcb[0,1]=np.dot(dNNNVdr[:],y[:])
        jcb[1,0]=np.dot(dNNNVds[:],x[:])
        jcb[1,1]=np.dot(dNNNVds[:],y[:])
        jcobq=np.linalg.det(jcb)
        area+=jcobq*weightq

   print('nqperdim=',nqperdim,area,'rel. error',(area-area_th)/area_th)


