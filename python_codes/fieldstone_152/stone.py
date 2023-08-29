import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg import *
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

###############################################################################

def NNN(r,s,space):
    if space=='Q1':
       val = np.zeros(4,dtype=np.float64)
       val[0]=0.25*(1.-r)*(1.-s)
       val[1]=0.25*(1.+r)*(1.-s)
       val[2]=0.25*(1.+r)*(1.+s)
       val[3]=0.25*(1.-r)*(1.+s)
    if space=='Q2':
       val = np.zeros(9,dtype=np.float64)
       val[0]= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       val[1]= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       val[2]= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       val[3]= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       val[4]=    (1.-r**2) * 0.5*s*(s-1.)
       val[5]= 0.5*r*(r+1.) *    (1.-s**2)
       val[6]=    (1.-r**2) * 0.5*s*(s+1.)
       val[7]= 0.5*r*(r-1.) *    (1.-s**2)
       val[8]=    (1.-r**2) *    (1.-s**2)
    if space=='Q3':
       val = np.zeros(16,dtype=np.float64)
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       val[ 0]= N1r*N1t ; val[ 1]= N2r*N1t ; val[ 2]= N3r*N1t ; val[ 3]= N4r*N1t
       val[ 4]= N1r*N2t ; val[ 5]= N2r*N2t ; val[ 6]= N3r*N2t ; val[ 7]= N4r*N2t
       val[ 8]= N1r*N3t ; val[ 9]= N2r*N3t ; val[10]= N3r*N3t ; val[11]= N4r*N3t
       val[12]= N1r*N4t ; val[13]= N2r*N4t ; val[14]= N3r*N4t ; val[15]= N4r*N4t
    if space=='Q4':
       val = np.zeros(25,dtype=np.float64)
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4)
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4)
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       val[0] = N1r*N1s ; val[1] = N2r*N1s ; val[2] = N3r*N1s ; val[3] = N4r*N1s ; val[4] = N5r*N1s
       val[5] = N1r*N2s ; val[6] = N2r*N2s ; val[7] = N3r*N2s ; val[8] = N4r*N2s ; val[9] = N5r*N2s
       val[10]= N1r*N3s ; val[11]= N2r*N3s ; val[12]= N3r*N3s ; val[13]= N4r*N3s ; val[14]= N5r*N3s
       val[15]= N1r*N4s ; val[16]= N2r*N4s ; val[17]= N3r*N4s ; val[18]= N4r*N4s ; val[19]= N5r*N4s
       val[20]= N1r*N5s ; val[21]= N2r*N5s ; val[22]= N3r*N5s ; val[23]= N4r*N5s ; val[24]= N5r*N5s
    return val

def dNNNdr(r,s,space):
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
       val[ 0]=dN1dr*N1s ; val[ 1]=dN2dr*N1s ; val[ 2]=dN3dr*N1s  
       val[ 3]=dN4dr*N1s ; val[ 4]=dN5dr*N1s ; val[ 5]=dN1dr*N2s  
       val[ 6]=dN2dr*N2s ; val[ 7]=dN3dr*N2s ; val[ 8]=dN4dr*N2s  
       val[ 9]=dN5dr*N2s ; val[10]=dN1dr*N3s ; val[11]=dN2dr*N3s 
       val[12]=dN3dr*N3s ; val[13]=dN4dr*N3s ; val[14]=dN5dr*N3s 
       val[15]=dN1dr*N4s ; val[16]=dN2dr*N4s ; val[17]=dN3dr*N4s  
       val[18]=dN4dr*N4s ; val[19]=dN5dr*N4s ; val[20]=dN1dr*N5s  
       val[21]=dN2dr*N5s ; val[22]=dN3dr*N5s ; val[23]=dN4dr*N5s  
       val[24]=dN5dr*N5s
   return val

def dNNNds(r,s,space):
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
       val[ 0]=N1r*dN1ds ; val[ 1]=N2r*dN1ds ; val[ 2]=N3r*dN1ds 
       val[ 3]=N4r*dN1ds ; val[ 4]=N5r*dN1ds ; val[ 5]=N1r*dN2ds 
       val[ 6]=N2r*dN2ds ; val[ 7]=N3r*dN2ds ; val[ 8]=N4r*dN2ds 
       val[ 9]=N5r*dN2ds ; val[10]=N1r*dN3ds ; val[11]=N2r*dN3ds 
       val[12]=N3r*dN3ds ; val[13]=N4r*dN3ds ; val[14]=N5r*dN3ds
       val[15]=N1r*dN4ds ; val[16]=N2r*dN4ds ; val[17]=N3r*dN4ds 
       val[18]=N4r*dN4ds ; val[19]=N5r*dN4ds ; val[20]=N1r*dN5ds 
       val[21]=N2r*dN5ds ; val[22]=N3r*dN5ds ; val[23]=N4r*dN5ds 
       val[24]=N5r*dN5ds
   return val

###############################################################################

def density(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
    gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
    alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
    val=k*math.sin(k*theta)*alephr + rho0 
    return val

def Psi(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    val=-r*gr*math.cos(k*theta)
    return val

def velocity_x(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.cos(theta)-vtheta*math.sin(theta)
    return val

def velocity_y(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.sin(theta)+vtheta*math.cos(theta)
    return val

def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    return val

def sr_xx(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.cos(theta))**2\
       +ett*(math.sin(theta))**2\
       -2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_yy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.sin(theta))**2\
       +ett*(math.cos(theta))**2\
       +2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_xy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=ert*(math.cos(theta)**2-math.sin(theta)**2)\
       +(err-ett)*math.cos(theta)*math.sin(theta)
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

###############################################################################

print("-----------------------------")
print("--------stone 152------------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=9     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 5):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
   nqperdim = int(sys.argv[3])
   mapping = int(sys.argv[4])
   if mapping==1: mapping='Q1'
   if mapping==2: mapping='Q2'
   if mapping==3: mapping='Q3'
   if mapping==4: mapping='Q4'
else:
   nelr = 20
   visu = 1
   nqperdim=3
   mapping='Q2' 

R1=1.
R2=2.

dr=(R2-R1)/nelr
nelt=12*nelr 
nel=nelr*nelt  

rho0=0.
kk=4
g0=1.

viscosity=1.  # dynamic viscosity \eta

eps=1.e-10

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

###############################################################################

if nqperdim==2:
   coords=[-1/np.sqrt(3),1/np.sqrt(3)]
   weights=[1,1]

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


###############################################################################
# grid point setup
###############################################################################
start = timing.time()

nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates
r=np.empty(nnp,dtype=np.float64)  
theta=np.empty(nnp,dtype=np.float64) 

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sz = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        xV[counter]=i*sx
        yV[counter]=j*sz
        counter += 1

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=xV[counter]
        yi=yV[counter]
        t=xi/Louter*2.*math.pi    
        xV[counter]=math.cos(t)*(R1+yi)
        yV[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(yV[counter],xV[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1

print("building coordinate arrays (%.3fs)" % (timing.time() - start))

###############################################################################
# build iconQ1 array needed for vtu file
###############################################################################

iconQ1 =np.zeros((4,nel),dtype=np.int32)
counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconQ1[0,counter] = icon2 
        iconQ1[1,counter] = icon1
        iconQ1[2,counter] = icon4
        iconQ1[3,counter] = icon3
        counter += 1
    #end for

###############################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
###############################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4

NfemV=nnp*ndofV           # Total number of degrees of V freedom 
NfemP=nelt*(nelr+1)*ndofP # Total number of degrees of P freedom
Nfem=NfemV+NfemP          # total number of dofs

print('nelr=',nelr)
print('nelr=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('nqel=',nqel)
print('mapping=',mapping)

###############################################################################
# connectivity
###############################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        iconV[0,counter]=2*counter+2 +2*j*nelt
        iconV[1,counter]=2*counter   +2*j*nelt
        iconV[2,counter]=iconV[1,counter]+4*nelt
        iconV[3,counter]=iconV[1,counter]+4*nelt+2
        iconV[4,counter]=iconV[0,counter]-1
        iconV[5,counter]=iconV[1,counter]+2*nelt
        iconV[6,counter]=iconV[2,counter]+1
        iconV[7,counter]=iconV[5,counter]+2
        iconV[8,counter]=iconV[5,counter]+1
        if i==nelt-1:
           iconV[0,counter]-=2*nelt
           iconV[7,counter]-=2*nelt
           iconV[3,counter]-=2*nelt
        #print(j,i,counter,'|',iconV[0:mV,counter])
        counter += 1


iconP =np.zeros((mP,nel),dtype=np.int32)
counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconP[0,counter] = icon2 
        iconP[1,counter] = icon1
        iconP[2,counter] = icon4
        iconP[3,counter] = icon3
        counter += 1
    #end for


#for iel in range(0,nel):
#    print(iel,'|',iconP[:,iel])

#now that I have both connectivity arrays I can 
# easily build xP,yP

NP=NfemP
xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]]
    yP[iconP[0,iel]]=yV[iconV[0,iel]]
    yP[iconP[1,iel]]=yV[iconV[1,iel]]
    yP[iconP[2,iel]]=yV[iconV[2,iel]]
    yP[iconP[3,iel]]=yV[iconV[3,iel]]

print("building connectivity array (%.3fs)" % (timing.time() - start))

###############################################################################
# Q1 nodes for mapping are corners of Q2 basis functions
# Q2 nodes for mapping are same as Q2 basis functions
# Q3 nodes for mapping are built
# Q4 nodes for mapping are built
###############################################################################

if mapping=='Q1':
   mmaping=4
   xmapping=np.zeros((4,nel),dtype=np.float64)
   ymapping=np.zeros((4,nel),dtype=np.float64)
   for iel in range(0,nel):
       xmapping[0,iel]=xV[iconV[0,iel]] ; ymapping[0,iel]=yV[iconV[0,iel]]
       xmapping[1,iel]=xV[iconV[1,iel]] ; ymapping[1,iel]=yV[iconV[1,iel]]
       xmapping[2,iel]=xV[iconV[2,iel]] ; ymapping[2,iel]=yV[iconV[2,iel]]
       xmapping[3,iel]=xV[iconV[3,iel]] ; ymapping[3,iel]=yV[iconV[3,iel]]

if mapping=='Q2':
   mmaping=9
   xmapping=np.zeros((9,nel),dtype=np.float64)
   ymapping=np.zeros((9,nel),dtype=np.float64)
   for iel in range(0,nel):
       xmapping[0,iel]=xV[iconV[0,iel]] ; ymapping[0,iel]=yV[iconV[0,iel]]
       xmapping[1,iel]=xV[iconV[1,iel]] ; ymapping[1,iel]=yV[iconV[1,iel]]
       xmapping[2,iel]=xV[iconV[2,iel]] ; ymapping[2,iel]=yV[iconV[2,iel]]
       xmapping[3,iel]=xV[iconV[3,iel]] ; ymapping[3,iel]=yV[iconV[3,iel]]
       xmapping[4,iel]=xV[iconV[4,iel]] ; ymapping[4,iel]=yV[iconV[4,iel]]
       xmapping[5,iel]=xV[iconV[5,iel]] ; ymapping[5,iel]=yV[iconV[5,iel]]
       xmapping[6,iel]=xV[iconV[6,iel]] ; ymapping[6,iel]=yV[iconV[6,iel]]
       xmapping[7,iel]=xV[iconV[7,iel]] ; ymapping[7,iel]=yV[iconV[7,iel]]
       xmapping[8,iel]=xV[iconV[8,iel]] ; ymapping[8,iel]=yV[iconV[8,iel]]

if mapping=='Q3':
   mmapping=16
   dtheta=2*np.pi/nelt/3
   xmapping=np.zeros((16,nel),dtype=np.float64)
   ymapping=np.zeros((16,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=r[iconV[0,iel]]
       rmax=r[iconV[2,iel]]
       counter=0
       for j in range(0,4):
           for i in range(0,4):
               ttt=thetamin-i*dtheta
               rrr=rmin+j*(rmax-rmin)/3
               xmapping[counter,iel]=math.cos(ttt)*rrr
               ymapping[counter,iel]=math.sin(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1

if mapping=='Q4':
   mmapping=25
   dtheta=2*np.pi/nelt/4
   xmapping=np.zeros((25,nel),dtype=np.float64)
   ymapping=np.zeros((25,nel),dtype=np.float64)
   for iel in range(0,nel):
       thetamin=theta[iconV[0,iel]]
       rmin=r[iconV[0,iel]]
       rmax=r[iconV[2,iel]]
       counter=0
       for j in range(0,5):
           for i in range(0,5):
               ttt=thetamin-i*dtheta
               rrr=rmin+j*(rmax-rmin)/4
               xmapping[counter,iel]=math.cos(ttt)*rrr
               ymapping[counter,iel]=math.sin(ttt)*rrr
               #print(xmapping[counter,iel],ymapping[counter,iel])
               counter+=1


###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix = np.zeros(Nfem, dtype=bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0,nnp):
    if r[i]<R1+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)
    if r[i]>(R2-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

###############################################################################
# compute area and volume of elements
# note that for the volume calculation we only consider the elements x>0
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
vol=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):
    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNN(rq,sq,mapping)
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
        if xq>0: vol[iel]+=jcob*weightq*2*np.pi*xq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.12e | nel= %d" %(area.sum(),nel))
print("     -> total area (anal) %e " %(np.pi*(R2**2-R1**2)))
print("     -> total volume (meas) %.12e | nel= %d" %(vol.sum(),nel))
print("     -> total volume (anal) %e" %(4*np.pi/3*(R2**3-R1**3)))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
dNNNVdx  = np.zeros(mV,dtype=np.float64) 
dNNNVdy  = np.zeros(mV,dtype=np.float64) 
#NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
#NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
#dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
#dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        # compute dNdx & dNdy
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for 

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]
        #end for 

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
            f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
        #end for 

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.
        #end for 

        G_el-=b_mat.T.dot(N_mat)*weightq*jcob

    #end for kq

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

#end for iel

print("build FE matrixs & rhs (%.3fs)" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs
    
sparse_matrix=A_sparse.tocsr()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (timing.time() - start))


###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (timing.time() - start))

###############################################################################
# compute strain rate - center to nodes - method 1
###############################################################################

count = np.zeros(nnp,dtype=np.int32)  
Lxx1 = np.zeros(nnp,dtype=np.float64)  
Lxy1 = np.zeros(nnp,dtype=np.float64)  
Lyx1 = np.zeros(nnp,dtype=np.float64)  
Lyy1 = np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):

    rq=qcoords_r[kq]
    sq=qcoords_s[kq]
    weightq=qweights[kq]

    #compute coords of quadrature points
    NNNV=NNN(rq,sq,mapping)
    xq=np.dot(NNNV[:],xmapping[:,iel])
    yq=np.dot(NNNV[:],ymapping[:,iel])

    #compute jacobian matrix
    dNNNVdr=dNNNdr(rq,sq,mapping)
    dNNNVds=dNNNds(rq,sq,mapping)
    jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
    jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
    jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
    jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    #basis functions
    dNNNVdr=dNNNdr(rq,sq,'Q2')
    dNNNVds=dNNNds(rq,sq,'Q2')

    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    #end for

    L_xx=0.
    L_xy=0.
    L_yx=0.
    L_yy=0.
    for k in range(0,mV):
        L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
        L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
        L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
        L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
    #end for
    for i in range(0,mV):
        inode=iconV[i,iel]
        Lxx1[inode]+=L_xx
        Lxy1[inode]+=L_xy
        Lyx1[inode]+=L_yx
        Lyy1[inode]+=L_yy
        count[inode]+=1
    #end for
#end for
Lxx1/=count
Lxy1/=count
Lyx1/=count
Lyy1/=count

print("     -> Lxx1 (m,M) %.4f %.4f " %(np.min(Lxx1),np.max(Lxx1)))
print("     -> Lyy1 (m,M) %.4f %.4f " %(np.min(Lyy1),np.max(Lyy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lxy1),np.max(Lxy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lyx1),np.max(Lyx1)))

print("compute vel gradient meth-1 (%.3fs)" % (timing.time() - start))

exx1 = np.zeros(nnp,dtype=np.float64)  
eyy1 = np.zeros(nnp,dtype=np.float64)  
exy1 = np.zeros(nnp,dtype=np.float64)  

exx1[:]=Lxx1[:]
eyy1[:]=Lyy1[:]
exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

###############################################################################
# compute strain rate - corners to nodes - method 2
###############################################################################
start = timing.time()

count = np.zeros(nnp,dtype=np.int32)  
q=np.zeros(nnp,dtype=np.float64)
Lxx2 = np.zeros(nnp,dtype=np.float64)  
Lxy2 = np.zeros(nnp,dtype=np.float64)  
Lyx2 = np.zeros(nnp,dtype=np.float64)  
Lyy2 = np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,mV):
        inode=iconV[i,iel]
        rq=rVnodes[i]
        sq=sVnodes[i]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        # compute dNdx & dNdy
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for 
	
        L_xx=0.
        L_xy=0.
        L_yx=0.
        L_yy=0.
        for k in range(0,mV):
            L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
            L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
            L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
            L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
        #end for
        Lxx2[inode]+=L_xx
        Lxy2[inode]+=L_xy
        Lyx2[inode]+=L_yx
        Lyy2[inode]+=L_yy
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

print("     -> Lxx2 (m,M) %.4f %.4f " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4f %.4f " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lyx2),np.max(Lyx2)))

#np.savetxt('pressure.ascii',np.array([xV,yV,q]).T)
#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

print("compute vel gradient meth-2 (%.3fs)" % (timing.time() - start))

exx2 = np.zeros(nnp,dtype=np.float64)  
eyy2 = np.zeros(nnp,dtype=np.float64)  
exy2 = np.zeros(nnp,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

###############################################################################
start = timing.time()

M_mat= np.zeros((nnp,nnp),dtype=np.float64)
rhsLxx=np.zeros(nnp,dtype=np.float64)
rhsLyy=np.zeros(nnp,dtype=np.float64)
rhsLxy=np.zeros(nnp,dtype=np.float64)
rhsLyx=np.zeros(nnp,dtype=np.float64)

for iel in range(0,nel):

    M_el =np.zeros((mV,mV),dtype=np.float64)
    fLxx_el=np.zeros(mV,dtype=np.float64)
    fLyy_el=np.zeros(mV,dtype=np.float64)
    fLxy_el=np.zeros(mV,dtype=np.float64)
    fLyx_el=np.zeros(mV,dtype=np.float64)
    NNNV1 =np.zeros((mV,1),dtype=np.float64) 

    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        #basis functions
        NNNV1[:,0]=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')

        # compute dNdx & dNdy
        Lxxq=0.
        Lyyq=0.
        Lxyq=0.
        Lyxq=0.
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            Lxxq+=dNNNVdx[k]*u[iconV[k,iel]]
            Lyyq+=dNNNVdy[k]*v[iconV[k,iel]]
            Lxyq+=dNNNVdx[k]*v[iconV[k,iel]]
            Lyxq+=dNNNVdy[k]*u[iconV[k,iel]]
        #end for 

        M_el +=NNNV1.dot(NNNV1.T)*weightq*jcob

        fLxx_el[:]+=NNNV1[:,0]*Lxxq*jcob*weightq
        fLyy_el[:]+=NNNV1[:,0]*Lyyq*jcob*weightq
        fLxy_el[:]+=NNNV1[:,0]*Lxyq*jcob*weightq
        fLyx_el[:]+=NNNV1[:,0]*Lyxq*jcob*weightq

    #end for kq

    for k1 in range(0,mV):
        m1=iconV[k1,iel]
        for k2 in range(0,mV):
            m2=iconV[k2,iel]
            M_mat[m1,m2]+=M_el[k1,k2]
        #end for
        rhsLxx[m1]+=fLxx_el[k1]
        rhsLyy[m1]+=fLyy_el[k1]
        rhsLxy[m1]+=fLxy_el[k1]
        rhsLyx[m1]+=fLyx_el[k1]
    #end for

#end for

Lxx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxx)
Lyy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyy)
Lxy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxy)
Lyx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyx)

print("     -> Lxx3 (m,M) %.4f %.4f " %(np.min(Lxx3),np.max(Lxx3)))
print("     -> Lyy3 (m,M) %.4f %.4f " %(np.min(Lyy3),np.max(Lyy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lxy3),np.max(Lxy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lyx3),np.max(Lyx3)))

print("compute vel gradient meth-3 (%.3fs)" % (timing.time() - start))

exx3 = np.zeros(nnp,dtype=np.float64)  
eyy3 = np.zeros(nnp,dtype=np.float64)  
exy3 = np.zeros(nnp,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

###############################################################################
# normalise pressure
###############################################################################
start = timing.time()

#print(np.sum(q[0:2*nelt])/(2*nelt))
#print(np.sum(q[nnp-2*nelt:nnp])/(2*nelt))
#print(np.sum(p[0:nelt])/(nelt))

poffset=np.sum(q[0:2*nelt])/(2*nelt)

q-=poffset
p-=poffset

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

print("normalise pressure (%.3fs)" % (timing.time() - start))

###############################################################################
# export pressure at both surfaces
###############################################################################
start = timing.time()

np.savetxt('q_R1.ascii',np.array([xV[0:2*nelt],yV[0:2*nelt],q[0:2*nelt],theta[0:2*nelt]]).T)
np.savetxt('q_R2.ascii',np.array([xV[nnp-2*nelt:nnp],\
                                  yV[nnp-2*nelt:nnp],\
                                   q[nnp-2*nelt:nnp],\
                               theta[nnp-2*nelt:nnp]]).T)

np.savetxt('p_R1.ascii',np.array([xP[0:nelt],yP[0:nelt],p[0:nelt]]).T)
np.savetxt('p_R2.ascii',np.array([xP[NP-nelt:NP],yP[NP-nelt:NP],p[NP-nelt:NP]]).T)

print("export p&q on R1,R2 (%.3fs)" % (timing.time() - start))

###############################################################################
# compute error
###############################################################################
start = timing.time()

NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

errv=0.
errp=0.
errq=0.
errexx1=0.
erreyy1=0.
errexy1=0.
errexx2=0.
erreyy2=0.
errexy2=0.
errexx3=0.
erreyy3=0.
errexy3=0.
vrms=0.
for iel in range (0,nel):

    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        #compute coords of quadrature points
        NNNV=NNN(rq,sq,mapping)
        xq=np.dot(NNNV[:],xmapping[:,iel])
        yq=np.dot(NNNV[:],ymapping[:,iel])

        #compute jacobian matrix
        dNNNVdr=dNNNdr(rq,sq,mapping)
        dNNNVds=dNNNds(rq,sq,mapping)
        jcb[0,0]=np.dot(dNNNVdr[:],xmapping[:,iel])
        jcb[0,1]=np.dot(dNNNVdr[:],ymapping[:,iel])
        jcb[1,0]=np.dot(dNNNVds[:],xmapping[:,iel])
        jcb[1,1]=np.dot(dNNNVds[:],ymapping[:,iel])
        jcob=np.linalg.det(jcb)

        #basis functions
        NNNV=NNN(rq,sq,'Q2')
        dNNNVdr=dNNNdr(rq,sq,'Q2')
        dNNNVds=dNNNds(rq,sq,'Q2')
        NNNP=NNN(rq,sq,'Q1')

        xq=0.
        yq=0.
        uq=0.
        vq=0.
        qq=0.
        exx1q=0.
        eyy1q=0.
        exy1q=0.
        exx2q=0.
        eyy2q=0.
        exy2q=0.
        exx3q=0.
        eyy3q=0.
        exy3q=0.
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
            yq+=NNNV[k]*yV[iconV[k,iel]]
            uq+=NNNV[k]*u[iconV[k,iel]]
            vq+=NNNV[k]*v[iconV[k,iel]]
            qq+=NNNV[k]*q[iconV[k,iel]]
            exx1q+=NNNV[k]*exx1[iconV[k,iel]]
            eyy1q+=NNNV[k]*eyy1[iconV[k,iel]]
            exy1q+=NNNV[k]*exy1[iconV[k,iel]]
            exx2q+=NNNV[k]*exx2[iconV[k,iel]]
            eyy2q+=NNNV[k]*eyy2[iconV[k,iel]]
            exy2q+=NNNV[k]*exy2[iconV[k,iel]]
            exx3q+=NNNV[k]*exx3[iconV[k,iel]]
            eyy3q+=NNNV[k]*eyy3[iconV[k,iel]]
            exy3q+=NNNV[k]*exy3[iconV[k,iel]]
        errv+=((uq-velocity_x(xq,yq,R1,R2,kk,rho0,g0))**2+\
               (vq-velocity_y(xq,yq,R1,R2,kk,rho0,g0))**2)*weightq*jcob
        errq+=(qq-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*weightq*jcob

        errexx1+=(exx1q-sr_xx(xq,yq,R1,R2,kk))**2*weightq*jcob
        erreyy1+=(eyy1q-sr_yy(xq,yq,R1,R2,kk))**2*weightq*jcob
        errexy1+=(exy1q-sr_xy(xq,yq,R1,R2,kk))**2*weightq*jcob
        errexx2+=(exx2q-sr_xx(xq,yq,R1,R2,kk))**2*weightq*jcob
        erreyy2+=(eyy2q-sr_yy(xq,yq,R1,R2,kk))**2*weightq*jcob
        errexy2+=(exy2q-sr_xy(xq,yq,R1,R2,kk))**2*weightq*jcob
        errexx3+=(exx3q-sr_xx(xq,yq,R1,R2,kk))**2*weightq*jcob
        erreyy3+=(eyy3q-sr_yy(xq,yq,R1,R2,kk))**2*weightq*jcob
        errexy3+=(exy3q-sr_xy(xq,yq,R1,R2,kk))**2*weightq*jcob

        vrms+=(uq**2+vq**2)*weightq*jcob

        xq=0.
        yq=0.
        pq=0.
        for k in range(0,mP):
            xq+=NNNP[k]*xP[iconP[k,iel]]
            yq+=NNNP[k]*yP[iconP[k,iel]]
            pq+=NNNP[k]*p[iconP[k,iel]]
        errp+=(pq-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*weightq*jcob

    # end for kq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)
errexx1=np.sqrt(errexx1)
erreyy1=np.sqrt(erreyy1)
errexy1=np.sqrt(errexy1)
errexx2=np.sqrt(errexx2)
erreyy2=np.sqrt(erreyy2)
errexy2=np.sqrt(errexy2)
errexx3=np.sqrt(errexx3)
erreyy3=np.sqrt(erreyy3)
errexy3=np.sqrt(errexy3)

vrms=np.sqrt(vrms/np.pi/(R2**2-R1**2))

print('     -> nelr=',nelr,' vrms=',vrms)
print("     -> nelr= %6d ; errv= %.8e ; errp= %.8e ; errq= %.8e" %(nelr,errv,errp,errq))
print("     -> nelr= %6d ; errexx1= %.8e ; erreyy1= %.8e ; errexy1= %.8e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %6d ; errexx2= %.8e ; erreyy2= %.8e ; errexy2= %.8e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %6d ; errexx3= %.8e ; erreyy3= %.8e ; errexy3= %.8e" %(nelr,errexx3,erreyy3,errexy3))

print("compute errors (%.3fs)" % (timing.time() - start))

###############################################################################
# plot of solution
###############################################################################
start = timing.time()

if visu==1:

   vtufile=open("solutionQ2.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                   iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %23)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   ####################################

   vtufile=open("solutionQ1.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%13e %13e %13e \n" %(velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                           velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                           v[i]-velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %density(xV[i],yV[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='Psi' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %Psi(xV[i],yV[i],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxx (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxx2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyy (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyy2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxy (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxy2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyx (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyx2[i])
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %(sr_xx(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %(sr_yy(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %(sr_xy(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyy' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyy[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxy' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxy[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyx' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyx[i])
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy1[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
   for i in range (0,nnp):
       vtufile.write("%f\n" % pressure(xV[i],yV[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu file (%.3fs)" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
