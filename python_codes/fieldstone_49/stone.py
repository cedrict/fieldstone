import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing

#------------------------------------------------------------------------------

def Q1(r):
    return 0.5*(1-r),\
           0.5*(1+r)

def dQ1d(r):
    return -0.5,\
           +0.5

#------------------------------------------------------------------------------
 
def Q2(r):
    return 0.5*r*(r-1.),\
           (1.-r**2),\
           0.5*r*(r+1.)

def dQ2d(r):
    return 0.5*(2.*r-1.),\
           (-2.*r),\
           0.5*(2.*r+1.)

#------------------------------------------------------------------------------

def Q3(r):
    return (-1    +r +9*r**2 - 9*r**3)/16,\
           (+9 -27*r -9*r**2 +27*r**3)/16,\
           (+9 +27*r -9*r**2 -27*r**3)/16,\
           (-1    -r +9*r**2 + 9*r**3)/16

def dQ3d(r):
    return ( +1 +18*r -27*r**2)/16,\
           (-27 -18*r +81*r**2)/16,\
           (+27 -18*r -81*r**2)/16,\
           ( -1 +18*r +27*r**2)/16

#------------------------------------------------------------------------------

def Q4(r):
    return (    r -   r**2 -4*r**3 + 4*r**4)/6,\
           ( -8*r +16*r**2 +8*r**3 -16*r**4)/6,\
           (1     - 5*r**2         + 4*r**4)  ,\
           (  8*r +16*r**2 -8*r**3 -16*r**4)/6,\
           (   -r -   r**2 +4*r**3 + 4*r**4)/6

def dQ4d(r):
    return (    1 - 2*r -12*r**2 +16*r**3)/6,\
           (   -8 +32*r +24*r**2 -64*r**3)/6,\
           (      -10*r          +16*r**3)  ,\
           (  8   +32*r -24*r**2 -64*r**3)/6,\
           (   -1 - 2*r +12*r**2 +16*r**3)/6

#------------------------------------------------------------------------------

def NNV(r,s,order):
    if order==1:
       N0r,N1r=Q1(r)
       N0s,N1s=Q1(s)
       return N0r*N0s,N1r*N0s,\
              N0r*N1s,N1r*N1s
    if order==2:
       N0r,N1r,N2r=Q2(r)
       N0s,N1s,N2s=Q2(s)
       return N0r*N0s,N1r*N0s,N2r*N0s,\
              N0r*N1s,N1r*N1s,N2r*N1s,\
              N0r*N2s,N1r*N2s,N2r*N2s
    if order==3:
       N0r,N1r,N2r,N3r=Q3(r)
       N0s,N1s,N2s,N3s=Q3(s)
       return N0r*N0s,N1r*N0s,N2r*N0s,N3r*N0s,\
              N0r*N1s,N1r*N1s,N2r*N1s,N3r*N1s,\
              N0r*N2s,N1r*N2s,N2r*N2s,N3r*N2s,\
              N0r*N3s,N1r*N3s,N2r*N3s,N3r*N3s
    if order==4:
       N0r,N1r,N2r,N3r,N4r=Q4(r)
       N0s,N1s,N2s,N3s,N4s=Q4(s)
       return N0r*N0s,N1r*N0s,N2r*N0s,N3r*N0s,N4r*N0s,\
              N0r*N1s,N1r*N1s,N2r*N1s,N3r*N1s,N4r*N1s,\
              N0r*N2s,N1r*N2s,N2r*N2s,N3r*N2s,N4r*N2s,\
              N0r*N3s,N1r*N3s,N2r*N3s,N3r*N3s,N4r*N3s,\
              N0r*N4s,N1r*N4s,N2r*N4s,N3r*N4s,N4r*N4s

#------------------------------------------------------------------------------

def dNNVdr(r,s,order):
    if order==1:
       dN0dr,dN1dr=dQ1d(r)
       N0s,N1s=Q1(s)
       return dN0dr*N0s,dN1dr*N0s,\
              dN0dr*N1s,dN1dr*N1s
    if order==2:
       dN0dr,dN1dr,dN2dr=dQ2d(r)
       N0s,N1s,N2s=Q2(s)
       return dN0dr*N0s,dN1dr*N0s,dN2dr*N0s,\
              dN0dr*N1s,dN1dr*N1s,dN2dr*N1s,\
              dN0dr*N2s,dN1dr*N2s,dN2dr*N2s
    if order==3:
       dN0dr,dN1dr,dN2dr,dN3dr=dQ3d(r)
       N0s,N1s,N2s,N3s=Q3(s)
       return dN0dr*N0s,dN1dr*N0s,dN2dr*N0s,dN3dr*N0s,\
              dN0dr*N1s,dN1dr*N1s,dN2dr*N1s,dN3dr*N1s,\
              dN0dr*N2s,dN1dr*N2s,dN2dr*N2s,dN3dr*N2s,\
              dN0dr*N3s,dN1dr*N3s,dN2dr*N3s,dN3dr*N3s 
    if order==4:
       dN0dr,dN1dr,dN2dr,dN3dr,dN4dr=dQ4d(r)
       N0s,N1s,N2s,N3s,N4s=Q4(s)
       return dN0dr*N0s,dN1dr*N0s,dN2dr*N0s,dN3dr*N0s,dN4dr*N0s,\
              dN0dr*N1s,dN1dr*N1s,dN2dr*N1s,dN3dr*N1s,dN4dr*N1s,\
              dN0dr*N2s,dN1dr*N2s,dN2dr*N2s,dN3dr*N2s,dN4dr*N2s,\
              dN0dr*N3s,dN1dr*N3s,dN2dr*N3s,dN3dr*N3s,dN4dr*N3s,\
              dN0dr*N4s,dN1dr*N4s,dN2dr*N4s,dN3dr*N4s,dN4dr*N4s

#------------------------------------------------------------------------------

def dNNVds(r,s,order):
    if order==1:
       N0r,N1r=Q1(r)
       dN0ds,dN1ds=dQ1d(s)
       return N0r*dN0ds,N1r*dN0ds,\
              N0r*dN1ds,N1r*dN1ds
    if order==2:
       N0r,N1r,N2r=Q2(r)
       dN0ds,dN1ds,dN2ds=dQ2d(s)
       return N0r*dN0ds,N1r*dN0ds,N2r*dN0ds,\
              N0r*dN1ds,N1r*dN1ds,N2r*dN1ds,\
              N0r*dN2ds,N1r*dN2ds,N2r*dN2ds
    if order==3:
       N0r,N1r,N2r,N3r=Q3(r)
       dN0ds,dN1ds,dN2ds,dN3ds=dQ3d(s)
       return N0r*dN0ds,N1r*dN0ds,N2r*dN0ds,N3r*dN0ds,\
              N0r*dN1ds,N1r*dN1ds,N2r*dN1ds,N3r*dN1ds,\
              N0r*dN2ds,N1r*dN2ds,N2r*dN2ds,N3r*dN2ds,\
              N0r*dN3ds,N1r*dN3ds,N2r*dN3ds,N3r*dN3ds
    if order==4:
       N0r,N1r,N2r,N3r,N4r=Q4(r)
       dN0ds,dN1ds,dN2ds,dN3ds,dN4ds=dQ4d(s)
       return N0r*dN0ds,N1r*dN0ds,N2r*dN0ds,N3r*dN0ds,N4r*dN0ds,\
              N0r*dN1ds,N1r*dN1ds,N2r*dN1ds,N3r*dN1ds,N4r*dN1ds,\
              N0r*dN2ds,N1r*dN2ds,N2r*dN2ds,N3r*dN2ds,N4r*dN2ds,\
              N0r*dN3ds,N1r*dN3ds,N2r*dN3ds,N3r*dN3ds,N4r*dN3ds,\
              N0r*dN4ds,N1r*dN4ds,N2r*dN4ds,N3r*dN4ds,N4r*dN4ds

#------------------------------------------------------------------------------

def NNP(r,s,order):
    if order==1:
       return 1.
    if order==2:
       N0r,N1r=Q1(r)
       N0s,N1s=Q1(s)
       return N0r*N0s,N1r*N0s,\
              N0r*N1s,N1r*N1s
    if order==3:
       N0r,N1r,N2r=Q2(r)
       N0s,N1s,N2s=Q2(s)
       return N0r*N0s,N1r*N0s,N2r*N0s,\
              N0r*N1s,N1r*N1s,N2r*N1s,\
              N0r*N2s,N1r*N2s,N2r*N2s
    if order==4:
       N0r,N1r,N2r,N3r=Q3(r)
       N0s,N1s,N2s,N3s=Q3(s)
       return N0r*N0s,N1r*N0s,N2r*N0s,N3r*N0s,\
              N0r*N1s,N1r*N1s,N2r*N1s,N3r*N1s,\
              N0r*N2s,N1r*N2s,N2r*N2s,N3r*N2s,\
              N0r*N3s,N1r*N3s,N2r*N3s,N3r*N3s

#------------------------------------------------------------------------------

def bx(x,y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x,y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def density(x,y):
    lambdaa=1
    k=2*np.pi/lambdaa
    y0=62./64.
    rho_alpha=64.
    if abs(y-y0)<1e-6:
       val=rho_alpha*np.cos(k*x)#+1.
    else:
       val=0.#+1.
    return val

def velocity_x(x,y,ibench):
    if ibench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    else:
       val=0.
    return val

def velocity_y(x,y,ibench):
    if ibench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    else:
       val=0.
    return val

def pressure(x,y,ibench):
    if ibench==1:
       val=x*(1.-x)#-1./6.
    else:
       val=0.
    return val

def sr_xx(x,y,ibench):
    if ibench==1:
       val=x**2*(2*x-2)*(4*y**3-6*y**2+2*y)+2*x*(-x+1)**2*(4*y**3-6*y**2+2*y)
    else:
       val=0.
    return val

def sr_xy(x,y,ibench):
    if ibench==1:
       val=(x**2*(-x+1)**2*(12*y**2-12*y+2)-y**2*(-y+1)**2*(12*x**2-12*x+2))/2
    else:
       val=0.
    return val

def sigma_xx(x,y,ibench):
    if ibench==1:
       val=2*x**2*(2*x-2)*(4*y**3-6*y**2+2*y)+4*x*(-x+1)**2*(4*y**3-6*y**2+2*y)-x*(-x+1)+1/6
    else:
       val=0.
    return val

def sigma_xy(x,y,ibench):
    if ibench==1:
       val=x**2*(-x+1)**2*(12*y**2-12*y+2)-y**2*(-y+1)**2*(12*x**2-12*x+2)
    else:
       val=0.
    return val

def sigma_yy(x,y,ibench):
    if ibench==1:
       val=-x*(-x+1)-2*y**2*(2*y-2)*(4*x**3-6*x**2+2*x)-4*y*(-y+1)**2*(4*x**3-6*x**2+2*x)+1/6
    else:
       lambdaa=1.
       y0=62./64.
       k=2*np.pi/lambdaa
       val=np.cos(k*x)/np.sinh(k)**2*\
          (k*(1.-y0)*np.sinh(k)*np.cosh(k*y0)\
          -k*np.sinh(k*(1.-y0))\
          +np.sinh(k)*np.sinh(k*y0) )
    return val

def sigma_yx(x,y,ibench):
    if ibench==1:
       val=sigma_xy(x,y)
    else:
       val=0.
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.
Ly=1.

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   order= int(sys.argv[4])
   Mlump= int(sys.argv[5])
else:
   nelx = 64
   nely = 64
   visu = 1
   order= 1
   Mlump= 0

ibench=2

gx=0.
gy=-1.

nel=nelx*nely
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
NV=nnx*nny

if order==1:
   NP=nelx*nely
   mV=4     # number of velocity nodes making up an element
   mP=1     # number of pressure nodes making up an element
   rVnodes=[-1,+1,-1,+1]
   sVnodes=[-1,-1,+1,+1]
   rPnodes=[0]
   sPnodes=[0]
if order==2:
   NP=(nelx+1)*(nely+1)
   mV=9     # number of velocity nodes making up an element
   mP=4     # number of pressure nodes making up an element
   rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
   rPnodes=[-1,+1,-1,+1]
   sPnodes=[-1,-1,+1,+1]
if order==3:
   NP=(2*nelx+1)*(2*nely+1)
   mV=16    # number of velocity nodes making up an element
   mP=9     # number of pressure nodes making up an element
   rVnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sVnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]
   rPnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sPnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
if order==4:
   NP=(3*nelx+1)*(3*nely+1)
   mV=25    # number of velocity nodes making up an element
   mP=16     # number of pressure nodes making up an element
   rVnodes=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]
   sVnodes=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
   rPnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sPnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]

ndofV=2
ndofP=1

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs

eps=1e-9
eta=1.

hx=Lx/nelx
hy=Ly/nely

sparse=True

#################################################################

if order==1:
   nqperdim=2
if order==2:
   nqperdim=3
if order==3:
   nqperdim=4
if order==4:
   nqperdim=5

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

#################################################################

print ('order    =',order)
print ('nnx      =',nnx)
print ('nny      =',nny)
print ('NV       =',NV)
print ('NP       =',NP)
print ('nel      =',nel)
print ('NfemV    =',NfemV)
print ('NfemP    =',NfemP)
print ('Nfem     =',Nfem)
print ('nqperdim =',nqperdim)
print("-----------------------------")

#################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i],order))
#################################################################
# checking that all pressure shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i],order))
#################################################################
# build velocity nodes coordinates 
#################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/order
        yV[counter]=j*hy/order
        counter+=1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("build V grid: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                iconV[counter2,counter]=i*order+l+j*order*nnx+nnx*k
                counter2+=1
        counter += 1

#print("-------iconV--------")
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range (0,mV):
#        print ("node ",i,':',iconV[i,iel],"at pos.",xV[iconV[i,iel]], yV[iconV[i,iel]])

print("build iconV: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

if order==1:
   for iel in range(0,nel):
       xP[iel]=sum(xV[iconV[0:mV,iel]])*0.25
       yP[iel]=sum(yV[iconV[0:mV,iel]])*0.25
      
if order>1:
   counter=0    
   for j in range(0,(order-1)*nely+1):
       for i in range(0,(order-1)*nelx+1):
           xP[counter]=i*hx/(order-1)
           yP[counter]=j*hy/(order-1)
           counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# build pressure connectivity array 
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)

if order==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           iconP[0,counter]=counter
           counter += 1

if order>1:
   om1=order-1
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   iconP[counter2,counter]=i*om1+l+j*om1*(om1*nelx+1)+(om1*nelx+1)*k 
                   counter2+=1
           counter += 1

#print("-------iconP--------")
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range(0,mP):
#        print ("node ",i,':',iconP[i,iel],"at pos.",xP[iconP[i,iel]], yP[iconP[i,iel]])

print("build iconP: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if ibench==1:
   for i in range(0,NV): # no slip
       if xV[i]<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
else:
   for i in range(0,NV): # free slip
       if xV[i]<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.


print("boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# building density array
#################################################################

rho = np.empty(NV, dtype=np.float64)  
if ibench==1:
   rho[:]=0.
else:
   for i in range(0,NV):
       rho[i]=density(xV[i],yV[i])

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

if sparse:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq,order)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            NNNP[0:mP]=NNP(rq,sq,order)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            rhoq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                rhoq+=NNNV[k]*rho[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            if ibench==1:
               for i in range(0,mV):
                   f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)
            else:
               for i in range(0,mV):
                   f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rhoq*gx
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq*gy


            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # end for jq
    # end for iq

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]

if not sparse:
   print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

if not sparse:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# assign extra pressure b.c. to remove null space
######################################################################

if sparse:
   A_sparse[Nfem-1,:]=0
   A_sparse[:,Nfem-1]=0
   A_sparse[Nfem-1,Nfem-1]=1
   rhs[Nfem-1]=0
else:
   a_mat[Nfem-1,:]=0
   a_mat[:,Nfem-1]=0
   a_mat[Nfem-1,Nfem-1]=1
   rhs[Nfem-1]=0

######################################################################
# solve system
######################################################################
start = timing.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

#####################################################################
# normalise pressure field 
#####################################################################

avrg_p=0.
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNP[0:mP]=NNP(rq,sq,order)

            # compure pressure at q point
            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)

            avrg_p+=pq*jcob*weightq

print('avrg pressure',avrg_p)

p[:]-=avrg_p

#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

#####################################################################
# compute strainrate at element center
#####################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.
    sq = 0.
    weightq = 2 
    NNNV[0:mV]=NNV(rq,sq,order)
    dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
    dNNNVds[0:mV]=dNNVds(rq,sq,order)
    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    for k in range(0,mV):
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (timing.time() - start))

#####################################################################
# project pressure onto velocity grid
#####################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)
c=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        NNNP[0:mP]=NNP(rVnodes[i],sVnodes[i],order)
        q[iconV[i,iel]]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        c[iconV[i,iel]]+=1.
    # end for i
# end for iel

q=q/c

#np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (timing.time() - start))

#####################################################################
# project strainrate onto velocity grid
#####################################################################

exxn=np.zeros(NV,dtype=np.float64)
eyyn=np.zeros(NV,dtype=np.float64)
exyn=np.zeros(NV,dtype=np.float64)
c=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i],order)
        dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i],order)
        dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i],order)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        e_xx=0.
        e_yy=0.
        e_xy=0.
        for k in range(0,mV):
            e_xx += dNNNVdx[k]*u[iconV[k,iel]]
            e_yy += dNNNVdy[k]*v[iconV[k,iel]]
            e_xy += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
        exxn[iconV[i,iel]]+=e_xx
        eyyn[iconV[i,iel]]+=e_yy
        exyn[iconV[i,iel]]+=e_xy
        c[iconV[i,iel]]+=1.
    # end for i
# end for iel

exxn/=c
eyyn/=c
exyn/=c

sigmaxxn=-q+2*eta*exxn
sigmayyn=-q+2*eta*eyyn
sigmaxyn=   2*eta*exyn

print("     -> exxn (m,M) %.4e %.4e " %(np.min(exxn),np.max(exxn)))
print("     -> eyyn (m,M) %.4e %.4e " %(np.min(eyyn),np.max(eyyn)))
print("     -> exyn (m,M) %.4e %.4e " %(np.min(exyn),np.max(exyn)))

#####################################################################
# compute error fields for plotting
#####################################################################
start = timing.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_p = np.empty(NP,dtype=np.float64)
error_q = np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i],ibench)
    error_v[i]=v[i]-velocity_y(xV[i],yV[i],ibench)
    error_q[i]=q[i]-pressure(xV[i],yV[i],ibench)

for i in range(0,NP): 
    error_p[i]=p[i]-pressure(xP[i],yP[i],ibench)

print("     -> error_u (m,M) %.4e %.4e " %(np.min(error_u),np.max(error_u)))
print("     -> error_v (m,M) %.4e %.4e " %(np.min(error_v),np.max(error_v)))
print("     -> error_p (m,M) %.4e %.4e " %(np.min(error_p),np.max(error_p)))
print("     -> error_q (m,M) %.4e %.4e " %(np.min(error_q),np.max(error_q)))

print("compute error fields: %.3f s" % (timing.time() - start))

#####################################################################
# compute L2 errors
#####################################################################
start = timing.time()

errv=0.
errp=0.
errq=0.
for iel in range (0,nel):

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq,order)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            NNNP[0:mP]=NNP(rq,sq,order)

            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            xq=0.
            yq=0.
            uq=0.
            vq=0.
            qq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                qq+=NNNV[k]*q[iconV[k,iel]]
            errv+=((uq-velocity_x(xq,yq,ibench))**2+(vq-velocity_y(xq,yq,ibench))**2)*weightq*jcob
            errq+=(qq-pressure(xq,yq,ibench))**2*weightq*jcob

            xq=0.
            yq=0.
            pq=0.
            for k in range(0,mP):
                xq+=NNNP[k]*xP[iconP[k,iel]]
                yq+=NNNP[k]*yP[iconP[k,iel]]
                pq+=NNNP[k]*p[iconP[k,iel]]
            errp+=(pq-pressure(xq,yq,ibench))**2*weightq*jcob

        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)

print("     -> nel= %6d ; errv= %.8e ; errp= %.8e ; errq= %.8e" %(nel,errv,errp,errq))

print("compute errors: %.3f s" % (timing.time() - start))

#####################################################################
# computing analytical stress values on V nodes
#####################################################################

stress_xx = np.zeros(NV,np.float64)
stress_yy = np.zeros(NV,np.float64)
stress_xy = np.zeros(NV,np.float64)

for i in range(0,NV):
    stress_xx[i]=sigma_xx(xV[i],yV[i],ibench)
    stress_yy[i]=sigma_yy(xV[i],yV[i],ibench)
    stress_xy[i]=sigma_xy(xV[i],yV[i],ibench)

#####################################################################
# compute derivatives on domain edges with CBF 
#####################################################################
start = timing.time()

if order==1:
   if Mlump==1:
      Mmat = np.array([[1.,0.],\
                       [0.,1.]],dtype=np.float64)
   else: 
      Mmat = np.array([[2.,1.],\
                       [1.,2.]],dtype=np.float64)/3.

if order==2:
   if Mlump==1:
      Mmat = np.array([[1.,0.,0.],\
                       [0.,4.,0.],\
                       [0.,0.,1.]],dtype=np.float64)/3.
   else: 
      Mmat = np.array([[ 8., 4.,-2.],\
                       [ 4.,32., 4.],\
                       [-2., 4., 8.]],dtype=np.float64)/30. 

if order==3:
   if Mlump==1:
      Mmat = np.array([[1.,0.,0.,0.],\
                       [0.,3.,0.,0.],\
                       [0.,0.,3.,0.],\
                       [0.,0.,0.,1.]],dtype=np.float64)/4.
   else: 
      Mmat = np.array([[256., 198., -72., 38.],\
                       [198.,1296.,-162.,-72.],\
                       [-72.,-162.,1296.,198.],\
                       [ 38., -72., 198.,256.]],dtype=np.float64)/16./105.

if order==4:
   if Mlump==1:
      Mmat = np.array([[7., 0., 0., 0.,0.],\
                       [0.,32., 0., 0.,0.],\
                       [0., 0.,12., 0.,0.],\
                       [0., 0., 0.,32.,0.],\
                       [0., 0., 0., 0.,7.]],dtype=np.float64)/45. 
   else: 
      Mmat = np.array([[1168., 1184., -696.,  224.,-116.],\
                       [1184., 7168.,-1536., 1024., 224.],\
                       [-696.,-1536., 7488.,-1536.,-696.],\
                       [ 224., 1024.,-1536., 7168.,1184.],\
                       [-116.,  224., -696., 1184.,1168.]],dtype=np.float64)/36./315. 

# compute the nb of traction dofs
NfemTr=np.sum(bc_fix)
print ('NfemTr=',NfemTr)

# build array which maps vel dofs which are fixed 
# to traction dofs
bc_nb=np.zeros(NfemV,dtype=np.int32)  
counter=0
for i in range(0,NfemV):
    if (bc_fix[i]):
       bc_nb[i]=counter
       counter+=1

M_cbf = np.zeros((NfemTr,NfemTr),np.float64)
rhs_cbf = np.zeros(NfemTr,np.float64)

for iel in range(0,nel):

    rhs_el =np.zeros((mV*ndofV),dtype=np.float64)

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq,order)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            NNNP[0:mP]=NNP(rq,sq,order)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            exxq=0.
            eyyq=0.
            exyq=0.
            rhoq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                rhoq+=NNNV[k]*rho[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq += dNNNVdx[k]*u[iconV[k,iel]]
                eyyq += dNNNVdy[k]*v[iconV[k,iel]]
                exyq += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                        0.5*dNNNVdx[k]*v[iconV[k,iel]]

            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]

            sigmaxxq=-pq+2*eta*exxq
            sigmayyq=-pq+2*eta*eyyq
            sigmaxyq=    2*eta*exyq

            # compute elemental rhs vector
            if ibench==1:
               for i in range(0,mV):
                   rhs_el[ndofV*i  ]+=(dNNNVdx[i]*sigmaxxq+dNNNVdy[i]*sigmaxyq\
                                      -NNNV[i]*bx(xq,yq))*jcob*weightq
                   rhs_el[ndofV*i+1]+=(dNNNVdx[i]*sigmaxyq+dNNNVdy[i]*sigmayyq\
                                      -NNNV[i]*by(xq,yq))*jcob*weightq
            else:
               for i in range(0,mV):
                   rhs_el[ndofV*i  ]+=(dNNNVdx[i]*sigmaxxq+dNNNVdy[i]*sigmaxyq\
                                      -NNNV[i]*rhoq*gx)*jcob*weightq
                   rhs_el[ndofV*i+1]+=(dNNNVdx[i]*sigmaxyq+dNNNVdy[i]*sigmayyq\
                                      -NNNV[i]*rhoq*gy)*jcob*weightq
            # end if

        # end for jq
    # end for iq

    # assemble terms for bottom boundary
    for idof in range(0,ndofV):
        if order==1:
           mylist=[0,1]
        if order==2:
           mylist=[0,1,2]
        if order==3:
           mylist=[0,1,2,3]
        if order==4:
           mylist=[0,1,2,3,4]
        # making sure edge is on boundary
        if bc_fix[ndofV*iconV[mylist[    0],iel]+idof] and \
           bc_fix[ndofV*iconV[mylist[order],iel]+idof]:
           for j in range(0,order+1):
              dofj=ndofV*iconV[mylist[j],iel]+idof
              rhs_cbf[bc_nb[dofj]]+=rhs_el[ndofV*mylist[j]+idof]   
              for k in range(0,order+1):
                  dofk=ndofV*iconV[mylist[k],iel]+idof
                  M_cbf[bc_nb[dofj],bc_nb[dofk]]+=Mmat[j,k]*hx/2

    # assemble terms for right boundary
        if order==1:
           mylist=[1,3]
        if order==2:
           mylist=[2,5,8]
        if order==3:
           mylist=[3,7,11,15]
        if order==4:
           mylist=[4,9,14,19,24]
        # making sure edge is on boundary
        if bc_fix[ndofV*iconV[mylist[    0],iel]+idof] and \
           bc_fix[ndofV*iconV[mylist[order],iel]+idof]:
           for j in range(0,order+1):
              dofj=ndofV*iconV[mylist[j],iel]+idof
              rhs_cbf[bc_nb[dofj]]+=rhs_el[ndofV*mylist[j]+idof]   
              for k in range(0,order+1):
                  dofk=ndofV*iconV[mylist[k],iel]+idof
                  M_cbf[bc_nb[dofj],bc_nb[dofk]]+=Mmat[j,k]*hy/2

    # assemble terms for top boundary
        if order==1:
           mylist=[3,2]
        if order==2:
           mylist=[8,7,6]
        if order==3:
           mylist=[15,14,13,12]
        if order==4:
           mylist=[24,23,22,21,20]
        # making sure edge is on boundary
        if bc_fix[ndofV*iconV[mylist[    0],iel]+idof] and \
           bc_fix[ndofV*iconV[mylist[order],iel]+idof]:
           for j in range(0,order+1):
              dofj=ndofV*iconV[mylist[j],iel]+idof
              rhs_cbf[bc_nb[dofj]]+=rhs_el[ndofV*mylist[j]+idof]   
              for k in range(0,order+1):
                  dofk=ndofV*iconV[mylist[k],iel]+idof
                  M_cbf[bc_nb[dofj],bc_nb[dofk]]+=Mmat[j,k]*hx/2

    # assemble terms for left boundary
        if order==1:
           mylist=[2,0]
        if order==2:
           mylist=[6,3,0]
        if order==3:
           mylist=[12,8,4,0]
        if order==4:
           mylist=[20,15,10,5,0]
        # making sure edge is on boundary
        if bc_fix[ndofV*iconV[mylist[    0],iel]+idof] and \
           bc_fix[ndofV*iconV[mylist[order],iel]+idof]:
           for j in range(0,order+1):
              dofj=ndofV*iconV[mylist[j],iel]+idof
              rhs_cbf[bc_nb[dofj]]+=rhs_el[ndofV*mylist[j]+idof]   
              for k in range(0,order+1):
                  dofk=ndofV*iconV[mylist[k],iel]+idof
                  M_cbf[bc_nb[dofj],bc_nb[dofk]]+=Mmat[j,k]*hy/2

    # end for idof 

# end for iel

print("     -> M_cbf   (m,M) %.4e %.4e " %(np.min(M_cbf),np.max(M_cbf)))
print("     -> rhs_cbf (m,M) %.4e %.4e " %(np.min(rhs_cbf),np.max(rhs_cbf)))

sol=sps.linalg.spsolve(sps.csr_matrix(M_cbf),rhs_cbf)

tx = np.zeros(NV,np.float64)
ty = np.zeros(NV,np.float64)
for i in range(0,NV):
    if bc_fix[ndofV*i+0]:
       tx[i]=sol[bc_nb[ndofV*i+0]]
    if bc_fix[ndofV*i+1]:
       ty[i]=sol[bc_nb[ndofV*i+1]]

print("     -> tx (m,M) %.4e %.4e " %(np.min(tx),np.max(tx)))
print("     -> ty (m,M) %.4e %.4e " %(np.min(ty),np.max(ty)))

np.savetxt('sigmayy_top.ascii',np.array([xV[NV-nnx:NV],\
                                         ty[NV-nnx:NV],\
                                         stress_yy[NV-nnx:NV],\
                                         sigmayyn[NV-nnx:NV]]).T,header='# x,sigmayy')

np.savetxt('sigmaxy_top.ascii',np.array([xV[NV-nnx:NV],\
                                         tx[NV-nnx:NV],\
                                         stress_xy[NV-nnx:NV],\
                                         sigmaxyn[NV-nnx:NV]]).T,header='# x,sigmaxy')

if ibench==2:
   print ('ty at top right point:',ty[NV-1],sigma_yy(1.,1.,ibench))

#np.savetxt('sigmaxy_bot.ascii',np.array([xV[0:nnx],
#                                         tx[0:nnx],
#                                         -stress_xy[0:nnx]
#                                         -sigmaxyn[0:nnx]]).T,header='# x,sigmaxy')

#np.savetxt('sigmayy_bot.ascii',np.array([xV[0:nnx],\
#                                         ty[0:nnx],\
#                                         -stress_yy[0:nnx],\
#                                         -sigmayyn[0:nnx]]).T,header='# x,sigmayy')

#np.savetxt('tractions.ascii',np.array([xV,yV,tx,ty,stress_xx,stress_yy,stress_xy]).T,header='# x,y,tx,ty')


#####################################################################
# plot of solution
#####################################################################

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    if order==1:
       vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
       for i in range(0,nel):
           vtufile.write("%10e \n" %p[i])
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    if ibench==2:
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for i in range(0,NV):
          vtufile.write("%10e \n" %rho[i])
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error u' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_u[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error v' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_v[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_q[i])
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %exxn[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %eyyn[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %exyn[i])
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='exxn (analytical)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %(sr_xx(xV[i],yV[i],ibench)))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exyn (analytical)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %(sr_xy(xV[i],yV[i],ibench)))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    if order==1:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[3,iel],iconV[2,iel]))
    if order==2:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[2,iel],iconV[8,iel],iconV[6,iel]))
    if order==3:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[3,iel],iconV[15,iel],iconV[12,iel]))
    if order==4:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[4,iel],iconV[24,iel],iconV[20,iel]))

    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
