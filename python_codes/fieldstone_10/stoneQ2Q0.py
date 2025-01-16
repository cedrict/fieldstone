import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
import time as time
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------

def NNN(rq,sq,tq):
    NV_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    NV_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.)
    NV_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    NV_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    NV_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    NV_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq-1.)
    NV_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (1.-tq**2)
    NV_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (1.-tq**2)
    NV_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (1.-tq**2)
    NV_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (1.-tq**2)
    NV_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*tq*(tq+1.)
    NV_26= (1.-rq**2)     * (1.-sq**2)     * (1.-tq**2)
    return NV_00,NV_01,NV_02,NV_03,NV_04,NV_05,NV_06,NV_07,NV_08,\
           NV_09,NV_10,NV_11,NV_12,NV_13,NV_14,NV_15,NV_16,NV_17,\
           NV_18,NV_19,NV_20,NV_21,NV_22,NV_23,NV_24,NV_25,NV_26

def dNNNdr(rq,sq,tq):
    dNVdr_00= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_01= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.) 
    dNVdr_02= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_03= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.) 
    dNVdr_04= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_05= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.) 
    dNVdr_06= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.) 
    dNVdr_07= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    dNVdr_08= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq-1.)
    dNVdr_09= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    dNVdr_10= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq-1.)
    dNVdr_11= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq-1.)
    dNVdr_12= (-2*rq)       * 0.5*sq*(sq-1.) * 0.5*tq*(tq+1.)
    dNVdr_13= 0.5*(2*rq+1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    dNVdr_14= (-2*rq)       * 0.5*sq*(sq+1.) * 0.5*tq*(tq+1.)
    dNVdr_15= 0.5*(2*rq-1.) * (1.-sq**2)     * 0.5*tq*(tq+1.)
    dNVdr_16= 0.5*(2*rq-1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    dNVdr_17= 0.5*(2*rq+1.) * 0.5*sq*(sq-1.) * (1.-tq**2)
    dNVdr_18= 0.5*(2*rq+1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    dNVdr_19= 0.5*(2*rq-1.) * 0.5*sq*(sq+1.) * (1.-tq**2)
    dNVdr_20= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq-1.)
    dNVdr_21= (-2*rq)       * 0.5*sq*(sq-1.) * (1.-tq**2)
    dNVdr_22= 0.5*(2*rq+1.) * (1.-sq**2)     * (1.-tq**2)
    dNVdr_23= (-2*rq)       * 0.5*sq*(sq+1.) * (1.-tq**2)
    dNVdr_24= 0.5*(2*rq-1.) * (1.-sq**2)     * (1.-tq**2)
    dNVdr_25= (-2*rq)       * (1.-sq**2)     * 0.5*tq*(tq+1.)
    dNVdr_26= (-2*rq)       * (1.-sq**2)     * (1.-tq**2)
    return dNVdr_00,dNVdr_01,dNVdr_02,dNVdr_03,dNVdr_04,dNVdr_05,dNVdr_06,dNVdr_07,dNVdr_08,\
           dNVdr_09,dNVdr_10,dNVdr_11,dNVdr_12,dNVdr_13,dNVdr_14,dNVdr_15,dNVdr_16,dNVdr_17,\
           dNVdr_18,dNVdr_19,dNVdr_20,dNVdr_21,dNVdr_22,dNVdr_23,dNVdr_24,dNVdr_25,dNVdr_26

def dNNNds(rq,sq,tq):
    dNVds_00= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.)
    dNVds_01= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.)
    dNVds_02= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.)
    dNVds_03= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.)
    dNVds_04= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.)
    dNVds_05= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.)
    dNVds_06= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.)
    dNVds_07= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.)
    dNVds_08= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq-1.)
    dNVds_09= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq-1.)
    dNVds_10= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq-1.)
    dNVds_11= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq-1.)
    dNVds_12= (1.-rq**2)     * 0.5*(2*sq-1.) * 0.5*tq*(tq+1.)
    dNVds_13= 0.5*rq*(rq+1.) * (-2*sq)       * 0.5*tq*(tq+1.)
    dNVds_14= (1.-rq**2)     * 0.5*(2*sq+1.) * 0.5*tq*(tq+1.)
    dNVds_15= 0.5*rq*(rq-1.) * (-2*sq)       * 0.5*tq*(tq+1.)
    dNVds_16= 0.5*rq*(rq-1.) * 0.5*(2*sq-1.) * (1.-tq**2)
    dNVds_17= 0.5*rq*(rq+1.) * 0.5*(2*sq-1.) * (1.-tq**2)
    dNVds_18= 0.5*rq*(rq+1.) * 0.5*(2*sq+1.) * (1.-tq**2)
    dNVds_19= 0.5*rq*(rq-1.) * 0.5*(2*sq+1.) * (1.-tq**2)
    dNVds_20= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq-1.)
    dNVds_21= (1.-rq**2)     * 0.5*(2*sq-1.) * (1.-tq**2)
    dNVds_22= 0.5*rq*(rq+1.) * (-2*sq)       * (1.-tq**2)
    dNVds_23= (1.-rq**2)     * 0.5*(2*sq+1.) * (1.-tq**2)
    dNVds_24= 0.5*rq*(rq-1.) * (-2*sq)       * (1.-tq**2)
    dNVds_25= (1.-rq**2)     * (-2*sq)       * 0.5*tq*(tq+1.)
    dNVds_26= (1.-rq**2)     * (-2*sq)       * (1.-tq**2)
    return dNVds_00,dNVds_01,dNVds_02,dNVds_03,dNVds_04,dNVds_05,dNVds_06,dNVds_07,dNVds_08,\
           dNVds_09,dNVds_10,dNVds_11,dNVds_12,dNVds_13,dNVds_14,dNVds_15,dNVds_16,dNVds_17,\
           dNVds_18,dNVds_19,dNVds_20,dNVds_21,dNVds_22,dNVds_23,dNVds_24,dNVds_25,dNVds_26

def dNNNdt(rq,sq,tq):
    dNVdt_00= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.)
    dNVdt_01= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.)
    dNVdt_02= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.)
    dNVdt_03= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.)
    dNVdt_04= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.)
    dNVdt_05= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.)
    dNVdt_06= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.)
    dNVdt_07= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.)
    dNVdt_08= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq-1.)
    dNVdt_09= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq-1.)
    dNVdt_10= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq-1.)
    dNVdt_11= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq-1.)
    dNVdt_12= (1.-rq**2)     * 0.5*sq*(sq-1.) * 0.5*(2*tq+1.)
    dNVdt_13= 0.5*rq*(rq+1.) * (1.-sq**2)     * 0.5*(2*tq+1.)
    dNVdt_14= (1.-rq**2)     * 0.5*sq*(sq+1.) * 0.5*(2*tq+1.)
    dNVdt_15= 0.5*rq*(rq-1.) * (1.-sq**2)     * 0.5*(2*tq+1.)
    dNVdt_16= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.) * (-2*tq)
    dNVdt_17= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.) * (-2*tq)
    dNVdt_18= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.) * (-2*tq)
    dNVdt_19= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.) * (-2*tq)
    dNVdt_20= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq-1.)
    dNVdt_21= (1.-rq**2)     * 0.5*sq*(sq-1.) * (-2*tq)
    dNVdt_22= 0.5*rq*(rq+1.) * (1.-sq**2)     * (-2*tq)
    dNVdt_23= (1.-rq**2)     * 0.5*sq*(sq+1.) * (-2*tq)
    dNVdt_24= 0.5*rq*(rq-1.) * (1.-sq**2)     * (-2*tq)
    dNVdt_25= (1.-rq**2)     * (1.-sq**2)     * 0.5*(2*tq+1.)
    dNVdt_26= (1.-rq**2)     * (1.-sq**2)     * (-2*tq)
    return dNVdt_00,dNVdt_01,dNVdt_02,dNVdt_03,dNVdt_04,dNVdt_05,dNVdt_06,dNVdt_07,dNVdt_08,\
           dNVdt_09,dNVdt_10,dNVdt_11,dNVdt_12,dNVdt_13,dNVdt_14,dNVdt_15,dNVdt_16,dNVdt_17,\
           dNVdt_18,dNVdt_19,dNVdt_20,dNVdt_21,dNVdt_22,dNVdt_23,dNVdt_24,dNVdt_25,dNVdt_26

#------------------------------------------------------------------------------

def bx(x,y,z):
    if experiment==0:
       val=0
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       val=0
    if experiment==5:
       val=4*(2*y-1)*(2*z-1)
    if experiment==6:
       fpp=2*(6*x**2-6*x+1)
       gp=2*y*(2*y**2-3*y+1)
       gppp=24*y-12
       hppp=24*z-12
       hp=2*z*(2*z**2-3*z+1)
       f=x**2*(1-x)**2
       val=-(2*fpp*gp*hp+f*gppp*hp+f*gp*hppp)
    return val

def by(x,y,z):
    if experiment==0:
       val=0
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       val=0
    if experiment==5:
       val=4*(2*x-1)*(2*z-1)
    if experiment==6:
       fp=2*x*(2*x**2-3*x+1)
       fppp=24*x-12
       g=y**2*(1-y)**2 
       gpp=2*(6*y**2-6*y+1) 
       hp=2*z*(2*z**2-3*z+1)
       hppp=24*z-12
       val=-(fppp*g*hp+2*fp*gpp*hp+fp*g*hppp)
    return val

def bz(x,y,z):
    if experiment==0:
       val=-1
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1.01*gz +1
       else:
          val=1.*gz +1
    if experiment==5:
       val=-2*(2*x-1)*(2*y-1) 
    if experiment==6:
       fp=2*x*(2*x**2-3*x+1)
       fppp=24*x-12
       gp=2*y*(2*y**2-3*y+1)
       gppp=24*y-12
       h=z**2*(1-z)**2 
       hpp=2*(6*z**2-6*z+1)
       val=2*fppp*gp*h+2*fp*gppp*h+fp*gp*hpp
    return val

#------------------------------------------------------------------------------

def viscosity(x,y,z):
    if experiment==0:
       val=1
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1e3
       else:
          val=1.
    if experiment==5:
       val=1.
    if experiment==6:
       val=1.
    return val

#------------------------------------------------------------------------------

def uth(x,y,z):
    if experiment==0:
       val=0
    if experiment==5:
       val=x*(1-x)*(1-2*y)*(1-2*z)
    if experiment==6:
       val=x**2*(1-x)**2* 2*y*(2*y**2-3*y+1)* 2*z*(2*z**2-3*z+1)
    return val

def vth(x,y,z):
    if experiment==0:
       val=0
    if experiment==5:
       val=(1-2*x)*y*(1-y)*(1-2*z)
    if experiment==6:
       val=2*x*(2*x**2-3*x+1)* y**2*(1-y)**2 *2*z*(2*z**2-3*z+1)
    return val

def wth(x,y,z):
    if experiment==0:
       val=0
    if experiment==5:
       val=-2*(1-2*x)*(1-2*y)*z*(1-z)
    if experiment==6:
       val=-2* 2*x*(2*x**2-3*x+1)* 2*y*(2*y**2-3*y+1)* z**2*(1-z)**2
    return val

def pth(x,y,z):
    if experiment==0:
       val=0.5-z
    if experiment==5:
       val=(2*x-1)*(2*y-1)*(2*z-1)
    if experiment==6:
       val=-2*x*(2*x**2-3*x+1)\
           *2*y*(2*y**2-3*y+1)\
           *2*z*(2*z**2-3*z+1)
    return val

#------------------------------------------------------------------------------

experiment=6

print("-----------------------------")
print("--------- stone 10 ----------")
print("-----------------------------")

m=27    # number of velocity nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   visu = int(sys.argv[2])
   nqperdimP= int(sys.argv[3])
else:
   nelx = 10
   visu = 1
   nqperdimP=2

if experiment==0: quarter=False
if experiment==1: quarter=False
if experiment==2: quarter=False
if experiment==3: quarter=False
if experiment==4: quarter=True
if experiment==5: quarter=False
if experiment==6: quarter=False

if quarter:
   nely=nelx
   nelz=2*nelx
   Lx=0.5 
   Ly=0.5
   Lz=1.
else: 
   nely=nelx
   nelz=nelx
   Lx=1.
   Ly=1.
   Lz=1.

FS=True
NS=False
OT=False
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
nnz=2*nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

penalty=1.e6  # penalty coefficient value

Nfem=NV*ndofV  # Total number of degrees of freedom

eps=1.e-10

gz=-1.  # gravity vector, z component

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

#################################################################

qcoords3=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights3=[5./9.,8./9.,5./9.]

qcoords2=[-np.sqrt(1./3.),np.sqrt(1./3.)]
qweights2=[1.,1.]

qcoords1=[0.]
qweights1=[2.]

nqperdimV=3

if nqperdimP==1:
   qcoordsP=qcoords1
   qweightsP=qweights1
if nqperdimP==2:
   qcoordsP=qcoords2
   qweightsP=qweights2
if nqperdimP==3:
   qcoordsP=qcoords3
   qweightsP=qweights3

#################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('NV=',NV)
print('Nfem=',Nfem)
print('nqperdimP=',nqperdimP)
print("-----------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*hx/2
            y[counter]=j*hy/2
            z[counter]=k*hz/2
            counter += 1
        #end for
    #end for
#end for
   
print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon=np.zeros((m,nel),dtype=np.int32)
counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[ 0,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon[ 1,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon[ 2,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon[ 3,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon[ 4,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon[ 5,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon[ 6,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon[ 7,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon[ 8,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon[ 9,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon[10,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon[11,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon[12,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon[13,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon[14,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon[15,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon[16,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon[17,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon[18,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon[19,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon[20,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            icon[21,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon[22,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon[23,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon[24,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon[25,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            icon[26,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=float)  # boundary condition, value

if experiment==0 or experiment==1 or experiment==2 or experiment==3 or experiment==4:

   if FS or OT:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if x[i]>(Lx-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if y[i]<eps:
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
          if y[i]>(Ly-eps):
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
          if z[i]<eps:
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if not OT and z[i]>(Lz-eps):
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          #end if
      #end for

   if NS:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if x[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if y[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if y[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if z[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if z[i]>(Lz-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if quarter and x[i]>(0.5-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if quarter and y[i]>(0.5-eps):
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
      #end for

if experiment==5 or experiment==6:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if x[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if z[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if z[i]>(Lz-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
      #end for

print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#   /1 1 1 0 0 0\      /2 0 0 0 0 0\ 
#   |1 1 1 0 0 0|      |0 2 0 0 0 0|
# K=|1 1 1 0 0 0|    C=|0 0 2 0 0 0|
#   |0 0 0 0 0 0|      |0 0 0 1 0 0|
#   |0 0 0 0 0 0|      |0 0 0 0 1 0|
#   \0 0 0 0 0 0/      \0 0 0 0 0 1/
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
w     = np.zeros(NV,dtype=np.float64)            # z-component velocity
jcb   = np.zeros((3,3),dtype=np.float64)
jcbi  = np.zeros((3,3),dtype=np.float64)
k_mat = np.zeros((6,6),dtype=np.float64) 
c_mat = np.zeros((6,6),dtype=np.float64) 

k_mat[0,0]=1. ; k_mat[0,1]=1. ; k_mat[0,2]=1.  
k_mat[1,0]=1. ; k_mat[1,1]=1. ; k_mat[1,2]=1.  
k_mat[2,0]=1. ; k_mat[2,1]=1. ; k_mat[2,2]=1.  

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

jcob=hx*hy*hz/8 
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcbi[2,2]=2/hz

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m*ndofV,dtype=np.float64)
    a_el=np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

    # integrate viscous term at 3*3*3 quadrature points
    for iq in range(0,nqperdimV):
        for jq in range(0,nqperdimV):
            for kq in range(0,nqperdimV):

                # position & weight of quad. point
                rq=qcoords3[iq]
                sq=qcoords3[jq]
                tq=qcoords3[kq]
                weightq=qweights3[iq]*qweights3[jq]*qweights3[kq]

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                # calculate the determinant of the jacobian
                #jcob = np.linalg.det(jcb)
                # calculate inverse of the jacobian matrix
                #jcbi = np.linalg.inv(jcb)

                # compute coordinates of quadrature points
                xq=N.dot(x[icon[0:m,iel]])
                yq=N.dot(y[icon[0:m,iel]])
                zq=N.dot(z[icon[0:m,iel]])

                # compute dNdx, dNdy, dNdz
                for k in range(0, m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
                #end for 

                # construct b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for 

                # compute elemental matrix
                a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,zq)*weightq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[ndofV*i+0]+=N[i]*jcob*weightq*bx(xq,yq,zq)
                    b_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq,zq)
                    b_el[ndofV*i+2]+=N[i]*jcob*weightq*bz(xq,yq,zq)
                #end for 

            #end for kq 
        #end for jq  
    #end for iq  

    for iq in range(0,nqperdimP):
        for jq in range(0,nqperdimP):
            for kq in range(0,nqperdimP):

                rq=qcoordsP[iq]
                sq=qcoordsP[jq]
                tq=qcoordsP[kq]
                weightq=qweightsP[iq]*qweightsP[jq]*qweightsP[kq]

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)

                # compute coordinates of quadrature points
                xq=N.dot(x[icon[0:m,iel]])
                yq=N.dot(y[icon[0:m,iel]])
                zq=N.dot(z[icon[0:m,iel]])

                # compute dNdx, dNdy, dNdz
                for k in range(0, m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
                #end for 

                # construct b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for 

                # compute elemental matrix
                a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

            #end for kq 
        #end for jq  
    #end for iq  

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndofV*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndofV):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               #end for
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                #end for
            #end for
            rhs[m1]+=b_el[ikk]
        #end for
    #end for

#end for iel

a_mat=csr_matrix(a_mat)

print("build FE system: %.3f s | nel= %d" % (time.time() - start,nel))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(a_mat,rhs)

print("solve time: %.3f s | Nfem= %d " % (time.time() - start,Nfem))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u,v,w=np.reshape(sol,(NV,3)).T

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.5e %.5e " %(np.min(w),np.max(w)))

np.savetxt('velocity.ascii',np.array([x,y,z,u*100,v*100,w*100]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

M_mat = lil_matrix((NV,NV),dtype=np.float64) # matrix of Ax=b
rhs   = np.zeros(NV,dtype=np.float64)        # right hand side of Ax=b

for iel in range(0,nel):

    M_el=np.zeros((m,m),dtype=np.float64)
    b_el=np.zeros(m,dtype=np.float64)

    # integrate mass matrix at 3x3x3 quadrature points
    for iq in range(0,nqperdimV):
        for jq in range(0,nqperdimV):
            for kq in range(0,nqperdimV):

                # position & weight of quad. point
                rq=qcoords3[iq]
                sq=qcoords3[jq]
                tq=qcoords3[kq]
                weightq=qweights3[iq]*qweights3[jq]*qweights3[kq]

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)
 
                for k1 in range(0,m):
                    for k2 in range(0,m):
                        M_el[k1,k2]+=N[k1]*N[k2]*weightq*jcob

            #end for kq 
        #end for jq  
    #end for iq  

    #----------------------------------
    for iq in range(0,nqperdimP):
        for jq in range(0,nqperdimP):
            for kq in range(0,nqperdimP):

                rq=qcoordsP[iq]
                sq=qcoordsP[jq]
                tq=qcoordsP[kq]
                weightq=qweightsP[iq]*qweightsP[jq]*qweightsP[kq]

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)

                # compute dNdx, dNdy, dNdz
                for k in range(0, m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
                #end for 

                dudxq=np.sum(dNdx*u[icon[:,iel]]) 
                dvdyq=np.sum(dNdy*v[icon[:,iel]]) 
                dwdzq=np.sum(dNdz*w[icon[:,iel]]) 

                # compute elemental rhs vector
                for i in range(0, m):
                    b_el[i]-=N[i]*jcob*weightq*(dudxq+dvdyq+dwdzq)*penalty
                #end for 

            #end for kq 
        #end for jq  
    #end for iq  

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        m1 =icon[k1,iel]
        for k2 in range(0,m):
            m2 =icon[k2,iel]
            M_mat[m1,m2]+=M_el[k1,k2]
        #end for
        rhs[m1]+=b_el[k1]
    #end for

#end for iel

M_mat=csr_matrix(M_mat)

#plt.spy(M_mat, markersize=0.01)
#plt.savefig('M.pdf', bbox_inches='tight')

q = sps.linalg.spsolve(M_mat,rhs)

np.savetxt('pressure.ascii',np.array([x,y,z,q]).T,header='# x,y,z,q')

print("     -> q (m,M) %.5e %.5e " %(np.min(q),np.max(q)))

print("compute q: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
dens = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  
jcb=np.zeros((3,3),dtype=np.float64)

for iel in range(0,nel):

    xc[iel]=x[icon[0,iel]]+hx/2
    yc[iel]=y[icon[0,iel]]+hy/2
    zc[iel]=z[icon[0,iel]]+hz/2

    rq=0.
    sq=0.
    tq=0.
    weightq=2.*2.*2.

    N[0:m]=NNN(rq,sq,tq)
    dNdr[0:m]=dNNNdr(rq,sq,tq)
    dNds[0:m]=dNNNds(rq,sq,tq)
    dNdt[0:m]=dNNNdt(rq,sq,tq)

    #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
    #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
    #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
    #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
    #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
    #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
    #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
    #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
    #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
    #jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
    #end for

    for k in range(0,m):
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        ezz[iel]+=dNdz[k]*w[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]
        exz[iel]+=0.5*dNdz[k]*u[icon[k,iel]]+0.5*dNdx[k]*w[icon[k,iel]]
        eyz[iel]+=0.5*dNdz[k]*v[icon[k,iel]]+0.5*dNdy[k]*w[icon[k,iel]]
    #end for

    p[iel]=-penalty*(exx[iel]+eyy[iel]+ezz[iel])
    visc[iel]=viscosity(xc[iel],yc[iel],zc[iel])
    dens[iel]=bz(xc[iel],yc[iel],zc[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                    +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])
    
#end for

print("     -> p   (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.4e %.4e " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyz),np.max(eyz)))
print("     -> visc (m,M) %.4e %.4e " %(np.min(visc),np.max(visc)))
print("     -> dens (m,M) %.4e %.4e " %(np.min(dens),np.max(dens)))

np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,zc,exx,eyy,exy')

print("compute p and strainrate: %.3f s" % (time.time() - start))

#####################################################################
# compute vrms and errors
#####################################################################

errv=0
errp=0
errq=0
vrms=0.

for iel in range(0,nel):

    # integrate 3x3x3 quadrature points
    for iq in range(0,nqperdimV):
        for jq in range(0,nqperdimV):
            for kq in range(0,nqperdimV):

                # position & weight of quad. point
                rq=qcoords3[iq]
                sq=qcoords3[jq]
                tq=qcoords3[kq]
                weightq=qweights3[iq]*qweights3[jq]*qweights3[kq]

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                #jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                #jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                #jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                #jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                #jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                #jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                #jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                #jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                #jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                #jcob = np.linalg.det(jcb)

                xq=N.dot(x[icon[:,iel]])
                yq=N.dot(y[icon[:,iel]])
                zq=N.dot(z[icon[:,iel]])
                uq=N.dot(u[icon[:,iel]])
                vq=N.dot(v[icon[:,iel]])
                wq=N.dot(w[icon[:,iel]])
                qq=N.dot(q[icon[:,iel]])

                vrms+=(uq**2+vq**2+wq**2)*jcob*weightq

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*weightq*jcob

                errq+=(qq-pth(xq,yq,zq))**2*weightq*jcob

                errp+=(p[iel]-pth(xq,yq,zq))**2*weightq*jcob

            #end for
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)

vrms=np.sqrt(vrms/Lx/Ly/Lz)

print("     -> nel= %6d ; errv: %e ; errp: %e ; errq: %e " %(nel,errv,errp,errq))

print("     -> nel= %6d ; vrms: %e" % (nel,vrms))

#####################################################################
# export various measurements for stokes sphere benchmark 
#####################################################################

vel=np.sqrt(u**2+v**2+w**2)
print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      np.min(w),np.max(w),\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      np.min(q),np.max(q),
      vrms)

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pth(xc[iel],yc[iel],zc[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='density*gz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % dens[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % visc[iel])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % ezz[iel])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%f\n" % pth(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%f\n" % q[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.20f %.20f %.20f \n" %(uth(x[i],y[i],z[i]),\
                                              vth(x[i],y[i],z[i]),\
                                              wth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
