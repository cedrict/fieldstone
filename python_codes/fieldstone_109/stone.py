import sys as sys
import numpy as np
import time as timing
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import math

cm=0.01
year=365.25*24*3600

#------------------------------------------------------------------------------
# ordering of nodes is a bit weird but it follows the VTK format 25
#------------------------------------------------------------------------------

def NNV(r,s,t):
    NNN01= 0.5*r*(r-1) * 0.5*s*(s-1) * 0.5*t*(t-1) # -1,-1,-1 
    NNN02= 0.5*r*(r+1) * 0.5*s*(s-1) * 0.5*t*(t-1) # +1,-1,-1
    NNN03= 0.5*r*(r+1) * 0.5*s*(s+1) * 0.5*t*(t-1) # +1, 1,-1
    NNN04= 0.5*r*(r-1) * 0.5*s*(s+1) * 0.5*t*(t-1) # -1, 1,-1
    NNN05= 0.5*r*(r-1) * 0.5*s*(s-1) * 0.5*t*(t+1) # -1,-1,+1
    NNN06= 0.5*r*(r+1) * 0.5*s*(s-1) * 0.5*t*(t+1) # +1,-1,+1
    NNN07= 0.5*r*(r+1) * 0.5*s*(s+1) * 0.5*t*(t+1) # +1, 1,+1
    NNN08= 0.5*r*(r-1) * 0.5*s*(s+1) * 0.5*t*(t+1) # -1, 1,+1

    NNN09= (1.-r**2)   * 0.5*s*(s-1) * 0.5*t*(t-1) #  0,-1,-1 
    NNN10= 0.5*r*(r+1) * (1.-s**2)   * 0.5*t*(t-1) # +1, 0,-1
    NNN11= (1.-r**2)   * 0.5*s*(s+1) * 0.5*t*(t-1) #  0,+1,-1
    NNN12= 0.5*r*(r-1) * (1.-s**2)   * 0.5*t*(t-1) # -1, 0,-1

    NNN13= (1.-r**2)   * 0.5*s*(s-1) * 0.5*t*(t+1) #  0,-1,+1
    NNN14= 0.5*r*(r+1) * (1.-s**2)   * 0.5*t*(t+1) # +1, 0,+1
    NNN15= (1.-r**2)   * 0.5*s*(s+1) * 0.5*t*(t+1) #  0,+1,+1
    NNN16= 0.5*r*(r-1) * (1.-s**2)   * 0.5*t*(t+1) # -1, 0,+1 

    NNN17= 0.5*r*(r-1) * 0.5*s*(s-1) * (1.-t**2)   # -1,-1, 0
    NNN18= 0.5*r*(r+1) * 0.5*s*(s-1) * (1.-t**2)   # +1,-1, 0
    NNN19= 0.5*r*(r+1) * 0.5*s*(s+1) * (1.-t**2)   # +1,+1, 0
    NNN20= 0.5*r*(r-1) * 0.5*s*(s+1) * (1.-t**2)   # -1,+1, 0

    NNN21= (1.-r**2)   * (1.-s**2)   * 0.5*t*(t-1) #  0, 0,-1
    NNN22= (1.-r**2)   * 0.5*s*(s-1) * (1.-t**2)   #  0,-1, 0
    NNN23= 0.5*r*(r+1) * (1.-s**2)   * (1.-t**2)   # +1, 0, 0
    NNN24= (1.-r**2)   * 0.5*s*(s+1) * (1.-t**2)   #  0,+1, 0
    NNN25= 0.5*r*(r-1) * (1.-s**2)   * (1.-t**2)   # -1, 0, 0
    NNN26= (1.-r**2)   * (1.-s**2)   * 0.5*t*(t+1) #  0, 0,+1
    NNN27= (1.-r**2)   * (1.-s**2)   * (1.-t**2)   #  0, 0, 0
    return NNN01,NNN02,NNN03,NNN04,NNN05,NNN06,NNN07,NNN08,NNN09,\
           NNN10,NNN11,NNN12,NNN13,NNN14,NNN15,NNN16,NNN17,NNN18,\
           NNN19,NNN20,NNN21,NNN22,NNN23,NNN24,NNN25,NNN26,NNN27

def dNNVdr(r,s,t):
    dNNNdr01= 0.5*(2*r-1) * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNNNdr02= 0.5*(2*r+1) * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNNNdr03= 0.5*(2*r+1) * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNNNdr04= 0.5*(2*r-1) * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNNNdr05= 0.5*(2*r-1) * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNNNdr06= 0.5*(2*r+1) * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNNNdr07= 0.5*(2*r+1) * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNNNdr08= 0.5*(2*r-1) * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNNNdr09= (-2*r)           * 0.5*s*(s-1.) * 0.5*t*(t-1.) 
    dNNNdr10= 0.5*(2*r+1) * (1.-s**2)      * 0.5*t*(t-1.) 
    dNNNdr11= (-2*r)           * 0.5*s*(s+1.) * 0.5*t*(t-1.) 
    dNNNdr12= 0.5*(2*r-1) * (1.-s**2)      * 0.5*t*(t-1.) 
    dNNNdr13= (-2*r)           * 0.5*s*(s-1.) * 0.5*t*(t+1.) 
    dNNNdr14= 0.5*(2*r+1) * (1.-s**2)      * 0.5*t*(t+1.) 
    dNNNdr15= (-2*r)           * 0.5*s*(s+1.) * 0.5*t*(t+1.) 
    dNNNdr16= 0.5*(2*r-1) * (1.-s**2)      * 0.5*t*(t+1.) 
    dNNNdr17= 0.5*(2*r-1) * 0.5*s*(s-1.) * (1.-t**2) 
    dNNNdr18= 0.5*(2*r+1) * 0.5*s*(s-1.) * (1.-t**2) 
    dNNNdr19= 0.5*(2*r+1) * 0.5*s*(s+1.) * (1.-t**2) 
    dNNNdr20= 0.5*(2*r-1) * 0.5*s*(s+1.) * (1.-t**2) 
    dNNNdr21= (-2*r)           * (1.-s**2)      * 0.5*t*(t-1.) 
    dNNNdr22= (-2*r)           * 0.5*s*(s-1.) * (1.-t**2) 
    dNNNdr23= 0.5*(2*r+1) * (1.-s**2)      * (1.-t**2) 
    dNNNdr24= (-2*r)           * 0.5*s*(s+1.) * (1.-t**2) 
    dNNNdr25= 0.5*(2*r-1) * (1.-s**2)      * (1.-t**2) 
    dNNNdr26= (-2*r)           * (1.-s**2)      * 0.5*t*(t+1.) 
    dNNNdr27= (-2*r)           * (1.-s**2)      * (1.-t**2)
    return dNNNdr01,dNNNdr02,dNNNdr03,dNNNdr04,dNNNdr05,dNNNdr06,dNNNdr07,dNNNdr08,dNNNdr09,\
           dNNNdr10,dNNNdr11,dNNNdr12,dNNNdr13,dNNNdr14,dNNNdr15,dNNNdr16,dNNNdr17,dNNNdr18,\
           dNNNdr19,dNNNdr20,dNNNdr21,dNNNdr22,dNNNdr23,dNNNdr24,dNNNdr25,dNNNdr26,dNNNdr27

def dNNVds(r,s,t):
    dNNNds01= 0.5*r*(r-1.) * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNNNds02= 0.5*r*(r+1.) * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNNNds03= 0.5*r*(r+1.) * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNNNds04= 0.5*r*(r-1.) * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNNNds05= 0.5*r*(r-1.) * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNNNds06= 0.5*r*(r+1.) * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNNNds07= 0.5*r*(r+1.) * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNNNds08= 0.5*r*(r-1.) * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNNNds09= (1.-r**2)    * 0.5*(2*s-1.) * 0.5*t*(t-1.) 
    dNNNds10= 0.5*r*(r+1.) * (-2*s)       * 0.5*t*(t-1.) 
    dNNNds11= (1.-r**2)    * 0.5*(2*s+1.) * 0.5*t*(t-1.) 
    dNNNds12= 0.5*r*(r-1.) * (-2*s)       * 0.5*t*(t-1.) 
    dNNNds13= (1.-r**2)    * 0.5*(2*s-1.) * 0.5*t*(t+1.) 
    dNNNds14= 0.5*r*(r+1.) * (-2*s)       * 0.5*t*(t+1.) 
    dNNNds15= (1.-r**2)    * 0.5*(2*s+1.) * 0.5*t*(t+1.) 
    dNNNds16= 0.5*r*(r-1.) * (-2*s)       * 0.5*t*(t+1.) 
    dNNNds17= 0.5*r*(r-1.) * 0.5*(2*s-1.) * (1.-t**2) 
    dNNNds18= 0.5*r*(r+1.) * 0.5*(2*s-1.) * (1.-t**2) 
    dNNNds19= 0.5*r*(r+1.) * 0.5*(2*s+1.) * (1.-t**2) 
    dNNNds20= 0.5*r*(r-1.) * 0.5*(2*s+1.) * (1.-t**2) 
    dNNNds21= (1.-r**2)    * (-2*s)       * 0.5*t*(t-1.) 
    dNNNds22= (1.-r**2)    * 0.5*(2*s-1.) * (1.-t**2) 
    dNNNds23= 0.5*r*(r+1.) * (-2*s)       * (1.-t**2) 
    dNNNds24= (1.-r**2)    * 0.5*(2*s+1.) * (1.-t**2) 
    dNNNds25= 0.5*r*(r-1.) * (-2*s)       * (1.-t**2) 
    dNNNds26= (1.-r**2)    * (-2*s)       * 0.5*t*(t+1.) 
    dNNNds27= (1.-r**2)    * (-2*s)       * (1.-t**2) 
    return dNNNds01,dNNNds02,dNNNds03,dNNNds04,dNNNds05,dNNNds06,dNNNds07,dNNNds08,dNNNds09,\
           dNNNds10,dNNNds11,dNNNds12,dNNNds13,dNNNds14,dNNNds15,dNNNds16,dNNNds17,dNNNds18,\
           dNNNds19,dNNNds20,dNNNds21,dNNNds22,dNNNds23,dNNNds24,dNNNds25,dNNNds26,dNNNds27

def dNNVdt(r,s,t):
    dNNNdt01= 0.5*r*(r-1.) * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNNNdt02= 0.5*r*(r+1.) * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNNNdt03= 0.5*r*(r+1.) * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNNNdt04= 0.5*r*(r-1.) * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNNNdt05= 0.5*r*(r-1.) * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNNNdt06= 0.5*r*(r+1.) * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNNNdt07= 0.5*r*(r+1.) * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNNNdt08= 0.5*r*(r-1.) * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNNNdt09= (1.-r**2)      * 0.5*s*(s-1.) * 0.5*(2*t-1.) 
    dNNNdt10= 0.5*r*(r+1.) * (1.-s**2)      * 0.5*(2*t-1.) 
    dNNNdt11= (1.-r**2)      * 0.5*s*(s+1.) * 0.5*(2*t-1.) 
    dNNNdt12= 0.5*r*(r-1.) * (1.-s**2)      * 0.5*(2*t-1.) 
    dNNNdt13= (1.-r**2)      * 0.5*s*(s-1.) * 0.5*(2*t+1.) 
    dNNNdt14= 0.5*r*(r+1.) * (1.-s**2)      * 0.5*(2*t+1.) 
    dNNNdt15= (1.-r**2)      * 0.5*s*(s+1.) * 0.5*(2*t+1.) 
    dNNNdt16= 0.5*r*(r-1.) * (1.-s**2)      * 0.5*(2*t+1.) 
    dNNNdt17= 0.5*r*(r-1.) * 0.5*s*(s-1.) * (-2*t) 
    dNNNdt18= 0.5*r*(r+1.) * 0.5*s*(s-1.) * (-2*t) 
    dNNNdt19= 0.5*r*(r+1.) * 0.5*s*(s+1.) * (-2*t) 
    dNNNdt20= 0.5*r*(r-1.) * 0.5*s*(s+1.) * (-2*t) 
    dNNNdt21= (1.-r**2)      * (1.-s**2)      * 0.5*(2*t-1.) 
    dNNNdt22= (1.-r**2)      * 0.5*s*(s-1.) * (-2*t) 
    dNNNdt23= 0.5*r*(r+1.) * (1.-s**2)      * (-2*t) 
    dNNNdt24= (1.-r**2)      * 0.5*s*(s+1.) * (-2*t) 
    dNNNdt25= 0.5*r*(r-1.) * (1.-s**2)      * (-2*t) 
    dNNNdt26= (1.-r**2)      * (1.-s**2)      * 0.5*(2*t+1.) 
    dNNNdt27= (1.-r**2)      * (1.-s**2)      * (-2*t) 
    return dNNNdt01,dNNNdt02,dNNNdt03,dNNNdt04,dNNNdt05,dNNNdt06,dNNNdt07,dNNNdt08,dNNNdt09,\
           dNNNdt10,dNNNdt11,dNNNdt12,dNNNdt13,dNNNdt14,dNNNdt15,dNNNdt16,dNNNdt17,dNNNdt18,\
           dNNNdt19,dNNNdt20,dNNNdt21,dNNNdt22,dNNNdt23,dNNNdt24,dNNNdt25,dNNNdt26,dNNNdt27

def NNP(r,s,t):
    NNN01=0.125*(1-r)*(1-s)*(1-t)    
    NNN02=0.125*(1+r)*(1-s)*(1-t)    
    NNN03=0.125*(1+r)*(1+s)*(1-t)    
    NNN04=0.125*(1-r)*(1+s)*(1-t)    
    NNN05=0.125*(1-r)*(1-s)*(1+t)    
    NNN06=0.125*(1+r)*(1-s)*(1+t)    
    NNN07=0.125*(1+r)*(1+s)*(1+t)    
    NNN08=0.125*(1-r)*(1+s)*(1+t)    
    return NNN01,NNN02,NNN03,NNN04,\
           NNN05,NNN06,NNN07,NNN08

#------------------------------------------------------------------------------

def eta(x,y,z):
    if (x-Lx/2)**2+(y)**2<200e3**2:
       val=2e21
    else:
       val=2e18
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------stone 109----------")
print("-----------------------------")

ndim=3
ndofV=3
ndofP=1
mV=27
mP=8

Lx=1000e3
Ly=500e3
Lz=15e3

nelx=28
nely=14
nelz=5

nnx=2*nelx+1
nny=2*nely+1
nnz=2*nelz+1
nel=nelx*nely*nelz
NV=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
NP=(nelx+1)*(nely+1)*(nelz+1)
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

print('nelx =',nelx)
print('nely =',nely)
print('nelz =',nelz)
print('nnx =',nnx)
print('nny =',nny)
print('nnz =',nnz)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

eps=1e-8

gx=0
gy=0
gz=0

U0=8*cm/year

rho=3000

sparse=False
pnormalise=True

rVnodes=[-1,1,1,-1,-1,1,1,-1, 0,1,0,-1,0,1,0,-1,-1,1,1,-1, 0,0,1,0,-1,0,0 ] 
sVnodes=[-1,-1,1,1,-1,-1,1,1, -1,0,1,0,-1,0,1,0,-1,-1,1,1, 0,-1,0,1,0,0,0 ]  
tVnodes=[-1,-1,-1,-1,1,1,1,1, -1,-1,-1,-1,1,1,1,1,0,0,0,0, -1,0,0,0,0,1,0 ] 

eta_ref=1e21

Pi2=8*U0*2e18/Lz**2 * Lx /2
print('Pi/2=',Pi2)

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates
zV = np.empty(NV,dtype=np.float64)  # y coordinates

counter=0    
for i in range(0,2*nelx+1):
    for j in range(0,2*nely+1):
        for k in range(0,2*nelz+1):
            xV[counter]=i*hx/2.
            yV[counter]=j*hy/2.
            zV[counter]=k*hz/2.
            counter += 1
        #end for
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV,zV]).T,header='# x,y,z')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0    
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconV[ 0,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            iconV[ 1,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            iconV[ 2,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            iconV[ 3,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            iconV[ 4,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            iconV[ 5,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            iconV[ 6,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            iconV[ 7,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            iconV[ 8,counter]=(2*(k)+1)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            iconV[ 9,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            iconV[10,counter]=(2*(k)+1)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            iconV[11,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            iconV[12,counter]=(2*(k)+3)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            iconV[13,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            iconV[14,counter]=(2*(k)+3)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            iconV[15,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            iconV[16,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+0) -1 
            iconV[17,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+2) -1 
            iconV[18,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+2) -1 
            iconV[19,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+0) -1 
            iconV[20,counter]=(2*(k)+1)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            iconV[21,counter]=(2*(k)+2)+ nnz*(2*(j)+0) + nny*nnz*(2*(i)+1) -1 
            iconV[22,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+2) -1 
            iconV[23,counter]=(2*(k)+2)+ nnz*(2*(j)+2) + nny*nnz*(2*(i)+1) -1 
            iconV[24,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+0) -1 
            iconV[25,counter]=(2*(k)+3)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            iconV[26,counter]=(2*(k)+2)+ nnz*(2*(j)+1) + nny*nnz*(2*(i)+1) -1 
            counter=counter+1   
        #end for
    #end for
#end for

print("setup: connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64) 
yP=np.empty(NP,dtype=np.float64)  
zP=np.empty(NP,dtype=np.float64)   
iconP=np.zeros((mP,nel),dtype=np.int32)

counter=0    
for i in range(0,nelx+1):
    for j in range(0,nely+1):
        for k in range(0,nelz+1):
            xP[counter]=i*Lx/nelx    
            yP[counter]=j*Ly/nely
            zP[counter]=k*Lz/nelz
            counter+=1    
        #end for
    #end for
#end for

counter = 0 
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconP[0,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k
            iconP[1,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k
            iconP[2,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k
            iconP[3,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k
            iconP[4,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k+1
            iconP[5,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k+1
            iconP[6,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k+1
            iconP[7,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

#np.savetxt('gridP.ascii',np.array([xP,yP,zP]).T,header='# x,y,z')

print("setup: build P grid: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(NfemV, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV+0]   = U0*zV[i]*(Lz-zV[i])*4/Lz**2
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       bc_fix[i*ndofV+2] = True ; bc_val[i*ndofV+2] = 0.
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV+0]   = U0*zV[i]*(Lz-zV[i])*4/Lz**2
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       bc_fix[i*ndofV+2] = True ; bc_val[i*ndofV+2] = 0.
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if zV[i]/Lz<eps:
       bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       bc_fix[i*ndofV+2] = True ; bc_val[i*ndofV+2] = 0.
    if zV[i]/Lz>(1-eps):
       bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       bc_fix[i*ndofV+2] = True ; bc_val[i*ndofV+2] = 0.

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

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

constr  = np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat   = np.zeros((6,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((6,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
c_mat   = np.zeros((6,6),dtype=np.float64)
c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64) # int of shape functions P

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):

                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                NNNV[0:mV]=NNV(rq,sq,tq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
                dNNNVds[0:mV]=dNNVds(rq,sq,tq)
                dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)
                NNNP[0:mP]=NNP(rq,sq,tq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[0,2] += dNNNVdr[k]*zV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                    jcb[1,2] += dNNNVds[k]*zV[iconV[k,iel]]
                    jcb[2,0] += dNNNVdt[k]*xV[iconV[k,iel]]
                    jcb[2,1] += dNNNVdt[k]*yV[iconV[k,iel]]
                    jcb[2,2] += dNNNVdt[k]*zV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx, dNdy, dNdz
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    zq+=NNNV[k]*zV[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]+jcbi[0,2]*dNNNVdt[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]+jcbi[1,2]*dNNNVdt[k]
                    dNNNVdz[k]=jcbi[2,0]*dNNNVdr[k]+jcbi[2,1]*dNNNVds[k]+jcbi[2,2]*dNNNVdt[k]

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:6, 3*i:3*i+3] = [[dNNNVdx[i],0.        ,0.        ],
                                             [0.        ,dNNNVdy[i],0.        ],
                                             [0.        ,0.        ,dNNNVdz[i]],
                                             [dNNNVdy[i],dNNNVdx[i],0.        ],
                                             [dNNNVdz[i],0.        ,dNNNVdx[i]],
                                             [0.        ,dNNNVdz[i],dNNNVdy[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq,zq)*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*rho*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho*gy
                    f_el[ndofV*i+2]+=NNNV[i]*jcob*weightq*rho*gz

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=NNNP[i]
                    N_mat[3,i]=0.
                    N_mat[4,i]=0.
                    N_mat[5,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                NNNNP[:]+=NNNP[:]*jcob*weightq

            # end for kq
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

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

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
        constr[m2]+=NNNNP[k2]

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

if not sparse:
   if pnormalise:
      a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
      a_mat[Nfem,NfemV:Nfem]=constr
      a_mat[NfemV:Nfem,Nfem]=constr
   else:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   #end if
#else:

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

if True:
   plt.spy(sparse_matrix, markersize=0.1)
   plt.savefig('matrix.png', bbox_inches='tight')
   plt.clf()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.4e %.4e " %(np.min(w),np.max(w)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

np.savetxt('velocity.ascii',np.array([xV,yV,zV,u,v,w]).T,header='# x,y,z,u,v,w')
np.savetxt('pressure.ascii',np.array([xP,yP,zP,p]).T,header='# x,y,z,p')

print("split vel into u,v: %.3f s" % (timing.time() - start))

#####################################################################
# project pressure onto velocity grid
#####################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)
c=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        NNNP[0:mP]=NNP(rVnodes[i],sVnodes[i],tVnodes[i])
        q[iconV[i,iel]]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        c[iconV[i,iel]]+=1.
    # end for i
# end for iel

q=q/c

np.savetxt('q.ascii',np.array([xV,yV,zV,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (timing.time() - start))

#####################################################################
# create analytical pressure field
#####################################################################
start = timing.time()

p_analytical=np.zeros(NV,dtype=np.float64)
p_analytical1=np.zeros(NV,dtype=np.float64)
p_analytical2=np.zeros(NV,dtype=np.float64)
theta=np.zeros(NV,dtype=np.float64)

kappa=(Lz/2)**2/3/eta(0,0,0)

a=200e3

U=U0*2/3
    
plinefile=open('p_line.ascii',"w")
psurffile=open('p_surf.ascii',"w")

for i in range(0,NV):
    ri=np.sqrt((xV[i]-Lx/2)**2+(yV[i])**2)
    theta[i]=math.atan2(yV[i],xV[i]-Lx/2)    
    if ri>=a:
       p_analytical[i]=-U/kappa*(ri+a**2/ri)*np.cos(theta[i])
       p_analytical1[i]=-U/kappa*(ri)*np.cos(theta[i])
       p_analytical2[i]=-U/kappa*(a**2/ri)*np.cos(theta[i])

       if abs(yV[i]/Ly)<eps and abs(zV[i]-Lz)/Lz<eps:
          plinefile.write("%e %e %e %e %e %e %e \n" %(xV[i],q[i],p_analytical[i],p_analytical1[i],\
                                                p_analytical2[i],q[i]-p_analytical[i],p_analytical[i]/q[i]))
       if abs(zV[i]-Lz)/Lz<eps:
          psurffile.write("%e %e %e %e %e %e %e %e\n" %(xV[i],yV[i],q[i],p_analytical[i],p_analytical1[i],\
                                                p_analytical2[i],q[i]-p_analytical[i],p_analytical[i]/q[i]))
    else:
       p_analytical[i]=0
       p_analytical1[i]=0
       p_analytical2[i]=0

plinefile.close()
psurffile.close()

print("compute analytical pressure: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = timing.time()

if True:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],zV[i]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%10e \n" % np.sum(p[iconP[0:8,iel]]*0.125))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,w[i]/cm*year))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %p_analytical[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical(1)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %p_analytical1[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p_analytical(2)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %p_analytical2[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %theta[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(eta(xV[i],yV[i],zV[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n" \
                       %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                         iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel],\
                         iconV[8,iel],iconV[9,iel],iconV[10,iel],iconV[11,iel],\
                         iconV[12,iel],iconV[13,iel],iconV[14,iel],iconV[15,iel],\
                         iconV[16,iel],iconV[17,iel],iconV[18,iel],iconV[19,iel],\
                         iconV[20,iel],iconV[21,iel],iconV[22,iel],iconV[23,iel],\
                         iconV[24,iel],iconV[25,iel],iconV[26,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*27))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %25)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("export to vtu: %.3f s" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
