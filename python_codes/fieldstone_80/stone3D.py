import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

#------------------------------------------------------------------------------

def rho(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=1.001
    else:
       val=1.
    return val

def eta(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
       val=1.
    else:
       val=1.
    return val

#------------------------------------------------------------------------------

def B9u(r,s,t):
    return 0.5*(1-r)*(1-s**2)*(1-t**2)
def dB9udr(r,s,t):
    return -0.5*(1-s**2)*(1-t**2)
def dB9uds(r,s,t):
    return -(1-r)*s*(1-t**2)
def dB9udt(r,s,t):
    return -(1-r)*(1-s**2)*t

def B10u(r,s,t):
    return 0.5*(1+r)*(1-s**2)*(1-t**2)
def dB10udr(r,s,t):
    return 0.5*(1-s**2)*(1-t**2)
def dB10uds(r,s,t):
    return -(1+r)*s*(1-t**2)
def dB10udt(r,s,t):
    return -(1+r)*(1-s**2)*t

#------------------------------------------------------------------------------

def B9v(r,s,t):
    return 0.5*(1-r**2)*(1-s)*(1-t**2)
def dB9vdr(r,s,t):
    return -r*(1-s)*(1-t**2)
def dB9vds(r,s,t):
    return -0.5*(1-r**2)*(1-t**2)
def dB9vdt(r,s,t):
    return -(1-r**2)*(1-s)*t

def B10v(r,s,t):
    return 0.5*(1-r**2)*(1+s)*(1-t**2)
def dB10vdr(r,s,t):
    return -r*(1+s)*(1-t**2)
def dB10vds(r,s,t):
    return 0.5*(1-r**2)*(1-t**2)
def dB10vdt(r,s,t):
    return -(1-r**2)*(1+s)*t

#------------------------------------------------------------------------------

def B9w(r,s,t):
    return 0.5*(1-r**2)*(1-s**2)*(1-t)
def dB9wdr(r,s,t):
    return -r*(1-s**2)*(1-t)
def dB9wds(r,s,t):
    return -(1-r**2)*s*(1-t)
def dB9wdt(r,s,t):
    return -0.5*(1-r**2)*(1-s**2)

def B10w(r,s,t):
    return 0.5*(1-r**2)*(1-s**2)*(1+t)
def dB10wdr(r,s,t):
    return -r*(1-s**2)*(1+t)
def dB10wds(r,s,t):
    return -(1-r**2)*s*(1+t)
def dB10wdt(r,s,t):
    return 0.5*(1-r**2)*(1-s**2)

#------------------------------------------------------------------------------

def NNVu(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) -0.25*B9u(r,s,t)
    N_1=0.125*(1+r)*(1-s)*(1-t) -0.25*B10u(r,s,t)
    N_2=0.125*(1+r)*(1+s)*(1-t) -0.25*B10u(r,s,t)
    N_3=0.125*(1-r)*(1+s)*(1-t) -0.25*B9u(r,s,t)
    N_4=0.125*(1-r)*(1-s)*(1+t) -0.25*B9u(r,s,t)
    N_5=0.125*(1+r)*(1-s)*(1+t) -0.25*B10u(r,s,t)
    N_6=0.125*(1+r)*(1+s)*(1+t) -0.25*B10u(r,s,t)
    N_7=0.125*(1-r)*(1+s)*(1+t) -0.25*B9u(r,s,t)
    N_8= B9u(r,s,t)
    N_9= B10u(r,s,t)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9

def dNNVudr(r,s,t):
    dNdr_0=-0.125*(1-s)*(1-t) -0.25*dB9udr(r,s,t) 
    dNdr_1=+0.125*(1-s)*(1-t) -0.25*dB10udr(r,s,t) 
    dNdr_2=+0.125*(1+s)*(1-t) -0.25*dB10udr(r,s,t) 
    dNdr_3=-0.125*(1+s)*(1-t) -0.25*dB9udr(r,s,t) 
    dNdr_4=-0.125*(1-s)*(1+t) -0.25*dB9udr(r,s,t) 
    dNdr_5=+0.125*(1-s)*(1+t) -0.25*dB10udr(r,s,t) 
    dNdr_6=+0.125*(1+s)*(1+t) -0.25*dB10udr(r,s,t) 
    dNdr_7=-0.125*(1+s)*(1+t) -0.25*dB9udr(r,s,t) 
    dNdr_8= dB9udr(r,s,t)
    dNdr_9= dB10udr(r,s,t)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8,dNdr_9

def dNNVuds(r,s,t):
    dNds_0=-0.125*(1-r)*(1-t) -0.25*dB9uds(r,s,t) 
    dNds_1=-0.125*(1+r)*(1-t) -0.25*dB10uds(r,s,t) 
    dNds_2=+0.125*(1+r)*(1-t) -0.25*dB10uds(r,s,t) 
    dNds_3=+0.125*(1-r)*(1-t) -0.25*dB9uds(r,s,t) 
    dNds_4=-0.125*(1-r)*(1+t) -0.25*dB9uds(r,s,t) 
    dNds_5=-0.125*(1+r)*(1+t) -0.25*dB10uds(r,s,t) 
    dNds_6=+0.125*(1+r)*(1+t) -0.25*dB10uds(r,s,t) 
    dNds_7=+0.125*(1-r)*(1+t) -0.25*dB9uds(r,s,t) 
    dNds_8= dB9uds(r,s,t)
    dNds_9= dB10uds(r,s,t)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8,dNds_9

def dNNVudt(r,s,t):
    dNdt_0=-0.125*(1-r)*(1-s) -0.25*dB9udt(r,s,t) 
    dNdt_1=-0.125*(1+r)*(1-s) -0.25*dB10udt(r,s,t) 
    dNdt_2=-0.125*(1+r)*(1+s) -0.25*dB10udt(r,s,t) 
    dNdt_3=-0.125*(1-r)*(1+s) -0.25*dB9udt(r,s,t) 
    dNdt_4=+0.125*(1-r)*(1-s) -0.25*dB9udt(r,s,t) 
    dNdt_5=+0.125*(1+r)*(1-s) -0.25*dB10udt(r,s,t) 
    dNdt_6=+0.125*(1+r)*(1+s) -0.25*dB10udt(r,s,t) 
    dNdt_7=+0.125*(1-r)*(1+s) -0.25*dB9udt(r,s,t) 
    dNdt_8= dB9udt(r,s,t)
    dNdt_9= dB10udt(r,s,t)
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8,dNdt_9


#------------------------------------------------------------------------------

def NNVv(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) -0.25*B9v(r,s,t)
    N_1=0.125*(1+r)*(1-s)*(1-t) -0.25*B9v(r,s,t)
    N_2=0.125*(1+r)*(1+s)*(1-t) -0.25*B10v(r,s,t)
    N_3=0.125*(1-r)*(1+s)*(1-t) -0.25*B10v(r,s,t)
    N_4=0.125*(1-r)*(1-s)*(1+t) -0.25*B9v(r,s,t)
    N_5=0.125*(1+r)*(1-s)*(1+t) -0.25*B9v(r,s,t)
    N_6=0.125*(1+r)*(1+s)*(1+t) -0.25*B10v(r,s,t)
    N_7=0.125*(1-r)*(1+s)*(1+t) -0.25*B10v(r,s,t)
    N_8= B9v(r,s,t)
    N_9= B10v(r,s,t)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9

def dNNVvdr(r,s,t):
    dNdr_0=-0.125*(1-s)*(1-t) -0.25*dB9vdr(r,s,t) 
    dNdr_1=+0.125*(1-s)*(1-t) -0.25*dB9vdr(r,s,t) 
    dNdr_2=+0.125*(1+s)*(1-t) -0.25*dB10vdr(r,s,t) 
    dNdr_3=-0.125*(1+s)*(1-t) -0.25*dB10vdr(r,s,t) 
    dNdr_4=-0.125*(1-s)*(1+t) -0.25*dB9vdr(r,s,t) 
    dNdr_5=+0.125*(1-s)*(1+t) -0.25*dB9vdr(r,s,t) 
    dNdr_6=+0.125*(1+s)*(1+t) -0.25*dB10vdr(r,s,t) 
    dNdr_7=-0.125*(1+s)*(1+t) -0.25*dB10vdr(r,s,t) 
    dNdr_8= dB9vdr(r,s,t)
    dNdr_9= dB10vdr(r,s,t)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8,dNdr_9

def dNNVvds(r,s,t):
    dNds_0=-0.125*(1-r)*(1-t) -0.25*dB9vds(r,s,t) 
    dNds_1=-0.125*(1+r)*(1-t) -0.25*dB9vds(r,s,t) 
    dNds_2=+0.125*(1+r)*(1-t) -0.25*dB10vds(r,s,t) 
    dNds_3=+0.125*(1-r)*(1-t) -0.25*dB10vds(r,s,t) 
    dNds_4=-0.125*(1-r)*(1+t) -0.25*dB9vds(r,s,t) 
    dNds_5=-0.125*(1+r)*(1+t) -0.25*dB9vds(r,s,t) 
    dNds_6=+0.125*(1+r)*(1+t) -0.25*dB10vds(r,s,t) 
    dNds_7=+0.125*(1-r)*(1+t) -0.25*dB10vds(r,s,t) 
    dNds_8= dB9vds(r,s,t)
    dNds_9= dB10vds(r,s,t)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8,dNds_9

def dNNVvdt(r,s,t):
    dNdt_0=-0.125*(1-r)*(1-s) -0.25*dB9vdt(r,s,t) 
    dNdt_1=-0.125*(1+r)*(1-s) -0.25*dB9vdt(r,s,t) 
    dNdt_2=-0.125*(1+r)*(1+s) -0.25*dB10vdt(r,s,t) 
    dNdt_3=-0.125*(1-r)*(1+s) -0.25*dB10vdt(r,s,t) 
    dNdt_4=+0.125*(1-r)*(1-s) -0.25*dB9vdt(r,s,t) 
    dNdt_5=+0.125*(1+r)*(1-s) -0.25*dB9vdt(r,s,t) 
    dNdt_6=+0.125*(1+r)*(1+s) -0.25*dB10vdt(r,s,t) 
    dNdt_7=+0.125*(1-r)*(1+s) -0.25*dB10vdt(r,s,t) 
    dNdt_8= dB9vdt(r,s,t)
    dNdt_9= dB10vdt(r,s,t)
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8,dNdt_9

#------------------------------------------------------------------------------

def NNVw(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) -0.25*B9w(r,s,t)
    N_1=0.125*(1+r)*(1-s)*(1-t) -0.25*B9w(r,s,t)
    N_2=0.125*(1+r)*(1+s)*(1-t) -0.25*B9w(r,s,t)
    N_3=0.125*(1-r)*(1+s)*(1-t) -0.25*B9w(r,s,t)
    N_4=0.125*(1-r)*(1-s)*(1+t) -0.25*B10w(r,s,t)
    N_5=0.125*(1+r)*(1-s)*(1+t) -0.25*B10w(r,s,t)
    N_6=0.125*(1+r)*(1+s)*(1+t) -0.25*B10w(r,s,t)
    N_7=0.125*(1-r)*(1+s)*(1+t) -0.25*B10w(r,s,t)
    N_8= B9w(r,s,t)
    N_9= B10w(r,s,t)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9

def dNNVwdr(r,s,t):
    dNdr_0=-0.125*(1-s)*(1-t) -0.25*dB9wdr(r,s,t) 
    dNdr_1=+0.125*(1-s)*(1-t) -0.25*dB9wdr(r,s,t) 
    dNdr_2=+0.125*(1+s)*(1-t) -0.25*dB9wdr(r,s,t) 
    dNdr_3=-0.125*(1+s)*(1-t) -0.25*dB9wdr(r,s,t) 
    dNdr_4=-0.125*(1-s)*(1+t) -0.25*dB10wdr(r,s,t) 
    dNdr_5=+0.125*(1-s)*(1+t) -0.25*dB10wdr(r,s,t) 
    dNdr_6=+0.125*(1+s)*(1+t) -0.25*dB10wdr(r,s,t) 
    dNdr_7=-0.125*(1+s)*(1+t) -0.25*dB10wdr(r,s,t) 
    dNdr_8= dB9wdr(r,s,t)
    dNdr_9= dB10wdr(r,s,t)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8,dNdr_9

def dNNVwds(r,s,t):
    dNds_0=-0.125*(1-r)*(1-t) -0.25*dB9wds(r,s,t) 
    dNds_1=-0.125*(1+r)*(1-t) -0.25*dB9wds(r,s,t) 
    dNds_2=+0.125*(1+r)*(1-t) -0.25*dB9wds(r,s,t) 
    dNds_3=+0.125*(1-r)*(1-t) -0.25*dB9wds(r,s,t) 
    dNds_4=-0.125*(1-r)*(1+t) -0.25*dB10wds(r,s,t) 
    dNds_5=-0.125*(1+r)*(1+t) -0.25*dB10wds(r,s,t) 
    dNds_6=+0.125*(1+r)*(1+t) -0.25*dB10wds(r,s,t) 
    dNds_7=+0.125*(1-r)*(1+t) -0.25*dB10wds(r,s,t) 
    dNds_8= dB9wds(r,s,t)
    dNds_9= dB10wds(r,s,t)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8,dNds_9

def dNNVwdt(r,s,t):
    dNdt_0=-0.125*(1-r)*(1-s) -0.25*dB9wdt(r,s,t) 
    dNdt_1=-0.125*(1+r)*(1-s) -0.25*dB9wdt(r,s,t) 
    dNdt_2=-0.125*(1+r)*(1+s) -0.25*dB9wdt(r,s,t) 
    dNdt_3=-0.125*(1-r)*(1+s) -0.25*dB9wdt(r,s,t) 
    dNdt_4=+0.125*(1-r)*(1-s) -0.25*dB10wdt(r,s,t) 
    dNdt_5=+0.125*(1+r)*(1-s) -0.25*dB10wdt(r,s,t) 
    dNdt_6=+0.125*(1+r)*(1+s) -0.25*dB10wdt(r,s,t) 
    dNdt_7=+0.125*(1-r)*(1+s) -0.25*dB10wdt(r,s,t) 
    dNdt_8= dB9wdt(r,s,t)
    dNdt_9= dB10wdt(r,s,t)
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8,dNdt_9

def NNP(r,s,t):
    return 1

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 10--------")
print("-----------------------------")

ndofV=3  # number of degrees of freedom per node
ndofP=1
mV=10
mP=1

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx = 14 
   nely = 14 #nelx
   nelz = 14 #nelx

visu=1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

nel=nelx*nely*nelz  # number of elements, total

NP=nelx*nely*nelz
NV=nnx*nny*nnz+nnx*nely*nelz+nny*nelx*nelz+nnz*nelx*nely 

NfemV=ndofV*nnx*nny*nnz + nnx*nely*nelz + nny*nelx*nelz + nnz*nelx*nely 
NfemP=NP*ndofP  # Total number of degrees of freedom
Nfem=NfemV+NfemP

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

eps=1.e-10

gx=0.    # gravity vector, x component
gy=0.    # gravity vector, y component
gz=-10.  # gravity vector, z component

nqperdim=2

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

pnormalise=False

#################################################################
#################################################################

print("Lx",Lx)
print("Ly",Ly)
print("Lz",Lz)
print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("NV=",NV)
print("NP=",NP)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("hx",hx)
print("hy",hy)
print("hz",hz)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

xV = np.zeros(NV,dtype=np.float64)  # x coordinates
yV = np.zeros(NV,dtype=np.float64)  # y coordinates
zV = np.zeros(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0, nnx):
    for j in range(0, nny):
        for k in range(0, nnz):
            xV[counter]=i*hx
            yV[counter]=j*hy
            zV[counter]=k*hz
            counter += 1
        #end for
    #end for
#end for

for i in range(0, nnx):
    for j in range(0, nely):
        for k in range(0, nelz):
            xV[counter]=i*hx
            yV[counter]=(j+0.5)*hy
            zV[counter]=(k+0.5)*hz
            counter += 1

for j in range(0, nny):
    for i in range(0, nelx):
        for k in range(0, nelz):
            xV[counter]=(i+0.5)*hx
            yV[counter]=j*hy
            zV[counter]=(k+0.5)*hz
            counter += 1

for k in range(0, nnz):
    for i in range(0, nelx):
        for j in range(0, nely):
            xV[counter]=(i+0.5)*hx
            yV[counter]=(j+0.5)*hy
            zV[counter]=k*hz
            counter += 1

np.savetxt('grid.ascii',np.array([xV,yV,zV]).T,header='# x,y,z')
   
print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
# Each element is connected to 8+6=14 nodes, but in each direction 
# only to 8+2=10. We therefore make 3 icon arrays for all three 
# directions
#################################################################
start = time.time()

iconu =np.zeros((mV, nel),dtype=np.int16)
iconv =np.zeros((mV, nel),dtype=np.int16)
iconw =np.zeros((mV, nel),dtype=np.int16)

counter = 0
for i in range(0, nelx):
    for j in range(0, nely):
        for k in range(0, nelz):

            iconu[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            iconu[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            iconu[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            iconu[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            iconu[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            iconu[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            iconu[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            iconu[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            iconu[8,counter]=nnx*nny*nnz+k +j*nelz      +nely*nelz*i       # face 0
            iconu[9,counter]=nnx*nny*nnz+k +j*nelz      +nely*nelz*(i+1)   # face 1

            iconv[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            iconv[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            iconv[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            iconv[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            iconv[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            iconv[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            iconv[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            iconv[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            iconv[8,counter]=nnx*nny*nnz+ nnx*nely*nelz+k +  i*nelz   +j*nelx*nelz      # face 2
            iconv[9,counter]=nnx*nny*nnz+ nnx*nely*nelz+k +  i*nelz   +(j+1)*nelx*nelz  # face 3

            iconw[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            iconw[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            iconw[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            iconw[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            iconw[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            iconw[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            iconw[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            iconw[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            iconw[8,counter]=nnx*nny*nnz+nnx*nely*nelz+nny*nelx*nelz  + j +i*nely +k*nelx*nely      # face 4
            iconw[9,counter]=nnx*nny*nnz+nnx*nely*nelz+nny*nelx*nelz  + j +i*nely +(k+1)*nelx*nely  # face 5

            counter += 1
        #end for
    #end for
#end for

#for iel in [0,1,2,3,4,5]:
    #print ("node 8 |",iconu[8,iel],"at pos.",xV[iconu[8,iel]], yV[iconu[8,iel]], zV[iconu[8,iel]])
    #print ("node 9 |",iconu[9,iel],"at pos.",xV[iconu[9,iel]], yV[iconu[9,iel]], zV[iconu[9,iel]])
    #print ("node 8 |",iconv[8,iel],"at pos.",xV[iconv[8,iel]], yV[iconv[8,iel]], zV[iconv[8,iel]])
    #print ("node 9 |",iconv[9,iel],"at pos.",xV[iconv[9,iel]], yV[iconv[9,iel]], zV[iconv[9,iel]])
    #print ("node 8 |",iconw[8,iel],"at pos.",xV[iconw[8,iel]], yV[iconw[8,iel]], zV[iconw[8,iel]])
    #print ("node 9 |",iconw[9,iel],"at pos.",xV[iconw[9,iel]], yV[iconw[9,iel]], zV[iconw[9,iel]])

#for iel in range(0,nel):
#    print(iconu[:,iel])
#    print(iconv[:,iel])
#    print(iconw[:,iel])

print("connectivity setup: %.3f s" % (time.time() - start))

#################################################################
# compute xc,yc,zc,rho,eta
#################################################################

xc = np.zeros(nel,dtype=np.float64)  # x coordinates
yc = np.zeros(nel,dtype=np.float64)  # y coordinates
zc = np.zeros(nel,dtype=np.float64)  # z coordinates
rho_el = np.zeros(nel,dtype=np.float64)  # z coordinates
eta_el = np.zeros(nel,dtype=np.float64)  # z coordinates

for iel in range(0,nel):
    xc[iel]=0.5*(xV[iconu[0,iel]]+xV[iconu[6,iel]])
    yc[iel]=0.5*(yV[iconu[0,iel]]+yV[iconu[6,iel]])
    zc[iel]=0.5*(zV[iconu[0,iel]]+zV[iconu[6,iel]])
    rho_el[iel]=rho(xc[iel],yc[iel],zc[iel])
    eta_el[iel]=eta(xc[iel],yc[iel],zc[iel])

np.savetxt('gridc.ascii',np.array([xc,yc,zc]).T,header='# x,y,z')

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros((mV*ndofV,nel),dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros((mV*ndofV,nel),dtype=np.float64)  # boundary condition, value

for iel in range(0,nel):

    inode0=iconu[0,iel] 
    inode6=iconu[6,iel]

    if xV[inode0]<eps: #element is on face x=0
       bc_fix[0*ndofV,iel]=True ; bc_val[0*ndofV,iel]= 0.
       bc_fix[3*ndofV,iel]=True ; bc_val[3*ndofV,iel]= 0.
       bc_fix[4*ndofV,iel]=True ; bc_val[4*ndofV,iel]= 0.
       bc_fix[7*ndofV,iel]=True ; bc_val[7*ndofV,iel]= 0.
       bc_fix[     24,iel]=True ; bc_val[     24,iel]= 0.

    if xV[inode6]>Lx-eps: #element is on face x=Lx
       bc_fix[1*ndofV,iel]=True ; bc_val[1*ndofV,iel]= 0.
       bc_fix[2*ndofV,iel]=True ; bc_val[2*ndofV,iel]= 0.
       bc_fix[5*ndofV,iel]=True ; bc_val[5*ndofV,iel]= 0.
       bc_fix[6*ndofV,iel]=True ; bc_val[6*ndofV,iel]= 0.
       bc_fix[     27,iel]=True ; bc_val[     27,iel]= 0.


    if yV[inode0]<eps: #element is on face y=0
       bc_fix[0*ndofV+1,iel]=True ; bc_val[0*ndofV+1,iel]= 0.
       bc_fix[1*ndofV+1,iel]=True ; bc_val[1*ndofV+1,iel]= 0.
       bc_fix[4*ndofV+1,iel]=True ; bc_val[4*ndofV+1,iel]= 0.
       bc_fix[5*ndofV+1,iel]=True ; bc_val[5*ndofV+1,iel]= 0.
       bc_fix[       25,iel]=True ; bc_val[       25,iel]= 0.

    if yV[inode6]>Ly-eps: #element is on face y=Ly
       bc_fix[2*ndofV+1,iel]=True ; bc_val[2*ndofV+1,iel]= 0.
       bc_fix[3*ndofV+1,iel]=True ; bc_val[3*ndofV+1,iel]= 0.
       bc_fix[6*ndofV+1,iel]=True ; bc_val[6*ndofV+1,iel]= 0.
       bc_fix[7*ndofV+1,iel]=True ; bc_val[7*ndofV+1,iel]= 0.
       bc_fix[       28,iel]=True ; bc_val[       28,iel]= 0.


    if zV[inode0]<eps: #element is on face z=0 
       bc_fix[0*ndofV+2,iel]=True ; bc_val[0*ndofV+2,iel]= 0.
       bc_fix[1*ndofV+2,iel]=True ; bc_val[1*ndofV+2,iel]= 0.
       bc_fix[2*ndofV+2,iel]=True ; bc_val[2*ndofV+2,iel]= 0.
       bc_fix[3*ndofV+2,iel]=True ; bc_val[3*ndofV+2,iel]= 0.
       bc_fix[       26,iel]=True ; bc_val[       26,iel]= 0.

    if zV[inode6]>Lz-eps: #element is on face z=Lz 
       bc_fix[4*ndofV+2,iel]=True ; bc_val[4*ndofV+2,iel]= 0.
       bc_fix[5*ndofV+2,iel]=True ; bc_val[5*ndofV+2,iel]= 0.
       bc_fix[6*ndofV+2,iel]=True ; bc_val[6*ndofV+2,iel]= 0.
       bc_fix[7*ndofV+2,iel]=True ; bc_val[7*ndofV+2,iel]= 0.
       bc_fix[       29,iel]=True ; bc_val[       29,iel]= 0.

#end for

print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# compute volume of elements
#################################################################
start = time.time()

volume=np.zeros(nel,dtype=np.float64) 
N     = np.zeros(8,dtype=np.float64)           # z-component velocity
jcbi=np.zeros((3,3),dtype=np.float64)

NNNVu    = np.zeros(mV,dtype=np.float64)           # shape functions u
NNNVv    = np.zeros(mV,dtype=np.float64)           # shape functions v
NNNVw    = np.zeros(mV,dtype=np.float64)           # shape functions w
dNNNVudx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVuds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVwdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                # compute xq,yq,zq 
                N[0]=0.125*(1.-rq)*(1.-sq)*(1.-tq)
                N[1]=0.125*(1.+rq)*(1.-sq)*(1.-tq)
                N[2]=0.125*(1.+rq)*(1.+sq)*(1.-tq)
                N[3]=0.125*(1.-rq)*(1.+sq)*(1.-tq)
                N[4]=0.125*(1.-rq)*(1.-sq)*(1.+tq)
                N[5]=0.125*(1.+rq)*(1.-sq)*(1.+tq)
                N[6]=0.125*(1.+rq)*(1.+sq)*(1.+tq)
                N[7]=0.125*(1.-rq)*(1.+sq)*(1.+tq)
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0,8):
                    xq+=N[k]*xV[iconu[k,iel]]
                    yq+=N[k]*yV[iconu[k,iel]]
                    zq+=N[k]*zV[iconu[k,iel]]
                #end for

                #print(xq,yq,zq)

                NNNVu[0:mV]=NNVu(rq,sq,tq)
                dNNNVudr[0:mV]=dNNVudr(rq,sq,tq)
                dNNNVuds[0:mV]=dNNVuds(rq,sq,tq)
                dNNNVudt[0:mV]=dNNVudt(rq,sq,tq)
                NNNVv[0:10]=NNVv(rq,sq,tq)
                dNNNVvdr[0:10]=dNNVvdr(rq,sq,tq)
                dNNNVvds[0:10]=dNNVvds(rq,sq,tq)
                dNNNVvdt[0:10]=dNNVvdt(rq,sq,tq)
                NNNVw[0:10]=NNVw(rq,sq,tq)
                dNNNVwdr[0:10]=dNNVwdr(rq,sq,tq)
                dNNNVwds[0:10]=dNNVwds(rq,sq,tq)
                dNNNVwdt[0:10]=dNNVwdt(rq,sq,tq)

                #compute jacobian matrix and determinant
                jcob=hx*hy*hz/8
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                # compute dNdx, dNdy, dNdz
                a=0
                b=0
                c=0
                d=0
                e=0
                f=0
                g=0
                h=0
                i=0
                jx=0
                jy=0
                jz=0
                for k in range(0,mV):
                    dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                    dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
                    dNNNVudz[k]=jcbi[2,2]*dNNNVudt[k]
                    dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
                    dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
                    dNNNVvdz[k]=jcbi[2,2]*dNNNVvdt[k]
                    dNNNVwdx[k]=jcbi[0,0]*dNNNVwdr[k]
                    dNNNVwdy[k]=jcbi[1,1]*dNNNVwds[k]
                    dNNNVwdz[k]=jcbi[2,2]*dNNNVwdt[k]
                    a+=dNNNVudx[k]*xV[iconu[k,iel]]
                    b+=dNNNVudy[k]*xV[iconu[k,iel]]
                    c+=dNNNVudz[k]*xV[iconu[k,iel]]
                    d+=dNNNVvdx[k]*yV[iconv[k,iel]]
                    e+=dNNNVvdy[k]*yV[iconv[k,iel]]
                    f+=dNNNVvdz[k]*yV[iconv[k,iel]]
                    g+=dNNNVwdx[k]*zV[iconw[k,iel]]
                    h+=dNNNVwdy[k]*zV[iconw[k,iel]]
                    i+=dNNNVwdz[k]*zV[iconw[k,iel]]
                    jx+=NNNVu[k]
                    jy+=NNNVv[k]
                    jz+=NNNVw[k]
                #end for

                volume[iel]+=jcob*weightq
            #end for
        #end for
    #end for
#end for

print(a,b,c)
print(d,e,f)
print(g,h,i)
print(jx,jy,jz)

print("     -> vol  (m,M) %.6e %.6e " %(np.min(volume),np.max(volume)))
print("     -> total vol meas %.6f " %(volume.sum()))
print("     -> total vol anal %.6f " %(Lx*Ly*Lz))

print("compute elements volumes: %.3f s" % (time.time() - start))
exit
#################################################################
# build FE matrix
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((6,30),dtype=np.float64)    # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
u     = np.zeros(NV,dtype=np.float64)           # x-component velocity
v     = np.zeros(NV,dtype=np.float64)           # y-component velocity
w     = np.zeros(NV,dtype=np.float64)           # z-component velocity
N     = np.zeros(8,dtype=np.float64)           # z-component velocity
N_mat   = np.zeros((6,ndofP*mP),dtype=np.float64) # matrix  
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P

NNNVu    = np.zeros(10,dtype=np.float64)           # shape functions u
NNNVv    = np.zeros(10,dtype=np.float64)           # shape functions v
NNNVw    = np.zeros(10,dtype=np.float64)           # shape functions w
dNNNVudx = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVudy = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVudz = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVudr = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVuds = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVudt = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvdx = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvdy = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvdz = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvdr = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvds = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVvdt = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwdx = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwdy = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwdz = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwdr = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwds = np.zeros(10,dtype=np.float64)           # shape functions derivatives
dNNNVwdt = np.zeros(10,dtype=np.float64)           # shape functions derivatives

c_mat = np.zeros((6,6),dtype=np.float64) 
c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.
jcbi=np.zeros((3,3),dtype=np.float64)

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((30),dtype=np.float64)
    K_el =np.zeros((30,30),dtype=np.float64)
    G_el=np.zeros((30,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)           # int of shape functions P

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                # calculate shape functions
                NNNVu[0:10]=NNVu(rq,sq,tq)
                dNNNVudr[0:10]=dNNVudr(rq,sq,tq)
                dNNNVuds[0:10]=dNNVuds(rq,sq,tq)
                dNNNVudt[0:10]=dNNVudt(rq,sq,tq)

                NNNVv[0:10]=NNVv(rq,sq,tq)
                dNNNVvdr[0:10]=dNNVvdr(rq,sq,tq)
                dNNNVvds[0:10]=dNNVvds(rq,sq,tq)
                dNNNVvdt[0:10]=dNNVvdt(rq,sq,tq)

                NNNVw[0:10]=NNVw(rq,sq,tq)
                dNNNVwdr[0:10]=dNNVwdr(rq,sq,tq)
                dNNNVwds[0:10]=dNNVwds(rq,sq,tq)
                dNNNVwdt[0:10]=dNNVwdt(rq,sq,tq)

                NNNP[0:mP]=NNP(rq,sq,tq)

                #compute jacobian matrix and determinant
                jcob=hx*hy*hz/8
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                # compute xq,yq,zq 
                N[0]=0.125*(1.-rq)*(1.-sq)*(1.-tq)
                N[1]=0.125*(1.+rq)*(1.-sq)*(1.-tq)
                N[2]=0.125*(1.+rq)*(1.+sq)*(1.-tq)
                N[3]=0.125*(1.-rq)*(1.+sq)*(1.-tq)
                N[4]=0.125*(1.-rq)*(1.-sq)*(1.+tq)
                N[5]=0.125*(1.+rq)*(1.-sq)*(1.+tq)
                N[6]=0.125*(1.+rq)*(1.+sq)*(1.+tq)
                N[7]=0.125*(1.-rq)*(1.+sq)*(1.+tq)
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0,8):
                    xq+=N[k]*xV[iconu[k,iel]]
                    yq+=N[k]*yV[iconu[k,iel]]
                    zq+=N[k]*zV[iconu[k,iel]]
                #end for

                #print(xq,yq,zq)

                # compute dNdx, dNdy, dNdz
                for k in range(0,mV):
                    dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                    dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
                    dNNNVudz[k]=jcbi[2,2]*dNNNVudt[k]
                    dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
                    dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
                    dNNNVvdz[k]=jcbi[2,2]*dNNNVvdt[k]
                    dNNNVwdx[k]=jcbi[0,0]*dNNNVwdr[k]
                    dNNNVwdy[k]=jcbi[1,1]*dNNNVwds[k]
                    dNNNVwdz[k]=jcbi[2,2]*dNNNVwdt[k]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:6, 3*i:3*i+3] = [[dNNNVudx[i],0.         ,0.        ],
                                             [0.         ,dNNNVvdy[i],0.        ],
                                             [0.         ,0.         ,dNNNVwdz[i]],
                                             [dNNNVudy[i],dNNNVvdx[i],0.        ],
                                             [dNNNVudz[i],0.         ,dNNNVwdx[i]],
                                             [0.         ,dNNNVvdz[i],dNNNVwdy[i]]]
                #end for

                #K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq,zq)*weightq*jcob
                K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta_el[iel]*weightq*jcob

                for i in range(0,mV):
                    #f_el[ndofV*i+0]=NNNVu[i]*jcob*weightq*rho(xq,yq,zq)*gx
                    #f_el[ndofV*i+1]=NNNVv[i]*jcob*weightq*rho(xq,yq,zq)*gy
                    #f_el[ndofV*i+2]=NNNVw[i]*jcob*weightq*rho(xq,yq,zq)*gz
                    f_el[ndofV*i+0]+=NNNVu[i]*jcob*weightq*rho_el[iel]*gx
                    f_el[ndofV*i+1]+=NNNVv[i]*jcob*weightq*rho_el[iel]*gy
                    f_el[ndofV*i+2]+=NNNVw[i]*jcob*weightq*rho_el[iel]*gz
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=NNNP[i]
                    N_mat[3,i]=0.
                    N_mat[4,i]=0.
                    N_mat[5,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                NNNNP[:]+=NNNP[:]*jcob*weightq

            #end for kq
        #end for jq
    #end for iq

    # impose b.c. 
    for ikk in range(0,mV*ndofV): # loop over lines
        if bc_fix[ikk,iel]: 
           K_ref=K_el[ikk,ikk] 
           for jkk in range(0,mV*ndofV): 
               f_el[jkk]-=K_el[jkk,ikk]*bc_val[ikk,iel]
               K_el[ikk,jkk]=0
               K_el[jkk,ikk]=0
           #end for
           K_el[ikk,ikk]=K_ref
           f_el[ikk]=K_ref*bc_val[ikk,iel]
           h_el[0]-=G_el[ikk,0]*bc_val[ikk,iel]
           G_el[ikk,0]=0
        #end if
    #end for

    # assemble matrix K_mat and right hand side rhs

    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1  # local 
            if k1<8: # Q1 nodes
               m1 =ndofV*iconu[k1,iel]+i1  # which iconu/v/w does nto matter
            else: # bubbles
               if i1==0:
                  m1 =iconu[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               if i1==1:
                  m1 =iconv[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               if i1==2:
                  m1 =iconw[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2 +i2  # local
                    if k2<8: # Q1 nodes
                       m2 =ndofV*iconu[k2,iel]+i2  # which iconu/v/w does nto matter
                    else: # bubbles
                       if i2==0:
                          m2 =iconu[k2,iel]  -nnx*nny*nnz + ndofV*nnx*nny*nnz
                       if i2==1:
                          m2 =iconv[k2,iel]  -nnx*nny*nnz + ndofV*nnx*nny*nnz
                       if i2==2:
                          m2 =iconw[k2,iel]  -nnx*nny*nnz + ndofV*nnx*nny*nnz
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0]
    
#end for iel

#plt.spy(K_mat)
#plt.savefig('K.pdf', bbox_inches='tight')

print("build FE system: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[Nfem,NfemV:Nfem]=1
   a_mat[NfemV:Nfem,Nfem]=1
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
#end if

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

#plt.spy(a_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

######################################################################
# solve system
######################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))
######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

for i in range(0,nnx*nny*nnz):
    u[i]=sol[i*ndofV+0]
    v[i]=sol[i*ndofV+1]
    w[i]=sol[i*ndofV+2]

#u,v,w=np.reshape(sol[0:ndofV*nnx*nny*nnz],(nnx*nny*nnz,3)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %6f %6f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %6f %6f " %(np.min(v),np.max(v)))
print("     -> w (m,M) %6f %6f " %(np.min(w),np.max(w)))
print("     -> p (m,M) %6f %6f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4es" % sol[Nfem])

np.savetxt('velocity.ascii',np.array([xV,yV,zV,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################

#for iel in range(0,nel):
#    for k1 in range(0,mV):
#        if k1<8: # Q1 nodes
#           m1 =ndofV*iconu[k1,iel]  
#           print(xV[iconu[k1,iel]],yV[iconu[k1,iel]],zV[iconu[k1,iel]],sol[m1])
#        else: # bubbles
#           m1 =iconu[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
#           print(xV[iconu[k1,iel]],yV[iconu[k1,iel]],zV[iconu[k1,iel]],sol[m1])

if True:
       filename = 'u_dofs.vtu' 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mV*nel,mV*nel))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconu[k1,iel]  
               else: # bubbles
                  m1 =iconu[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%3e \n" %sol[m1])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconu[k1,iel]  
               else: # bubbles
                  m1 =iconu[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%10e %10e %10e \n" %(xV[iconu[k1,iel]],yV[iconu[k1,iel]],zV[iconu[k1,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % i) 
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % 1) 
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       filename = 'v_dofs.vtu' 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mV*nel,mV*nel))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='v' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconv[k1,iel]  +1
               else: # bubbles
                  m1 =iconv[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%3e \n" %sol[m1])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconv[k1,iel]  +1
               else: # bubbles
                  m1 =iconv[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%10e %10e %10e \n" %(xV[iconv[k1,iel]],yV[iconv[k1,iel]],zV[iconv[k1,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % i) 
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % 1) 
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       filename = 'w_dofs.vtu' 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mV*nel,mV*nel))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='w' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconw[k1,iel]  +2
               else: # bubbles
                  m1 =iconw[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%3e \n" %sol[m1])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<8: # Q1 nodes
                  m1 =ndofV*iconw[k1,iel]  +2
               else: # bubbles
                  m1 =iconw[k1,iel] -nnx*nny*nnz + ndofV*nnx*nny*nnz
               vtufile.write("%10e %10e %10e \n" %(xV[iconw[k1,iel]],yV[iconw[k1,iel]],zV[iconw[k1,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % i) 
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % 1) 
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

















#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny*nnz,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],zV[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % rho_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eta_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='strainrate' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f\n" % sr[iel])
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconu[0,iel],iconu[1,iel],iconu[2,iel],iconu[3,iel],
                                                   iconu[4,iel],iconu[5,iel],iconu[6,iel],iconu[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   #--
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
