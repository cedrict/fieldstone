import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.linalg import null_space

#------------------------------------------------------------------------------

def NNV(rq,sq,tq):
    N_0=0.125*(1-rq)*(1-sq)*(1-tq) - 
    N_1=0.125*(1+rq)*(1-sq)*(1-tq) -
    N_2=0.125*(1+rq)*(1+sq)*(1-tq) -
    N_3=0.125*(1-rq)*(1+sq)*(1-tq) -
    N_4=0.125*(1-rq)*(1-sq)*(1+tq) -
    N_5=0.125*(1+rq)*(1-sq)*(1+tq) -
    N_6=0.125*(1+rq)*(1+sq)*(1+tq) -
    N_7=0.125*(1-rq)*(1+sq)*(1+tq) -
    N_8=(27/32)**3*(1-rq**2)*(1-sq**2)*(1-tq**2)*(1-rq)*(1-sq)*(1-tq)  
    N_9=(27/32)**3*(1-rq**2)*(1-sq**2)*(1-tq**2)*(1+rq)*(1+sq)*(1+tq)  
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9





#------------------------------------------------------------------------------

def NNP(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) 
    N_1=0.125*(1+r)*(1-s)*(1-t) 
    N_2=0.125*(1+r)*(1+s)*(1-t) 
    N_3=0.125*(1-r)*(1+s)*(1-t) 
    N_4=0.125*(1-r)*(1-s)*(1+t) 
    N_5=0.125*(1+r)*(1-s)*(1+t) 
    N_6=0.125*(1+r)*(1+s)*(1+t) 
    N_7=0.125*(1-r)*(1+s)*(1+t) 
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 82--------")
print("-----------------------------")

mV=8+2   # number of V nodes making up an element
mP=8     # number of P nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx =2  # do not exceed 20 
   nely =nelx
   nelz =nelx
#end if

gx=0
gy=0
gz=-1

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

visu=1

pnormalise=True
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

nel=nelx*nely*nelz  # number of elements, total

NV=nnx*nny*nnz+2*nel 
NP=nnx*nny*nnz

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

sqrt3=np.sqrt(3.)

#################################################################
#################################################################

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
print("Nfem=",Nfem)
print("------------------------------")

######################################################################
# grid point setup
######################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates
zV = np.empty(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            xV[counter]=i*Lx/float(nelx)
            yV[counter]=j*Ly/float(nely)
            zV[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (timing.time() - start))

######################################################################
# connectivity
######################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconV[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            iconV[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            iconV[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            iconV[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            iconV[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            iconV[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            iconV[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            iconV[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            iconV[8,counter]=(nelx+1)*(nely+1)*(nelz+1)+2*counter+0
            iconV[9,counter]=(nelx+1)*(nely+1)*(nelz+1)+2*counter+1
            counter += 1
        #end for
    #end for
#end for

print("build connectivity: %.3f s" % (timing.time() - start))

######################################################################
# add bubble nodes
######################################################################

counter=0
for iel in range(0,nel):
    xV[nnx*nny*nnz+counter]=xV[iconV[0,iel]]+hx/3
    yV[nnx*nny*nnz+counter]=yV[iconV[0,iel]]+hy/3
    zV[nnx*nny*nnz+counter]=zV[iconV[0,iel]]+hz/3
    counter+=1
    xV[nnx*nny*nnz+counter]=xV[iconV[0,iel]]+2*hx/3
    yV[nnx*nny*nnz+counter]=yV[iconV[0,iel]]+2*hy/3
    zV[nnx*nny*nnz+counter]=zV[iconV[0,iel]]+2*hz/3
    counter+=1

np.savetxt('gridV.ascii',np.array([xV,yV,zV]).T,header='# x,y,z,u,v,w')

for iel in range (0,nel):
    print ("iel=",iel)
    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0,iel]],yV[iconV[0,iel]],zV[iconV[0,iel]])
    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1,iel]],yV[iconV[1,iel]],zV[iconV[1,iel]])
    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2,iel]],yV[iconV[2,iel]],zV[iconV[2,iel]])
    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3,iel]],yV[iconV[3,iel]],zV[iconV[3,iel]])
    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4,iel]],yV[iconV[4,iel]],zV[iconV[4,iel]])
    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5,iel]],yV[iconV[5,iel]],zV[iconV[5,iel]])
    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6,iel]],yV[iconV[6,iel]],zV[iconV[6,iel]])
    print ("node 7",iconV[7,iel],"at pos.",xV[iconV[7,iel]],yV[iconV[7,iel]],zV[iconV[7,iel]])
    print ("node 8",iconV[8,iel],"at pos.",xV[iconV[8,iel]],yV[iconV[8,iel]],zV[iconV[8,iel]])
    print ("node 9",iconV[9,iel],"at pos.",xV[iconV[9,iel]],yV[iconV[9,iel]],zV[iconV[9,iel]])

#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
zP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

xP[0:NP]=xV[0:NP]
yP[0:NP]=yV[0:NP]
zP[0:NP]=zV[0:NP]

iconP[0:mP,0:nel]=iconV[0:mP,0:nel]

np.savetxt('gridP.ascii',np.array([xP,yP,zP]).T,header='# x,y,z')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# compute volume of elements
#################################################################






