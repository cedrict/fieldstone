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
    
aa=8/27
bb=10/21
cc=4/21
dd=64/63
ee=8/63

def NNV(r,s,t):
    b8=(27/32)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1-r)*(1-s)*(1-t)  
    b9=(27/32)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1+r)*(1+s)*(1+t)  

    N_0=0.125*(1-r)*(1-s)*(1-t) -aa*b8  
    N_1=0.125*(1+r)*(1-s)*(1-t) -aa*bb*b8-aa*cc*b9
    N_2=0.125*(1+r)*(1+s)*(1-t) -aa*cc*b8-aa*bb*b9
    N_3=0.125*(1-r)*(1+s)*(1-t) -aa*bb*b8-aa*cc*b9
    N_4=0.125*(1-r)*(1-s)*(1+t) -aa*bb*b8-aa*cc*b9
    N_5=0.125*(1+r)*(1-s)*(1+t) -aa*cc*b8-aa*bb*b9
    N_6=0.125*(1+r)*(1+s)*(1+t) -aa*b9
    N_7=0.125*(1-r)*(1+s)*(1+t) -aa*cc*b8-aa*bb*b9
    N_8= dd*b8-ee*b9
    N_9=-ee*b8+dd*b9
 
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9

def dNNVdr(r,s,t):
    db8dr=(27/32)**3*(1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2) 
    db9dr=(27/32)**3*(1-s**2)*(1-t**2)*(1+s)*(1+t)*( 1-2*r-3*r**2)

    dNdr_0=-0.125*(1-s)*(1-t) -aa*db8dr  
    dNdr_1=+0.125*(1-s)*(1-t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr_2=+0.125*(1+s)*(1-t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr_3=-0.125*(1+s)*(1-t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr_4=-0.125*(1-s)*(1+t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr_5=+0.125*(1-s)*(1+t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr_6=+0.125*(1+s)*(1+t) -aa*db9dr
    dNdr_7=-0.125*(1+s)*(1+t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr_8= dd*db8dr-ee*db9dr
    dNdr_9=-ee*db8dr+dd*db9dr
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8,dNdr_9

def dNNVds(r,s,t):
    db8ds=(27/32)**3*(1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2) 
    db9ds=(27/32)**3*(1-r**2)*(1-t**2)*(1+r)*(1+t)*( 1-2*s-3*s**2)

    dNds_0=-0.125*(1-r)*(1-t) -aa*db8ds   
    dNds_1=-0.125*(1+r)*(1-t) -aa*bb*db8ds-aa*cc*db9ds 
    dNds_2=+0.125*(1+r)*(1-t) -aa*cc*db8ds-aa*bb*db9ds 
    dNds_3=+0.125*(1-r)*(1-t) -aa*bb*db8ds-aa*cc*db9ds
    dNds_4=-0.125*(1-r)*(1+t) -aa*bb*db8ds-aa*cc*db9ds
    dNds_5=-0.125*(1+r)*(1+t) -aa*cc*db8ds-aa*bb*db9ds
    dNds_6=+0.125*(1+r)*(1+t) -aa*db9ds
    dNds_7=+0.125*(1-r)*(1+t) -aa*cc*db8ds-aa*bb*db9ds
    dNds_8= dd*db8ds-ee*db9ds
    dNds_9=-ee*db8ds+dd*db9ds
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8,dNds_9

def dNNVdt(r,s,t):
    db8dt=(27/32)**3*(1-r**2)*(1-s**2)*(1-r)*(1-s)*(-1-2*t+3*t**2) 
    db9dt=(27/32)**3*(1-r**2)*(1-s**2)*(1+r)*(1+s)*( 1-2*t-3*t**2)

    dNdt_0=-0.125*(1-r)*(1-s) -aa*db8dt   
    dNdt_1=-0.125*(1+r)*(1-s) -aa*bb*db8dt-aa*cc*db9dt 
    dNdt_2=-0.125*(1+r)*(1+s) -aa*cc*db8dt-aa*bb*db9dt 
    dNdt_3=-0.125*(1-r)*(1+s) -aa*bb*db8dt-aa*cc*db9dt
    dNdt_4=+0.125*(1-r)*(1-s) -aa*bb*db8dt-aa*cc*db9dt
    dNdt_5=+0.125*(1+r)*(1-s) -aa*cc*db8dt-aa*bb*db9dt
    dNdt_6=+0.125*(1+r)*(1+s) -aa*db9dt
    dNdt_7=+0.125*(1-r)*(1+s) -aa*cc*db8dt-aa*bb*db9dt
    dNdt_8= dd*db8dt-ee*db9dt
    dNdt_9=-ee*db8dt+dd*db9dt
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8,dNdt_9

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

nqperdim=2

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

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0,iel]],yV[iconV[0,iel]],zV[iconV[0,iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1,iel]],yV[iconV[1,iel]],zV[iconV[1,iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2,iel]],yV[iconV[2,iel]],zV[iconV[2,iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3,iel]],yV[iconV[3,iel]],zV[iconV[3,iel]])
#    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4,iel]],yV[iconV[4,iel]],zV[iconV[4,iel]])
#    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5,iel]],yV[iconV[5,iel]],zV[iconV[5,iel]])
#    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6,iel]],yV[iconV[6,iel]],zV[iconV[6,iel]])
#    print ("node 7",iconV[7,iel],"at pos.",xV[iconV[7,iel]],yV[iconV[7,iel]],zV[iconV[7,iel]])
#    print ("node 8",iconV[8,iel],"at pos.",xV[iconV[8,iel]],yV[iconV[8,iel]],zV[iconV[8,iel]])
#    print ("node 9",iconV[9,iel],"at pos.",xV[iconV[9,iel]],yV[iconV[9,iel]],zV[iconV[9,iel]])

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

N     = np.zeros(8,dtype=np.float64)           # z-component velocity
NNNV  = np.zeros(mV,dtype=np.float64)           # shape functions u
field = np.zeros(NV,dtype=np.float64)
jcbi=np.zeros((3,3),dtype=np.float64)

dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

field[:]=zV[:]

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
                    xq+=N[k]*xV[iconV[k,iel]]
                    yq+=N[k]*yV[iconV[k,iel]]
                    zq+=N[k]*zV[iconV[k,iel]]
                #end for

                NNNV[0:mV]=NNV(rq,sq,tq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
                dNNNVds[0:mV]=dNNVds(rq,sq,tq)
                dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)

                jcob=hx*hy*hz/8
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                    dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                    dNNNVdz[k]=jcbi[2,2]*dNNNVdt[k]

                dfdxq=0
                dfdyq=0
                dfdzq=0
                for k in range(0,mV):
                    dfdxq+=dNNNVdx[k]*field[iconV[k,iel]]
                    dfdyq+=dNNNVdy[k]*field[iconV[k,iel]]
                    dfdzq+=dNNNVdz[k]*field[iconV[k,iel]]

                fq=0.0
                for k in range(0,mV):
                    fq+=NNNV[k]*field[iconV[k,iel]]

                print(xq,yq,zq,fq,dfdxq,dfdyq,dfdzq)

            #end kq
        #end jq
    #end iq
#end iel




