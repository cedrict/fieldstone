import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

#------------------------------------------------------------------------------

def B(r,s,t):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1-r)*(1-s)*(1-t)
    if bubble==2:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1+beta*(r+s+t))

def dBdr(r,s,t):
    if bubble==1:
       return (1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2)
    if bubble==2:
       return (1-s**2)*(1-t**2)*(-beta*(3*r**2+2*r*(s+t)-1)+2*r) 

def dBds(r,s,t):
    if bubble==1:
       return (1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2) 
    if bubble==2:
       return (1-r**2)*(1-t**2)*(-beta*(3*s**2+2*s*(r+t)-1)+2*s) 

def dBdt(r,s,t):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)*(-1-2*t+3*t**2) 
    if bubble==2:
       return (1-r**2)*(1-s**2)*(-beta*(3*t**2+2*t*(r+s)-1)+2*t) 

#------------------------------------------------------------------------------

def rho(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
    #if abs(x-.5)<2*hx and abs(y-0.5)<2*hy and abs(z-0.5)<2*hz:
    #if abs(x-.5)<0.25 and abs(y-0.5)<0.25 and abs(z-0.5)<0.25:
       val=2.
    else:
       val=1.
    return val

def eta(x,y,z):
    if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123**2:
    #if abs(x-.5)<2*hx and abs(y-0.5)<2*hy and abs(z-0.5)<2*hz:
    #if abs(x-.5)<0.25 and abs(y-0.5)<0.25 and abs(z-0.5)<0.25:
       val=10.
    else:
       val=1.
    return val

#------------------------------------------------------------------------------

def NNV(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) -0.125*B(r,s,t)
    N_1=0.125*(1+r)*(1-s)*(1-t) -0.125*B(r,s,t)
    N_2=0.125*(1+r)*(1+s)*(1-t) -0.125*B(r,s,t)
    N_3=0.125*(1-r)*(1+s)*(1-t) -0.125*B(r,s,t)
    N_4=0.125*(1-r)*(1-s)*(1+t) -0.125*B(r,s,t)
    N_5=0.125*(1+r)*(1-s)*(1+t) -0.125*B(r,s,t)
    N_6=0.125*(1+r)*(1+s)*(1+t) -0.125*B(r,s,t)
    N_7=0.125*(1-r)*(1+s)*(1+t) -0.125*B(r,s,t)
    N_8= B(r,s,t)
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8

def dNNVdr(r,s,t):
    dNdr_0=-0.125*(1-s)*(1-t) -0.125*dBdr(r,s,t) 
    dNdr_1=+0.125*(1-s)*(1-t) -0.125*dBdr(r,s,t) 
    dNdr_2=+0.125*(1+s)*(1-t) -0.125*dBdr(r,s,t) 
    dNdr_3=-0.125*(1+s)*(1-t) -0.125*dBdr(r,s,t) 
    dNdr_4=-0.125*(1-s)*(1+t) -0.125*dBdr(r,s,t) 
    dNdr_5=+0.125*(1-s)*(1+t) -0.125*dBdr(r,s,t) 
    dNdr_6=+0.125*(1+s)*(1+t) -0.125*dBdr(r,s,t) 
    dNdr_7=-0.125*(1+s)*(1+t) -0.125*dBdr(r,s,t) 
    dNdr_8= dBdr(r,s,t)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8

def dNNVds(r,s,t):
    dNds_0=-0.125*(1-r)*(1-t) -0.125*dBds(r,s,t) 
    dNds_1=-0.125*(1+r)*(1-t) -0.125*dBds(r,s,t) 
    dNds_2=+0.125*(1+r)*(1-t) -0.125*dBds(r,s,t) 
    dNds_3=+0.125*(1-r)*(1-t) -0.125*dBds(r,s,t) 
    dNds_4=-0.125*(1-r)*(1+t) -0.125*dBds(r,s,t) 
    dNds_5=-0.125*(1+r)*(1+t) -0.125*dBds(r,s,t) 
    dNds_6=+0.125*(1+r)*(1+t) -0.125*dBds(r,s,t) 
    dNds_7=+0.125*(1-r)*(1+t) -0.125*dBds(r,s,t) 
    dNds_8= dBds(r,s,t)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8

def dNNVdt(r,s,t):
    dNdt_0=-0.125*(1-r)*(1-s) -0.125*dBdt(r,s,t) 
    dNdt_1=-0.125*(1+r)*(1-s) -0.125*dBdt(r,s,t) 
    dNdt_2=-0.125*(1+r)*(1+s) -0.125*dBdt(r,s,t) 
    dNdt_3=-0.125*(1-r)*(1+s) -0.125*dBdt(r,s,t) 
    dNdt_4=+0.125*(1-r)*(1-s) -0.125*dBdt(r,s,t) 
    dNdt_5=+0.125*(1+r)*(1-s) -0.125*dBdt(r,s,t) 
    dNdt_6=+0.125*(1+r)*(1+s) -0.125*dBdt(r,s,t) 
    dNdt_7=+0.125*(1-r)*(1+s) -0.125*dBdt(r,s,t) 
    dNdt_8= dBdt(r,s,t)
    return dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8

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
print("--------fieldstone 75--------")
print("-----------------------------")

mV=9     # number of V nodes making up an element
mP=8     # number of P nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   nqperdim=int(sys.argv[4])
   bubble=int(sys.argv[5])
else:
   nelx =8   # do not exceed 20 
   nely =nelx
   nelz =nelx
   nqperdim=4
   bubble=1
#end if

gx=0
gy=0
gz=-1

visu=1

pnormalise=True
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

nel=nelx*nely*nelz  # number of elements, total

NV=(nelx+1)*(nely+1)*(nelz+1)+nel
NP=(nelx+1)*(nely+1)*(nelz+1)

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

eps=1.e-6

sqrt3=np.sqrt(3.)

beta=0.25

sparse=True

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


rVnodes=[-1,+1,+1,-1,-1,+1,+1,-1,0]
sVnodes=[-1,-1,+1,+1,-1,-1,+1,+1,0]
tVnodes=[-1,-1,-1,-1,+1,+1,+1,+1,0]


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
print("------------------------------")

######################################################################
# grid point setup
######################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates
zV=np.empty(NV,dtype=np.float64)  # z coordinates

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

iconV=np.zeros((mV,nel),dtype=np.int32)
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
            iconV[8,counter]=(nelx+1)*(nely+1)*(nelz+1)+counter
            counter += 1
        #end for
    #end for
#end for

print("build connectivity: %.3f s" % (timing.time() - start))

#################################################################
# bubble node position 
#################################################################

for iel in range(0,nel):
    xV[iconV[8,iel]]=0.125*xV[iconV[0,iel]]+0.125*xV[iconV[1,iel]]\
                    +0.125*xV[iconV[2,iel]]+0.125*xV[iconV[3,iel]]\
                    +0.125*xV[iconV[4,iel]]+0.125*xV[iconV[5,iel]]\
                    +0.125*xV[iconV[6,iel]]+0.125*xV[iconV[7,iel]]
    yV[iconV[8,iel]]=0.125*yV[iconV[0,iel]]+0.125*yV[iconV[1,iel]]\
                    +0.125*yV[iconV[2,iel]]+0.125*yV[iconV[3,iel]]\
                    +0.125*yV[iconV[4,iel]]+0.125*yV[iconV[5,iel]]\
                    +0.125*yV[iconV[6,iel]]+0.125*yV[iconV[7,iel]]
    zV[iconV[8,iel]]=0.125*zV[iconV[0,iel]]+0.125*zV[iconV[1,iel]]\
                    +0.125*zV[iconV[2,iel]]+0.125*zV[iconV[3,iel]]\
                    +0.125*zV[iconV[4,iel]]+0.125*zV[iconV[5,iel]]\
                    +0.125*zV[iconV[6,iel]]+0.125*zV[iconV[7,iel]]

#np.savetxt('gridV.ascii',np.array([xV,yV,zV]).T,header='# x,y,z')

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

#np.savetxt('gridP.ascii',np.array([xP,yP,zP]).T,header='# x,y,z')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# compute volume of elements
#################################################################
start = timing.time()

volume=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                NNNV[0:mV]=NNV(rq,sq,tq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
                dNNNVds[0:mV]=dNNVds(rq,sq,tq)
                dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[0,2]+=dNNNVdr[k]*zV[iconV[k,iel]]
                    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                    jcb[1,2]+=dNNNVds[k]*zV[iconV[k,iel]]
                    jcb[2,0]+=dNNNVdt[k]*xV[iconV[k,iel]]
                    jcb[2,1]+=dNNNVdt[k]*yV[iconV[k,iel]]
                    jcb[2,2]+=dNNNVdt[k]*zV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                volume[iel]+=jcob*weightq
            #end for
        #end for
    #end for
#end for

print("     -> vol  (m,M) %.6e %.6e " %(np.min(volume),np.max(volume)))
print("     -> total vol meas %.6f " %(volume.sum()))
print("     -> total vol anal %.6f " %(Lx*Ly*Lz))

print("compute elements volumes: %.3f s" % (timing.time() - start))

######################################################################
# define boundary conditions
######################################################################
start = timing.time()

bc_fix=np.zeros(Nfem,dtype=np.bool)    # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=0
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=0
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0
    if zV[i]/Lz<eps:
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=0
    if zV[i]/Lz>(1-eps):
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=0 
#end for

print("define b.c.: %.3f s" % (timing.time() - start))

######################################################################
# build FE matrix
######################################################################
start = timing.time()

if sparse:
   if pnormalise:
      A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   else:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

constr  = np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((6,ndofV*mV),dtype=np.float64)   # gradient matrix B 
u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
w     = np.zeros(NV,dtype=np.float64)          # y-component velocity
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat = np.zeros((6,6),dtype=np.float64) 
N_mat   = np.zeros((6,ndofP*mP),dtype=np.float64) # matrix  

NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)           # int of shape functions P

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]

                # calculate shape functions
                NNNV[0:mV]=NNV(rq,sq,tq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
                dNNNVds[0:mV]=dNNVds(rq,sq,tq)
                dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)
                NNNP[0:mP]=NNP(rq,sq,tq)

                # calculate jacobian matrix
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[0,2]+=dNNNVdr[k]*zV[iconV[k,iel]]
                    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                    jcb[1,2]+=dNNNVds[k]*zV[iconV[k,iel]]
                    jcb[2,0]+=dNNNVdt[k]*xV[iconV[k,iel]]
                    jcb[2,1]+=dNNNVdt[k]*yV[iconV[k,iel]]
                    jcb[2,2]+=dNNNVdt[k]*zV[iconV[k,iel]]
                #end for
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
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:6, 3*i:3*i+3] = [[dNNNVdx[i],0.        ,0.        ],
                                             [0.        ,dNNNVdy[i],0.        ],
                                             [0.        ,0.        ,dNNNVdz[i]],
                                             [dNNNVdy[i],dNNNVdx[i],0.        ],
                                             [dNNNVdz[i],0.        ,dNNNVdx[i]],
                                             [0.        ,dNNNVdz[i],dNNNVdy[i]]]
                #end for

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq,zq)*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*rho(xq,yq,zq)*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho(xq,yq,zq)*gy
                    f_el[ndofV*i+2]+=NNNV[i]*jcob*weightq*rho(xq,yq,zq)*gz
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
            #end if
        #end for
    #end for

    #G_el*=eta_ref/Ly
    #h_el*=eta_ref/Ly

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
    if sparse and pnormalise:
       A_sparse[Nfem,NfemV+m2]=constr[m2]
       A_sparse[NfemV+m2,Nfem]=constr[m2]

#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))

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
else:
   if pnormalise:
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   else:
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
#else:

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

#plt.spy(a_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

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

u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
p=sol[NfemV:Nfem]

print("     -> uu (m,M) %.7f %.7f %.7f" %(np.min(u),np.max(u),hx))
print("     -> vv (m,M) %.7f %.7f %.7f" %(np.min(v),np.max(v),hx))
print("     -> ww (m,M) %.7f %.7f %.7f" %(np.min(w),np.max(w),hx))
print("     -> pp (m,M) %.7f %.7f %.7f" %(np.min(p),np.max(p),hx))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

#np.savetxt('velocity.ascii',np.array([xV,yV,zV,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (timing.time() - start))

#####################################################################
# compute density and viscosity at center of element 
#####################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
dens = np.zeros(nel,dtype=np.float64)  
for iel in range(0,nel):
    xc[iel]=xV[iconV[8,iel]]
    yc[iel]=yV[iconV[8,iel]]
    zc[iel]=zV[iconV[8,iel]]
    visc[iel]=eta(xc[iel],yc[iel],zc[iel])
    dens[iel]=rho(xc[iel],yc[iel],zc[iel])

print("visc (m,M) %.5f %.5f " %(np.min(visc),np.max(visc)))
print("dens (m,M) %.5f %.5f " %(np.min(dens),np.max(dens)))

print("compute eta,rho in elt center: %.3f s" % (timing.time() - start))

#####################################################################
# compute strainrate 
#####################################################################
start = timing.time()

count = np.zeros(NV,dtype=np.int16)  
exx = np.zeros(NV,dtype=np.float64)  
eyy = np.zeros(NV,dtype=np.float64)  
ezz = np.zeros(NV,dtype=np.float64)  
exy = np.zeros(NV,dtype=np.float64)  
exz = np.zeros(NV,dtype=np.float64)  
eyz = np.zeros(NV,dtype=np.float64)  
sr = np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,mV):
        inode=iconV[i,iel]
        rq=rVnodes[i]
        sq=sVnodes[i]
        tq=tVnodes[i]

        NNNV[0:mV]=NNV(rq,sq,tq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
        dNNNVds[0:mV]=dNNVds(rq,sq,tq)
        dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)

        # calculate jacobian matrix
        jcb=np.zeros((3,3),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[0,2]+=dNNNVdr[k]*zV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            jcb[1,2]+=dNNNVds[k]*zV[iconV[k,iel]]
            jcb[2,0]+=dNNNVdt[k]*xV[iconV[k,iel]]
            jcb[2,1]+=dNNNVdt[k]*yV[iconV[k,iel]]
            jcb[2,2]+=dNNNVdt[k]*zV[iconV[k,iel]]
        #end for
        jcbi=np.linalg.inv(jcb)

        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]+jcbi[0,2]*dNNNVdt[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]+jcbi[1,2]*dNNNVdt[k]
            dNNNVdz[k]=jcbi[2,0]*dNNNVdr[k]+jcbi[2,1]*dNNNVds[k]+jcbi[2,2]*dNNNVdt[k]
        #end for

        for k in range(0,mV):
            exx[inode]+=dNNNVdx[k]*u[iconV[k,iel]]
            eyy[inode]+=dNNNVdy[k]*v[iconV[k,iel]]
            ezz[inode]+=dNNNVdz[k]*w[iconV[k,iel]]
            exy[inode]+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+0.5*dNNNVdx[k]*v[iconV[k,iel]]
            exz[inode]+=0.5*dNNNVdz[k]*u[iconV[k,iel]]+0.5*dNNNVdx[k]*w[iconV[k,iel]]
            eyz[inode]+=0.5*dNNNVdz[k]*v[iconV[k,iel]]+0.5*dNNNVdy[k]*w[iconV[k,iel]]
        #end for

        count[inode]+=1
    #end for
#end for iel
exx/=count
eyy/=count
ezz/=count
exy/=count
exz/=count
eyz/=count
        
#sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
#               +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

print("exx (m,M) %.5f %.5f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.5f %.5f " %(np.min(eyy),np.max(eyy)))
print("ezz (m,M) %.5f %.5f " %(np.min(ezz),np.max(ezz)))
print("exy (m,M) %.5f %.5f " %(np.min(exy),np.max(exy)))
print("exz (m,M) %.5f %.5f " %(np.min(exz),np.max(exz)))
print("eyz (m,M) %.5f %.5f " %(np.min(eyz),np.max(eyz)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute strainrate: %.3f s" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = timing.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny*nnz,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
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
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % visc[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % dens[iel])
   vtufile.write("</DataArray>\n")

   #--
   #vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f\n" % sr[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%f %f %f %f %f %f\n" % (exx[iel], eyy[iel], ezz[iel], exy[iel], eyz[iel], exz[iel]))
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%.10f %.10f %.10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % p[i])
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % exx[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % eyy[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % ezz[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % exy[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exz' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % exz[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyz' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % eyz[i])
   vtufile.write("</DataArray>\n")











   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],
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
       vtufile.write("%d \n" %12)
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

