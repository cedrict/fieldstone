import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.linalg import null_space
from scipy.sparse.csgraph import reverse_cuthill_mckee
import schur_complement_cg_solver as cg
from scipy.sparse.linalg import spsolve

###############################################################################

def bx(x,y,z,beta):
    if bench==1:
       mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
       mux=-beta*(1-2*x)*mu
       muy=-beta*(1-2*y)*mu
       muz=-beta*(1-2*z)*mu
       val=-(y*z+3*x**2*y**3*z) + mu * (2+6*x*y) \
           +(2+4*x+2*y+6*x**2*y) * mux \
           +(x+x**3+y+2*x*y**2 ) * muy \
           +(-3*z-10*x*y*z     ) * muz
    if bench==2:
       val=0
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=4*(2*y-1)*(2*z-1)  *(-1)
    if bench==-1:
       val=0
    return val

def by(x,y,z,beta):
    if bench==1:
       mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
       mux=-beta*(1-2*x)*mu
       muy=-beta*(1-2*y)*mu
       muz=-beta*(1-2*z)*mu
       val=-(x*z+3*x**3*y**2*z) + mu * (2 +2*x**2 + 2*y**2) \
          +(x+x**3+y+2*x*y**2   ) * mux \
          +(2+2*x+4*y+4*x**2*y  ) * muy \
          +(-3*z-5*x**2*z       ) * muz 
    if bench==2:
       val=0
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=4*(2*x-1)*(2*z-1) *(-1)
    if bench==-1:
       val=0
    return val

def bz(x,y,z,beta):
    if bench==1:
       mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
       mux=-beta*(1-2*x)*mu
       muy=-beta*(1-2*y)*mu
       muz=-beta*(1-2*z)*mu
       val=-(x*y+x**3*y**3) + mu * (-10*y*z) \
          +(-3*z-10*x*y*z        ) * mux \
          +(-3*z-5*x**2*z        ) * muy \
          +(-4-6*x-6*y-10*x**2*y ) * muz 
    if bench==2:
       val=0
    if bench==3:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1.01
       else:
          val=1.
    if bench==4:
       if abs(x-0.5)<0.125 and abs(y-0.5)<0.125 and abs(z-0.5)<0.125:
          val=0.01
       else:
          val=0.
    if bench==5:
       val=-2*(2*x-1)*(2*y-1) *(-1)
    if bench==-1:
       val=0
    return val

###############################################################################

def eta(x,y,z,beta):
    if bench==1:
       val=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    if bench==2:
       val=1
    if bench==3:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1000
       else:
          val=1.
    if bench==4:
       if abs(x-0.5)<0.125 and abs(y-0.5)<0.125 and abs(z-0.5)<0.125:
          val=1000
       else:
          val=1.
    if bench==5:
       val=1
    if bench==-1:
       val=0
    return val

###############################################################################

def uth(x,y,z):
    if bench==1:
       val=x+x*x+x*y+x*x*x*y
    if bench==2:
       val=0.5-z
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=x*(1-x)*(1-2*y)*(1-2*z)
    if bench==-1:
       val=0
    return val

def vth(x,y,z):
    if bench==1:
       val=y+x*y+y*y+x*x*y*y
    if bench==2:
       val=0
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=(1-2*x)*y*(1-y)*(1-2*z)
    if bench==-1:
       val=0
    return val

def wth(x,y,z):
    if bench==1:
       val=-2*z-3*x*z-3*y*z-5*x*x*y*z
    if bench==2:
       val=0
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=-2*(1-2*x)*(1-2*y)*z*(1-z)
    if bench==-1:
       val=0
    return val

def pth(x,y,z):
    if bench==1:
       val=x*y*z+x*x*x*y*y*y*z-5./32.
    if bench==2:
       val=0
    if bench==3:
       val=0
    if bench==4:
       val=0
    if bench==5:
       val=(2*x-1)*(2*y-1)*(2*z-1)
    if bench==-1:
       val=0
    return val

###############################################################################
    
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
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8,N_9],dtype=np.float64)

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
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8,dNdr_9],dtype=np.float64)

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
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8,dNds_9],dtype=np.float64)

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
    return np.array([dNdt_0,dNdt_1,dNdt_2,dNdt_3,dNdt_4,dNdt_5,dNdt_6,dNdt_7,dNdt_8,dNdt_9],dtype=np.float64)

###############################################################################

def NNP(r,s,t):
    N_0=0.125*(1-r)*(1-s)*(1-t) 
    N_1=0.125*(1+r)*(1-s)*(1-t) 
    N_2=0.125*(1+r)*(1+s)*(1-t) 
    N_3=0.125*(1-r)*(1+s)*(1-t) 
    N_4=0.125*(1-r)*(1-s)*(1+t) 
    N_5=0.125*(1+r)*(1-s)*(1+t) 
    N_6=0.125*(1+r)*(1+s)*(1+t) 
    N_7=0.125*(1-r)*(1+s)*(1+t) 
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7],dtype=np.float64)

###############################################################################

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

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   nqperdim=int(sys.argv[4])
   visu = int(sys.argv[5])
else:
   nelx = 8  # do not exceed 20 
   nely =nelx
   nelz =nelx
   nqperdim=2
   visu=1
#end if

bench=5

beeta=0 # beta parameter for mms

debug=False

# True: build fully assembled Stokes matrix
# False: use schur complement solver
globall=False

apply_RCM=False      # Reverse Cuthill-McKee algo
matrix_snapshot=False

if not globall:
   niter=200
   tol=1e-6
   use_precond=False

if bench==-1:
   nelx=2
   nely=2
   nelz=2
   globall=False

nnx=nelx+1           # number of elements, x direction
nny=nely+1           # number of elements, y direction
nnz=nelz+1           # number of elements, z direction
nel=nelx*nely*nelz   # number of elements, total
NV=nnx*nny*nnz+2*nel # total number of V nodes 
NP=nnx*nny*nnz       # total number of P nodes
NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total number of dofs
hx=Lx/nelx           # mesh size
hy=Ly/nely           # mesh size
hz=Lz/nelz           # mesh size

###############################################################################
# quadrature points and weights

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

stats_vel_file=open('stats_vel.ascii',"w")
stats_p_file=open('stats_p.ascii',"w")

###############################################################################

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

###############################################################################
# grid point setup
###############################################################################
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

###############################################################################
# connectivity
###############################################################################
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

###############################################################################
# add bubble nodes along 0-6 diagonal
###############################################################################
start = timing.time()

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

if debug: np.savetxt('gridV.ascii',np.array([xV,yV,zV]).T,header='# x,y,z,u,v,w')

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

print("compute bubble nodes coords: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure grid and iconP 
###############################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
zP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

xP[0:NP]=xV[0:NP]
yP[0:NP]=yV[0:NP]
zP[0:NP]=zV[0:NP]

iconP[0:mP,0:nel]=iconV[0:mP,0:nel]

if debug: np.savetxt('gridP.ascii',np.array([xP,yP,zP]).T,header='# x,y,z')

print("build P grid: %.3f s" % (timing.time() - start))

###############################################################################
# compute volume of elements
# test shape functions and derivatives
###############################################################################
start = timing.time()

N       = np.zeros(8,dtype=np.float64)      # shape functions Q1
NNNV    = np.zeros(mV,dtype=np.float64)     # shape functions Q1+bubbles
NNNP    = np.zeros(mP,dtype=np.float64)     # shape functions Q
jcbi    = np.zeros((3,3),dtype=np.float64)  # inverse of jacobian matrix
dNNNVdx = np.zeros(mV,dtype=np.float64)     # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)     # shape functions derivatives
dNNNVdz = np.zeros(mV,dtype=np.float64)     # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)     # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)     # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)     # shape functions derivatives

if debug:
   field   = np.zeros(NV,dtype=np.float64)
   field[:]=zV[:]**2

   for iel in range(0,nel):
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

                   jcob=hx*hy*hz/8
                   jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                   jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                   jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                   xq=0.0
                   yq=0.0
                   zq=0.0
                   for k in range(0,mV):
                       xq+=NNNV[k]*xV[iconV[k,iel]]
                       yq+=NNNV[k]*yV[iconV[k,iel]]
                       zq+=NNNV[k]*zV[iconV[k,iel]]
                       dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                       dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                       dNNNVdz[k]=jcbi[2,2]*dNNNVdt[k]

                   fq=0.0
                   dfdxq=0
                   dfdyq=0
                   dfdzq=0
                   for k in range(0,mV):
                       fq+=NNNV[k]*field[iconV[k,iel]]
                       dfdxq+=dNNNVdx[k]*field[iconV[k,iel]]
                       dfdyq+=dNNNVdy[k]*field[iconV[k,iel]]
                       dfdzq+=dNNNVdz[k]*field[iconV[k,iel]]
                   print(xq,yq,zq,fq,dfdxq,dfdyq,dfdzq)

               #end kq
           #end jq
       #end iq
   #end iel
#end if debug 

print("testing shape fcts: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

eps=1.e-10

bc_fix=np.zeros(Nfem,dtype=bool)    # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps or xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(xV[i],yV[i],zV[i])
    if yV[i]/Ly<eps or yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(xV[i],yV[i],zV[i])
    if zV[i]/Lz<eps or zV[i]/Lz>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(xV[i],yV[i],zV[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(xV[i],yV[i],zV[i])
#end for

print("define b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start = timing.time()

if globall:
   A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)
   rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
else:   
   K_mat = lil_matrix((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = lil_matrix((NfemV,NfemP),dtype=np.float64) # matrix G
   C_mat = lil_matrix((NfemP,NfemP),dtype=np.float64) # matrix C
   M_mat = lil_matrix((NfemV,NfemV),dtype=np.float64) # matrix M

f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
N_mat = np.zeros((6,ndofP*mP),dtype=np.float64) # matrix  
b_mat = np.zeros((6,ndofV*mV),dtype=np.float64)  # gradient matrix B 
c_mat = np.zeros((6,6),dtype=np.float64)         # C matrix 
c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0, nel):

    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

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

                #using Q1 mapping
                jcob=hx*hy*hz/8
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    zq+=NNNV[k]*zV[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                    dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                    dNNNVdz[k]=jcbi[2,2]*dNNNVdt[k]

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:6, 3*i:3*i+3] = [[dNNNVdx[i],0.        ,0.        ],
                                             [0.        ,dNNNVdy[i],0.        ],
                                             [0.        ,0.        ,dNNNVdz[i]],
                                             [dNNNVdy[i],dNNNVdx[i],0.        ],
                                             [dNNNVdz[i],0.        ,dNNNVdx[i]],
                                             [0.        ,dNNNVdz[i],dNNNVdy[i]]]
                #end for

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq,zq,beeta)*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i+0]-=NNNV[i]*jcob*weightq*bx(xq,yq,zq,beeta)
                    f_el[ndofV*i+1]-=NNNV[i]*jcob*weightq*by(xq,yq,zq,beeta)
                    f_el[ndofV*i+2]-=NNNV[i]*jcob*weightq*bz(xq,yq,zq,beeta)
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=NNNP[i]

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

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

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    if globall:
                       A_mat[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if globall:
                   A_mat[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_mat[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]

#end for iel

print("build FE matrix: %.3f s, h= %e" % (timing.time() - start, hx))

###############################################################################

if bench==-1:
   print ("-----bench=-1-------------------------------------------")
   print('size of G_mat:',G_mat.shape)
   for i in range(NfemV):
       print(i,G_mat[i,0:NfemP])
   G2 = np.zeros((51,NfemP),dtype=np.float64)
   G2[0:3,:]=G_mat[39:42,:] 
   G2[3:,:]=G_mat[81:,:]    
   ns = null_space(G2)
   opla=ns.shape
   print('size of nullspace=',opla[1])
   exit()

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start = timing.time()

if globall:
   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   A_csr=A_mat.tocsr()

print("assemble rhs & convert A to csr: %.3f s" % (timing.time() - start))

###############################################################################
# apply reverse Cuthill-McKee algorithm 
###############################################################################
start = timing.time()

#take snapshot of matrix before reordering
if matrix_snapshot:
   plt.spy(A_csr, markersize=0.1)
   plt.savefig('matrix_bef.png', bbox_inches='tight')
   plt.clf()

if apply_RCM:
   #compute reordering array
   perm = reverse_cuthill_mckee(A_csr,symmetric_mode=True)
   #build reverse perm array
   perm_inv=np.empty(len(perm),dtype=np.int32)
   for i in range(0,len(perm)):
       perm_inv[perm[i]]=i
   A_csr=A_csr[np.ix_(perm,perm)]
   rhs=rhs[np.ix_(perm)]

if matrix_snapshot:
   #take snapshot of matrix after reordering
   plt.spy(A_csr, markersize=1)
   plt.savefig('matrix_aft.png', bbox_inches='tight')
   plt.clf()

print("apply reordering: %.3f s" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################

if globall:
   sol=sps.linalg.spsolve(A_csr,rhs)
   if apply_RCM:
      sol=sol[np.ix_(perm_inv)]
   u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
   p=sol[NfemV:Nfem]

else:
   # convert matrices to CSR format
   G_mat=sps.csr_matrix(G_mat)
   K_mat=sps.csr_matrix(K_mat)
   M_mat=sps.csr_matrix(M_mat)
   C_mat=sps.csr_matrix(C_mat)
   print("     -> K_mat (m,M) %.7f %.7f " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.7f %.7f " %(np.min(G_mat),np.max(G_mat)))
   print("     -> M_mat (m,M) %.7f %.7f " %(np.min(M_mat),np.max(M_mat)))
   solV,solP,k=cg.schur_complement_cg_solver(K_mat,G_mat,C_mat,M_mat,\
                         f_rhs,h_rhs,NfemV,NfemP,niter,tol,use_precond)
   u,v,w=np.reshape(solV,(NV,3)).T
   p=solP[:]

print("     -> uu (m,M) %.7f %.7f %.7f" %(np.min(u),np.max(u),hx))
print("     -> vv (m,M) %.7f %.7f %.7f" %(np.min(v),np.max(v),hx))
print("     -> ww (m,M) %.7f %.7f %.7f" %(np.min(w),np.max(w),hx))
print("     -> pp (m,M) %.7f %.7f %.7f" %(np.min(p),np.max(p),hx))

stats_vel_file.write("%e %e %e %e %e %e %e \n" %(np.min(u),np.max(u),\
                                                 np.min(v),np.max(v),\
                                                 np.min(w),np.max(w),hx))
stats_vel_file.flush()

if debug: np.savetxt('velocity.ascii',np.array([xV,yV,zV,u,v,w]).T,header='# x,y,z,u,v,w')

print("solve time: %.3f s, h= %e" % (timing.time() - start, hx))

###############################################################################
# make sure int p dV = 0 
###############################################################################
start = timing.time()

int_p=0
print("     -> pp (m,M) %.7f %.7f %.7f" %(np.min(p),np.max(p),hx))
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            for kq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                NNNP[0:mP]=NNP(rq,sq,tq)
                pq=np.sum(NNNP[0:mP]*p[iconP[0:mP,iel]])
                jcob=hx*hy*hz/8
                int_p+=pq*jcob*weightq
            #end for
        #end for
    #end for
#end for
int_p/=(Lx*Ly*Ly)
p[:]-=int_p

print("     -> pp (m,M) %.7f %.7f %.7f" %(np.min(p),np.max(p),hx))

stats_p_file.write("%e %e %e \n" %(np.min(p),np.max(p),hx))
stats_p_file.flush()

print("normalise pressure: %.3f s" % (timing.time() - start))

###############################################################################
# compute xc,yc,zc,rho,eta
###############################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  # x coordinates
yc = np.zeros(nel,dtype=np.float64)  # y coordinates
zc = np.zeros(nel,dtype=np.float64)  # z coordinates
bx_el = np.zeros(nel,dtype=np.float64)  
by_el = np.zeros(nel,dtype=np.float64)  
bz_el = np.zeros(nel,dtype=np.float64)  
eta_el = np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    xc[iel]=0.5*(xV[iconV[0,iel]]+xV[iconV[6,iel]])
    yc[iel]=0.5*(yV[iconV[0,iel]]+yV[iconV[6,iel]])
    zc[iel]=0.5*(zV[iconV[0,iel]]+zV[iconV[6,iel]])
    bx_el[iel]=bx(xc[iel],yc[iel],zc[iel],beeta)
    by_el[iel]=by(xc[iel],yc[iel],zc[iel],beeta)
    bz_el[iel]=bz(xc[iel],yc[iel],zc[iel],beeta)
    eta_el[iel]=eta(xc[iel],yc[iel],zc[iel],beeta)
#end for

if debug: np.savetxt('gridc.ascii',np.array([xc,yc,zc]).T,header='# x,y,z')

print("compute gridc: %.3f s" % (timing.time() - start))

###############################################################################
# compute error in L2 norm 
###############################################################################
start = timing.time()

avrg_u=0
avrg_v=0
avrg_w=0
avrg_absu=0
avrg_absv=0
avrg_absw=0
vrms=0.
errv=0.
errp=0.
for iel in range (0,nel):
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

                jcob=hx*hy*hz/8
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
                jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                    dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                    dNNNVdz[k]=jcbi[2,2]*dNNNVdt[k]
                #end for

                xq=0.0
                yq=0.0
                zq=0.0
                uq=0.0
                vq=0.0
                wq=0.0
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    zq+=NNNV[k]*zV[iconV[k,iel]]
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                    wq+=NNNV[k]*w[iconV[k,iel]]
                #end for

                pq=0.0
                for k in range(0,mP):
                    pq+=NNNP[k]*p[iconP[k,iel]]
                #end for

                avrg_u+=uq*weightq*jcob
                avrg_v+=vq*weightq*jcob
                avrg_w+=wq*weightq*jcob
                avrg_absu+=abs(uq)*weightq*jcob
                avrg_absv+=abs(vq)*weightq*jcob
                avrg_absw+=abs(wq)*weightq*jcob

                vrms+=(uq**2+vq**2+wq**2)*weightq*jcob

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*weightq*jcob

                errp+=(pq-pth(xq,yq,zq))**2*weightq*jcob

            #end for kq
        #end for jq
    #end for iq
#end for iel

errv=np.sqrt(errv/Lx/Ly/Lz)
errp=np.sqrt(errp/Lx/Ly/Lz)
vrms=np.sqrt(vrms/Lx/Ly/Lz)
avrg_u/=(Lx*Ly*Lz)
avrg_v/=(Lx*Ly*Lz)
avrg_w/=(Lx*Ly*Lz)
avrg_absu/=(Lx*Ly*Lz)
avrg_absv/=(Lx*Ly*Lz)
avrg_absw/=(Lx*Ly*Lz)

print("     -> nel= %6d ; errv: %e ; errp: %e " %(nel,errv,errp))
print("     -> nel= %6d ; vrms: %e" % (nel,vrms))
print("     -> nel= %6d ; averages u,v,w: %e %e %e %e %e %e  " % (nel,avrg_u,avrg_v,avrg_w,\
                                                                  avrg_absu,avrg_absv,avrg_absw))

print("compute errors: %.3f s" % (timing.time() - start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################

vel=np.sqrt(u**2+v**2+w**2)
print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      np.min(w),np.max(w),\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms)

###############################################################################
# export velocity and pressure on diagonals
###############################################################################
   
diagfile1=open("diag1.ascii","w")
diagfile2=open("diag2.ascii","w")
diagfile3=open("diag3.ascii","w")
diagfile4=open("diag4.ascii","w")

npts=100

for idiag in range(0,4):

    if idiag==0:
       x_start=0
       y_start=0
       z_start=0
       x_end=Lx
       y_end=Ly
       z_end=Lz

    if idiag==1:
       x_start=Lx
       y_start=0
       z_start=0
       x_end=0
       y_end=Ly
       z_end=Lz

    if idiag==2:
       x_start=Lx
       y_start=Ly
       z_start=0
       x_end=0
       y_end=0
       z_end=Lz

    if idiag==3:
       x_start=0
       y_start=Ly
       z_start=0
       x_end=Lx
       y_end=0
       z_end=Lz

    delta_x=(x_end-x_start)/npts
    delta_y=(y_end-y_start)/npts
    delta_z=(z_end-z_start)/npts

    for i in range(0,npts):
        xq=x_start+delta_x/2+i*delta_x
        yq=y_start+delta_y/2+i*delta_y
        zq=z_start+delta_z/2+i*delta_z
        qq=np.sqrt((xq-x_start)**2+(yq-y_start)**2+(zq-z_start)**2)
        for iel in range(0,nel):
            if xq>=xV[iconV[0,iel]] and xq<=xV[iconV[6,iel]] and \
               yq>=yV[iconV[0,iel]] and yq<=yV[iconV[6,iel]] and \
               zq>=zV[iconV[0,iel]] and zq<=zV[iconV[6,iel]] :
               rq=((xq-xV[iconV[0,iel]])/hx-0.5)*2
               sq=((yq-yV[iconV[0,iel]])/hy-0.5)*2
               tq=((zq-zV[iconV[0,iel]])/hz-0.5)*2
               NNNV[0:mV]=NNV(rq,sq,tq)
               NNNP[0:mP]=NNP(rq,sq,tq)
               uq=0.0
               vq=0.0
               wq=0.0
               for k in range(0,mV):
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
                   wq+=NNNV[k]*w[iconV[k,iel]]
               #end for
               pq=0.0
               for k in range(0,mP):
                   pq+=NNNP[k]*p[iconP[k,iel]]
               #end for
               if idiag==0:
                  diagfile1.write("%.10f %.10f %.10f %.20f %.20f %.20f %.20f %.10f\n" %(xq,yq,zq,uq,vq,wq,pq,qq))
               if idiag==1:
                  diagfile2.write("%.10f %.10f %.10f %.20f %.20f %.20f %.20f %.10f\n" %(xq,yq,zq,uq,vq,wq,pq,qq))
               if idiag==2:
                  diagfile3.write("%.10f %.10f %.10f %.20f %.20f %.20f %.20f %.10f\n" %(xq,yq,zq,uq,vq,wq,pq,qq))
               if idiag==3:
                  diagfile4.write("%.10f %.10f %.10f %.20f %.20f %.20f %.20f %.10f\n" %(xq,yq,zq,uq,vq,wq,pq,qq))
            #end if
        #end for
    #end for
#end for

###############################################################################
# export solution on vertical line
###############################################################################

vertfile=open("vert.ascii","w")

for i in range(0,NP):
    if abs(xV[i]-Lx/2)<1e-6 and abs(yV[i]-Ly/2)<1e-6:
       vertfile.write("%.10f %.20f %.20f %.20f %.20f\n" %(zV[i],u[i],v[i],w[i],p[i]))

###############################################################################
# plot of solution
###############################################################################
start = timing.time()

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
   vtufile.write("<DataArray type='Float32' Name='rho.gx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % bx_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho.gy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % by_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho.gz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % bz_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % eta_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%.20f\n" % p[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       vtufile.write("%f\n" % pth(xV[i],yV[i],zV[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (err)' Format='ascii'> \n")
   for i in range (0,nnx*nny*nnz):
       error_p=p[i]-pth(xV[i],yV[i],zV[i])
       vtufile.write("%f\n" % error_p)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%.20f %.20f %.20f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%.20f %.20f %.20f \n" %(uth(xV[i],yV[i],zV[i]),\
                                              vth(xV[i],yV[i],zV[i]),\
                                              wth(xV[i],yV[i],zV[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (err)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       error_u=u[i]-uth(xV[i],yV[i],zV[i])
       error_v=v[i]-vth(xV[i],yV[i],zV[i])
       error_w=w[i]-wth(xV[i],yV[i],zV[i])
       vtufile.write("%10f %10f %10f \n" %(error_u,error_v,error_w))
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

   filename = 'allVdofs.vtu' 
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,NV))
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],zV[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%d " % i) 
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,NV):
       vtufile.write("%d " % 1) 
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
