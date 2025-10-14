import numba
import sys as sys
import numpy as np
import time as clock
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.linalg import null_space
from scipy.sparse.csgraph import reverse_cuthill_mckee
import schur_complement_cg_solver as cg
from scipy.sparse.linalg import spsolve

###############################################################################
# bench=1: Dohrmann & Bochev manufactured solution 2004
# bench=2: horizontal shear 
# bench=3: Stokes sphere
# bench=4: Sinking cube
# bench=5: Manufactured solution
###############################################################################

bench=5

###############################################################################
beeta=0 # beta parameter for mms

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
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=4*(2*y-1)*(2*z-1)  *(-1)
    if bench==-1: val=0
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
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=4*(2*x-1)*(2*z-1) *(-1)
    if bench==-1: val=0
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
    if bench==2: val=0
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
    if bench==-1: val=0
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
    if bench==5: val=1
    if bench==-1: val=0
    return val

###############################################################################

def uth(x,y,z):
    if bench==1: val=x+x*x+x*y+x*x*x*y
    if bench==2: val=0.5-z
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=x*(1-x)*(1-2*y)*(1-2*z)
    if bench==-1: val=0
    return val

def vth(x,y,z):
    if bench==1: val=y+x*y+y*y+x*x*y*y
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=(1-2*x)*y*(1-y)*(1-2*z)
    if bench==-1: val=0
    return val

def wth(x,y,z):
    if bench==1: val=-2*z-3*x*z-3*y*z-5*x*x*y*z
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=-2*(1-2*x)*(1-2*y)*z*(1-z)
    if bench==-1: val=0
    return val

def pth(x,y,z):
    if bench==1: val=x*y*z+x*x*x*y*y*y*z-5./32.
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=0
    if bench==5: val=(2*x-1)*(2*y-1)*(2*z-1)
    if bench==-1: val=0
    return val

###############################################################################
    
aa=8/27
bb=10/21
cc=4/21
dd=64/63
ee=8/63

@numba.njit
def basis_functions_V(r,s,t):
    b8=(27/32)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1-r)*(1-s)*(1-t)  
    b9=(27/32)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1+r)*(1+s)*(1+t)  
    N0=0.125*(1-r)*(1-s)*(1-t) -aa*b8  
    N1=0.125*(1+r)*(1-s)*(1-t) -aa*bb*b8-aa*cc*b9
    N2=0.125*(1+r)*(1+s)*(1-t) -aa*cc*b8-aa*bb*b9
    N3=0.125*(1-r)*(1+s)*(1-t) -aa*bb*b8-aa*cc*b9
    N4=0.125*(1-r)*(1-s)*(1+t) -aa*bb*b8-aa*cc*b9
    N5=0.125*(1+r)*(1-s)*(1+t) -aa*cc*b8-aa*bb*b9
    N6=0.125*(1+r)*(1+s)*(1+t) -aa*b9
    N7=0.125*(1-r)*(1+s)*(1+t) -aa*cc*b8-aa*bb*b9
    N8= dd*b8-ee*b9
    N9=-ee*b8+dd*b9
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8,N9],dtype=np.float64)

@numba.njit
def basis_functions_V_dr(r,s,t):
    db8dr=(27/32)**3*(1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2) 
    db9dr=(27/32)**3*(1-s**2)*(1-t**2)*(1+s)*(1+t)*( 1-2*r-3*r**2)
    dNdr0=-0.125*(1-s)*(1-t) -aa*db8dr  
    dNdr1=+0.125*(1-s)*(1-t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr2=+0.125*(1+s)*(1-t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr3=-0.125*(1+s)*(1-t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr4=-0.125*(1-s)*(1+t) -aa*bb*db8dr-aa*cc*db9dr
    dNdr5=+0.125*(1-s)*(1+t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr6=+0.125*(1+s)*(1+t) -aa*db9dr
    dNdr7=-0.125*(1+s)*(1+t) -aa*cc*db8dr-aa*bb*db9dr
    dNdr8= dd*db8dr-ee*db9dr
    dNdr9=-ee*db8dr+dd*db9dr
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,\
                     dNdr6,dNdr7,dNdr8,dNdr9],dtype=np.float64)

@numba.njit
def basis_functions_V_ds(r,s,t):
    db8ds=(27/32)**3*(1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2) 
    db9ds=(27/32)**3*(1-r**2)*(1-t**2)*(1+r)*(1+t)*( 1-2*s-3*s**2)
    dNds0=-0.125*(1-r)*(1-t) -aa*db8ds   
    dNds1=-0.125*(1+r)*(1-t) -aa*bb*db8ds-aa*cc*db9ds 
    dNds2=+0.125*(1+r)*(1-t) -aa*cc*db8ds-aa*bb*db9ds 
    dNds3=+0.125*(1-r)*(1-t) -aa*bb*db8ds-aa*cc*db9ds
    dNds4=-0.125*(1-r)*(1+t) -aa*bb*db8ds-aa*cc*db9ds
    dNds5=-0.125*(1+r)*(1+t) -aa*cc*db8ds-aa*bb*db9ds
    dNds6=+0.125*(1+r)*(1+t) -aa*db9ds
    dNds7=+0.125*(1-r)*(1+t) -aa*cc*db8ds-aa*bb*db9ds
    dNds8= dd*db8ds-ee*db9ds
    dNds9=-ee*db8ds+dd*db9ds
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,\
                     dNds6,dNds7,dNds8,dNds9],dtype=np.float64)

@numba.njit
def basis_functions_V_dt(r,s,t):
    db8dt=(27/32)**3*(1-r**2)*(1-s**2)*(1-r)*(1-s)*(-1-2*t+3*t**2) 
    db9dt=(27/32)**3*(1-r**2)*(1-s**2)*(1+r)*(1+s)*( 1-2*t-3*t**2)
    dNdt0=-0.125*(1-r)*(1-s) -aa*db8dt   
    dNdt1=-0.125*(1+r)*(1-s) -aa*bb*db8dt-aa*cc*db9dt 
    dNdt2=-0.125*(1+r)*(1+s) -aa*cc*db8dt-aa*bb*db9dt 
    dNdt3=-0.125*(1-r)*(1+s) -aa*bb*db8dt-aa*cc*db9dt
    dNdt4=+0.125*(1-r)*(1-s) -aa*bb*db8dt-aa*cc*db9dt
    dNdt5=+0.125*(1+r)*(1-s) -aa*cc*db8dt-aa*bb*db9dt
    dNdt6=+0.125*(1+r)*(1+s) -aa*db9dt
    dNdt7=+0.125*(1-r)*(1+s) -aa*cc*db8dt-aa*bb*db9dt
    dNdt8= dd*db8dt-ee*db9dt
    dNdt9=-ee*db8dt+dd*db9dt
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,\
                     dNdt6,dNdt7,dNdt8,dNdt9],dtype=np.float64)

###############################################################################

@numba.njit
def basis_functions_P(r,s,t):
    N0=0.125*(1-r)*(1-s)*(1-t) 
    N1=0.125*(1+r)*(1-s)*(1-t) 
    N2=0.125*(1+r)*(1+s)*(1-t) 
    N3=0.125*(1-r)*(1+s)*(1-t) 
    N4=0.125*(1-r)*(1-s)*(1+t) 
    N5=0.125*(1+r)*(1-s)*(1+t) 
    N6=0.125*(1+r)*(1+s)*(1+t) 
    N7=0.125*(1-r)*(1+s)*(1+t) 
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 82 ***********")
print("*******************************")

ndim=3
m_V=8+2   # number of V nodes making up an element
m_P=8     # number of P nodes making up an element
ndof_V=3  # number of velocity degrees of freedom per node

Lx=1. # x- extent of the domain 
Ly=1. # y- extent of the domain 
Lz=1. # z- extent of the domain 

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   nq_per_dim=int(sys.argv[4])
   visu = int(sys.argv[5])
else:
   nelx = 14  # do not exceed 20 
   nely =nelx
   nelz =nelx
   nq_per_dim=2
   visu=1
#end if

debug=False

# True: build fully assembled Stokes matrix
# False: use schur complement solver
globall=True

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

nnx=nelx+1             # number of nodes, x direction
nny=nely+1             # number of nodes, y direction
nnz=nelz+1             # number of nodes, z direction
nel=nelx*nely*nelz     # number of elements, total
nn_V=nnx*nny*nnz+2*nel # total number of V nodes 
nn_P=nnx*nny*nnz       # total number of P nodes
Nfem_V=nn_V*ndof_V     # total number of velocity dofs
Nfem_P=nn_P            # total number of pressure dofs
Nfem=Nfem_V+Nfem_P     # total number of dofs
hx=Lx/nelx             # mesh size
hy=Ly/nely             # mesh size
hz=Lz/nelz             # mesh size
volume=Lx*Ly*Lz

###############################################################################
# quadrature points and weights

if nq_per_dim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x_V[counter]=i*Lx/float(nelx)
            y_V[counter]=j*Ly/float(nely)
            z_V[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_V[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon_V[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon_V[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon_V[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon_V[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon_V[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon_V[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon_V[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            icon_V[8,counter]=(nelx+1)*(nely+1)*(nelz+1)+2*counter+0
            icon_V[9,counter]=(nelx+1)*(nely+1)*(nelz+1)+2*counter+1
            counter += 1
        #end for
    #end for
#end for

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# add bubble nodes along 0-6 diagonal
###############################################################################
start=clock.time()

counter=0
for iel in range(0,nel):
    x_V[nnx*nny*nnz+counter]=x_V[icon_V[0,iel]]+hx/3
    y_V[nnx*nny*nnz+counter]=y_V[icon_V[0,iel]]+hy/3
    z_V[nnx*nny*nnz+counter]=z_V[icon_V[0,iel]]+hz/3
    counter+=1
    x_V[nnx*nny*nnz+counter]=x_V[icon_V[0,iel]]+2*hx/3
    y_V[nnx*nny*nnz+counter]=y_V[icon_V[0,iel]]+2*hy/3
    z_V[nnx*nny*nnz+counter]=z_V[icon_V[0,iel]]+2*hz/3
    counter+=1

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V,z_V]).T,header='# x,y,z,u,v,w')

#for iel in range(0,nel):
#    print ("iel=",iel)
#    print ("node 0",icon_V[0,iel],"at pos.",x_V[iconV[0,iel]],y_V[iconV[0,iel]],z_V[iconV[0,iel]])
#    print ("node 1",icon_V[1,iel],"at pos.",x_V[iconV[1,iel]],y_V[iconV[1,iel]],z_V[iconV[1,iel]])
#    print ("node 2",icon_V[2,iel],"at pos.",x_V[iconV[2,iel]],y_V[iconV[2,iel]],z-V[iconV[2,iel]])
#    print ("node 3",icon_V[3,iel],"at pos.",x_V[iconV[3,iel]],y_V[iconV[3,iel]],z_V[iconV[3,iel]])
#    print ("node 4",icon_V[4,iel],"at pos.",x_V[iconV[4,iel]],y-V[iconV[4,iel]],z_V[iconV[4,iel]])
#    print ("node 5",icon_V[5,iel],"at pos.",x_V[iconV[5,iel]],y-V[iconV[5,iel]],z_V[iconV[5,iel]])
#    print ("node 6",icon_V[6,iel],"at pos.",x_V[iconV[6,iel]],y_V[iconV[6,iel]],z_V[iconV[6,iel]])
#    print ("node 7",icon_V[7,iel],"at pos.",x_V[iconV[7,iel]],y_V[iconV[7,iel]],z_V[iconV[7,iel]])
#    print ("node 8",icon_V[8,iel],"at pos.",x_V[iconV[8,iel]],y_V[iconV[8,iel]],z_V[iconV[8,iel]])
#    print ("node 9",icon_V[9,iel],"at pos.",x_V[iconV[9,iel]],y_V[iconV[9,iel]],z_V[iconV[9,iel]])

print("compute bubble nodes coords: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid and iconP 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)
z_P=np.zeros(nn_P,dtype=np.float64)
x_P[0:nn_P]=x_V[0:nn_P]
y_P[0:nn_P]=y_V[0:nn_P]
z_P[0:nn_P]=z_V[0:nn_P]

icon_P=np.zeros((m_P,nel),dtype=np.int32)
icon_P[0:m_P,0:nel]=icon_V[0:m_P,0:nel]

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P,z_P]).T,header='# x,y,z')

print("build P grid: %.3f s" % (clock.time()-start))

###############################################################################
# Compute determinant of Jacobian matrix and its inverse.
# Only valid since all elements are identical cuboids.
###############################################################################

jcob=hx*hy*hz/8
jcb=np.zeros((ndim,ndim),dtype=np.float64)
jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

###############################################################################
# compute volume of elements, sanity check
# test shape functions and derivatives
###############################################################################
start=clock.time()

if debug:
   field=np.zeros(nn_V,dtype=np.float64)
   field[:]=z_V[:]**2

   for iel in range(0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               for kq in range(0,nq_per_dim):
                   rq=qcoords[iq]
                   sq=qcoords[jq]
                   tq=qcoords[kq]

                   N_V=basis_functions_V(rq,sq,tq)
                   dNdr_V=basis_functions_V_dr(rq,sq,tq)
                   dNds_V=basis_functions_V_ds(rq,sq,tq)
                   dNdt_V=basis_functions_V_dt(rq,sq,tq)

                   xq=np.dot(N_V,x_V[icon_V[:,iel]])
                   yq=np.dot(N_V,y_V[icon_V[:,iel]])
                   zq=np.dot(N_V,z_V[icon_V[:,iel]])

                   dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                   dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                   dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                   fq=np.dot(N_V,field[icon_V[:,iel]])
                   dfdxq=np.dot(dNdx_V,field[icon_V[:,iel]])
                   dfdyq=np.dot(dNdy_V,field[icon_V[:,iel]])
                   dfdzq=np.dot(dNdz_V,field[icon_V[:,iel]])

                   print(xq,yq,zq,fq,dfdxq,dfdyq,dfdzq)

               #end kq
           #end jq
       #end iq
   #end iel
#end if debug 

print("testing shape fcts: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)    # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps or x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=uth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+2]=True ; bc_val_V[i*ndof_V+2]=wth(x_V[i],y_V[i],z_V[i])
    if y_V[i]/Ly<eps or y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=uth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+2]=True ; bc_val_V[i*ndof_V+2]=wth(x_V[i],y_V[i],z_V[i])
    if z_V[i]/Lz<eps or z_V[i]/Lz>(1-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=uth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+2]=True ; bc_val_V[i*ndof_V+2]=wth(x_V[i],y_V[i],z_V[i])
#end for

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

if globall:
   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
   b_fem=np.zeros(Nfem,dtype=np.float64)
else:   
   K_mat=lil_matrix((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat=lil_matrix((Nfem_V,Nfem_P),dtype=np.float64) # matrix G
   C_mat=lil_matrix((Nfem_P,Nfem_P),dtype=np.float64) # matrix C
   M_mat=lil_matrix((Nfem_V,Nfem_V),dtype=np.float64) # matrix M

f_rhs = np.zeros(Nfem_V,dtype=np.float64) 
h_rhs = np.zeros(Nfem_P,dtype=np.float64) 
N_mat = np.zeros((6,m_P),dtype=np.float64)
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)
C=np.zeros((6,6),dtype=np.float64)     
C[0,0]=2.; C[1,1]=2.; C[2,2]=2.
C[3,3]=1.; C[4,4]=1.; C[5,5]=1.

for iel in range(0,nel):

    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                JxWq=jcob*qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])

                for i in range(0,m_V):
                    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                        [0.       ,dNdy_V[i],0.       ],
                                        [0.       ,0.       ,dNdz_V[i]],
                                        [dNdy_V[i],dNdx_V[i],0.       ],
                                        [dNdz_V[i],0.       ,dNdx_V[i]],
                                        [0.       ,dNdz_V[i],dNdy_V[i]]]

                K_el+=B.T.dot(C.dot(B))*eta(xq,yq,zq,beeta)*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i+0]-=N_V[i]*bx(xq,yq,zq,beeta)*JxWq
                    f_el[ndof_V*i+1]-=N_V[i]*by(xq,yq,zq,beeta)*JxWq
                    f_el[ndof_V*i+2]-=N_V[i]*bz(xq,yq,zq,beeta)*JxWq
                #end for

                N_mat[0,:]=N_P[:]
                N_mat[1,:]=N_P[:]
                N_mat[2,:]=N_P[:]

                G_el-=B.T.dot(N_mat)*JxWq

            #end for kq
        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1=ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    if globall:
                       A_fem[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,m_P):
                jkk=k2
                m2=icon_P[k2,iel]
                if globall:
                   A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]

#end for iel

print("build FE matrix: %.3f s, h= %e" % (clock.time()-start, hx))

###############################################################################
# looking for nullspace
###############################################################################

if bench==-1:
   print ("-----bench=-1-------------------------------------------")
   print('size of G_mat:',G_mat.shape)
   for i in range(Nfem_V):
       print(i,G_mat[i,0:Nfem_P])
   G2 = np.zeros((51,Nfem_P),dtype=np.float64)
   G2[0:3,:]=G_mat[39:42,:] 
   G2[3:,:]=G_mat[81:,:]    
   ns=null_space(G2)
   opla=ns.shape
   print('size of nullspace=',opla[1])
   exit()

###############################################################################
# assemble rhs
###############################################################################
start=clock.time()

if globall:
   b_fem[0:Nfem_V]=f_rhs
   b_fem[Nfem_V:Nfem]=h_rhs
   A_fem=A_fem.tocsr()

print("assemble rhs & convert A to csr: %.3f s" % (clock.time()-start))

###############################################################################
# apply reverse Cuthill-McKee algorithm 
###############################################################################
start=clock.time()

#take snapshot of matrix before reordering
if matrix_snapshot:
   plt.spy(A_fem, markersize=0.1)
   plt.savefig('matrix_bef.png', bbox_inches='tight')
   plt.clf()

if apply_RCM:
   #compute reordering array
   perm = reverse_cuthill_mckee(A_fem,symmetric_mode=True)
   #build reverse perm array
   perm_inv=np.zeros(len(perm),dtype=np.int32)
   for i in range(0,len(perm)):
       perm_inv[perm[i]]=i
   A_fem=A_fem[np.ix_(perm,perm)]
   b_fem=b_fem[np.ix_(perm)]

if matrix_snapshot:
   #take snapshot of matrix after reordering
   plt.spy(A_fem, markersize=0.1)
   plt.savefig('matrix_aft.png', bbox_inches='tight')
   plt.clf()

print("apply reordering: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

if globall:
   sol=sps.linalg.spsolve(A_fem,b_fem)
   if apply_RCM:
      sol=sol[np.ix_(perm_inv)]
   u,v,w=np.reshape(sol[0:Nfem_V],(nn_V,3)).T
   p=sol[Nfem_V:Nfem]

else:
   G_mat=sps.csr_matrix(G_mat)
   K_mat=sps.csr_matrix(K_mat)
   M_mat=sps.csr_matrix(M_mat)
   C_mat=sps.csr_matrix(C_mat)
   print("     -> K_mat (m,M) %e %e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %e %e " %(np.min(G_mat),np.max(G_mat)))
   print("     -> M_mat (m,M) %e %e " %(np.min(M_mat),np.max(M_mat)))
   solV,solP,k=cg.schur_complement_cg_solver(K_mat,G_mat,C_mat,M_mat,\
               f_rhs,h_rhs,Nfem_V,Nfem_P,niter,tol,use_precond)
   u,v,w=np.reshape(solV,(nn_V,3)).T
   p=solP[:]

print("     -> uu (m,M) %e %e %e" %(np.min(u),np.max(u),hx))
print("     -> vv (m,M) %e %e %e" %(np.min(v),np.max(v),hx))
print("     -> ww (m,M) %e %e %e" %(np.min(w),np.max(w),hx))
print("     -> pp (m,M) %e %e %e" %(np.min(p),np.max(p),hx))

stats_vel_file=open('stats_vel.ascii',"w")
stats_vel_file.write("%e %e %e %e %e %e %e \n" %(np.min(u),np.max(u),\
                                                 np.min(v),np.max(v),\
                                                 np.min(w),np.max(w),hx))
stats_vel_file.flush()

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')

print("solve time: %.3f s, h= %e" % (clock.time()-start,hx))

###############################################################################
# make sure int p dV = 0 
###############################################################################
start=clock.time()

print("     -> pp (m,M) %e %e %e" %(np.min(p),np.max(p),hx))

int_p=0
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                JxWq=jcob*qweights[iq]*qweights[jq]*qweights[kq]
                N_P=basis_functions_P(rq,sq,tq)
                pq=np.dot(N_P,p[icon_P[:,iel]])
                int_p+=pq*JxWq
            #end for
        #end for
    #end for
#end for

p[:]-=int_p/volume

print("     -> pp (m,M) %e %e %e" %(np.min(p),np.max(p),hx))

stats_p_file=open('stats_p.ascii',"w")
stats_p_file.write("%e %e %e \n" %(np.min(p),np.max(p),hx)) 
stats_p_file.flush()

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute xc,yc,zc,rho,eta (middle of elements)
###############################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)
yc=np.zeros(nel,dtype=np.float64)
zc=np.zeros(nel,dtype=np.float64)
bx_el=np.zeros(nel,dtype=np.float64)  
by_el=np.zeros(nel,dtype=np.float64)  
bz_el=np.zeros(nel,dtype=np.float64)  
eta_el=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    xc[iel]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[6,iel]])
    yc[iel]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[6,iel]])
    zc[iel]=0.5*(z_V[icon_V[0,iel]]+z_V[icon_V[6,iel]])
    bx_el[iel]=bx(xc[iel],yc[iel],zc[iel],beeta)
    by_el[iel]=by(xc[iel],yc[iel],zc[iel],beeta)
    bz_el[iel]=bz(xc[iel],yc[iel],zc[iel],beeta)
    eta_el[iel]=eta(xc[iel],yc[iel],zc[iel],beeta)
#end for

if debug: np.savetxt('gridc.ascii',np.array([xc,yc,zc]).T,header='# x,y,z')

print("compute gridc: %.3f s" % (clock.time()-start))

###############################################################################
# compute error in L2 norm 
###############################################################################
start=clock.time()

vrms=0. ; errv=0. ; errp=0.
avrg_u=0 ; avrg_v=0 ; avrg_w=0
avrg_absu=0 ; avrg_absv=0 ; avrg_absw=0

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                JxWq=jcob*qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                wq=np.dot(N_V,w[icon_V[:,iel]])
                pq=np.dot(N_P,p[icon_P[:,iel]])
                avrg_u+=uq*JxWq
                avrg_v+=vq*JxWq
                avrg_w+=wq*JxWq
                avrg_absu+=abs(uq)*JxWq
                avrg_absv+=abs(vq)*JxWq
                avrg_absw+=abs(wq)*JxWq
                vrms+=(uq**2+vq**2+wq**2)*JxWq
                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*JxWq
                errp+=(pq-pth(xq,yq,zq))**2*JxWq
            #end for kq
        #end for jq
    #end for iq
#end for iel

errv=np.sqrt(errv/volume)
errp=np.sqrt(errp/volume)
vrms=np.sqrt(vrms/volume)
avrg_u/=volume
avrg_v/=volume
avrg_w/=volume
avrg_absu/=volume
avrg_absv/=volume
avrg_absw/=volume

print("     -> nel= %6d ; errv: %e ; errp: %e " %(nel,errv,errp))
print("     -> nel= %6d ; vrms: %e" %(nel,vrms))
print("     -> nel= %6d ; averages u,v,w,|u|,|v|,|w|: %e %e %e %e %e %e" %(nel,\
                          avrg_u,avrg_v,avrg_w,avrg_absu,avrg_absv,avrg_absw))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################
start=clock.time()

vel=np.sqrt(u**2+v**2+w**2)

print('bench ',hx,nel,Nfem,np.min(u),np.max(u),np.min(v),np.max(v),\
               np.min(w),np.max(w),np.min(vel),np.max(vel),np.min(p),np.max(p),vrms)

print("print m/M values: %.3f s" % (clock.time()-start))

###############################################################################
# export velocity and pressure on diagonals
###############################################################################
start=clock.time()
   
diagf1=open("diag1.ascii","w")
diagf2=open("diag2.ascii","w")
diagf3=open("diag3.ascii","w")
diagf4=open("diag4.ascii","w")

npts=256

for idiag in range(0,4):
    if idiag==0:
       x_start=0 ; y_start=0 ; z_start=0 ; x_end=Lx ; y_end=Ly ; z_end=Lz
    if idiag==1:
       x_start=Lx ; y_start=0 ; z_start=0 ; x_end=0 ; y_end=Ly ; z_end=Lz
    if idiag==2:
       x_start=Lx ; y_start=Ly ; z_start=0 ; x_end=0 ; y_end=0 ; z_end=Lz
    if idiag==3:
       x_start=0 ; y_start=Ly ; z_start=0 ; x_end=Lx ; y_end=0 ; z_end=Lz

    delta_x=(x_end-x_start)/npts
    delta_y=(y_end-y_start)/npts
    delta_z=(z_end-z_start)/npts

    for i in range(0,npts):
        xq=x_start+delta_x/2+i*delta_x
        yq=y_start+delta_y/2+i*delta_y
        zq=z_start+delta_z/2+i*delta_z
        dist=np.sqrt((xq-x_start)**2+(yq-y_start)**2+(zq-z_start)**2)

        ielx=int(xq/hx)
        iely=int(yq/hy)
        ielz=int(zq/hz)
        iel=nely*nelz*ielx+nelz*iely+ielz 

        if xq<x_V[icon_V[0,iel]] or xq>x_V[icon_V[6,iel]] or \
           yq<y_V[icon_V[0,iel]] or yq>y_V[icon_V[6,iel]] or \
           zq<z_V[icon_V[0,iel]] or zq>z_V[icon_V[6,iel]] : exit('AH!')

        rq=((xq-x_V[icon_V[0,iel]])/hx-0.5)*2
        sq=((yq-y_V[icon_V[0,iel]])/hy-0.5)*2
        tq=((zq-z_V[icon_V[0,iel]])/hz-0.5)*2
        N_V=basis_functions_V(rq,sq,tq)
        N_P=basis_functions_P(rq,sq,tq)
        uq=np.dot(N_V,u[icon_V[:,iel]])
        vq=np.dot(N_V,v[icon_V[:,iel]])
        wq=np.dot(N_V,w[icon_V[:,iel]])
        pq=np.dot(N_P,p[icon_P[:,iel]])

        if idiag==0: diagf1.write("%e %e %e %e %e \n" %(dist,uq,vq,wq,pq))
        if idiag==1: diagf2.write("%e %e %e %e %e \n" %(dist,uq,vq,wq,pq))
        if idiag==2: diagf3.write("%e %e %e %e %e \n" %(dist,uq,vq,wq,pq))
        if idiag==3: diagf4.write("%e %e %e %e %e \n" %(dist,uq,vq,wq,pq))
    #end for
#end for

print("export diagonals: %.3f s" % (clock.time()-start))

###############################################################################
# export solution on vertical line
###############################################################################
start=clock.time()

vertfile=open("vert.ascii","w")

for i in range(0,nn_P):
    if abs(x_V[i]-Lx/2)/Lx<eps and abs(y_V[i]-Ly/2)/Ly<eps:
       vertfile.write("%e %e %e %e %e\n" %(z_V[i],u[i],v[i],w[i],p[i]))

print("export vertical line: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny*nnz,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho.gx' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%f\n" % bx_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho.gy' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%f\n" % by_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho.gz' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%f\n" % bz_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%f\n" % eta_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%e\n" % p[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%f\n" % pth(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (err)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%f\n" % (p[i]-pth(x_V[i],y_V[i],z_V[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       vtufile.write("%e %e %e \n" %(uth(x_V[i],y_V[i],z_V[i]),\
                                     vth(x_V[i],y_V[i],z_V[i]),\
                                     wth(x_V[i],y_V[i],z_V[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (err)' Format='ascii'> \n")
   for i in range(0,nnx*nny*nnz):
       error_u=u[i]-uth(x_V[i],y_V[i],z_V[i])
       error_v=v[i]-vth(x_V[i],y_V[i],z_V[i])
       error_w=w[i]-wth(x_V[i],y_V[i],z_V[i])
       vtufile.write("%e %e %e \n" %(error_u,error_v,error_w))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                   icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                   icon_V[6,iel],icon_V[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range(0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (clock.time()-start))

   vtufile=open("allVdofs.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nn_V))
   #--
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #--
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nn_V):
       vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #--
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%d " % i) 
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nn_V):
       vtufile.write("%d " % 1) 
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #--
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
