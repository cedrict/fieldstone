import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time
import random

#------------------------------------------------------------------------------

def gx(x,y):
    return 0

def gy(x,y):
    return -grav

#------------------------------------------------------------------------------

def stretch_towards_center(x,L,beta1,beta2):
    if x<beta1*L: 
       val = beta2/beta1*x
    elif x<(1.-beta1)*L: 
       val = (1-2*beta2)/(1-2*beta1)*(x-beta1*L)+beta2*L
    else:
       val=beta2/beta1*(x-(1-beta1)*L)+(1-beta2)*L
    return val

def stretch_towards_top(x,L,beta1,beta2):
    if x<beta1*L: 
       val=beta2/beta1*x
    else:
       val=(1-beta2)/(1-beta1)*(x-beta1*L)+beta2*L
    return val

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cm=0.01
year=3600.*24.*365.
eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

#model=1 # quinquis et al, EGU 2010
#model=2 # schmeling et al,PEPI 2008
#model=3 # falling block 
#model=4 # dripping instability
#model=5 # mato83 
#model=6 # quinquis-like 
model=7 # buks12, anhydrite in salt

#...........
if model==1:
   Lx=2680e3  # horizontal extent of the domain 
   Ly=670e3      # vertical extent of the domain 
   nelx=256
   nely=int(nelx*Ly/Lx)
   grav=9.81
   use_stretching_x=True
   use_stretching_y=True
   #material 1: OBSC   eta=1e19, rho=3300
   #material 2: SHB    eta=1e23, rho=3250
   #material 3: 40Myr  eta=1e22, rho=3240
   #material 4: OBSC   eta=1e19, rho=3300
   #material 5: SHB    eta=1e23, rho=3250
   #material 6: 70Myr  eta=1e22, rho=3250
   #material 7: mantle eta=1e20, rho=3200
   nmat=7
   rho_mat = np.array([3300,3250,3240,3300,3250,3250,3200],dtype=np.float64)   ; rho_mat-=np.min(rho_mat)
   eta_mat = np.array([1e19,1.e23,1e22,1e19,1e23,1e22,1e20],dtype=np.float64) 
   mass0=4
   rk=2
   nparticle_per_dim=10
   marker_random=True
   eta_ref=1e21      # scaling of G blocks

#...........
if model==2:
   Lx=3000e3
   Ly=750e3
   nelx=128
   nely=int(nelx*Ly/Lx)
   grav=10
   use_stretching_x=False
   use_stretching_y=False
   #material 1: sediments/air rho=0 eta=1e19
   #material 2: lithosphere rho=3300 eta=1e23
   #material 3: mantle rho=3200 eta=1e21
   nmat=3
   rho_mat = np.array([0,3300,3200],dtype=np.float64)   
   eta_mat = np.array([1e19,1e23,1e21],dtype=np.float64) 
   #eta_mat = np.array([1e21,1e21,1e21],dtype=np.float64) 
   #rho_mat = np.array([3300,3300,3300],dtype=np.float64) 
   mass0=Lx*50e3*rho_mat[0]+\
         (100e3*100e3+2000e3*100e3)*rho_mat[1]+\
         (1000e3*700e3+100e3*500e3+1900e3*600e3)*rho_mat[2]
   rk=2
   nparticle_per_dim=20
   marker_random=True
   eta_ref=1e21      # scaling of G blocks

#...........
if model==3: # falling block
   Lx=500e3
   Ly=500e3
   nelx=64
   nely=64
   grav=9.81
   use_stretching_x=False
   use_stretching_y=False
   #material 1: mantle 
   #material 2: block
   nmat=2
   rho_mat = np.array([3200,3300],dtype=np.float64) #; rho_mat-=np.min(rho_mat)
   eta_mat = np.array([1e21,1e22],dtype=np.float64) 
   mass0=100e3*100e3*rho_mat[1] + (Lx*Ly-100e3*100e3)*rho_mat[0]
   rk=2
   nparticle_per_dim=10
   marker_random=True
   eta_ref=1e21      # scaling of G blocks

#...........
if model==4: # dripping instability 
   Lx=1
   Ly=2
   nelx=33
   nely=65
   grav=1
   use_stretching_x=False
   use_stretching_y=False
   #material 1: drip 
   #material 2: upper mantle
   #material 3: lower mantle
   nmat=3
   rho_mat = np.array([1,0,0],dtype=np.float64) 
   eta_mat = np.array([1,1,200],dtype=np.float64) 
   mass0=0.2*0.1*rho_mat[0]+(1-0.2*0.1)*rho_mat[1]+1*rho_mat[2]
   rk=3
   nparticle_per_dim=10
   marker_random=True
   eta_ref=10      # scaling of G blocks

#...........
if model==5: # mato83 
   Lx=400e3
   Ly=180e3
   nelx=160
   nely=72
   grav=9.81
   use_stretching_x=False
   use_stretching_y=False
   #material 1: mantle left
   #material 2: mantle right 
   #material 3: lithosphere left
   #material 4: lithosphere right
   #material 5: water
   nmat=5
   rho_mat = np.array([3200,3200,3300,3300,1030],dtype=np.float64) 
   eta_mat = np.array([1e21,1e21,1e22,1e22,1e19],dtype=np.float64) 
   Dw=10e3
   dw=8e3
   Dl=50e3
   dl=10e3
   mass0=0.75*Lx*( (Ly-Dw-Dl)*rho_mat[0] + Dl*rho_mat[2] + Dw*rho_mat[4] )\
        +0.25*Lx*( (Ly-dw-dl)*rho_mat[1] + dl*rho_mat[3] + dw*rho_mat[4] )
   rk=3
   nparticle_per_dim=6
   marker_random=True
   eta_ref=1e21      # scaling of G blocks

#...........
if model==6: #  
   Lx=3000e3
   Ly=670e3
   nelx=150
   nely=int(nelx*Ly/Lx*1.4)
   grav=9.81
   use_stretching_x=True
   use_stretching_y=True
   #material 1: mantle 
   #material 2: 
   #material 3: 
   #material 4: upper crust left
   #material 5: 
   #material 6: 
   #material 7: upper crust right
   nmat=7
   rho_mat = np.array([3200,3240,3250,330,3250,3250,3300],dtype=np.float64) 
   eta_mat = np.array([1e20,1e22,1e23,1e19,1e22,1e23,1e19],dtype=np.float64) 
   mass0=10
   rk=1
   nparticle_per_dim=10
   marker_random=True
   eta_ref=1e21      # scaling of G blocks

#...........
if model==7:
   Lx=2500
   Ly=5000
   nelx=75 
   nely=150 
   grav=9.81
   use_stretching_x=False
   use_stretching_y=False
   #material 1: salt left
   #material 2: salt right
   #material 3: anhydrite
   nmat=3
   #case A
   #rho_mat = np.array([2200,2200,2900],dtype=np.float64) 
   #eta_mat = np.array([1e17,1e17,1e20],dtype=np.float64) 
   #case B
   #rho_mat = np.array([2200,2200,2900],dtype=np.float64) 
   #eta_mat = np.array([5e16,1e17,1e20],dtype=np.float64) 
   #case C
   rho_mat = np.array([2200,2200,2900],dtype=np.float64) 
   eta_mat = np.array([1e16,1e17,1e20],dtype=np.float64) 
   mass0=10
   rk=1
   nparticle_per_dim=3
   marker_random=False
   eta_ref=1e20  # scaling of G blocks

#...........
   
nnx=2*nelx+1                  # number of V nodes, x direction
nny=2*nely+1                  # number of V nodes, y direction
NV=nnx*nny                    # total number of nodes
NP=(nelx+1)*(nely+1)          # total number of P nodes
nel=nelx*nely                 # number of elements, total
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs

hx=Lx/nelx                    # mesh spacing in x direction
hy=Ly/nely                    # mesh spacing in y direction

nstep=250

CFL_nb=0.25

nq_per_dim=3                  # number of quad points per dimension
nq=nq_per_dim**2*nel          # number of quadrature points

nparticle_per_element=nparticle_per_dim**2
nparticle=nparticle_per_element*nel

#1: use elemental values for all q points
#2: use nodal values + Q1 shape functions to interp on q points
#3: use avrg nodal values to assign to all q points (elemental avrg)
#4: nodal rho, elemental eta
particle_projection=1

avrg=2

every=1

sparse=True

#################################################################

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

if nq_per_dim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

#################################################################
#################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("mass0=",mass0)
print("avrg=",avrg)
print("nparticle=",nparticle)
print("particle_projection=",particle_projection)
print("------------------------------")

vel_file=open('vel.ascii',"w")
vrms_file=open('vrms.ascii',"w")
mass_file=open('mass.ascii',"w")
nparticle_file=open('nparticle_per_element.ascii',"w")
dt_file=open('dt.ascii',"w")

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx/2.
        y[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid_bef.ascii',np.array([x,y]).T,header='# x,y')

print("grid setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

# creating a dedicated connectivity array to plot the solution on Q1 space
# different icon array but same velocity nodes.

nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (time.time() - start))

#################################################################
# mesh stretching
#################################################################

if use_stretching_x:
   beta1=0.25
   beta2=0.375
   for i in range(0,NV):
       x[i]=stretch_towards_center(x[i],Lx,beta1,beta2)

if use_stretching_y:
   beta1=0.25
   beta2=0.5
   for i in range(0,NV):
       y[i]=stretch_towards_top(y[i],Ly,beta1,beta2)

for iel in range(0,nel):
    x[iconV[4,iel]]=(x[iconV[0,iel]]+x[iconV[1,iel]])*0.5
    y[iconV[4,iel]]=(y[iconV[0,iel]]+y[iconV[1,iel]])*0.5
 
    x[iconV[5,iel]]=(x[iconV[1,iel]]+x[iconV[2,iel]])*0.5
    y[iconV[5,iel]]=(y[iconV[1,iel]]+y[iconV[2,iel]])*0.5

    x[iconV[6,iel]]=(x[iconV[2,iel]]+x[iconV[3,iel]])*0.5
    y[iconV[6,iel]]=(y[iconV[2,iel]]+y[iconV[3,iel]])*0.5

    x[iconV[7,iel]]=(x[iconV[3,iel]]+x[iconV[0,iel]])*0.5
    y[iconV[7,iel]]=(y[iconV[3,iel]]+y[iconV[0,iel]])*0.5

    x[iconV[8,iel]]=(x[iconV[0,iel]]+x[iconV[2,iel]])*0.5
    y[iconV[8,iel]]=(y[iconV[0,iel]]+y[iconV[2,iel]])*0.5

#np.savetxt('grid_aft.ascii',np.array([x,y]).T,header='# x,y')

#################################################################
# compute smallest element dimensions
#################################################################

hx_min=1e30
hy_min=1e30

if use_stretching_x:
   for ielx in range(0,nelx):
       xmin=x[iconV[0,ielx]]
       xmax=x[iconV[2,ielx]]
       hx_min=min(hx_min,xmax-xmin)
else:
   hx_min=hx


if use_stretching_y:
   for iely in range(0,nely):
       iel=iely*nelx
       ymin=y[iconV[0,iel]]
       ymax=y[iconV[2,iel]]
       hy_min=min(hy_min,ymax-ymin)
else:
   hy_min=hy

print("     ->hx_min=",hx_min)
print("     ->hy_min=",hy_min)

#################################################################
# compute coordinates of pressure nodes
#################################################################

xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=x[iconV[0,iel]]
    xP[iconP[1,iel]]=x[iconV[1,iel]]
    xP[iconP[2,iel]]=x[iconV[2,iel]]
    xP[iconP[3,iel]]=x[iconV[3,iel]]
    yP[iconP[0,iel]]=y[iconV[0,iel]]
    yP[iconP[1,iel]]=y[iconV[1,iel]]
    yP[iconP[2,iel]]=y[iconV[2,iel]]
    yP[iconP[3,iel]]=y[iconV[3,iel]]
#end for

#################################################################
# particle setup
#################################################################
start = time.time()

swarm_x=np.empty(nparticle,dtype=np.float64) 
swarm_y=np.empty(nparticle,dtype=np.float64) 
swarm_mat=np.empty(nparticle,dtype=np.int8)  
swarm_paint=np.empty(nparticle,dtype=np.float64) 
swarm_r=np.empty(nparticle,dtype=np.float64) 
swarm_s=np.empty(nparticle,dtype=np.float64) 
swarm_iel=np.empty(nparticle,dtype=np.int32) 

counter=0
for iel in range(0,nel):
    x1=x[iconV[0,iel]] ; y1=y[iconV[0,iel]]
    x2=x[iconV[1,iel]] ; y2=y[iconV[1,iel]]
    x3=x[iconV[2,iel]] ; y3=y[iconV[2,iel]]
    x4=x[iconV[3,iel]] ; y4=y[iconV[3,iel]]
    if marker_random:
       for j in range(0,nparticle_per_dim):
           for i in range(0,nparticle_per_dim):
               r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
               s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
               r+=random.uniform(-1,+1)/nparticle_per_dim/4.
               s+=random.uniform(-1,+1)/nparticle_per_dim/4.
               swarm_r[counter]=r
               swarm_s[counter]=s
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               swarm_iel[counter]=iel
               counter+=1
           #end for 
       #end for 
    else:
       for j in range(0,nparticle_per_dim):
           for i in range(0,nparticle_per_dim):
               r=-1.+i*2./nparticle_per_dim + 1./nparticle_per_dim
               s=-1.+j*2./nparticle_per_dim + 1./nparticle_per_dim
               swarm_r[counter]=r
               swarm_s[counter]=s
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               swarm_iel[counter]=iel
               counter+=1
           #end for 
       #end for 
    #end if
#end for 

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("marker setup: %.3f s" % (time.time() - start))


#################################################################
# assign material id to particles  
#################################################################
start = time.time()

if model==1:

   #points_file=open('points.ascii',"w")

   xA=Lx/2 ; yA=Ly
   xB=Lx/2 ; yB=Ly-7e3
   xC=Lx/2 ; yC=Ly-8e3
   xD=Lx/2 ; yD=Ly-39e3
   xE=Lx/2 ; yE=Ly-40e3
   xF=Lx/2 ; yF=Ly-82e3
   xG=Lx/2 ; yG=Ly-110e3

   for im in range (0,nparticle):
       swarm_mat[im]=7
       if swarm_x[im]<Lx/2:
          if swarm_y[im]>yB:
             swarm_mat[im]=1
          elif swarm_y[im]>yD:
             swarm_mat[im]=2
          elif swarm_y[im]>yF:
             swarm_mat[im]=3
       else:
          if swarm_y[im]>yC:
             swarm_mat[im]=4
          elif swarm_y[im]>yE:
             swarm_mat[im]=5
          elif swarm_y[im]>yG:
             swarm_mat[im]=6
       #end if
   #end for 

   #round part 
   xc=Lx/2
   yc=Ly-200e3
   for im in range (0,nparticle):
       if swarm_x[im]<Lx/2 and swarm_y[im]>yc:
          if (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2<200e3**2 and\
             (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2>192e3**2 and\
             swarm_y[im]>-(swarm_x[im]-xc)+yc:
             swarm_mat[im]=4
          if (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2<192e3**2 and\
             (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2>160e3**2 and\
             swarm_y[im]>-(swarm_x[im]-xc)+yc:
             swarm_mat[im]=5
          if (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2<160e3**2 and\
             (swarm_x[im]-xc)**2+(swarm_y[im]-yc)**2>90e3**2 and\
             swarm_y[im]>-(swarm_x[im]-xc)+yc:
             swarm_mat[im]=6
          #if swarm_mat[im]==4:
          #   mat4_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
          #if swarm_mat[im]==5:
          #   mat5_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
          #if swarm_mat[im]==6:
          #   mat6_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
       #end if
   #end for

   #subducted slab

   xH=Lx/2-np.sqrt(2.)/2.*200e3 ; yH=Ly-200e3+np.sqrt(2.)/2.*200e3
   xI=Lx/2-np.sqrt(2.)/2.*192e3 ; yI=Ly-200e3+np.sqrt(2.)/2.*192e3
   xJ=Lx/2-np.sqrt(2.)/2.*160e3 ; yJ=Ly-200e3+np.sqrt(2.)/2.*160e3
   xK=Lx/2-np.sqrt(2.)/2.*90e3  ; yK=Ly-200e3+np.sqrt(2.)/2.*90e3

   xL=xH-130e3*np.sqrt(2.)/2. ; yL=yH-130e3*np.sqrt(2.)/2.
   xM=xI-130e3*np.sqrt(2.)/2. ; yM=yI-130e3*np.sqrt(2.)/2.
   xN=xJ-130e3*np.sqrt(2.)/2. ; yN=yJ-130e3*np.sqrt(2.)/2.
   xO=xK-130e3*np.sqrt(2.)/2. ; yO=yK-130e3*np.sqrt(2.)/2.

   #points_file.write("%e %e\n" %(xH,yH))
   #points_file.write("%e %e\n" %(xI,yI))
   #points_file.write("%e %e\n" %(xJ,yJ))
   #points_file.write("%e %e\n" %(xK,yK))
   #points_file.write("%e %e\n" %(xL,yL))
   #points_file.write("%e %e\n" %(xM,yM))
   #points_file.write("%e %e\n" %(xN,yN))
   #points_file.write("%e %e\n" %(xO,yO))

   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       if yi< (xi-xL)+yL and yi> (xi-xM)+yM and yi<= -(xi-xH)+yH and yi>= -(xi-xL)+yL:
          swarm_mat[im]=4
       if yi< (xi-xM)+yM and yi> (xi-xN)+yN and yi<= -(xi-xH)+yH and yi>= -(xi-xL)+yL:
          swarm_mat[im]=5
       if yi< (xi-xN)+yN and yi> (xi-xO)+yO and yi<= -(xi-xH)+yH and yi>= -(xi-xL)+yL:
          swarm_mat[im]=6
       #if swarm_mat[im]==4:
       #   mat4_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
       #if swarm_mat[im]==5:
       #   mat5_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
       #if swarm_mat[im]==6:
       #   mat6_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
   #end for
    
   #for im in range(0,nparticle):
   #    if swarm_mat[im]==4:
   #       mat4_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
   #    if swarm_mat[im]==5:
   #       mat5_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))
   #    if swarm_mat[im]==6:
   #       mat6_file.write("%e %e\n" %(swarm_x[im],swarm_y[im]))

if model==2:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       swarm_mat[im]=3
       if yi>700e3:
          swarm_mat[im]=1
       if xi>1000e3 and yi<700e3 and yi>600e3:
          swarm_mat[im]=2
       if xi>1000e3 and xi<1100e3 and yi<700e3 and yi>500e3: 
          swarm_mat[im]=2
    #end for

if model==3:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       swarm_mat[im]=1
       if abs(xi-Lx/2)<50e3 and abs(yi-400e3)<50e3:
          swarm_mat[im]=2
    #end for

if model==4:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       if yi>1:
          if abs(xi-0.5)<0.1 and yi>1.9: 
             swarm_mat[im]=1
          else:          
             swarm_mat[im]=2
       else:
          swarm_mat[im]=3
    #end for

if model==5:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       swarm_mat[im]=5
       if xi<300e3 and yi<120e3:
          swarm_mat[im]=1
       if xi>300e3 and yi<162e3:
          swarm_mat[im]=2
       if xi<300e3 and yi>120e3 and yi<170e3:
          swarm_mat[im]=3
       if xi>300e3 and yi>162e3 and yi<172e3:
          swarm_mat[im]=4
    #end for

if model==6:
   xM=Lx/2
   yM=Ly-150e3
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       if xi<Lx/2:
          if yi>Ly-7e3:
             swarm_mat[im]=4
          elif yi>Ly-39e3:
             swarm_mat[im]=3
          elif yi>Ly-82e3:
             swarm_mat[im]=2
          else:
             swarm_mat[im]=1
       else:
          if yi>Ly-8e3:
             swarm_mat[im]=7
          elif yi>Ly-40e3:
             swarm_mat[im]=6
          elif yi>Ly-110e3:
             swarm_mat[im]=5
          else:
             swarm_mat[im]=1
       #end if
       if xi<xM and yi>yM:
          if np.sqrt((xi-xM)**2+(yi-yM)**2)<150e3 and\
             np.sqrt((xi-xM)**2+(yi-yM)**2)>142e3:
             swarm_mat[im]=7
          if np.sqrt((xi-xM)**2+(yi-yM)**2)<142e3 and\
             np.sqrt((xi-xM)**2+(yi-yM)**2)>110e3:
             swarm_mat[im]=6
          if np.sqrt((xi-xM)**2+(yi-yM)**2)<110e3 and\
             np.sqrt((xi-xM)**2+(yi-yM)**2)>40e3:
             swarm_mat[im]=5
       #end if

if model==7:
   for im in range(0,nparticle):
       xi=swarm_x[im]
       yi=swarm_y[im]
       if yi>Ly-200 and yi<Ly-100 and abs(xi-Lx/2)<100: 
          swarm_mat[im]=3
       else:
          if xi<Lx/2:
             swarm_mat[im]=1
          else:
             swarm_mat[im]=2
    #end for

print("marker layout: %.3f s" % (time.time() - start))

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y')
#exit()

#################################################################
# paint particles 
#################################################################
start = time.time()

for im in range (0,nparticle):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))
#end for 

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if model==1:
   u_in=-5*cm/year
   y_in=Ly-128e3
   y_out=Ly-160e3
   u_out=-u_in * (Ly-0.5*(y_in+y_out)) / (0.5*(y_in+y_out))

   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if x[i]/Lx>(1-eps):
          if y[i]<=y_out:
             bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = u_out
          elif y[i]<y_in:
             bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = (u_in-u_out)/(y_in-y_out)*(y[i]-y_out)+u_out
          else:
             bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = u_in
          #end if
       if y[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. 
   #end for 
#end if

if model==2 or model==3 or model==5 or model==6 or model==7:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if y[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # free slip
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # free slip
   #end for
#end if

if model==4:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       if y[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # no slip
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # no slip
   #end for
#end if

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# layout interface markers
#################################################################
start = time.time()

if model==1:
   nmarker=0

if model==2:
   nmarker=0

if model==3:
   hh=1e3
   nmarker1=np.int(100e3/hh)
   nmarker2=np.int(100e3/hh)
   nmarker3=np.int(100e3/hh)
   nmarker4=np.int(100e3/hh)
   nmarker=nmarker1+nmarker2+nmarker3+nmarker4
   print('     -> nmarker=',nmarker)

   m_x=np.empty(nmarker,dtype=np.float64) 
   m_y=np.empty(nmarker,dtype=np.float64) 
   m_r=np.empty(nmarker,dtype=np.float64) 
   m_s=np.empty(nmarker,dtype=np.float64) 
   m_u=np.empty(nmarker,dtype=np.float64) 
   m_v=np.empty(nmarker,dtype=np.float64) 

   counter=0
   for i in range(0,nmarker1):
       m_x[counter]=200e3
       m_y[counter]=450e3-i*hh
       counter+=1
   for i in range(0,nmarker2):
       m_x[counter]=200e3+i*hh
       m_y[counter]=350e3
       counter+=1
   for i in range(0,nmarker3):
       m_x[counter]=300e3
       m_y[counter]=350e3+i*hh
       counter+=1
   for i in range(0,nmarker4):
       m_x[counter]=300e3-i*hh
       m_y[counter]=450e3
       counter+=1

if model==4:
   hh=0.001
   nmarker1=np.int(0.1/hh)
   nmarker2=np.int(0.2/hh)
   nmarker3=np.int(0.1/hh)
   nmarker=nmarker1+nmarker2+nmarker3+1
   print('     -> nmarker=',nmarker)

   m_x=np.empty(nmarker,dtype=np.float64) 
   m_y=np.empty(nmarker,dtype=np.float64) 
   m_r=np.empty(nmarker,dtype=np.float64) 
   m_s=np.empty(nmarker,dtype=np.float64) 
   m_u=np.empty(nmarker,dtype=np.float64) 
   m_v=np.empty(nmarker,dtype=np.float64) 

   counter=0
   for i in range(0,nmarker1):
       m_x[counter]=0.4+eps
       m_y[counter]=2-i*hh-eps 
       counter+=1
   for i in range(0,nmarker2):
       m_x[counter]=0.4+i*hh  
       m_y[counter]=1.9+eps
       counter+=1
   for i in range(0,nmarker3+1):
       m_x[counter]=0.6-eps
       m_y[counter]=1.9+i*hh -eps
       counter+=1
    #end for

if model==5:
   nmarker=0

if model==6:
   nmarker=0

if model==7:
   nmarker=0

#np.savetxt('markers.ascii',np.array([m_x,m_y]).T,header='# x,y')

print("marker setup: %.3f s" % (time.time() - start))

#--------------------------------------------------------------------------------------------------
# time stepping loop
#--------------------------------------------------------------------------------------------------

Time=0.

for istep in range(0,nstep):

    print ('---------------------------------------')
    print ('-----------------istep= %i -------------' %istep)
    print ('---------------------------------------')

    #################################################################
    # locate particle
    #################################################################
    start = time.time()

    for im in range(0,nparticle):
        #-----
        if use_stretching_x:
           for ielx in range(0,nelx):
               xmin=x[iconV[0,ielx]]
               xmax=x[iconV[2,ielx]]
               if swarm_x[im]>xmin and swarm_x[im]<xmax:
                  break
        else:
           ielx=int(swarm_x[im]/Lx*nelx)
        #-----
        if use_stretching_y:
           for iely in range(0,nely):
               iel=iely*nelx
               ymin=y[iconV[0,iel]]
               ymax=y[iconV[2,iel]]
               if swarm_y[im]>ymin and swarm_y[im]<ymax:
                  break
        else:
           iely=int(swarm_y[im]/Ly*nely)
        #-----
        swarm_iel[im]=nelx*(iely)+ielx
    #end for

    #np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_iel]).T,header='# x,y')
    #exit()

    print("locate particles: %.3f s" % (time.time() - start))

    #################################################################
    # compute nodal averagings
    # with the tracer ratio method \cite{taki03}  
    #################################################################
    start = time.time()

    mat_nodal=np.zeros((nmat,NP),dtype=np.float64) 
    mat_nodal_counter=np.zeros(NP,dtype=np.float64) 
    nparticle_in_element=np.zeros(nel,dtype=np.float64) 

    for im in range(0,nparticle):
        imat=swarm_mat[im]-1
        iel=swarm_iel[im]
        N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
        N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
        N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
        N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
        mat_nodal[imat,iconP[0,iel]]+=N1
        mat_nodal[imat,iconP[1,iel]]+=N2
        mat_nodal[imat,iconP[2,iel]]+=N3
        mat_nodal[imat,iconP[3,iel]]+=N4
        mat_nodal_counter[iconP[0,iel]]+=N1
        mat_nodal_counter[iconP[1,iel]]+=N2
        mat_nodal_counter[iconP[2,iel]]+=N3
        mat_nodal_counter[iconP[3,iel]]+=N4

        #if swarm_r[im]<0:
        #   if swarm_s[im]<0:
        #      mat_nodal[imat,iconP[0,iel]]+=1
        #      mat_nodal_counter[iconP[0,iel]]+=1
        #   else:
        #      mat_nodal[imat,iconP[3,iel]]+=1
        #      mat_nodal_counter[iconP[3,iel]]+=1
        #   #end if
        #else:
        #   if swarm_s[im]<0:
        #      mat_nodal[imat,iconP[1,iel]]+=1
        #      mat_nodal_counter[iconP[1,iel]]+=1
        #   else:
        #      mat_nodal[imat,iconP[2,iel]]+=1
        #      mat_nodal_counter[iconP[2,iel]]+=1
        #   #end if
        #end if
        nparticle_in_element[iel]+=1
    #end for
    mat_nodal/=mat_nodal_counter

    nparticle_file.write("%d %e %e\n" %(istep,np.min(nparticle_in_element),np.max(nparticle_in_element))) 
    nparticle_file.flush()

    if np.min(nparticle_in_element)==0:
       exit('no particle left in an element')

    #np.savetxt('mat_nodal0.ascii',np.array([xP,yP,mat_nodal[0,:]]).T)
    #np.savetxt('mat_nodal1.ascii',np.array([xP,yP,mat_nodal[1,:]]).T)
    #np.savetxt('mat_nodal2.ascii',np.array([xP,yP,mat_nodal[2,:]]).T)
    #np.savetxt('mat_nodal3.ascii',np.array([xP,yP,mat_nodal[3,:]]).T)
    #np.savetxt('mat_nodal4.ascii',np.array([xP,yP,mat_nodal[4,:]]).T)
    #np.savetxt('mat_nodal5.ascii',np.array([xP,yP,mat_nodal[5,:]]).T)
    #np.savetxt('mat_nodal6.ascii',np.array([xP,yP,mat_nodal[6,:]]).T)

    #np.savetxt('rho_nodal0.ascii',np.array([xP,yP,rho_nodal]).T,header='# x,y,rho')

    print("     -> nparticle_in_elt(m,M) %.5e %.5e " %(np.min(nparticle_in_element),np.max(nparticle_in_element)))
    print("     -> mat_nodal     (m,M) %.5e %.5e " %(np.min(mat_nodal),np.max(mat_nodal)))

    print("particles onto Q1 grid: %.3f s" % (time.time() - start))

    ################################################################
    # compute rho and eta on P nodes
    #################################################################
    start = time.time()

    rho_nodal=np.zeros(NP,dtype=np.float64) 
    eta_nodal=np.zeros(NP,dtype=np.float64) 

    for i in range(0,NP):
        for imat in range(0,nmat):
            rho_nodal[i]+=mat_nodal[imat,i]*rho_mat[imat]

    if avrg==1:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]*eta_mat[imat]

    if avrg==2:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]*np.log10(eta_mat[imat])
           eta_nodal[i]=10.**eta_nodal[i]

    if avrg==3:
       for i in range(0,NP):
           for imat in range(0,nmat):
               eta_nodal[i]+=mat_nodal[imat,i]/eta_mat[imat]
           eta_nodal[i]=1./eta_nodal[i]

    print("     -> rho_nodal     (m,M) %.5e %.5e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_nodal     (m,M) %.5e %.5e " %(np.min(eta_nodal),np.max(eta_nodal)))

    #np.savetxt('rho_nodal.ascii',np.array([xP,yP,rho_nodal]).T,header='# x,y,rho')
    #np.savetxt('eta_nodal.ascii',np.array([xP,yP,eta_nodal]).T,header='# x,y,eta')

    print("compute rho,eta on P nodes: %.3f s" % (time.time() - start))

    ################################################################
    # compute elemental rho and eta 
    #################################################################
    start = time.time()
    
    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    c=np.zeros(nel,dtype=np.float64)

    for im in range(0,nparticle):
        iel=swarm_iel[im]
        imat=swarm_mat[im]-1
        rho_elemental[iel]+=rho_mat[imat]
        c[iel]+=1
    rho_elemental/=c

    c=np.zeros(nel,dtype=np.float64)

    if avrg==1:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=eta_mat[imat]
           c[iel]+=1
       eta_elemental/=c

    if avrg==2:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=np.log10(eta_mat[imat])
           c[iel]+=1
       eta_elemental/=c
       eta_elemental=10.**eta_elemental

    if avrg==3:
       for im in range(0,nparticle):
           iel=swarm_iel[im]
           imat=swarm_mat[im]-1
           eta_elemental[iel]+=1./eta_mat[imat]
           c[iel]+=1
       eta_elemental=c/eta_elemental

    print("     -> rho_elemental (m,M) %.5e %.5e " %(np.min(rho_elemental),np.max(rho_elemental)))
    print("     -> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))

    print("compute elemental rho,eta: %.3f s" % (time.time() - start))

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = time.time()

    if sparse:
       A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
    else:   
       K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
       G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
    b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
    N_mat = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)          # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)          # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)         # shape functions derivatives
    u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
    v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
    xq   = np.zeros(nq,dtype=np.float64)             # 
    yq   = np.zeros(nq,dtype=np.float64)             # 
    rhoq = np.zeros(nq,dtype=np.float64)             # 
    etaq = np.zeros(nq,dtype=np.float64)             # 
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

    counter=0
    for iel in range(0,nel):
        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   
        # integrate viscous term at quadrature points
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*y[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*y[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    xq[counter]+=NNNV[k]*x[iconV[k,iel]]
                    yq[counter]+=NNNV[k]*y[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for

                if particle_projection==1:
                   rhoq[counter]=rho_elemental[iel]
                   etaq[counter]=eta_elemental[iel]
                elif particle_projection==2:
                   for k in range(0,mP):
                       rhoq[counter]+=NNNP[k]*rho_nodal[iconP[k,iel]]
                       etaq[counter]+=NNNP[k]*eta_nodal[iconP[k,iel]]
                   #end for
                elif particle_projection==3:
                   for k in range(0,mP):
                       rhoq[counter]+=rho_nodal[iconP[k,iel]]*0.25
                       etaq[counter]+=eta_nodal[iconP[k,iel]]*0.25
                   #end for
                elif particle_projection==4:
                   for k in range(0,mP):
                       rhoq[counter]+=NNNP[k]*rho_nodal[iconP[k,iel]]
                   etaq[counter]=eta_elemental[iel]
                #end if

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq[counter],yq[counter])*rhoq[counter]
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq[counter],yq[counter])*rhoq[counter]
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.
                #end for

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                NNNNP[:]+=NNNP[:]*jcob*weightq
                counter+=1
            #end for
        #end for

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
                #end for
            #end for
        #end for

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
                        #end if
                        #K_mat[m1,m2]+=K_el[ikk,jkk] OLD
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    if sparse:
                       A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                       A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                    else:
                       G_mat[m1,m2]+=G_el[ikk,jkk]
                    #end if
                    #G_mat[m1,m2]+=G_el[ikk,jkk] old
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for

        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
            constr[m2]+=NNNNP[k2]
        #end for

    #end for

    #G_mat*=eta_ref/Ly
    #h_rhs*=eta_ref/Ly

    if not sparse:
       print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
       print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    #np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
    #np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')

    print("build FE matrix: %.3f s" % (time.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = time.time()

    rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b

    if not sparse:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
       a_mat[Nfem,NfemV:Nfem]=constr
       a_mat[NfemV:Nfem,Nfem]=constr
    else:
       for i in range(0,NfemP):
           A_sparse[Nfem,NfemV+i]=constr[i]
           A_sparse[NfemV+i,Nfem]=constr[i]

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (time.time() - start))

    ######################################################################
    # convert to csr
    ######################################################################
    start = time.time()

    if sparse:
       sparse_matrix=A_sparse.tocsr()
    else:
       sparse_matrix=sps.csr_matrix(a_mat)

    print("convert from lil to csr format: %.3f s" % (time.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = time.time()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (time.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))
    
    vel_file.write("%e %e %e %e %e\n" %(Time,np.min(u),np.max(u),np.min(v),np.max(v))) ; vel_file.flush()

    np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute timestep 
    ######################################################################

    dt=CFL_nb*min(hx_min,hy_min)/max(max(abs(u)),max(abs(v)))

    print("dt= %.3e yr" %(dt/year))

    dt_file.write("%d %e \n" %(istep,dt)) ; dt_file.flush()

    #################################################################
    # compute vrms 
    #################################################################
    start = time.time()

    vrms=0.
    mass=0.
    counterq=0
    for iel in range (0,nel):
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*y[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*y[iconV[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for
                uq=0.
                vq=0.
                for k in range(0,mV):
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                #end for
                #rhoq=0.
                #for k in range(0,mP):
                #    rhoq+=NNNP[k]*rho_nodal[iconP[k,iel]]
                #end for
                vrms+=(uq**2+vq**2)*weightq*jcob
                mass+=rhoq[counterq]*weightq*jcob
                counterq+=1
            #end for
        #end for
    #end for

    vrms=np.sqrt(vrms/(Lx*Ly))

    vrms_file.write("%e %e %e %e\n" %(Time,vrms,Time/year,vrms/cm*year)) ; vrms_file.flush()
    mass_file.write("%e %e %e %e %e\n" %(Time,mass,mass0,mass/mass0-1.,Time/year)) ; mass_file.flush()

    print("     -> vrms %.5e " %vrms)
    print("     -> mass %.5e " %mass)

    print("compute vrms & mass: %.3f s" % (time.time() - start))

    ######################################################################
    # advect particles 
    ######################################################################
    start = time.time()

    if rk==1:
       for im in range(0,nparticle):
           #-----
           if use_stretching_x:
              for ielx in range(0,nelx):
                  xmin=x[iconV[0,ielx]]
                  xmax=x[iconV[1,ielx]]
                  hx=xmax-xmin
                  if swarm_x[im]>xmin and swarm_x[im]<xmax: break
           else:
              ielx=int(swarm_x[im]/Lx*nelx)
           #-----
           if use_stretching_y:
              for iely in range(0,nely):
                  iel=iely*nelx
                  ymin=y[iconV[0,iel]]
                  ymax=y[iconV[2,iel]]
                  hy=ymax-ymin
                  if swarm_y[im]>ymin and swarm_y[im]<ymax: break
           else:
              iely=int(swarm_y[im]/Ly*nely)
           #-----
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
       #end for
       #np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_iel,swarm_r,swarm_s]).T)
       #exit()

    elif rk==2:
       if use_stretching_x or use_stretching_y: exit('rk2 not available with stretching')
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xm=swarm_x[im]+um*dt/2
           ym=swarm_y[im]+vm*dt/2

           ielx=int(xm/Lx*nelx)
           iely=int(ym/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(xm-x0)/hx
           swarm_s[im]=-1+2*(ym-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
       #end for

    elif rk==3:
       if use_stretching_x or use_stretching_y: exit('rk3 not available with stretching')
       for im in range(0,nparticle):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           uA=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vA=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xB=swarm_x[im]+uA*dt/2
           yB=swarm_y[im]+vA*dt/2

           ielx=int(xB/Lx*nelx)
           iely=int(yB/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(xB-x0)/hx
           s=-1+2*(yB-y0)/hy
           NNNV[0:mV]=NNV(r,s)
           uB=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vB=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           xC=swarm_x[im]+(2*uB-uA)*dt/2
           yC=swarm_y[im]+(2*vB-vA)*dt/2

           ielx=int(xC/Lx*nelx)
           iely=int(yC/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           swarm_r[im]=-1+2*(xC-x0)/hx
           swarm_s[im]=-1+2*(yC-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           uC=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vC=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=(uA+4*uB+uC)*dt/6
           swarm_y[im]+=(vA+4*vB+vC)*dt/6
       #end for

    #end if

    print("advect particles: %.3f s" % (time.time() - start))

    ######################################################################
    # advect markers
    ######################################################################
    start = time.time()

    if nmarker>0:
       if rk==1:
          if use_stretching_x or use_stretching_y: exit('not available with stretching')
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              #print (ielx,iely,iel)
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              m_r[im]=rm
              m_s[im]=sm
              NNNV[0:mV]=NNV(rm,sm)
              m_u[im]=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              m_v[im]=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_x[im]+=m_u[im]*dt
              m_y[im]+=m_v[im]*dt
          #end for

       elif rk==2:
          if use_stretching_x or use_stretching_y: exit('not available with stretching')
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              NNNV[0:mV]=NNV(rm,sm)
              um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              xm=m_x[im]+um*dt/2
              ym=m_y[im]+vm*dt/2

              ielx=int(xm/Lx*nelx)
              iely=int(ym/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              m_r[im]=-1+2*(xm-x0)/hx
              m_s[im]=-1+2*(ym-y0)/hy
              NNNV[0:mV]=NNV(m_r[im],m_s[im])
              m_u[im]=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              m_v[im]=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_x[im]+=m_u[im]*dt
              m_y[im]+=m_v[im]*dt
          #end for

       elif rk==3:
          if use_stretching_x or use_stretching_y: exit('not available with stretching')
          for im in range(0,nmarker):
              ielx=int(m_x[im]/Lx*nelx)
              iely=int(m_y[im]/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              rm=-1.+2*(m_x[im]-x0)/hx
              sm=-1.+2*(m_y[im]-y0)/hy
              NNNV[0:mV]=NNV(rm,sm)
              uA=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vA=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              xB=m_x[im]+uA*dt/2
              yB=m_y[im]+vA*dt/2

              ielx=int(xB/Lx*nelx)
              iely=int(yB/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              r=-1+2*(xB-x0)/hx
              s=-1+2*(yB-y0)/hy
              NNNV[0:mV]=NNV(r,s)
              uB=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vB=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              xC=m_x[im]+(2*uB-uA)*dt/2
              yC=m_y[im]+(2*vB-vA)*dt/2

              ielx=int(xC/Lx*nelx)
              iely=int(yC/Ly*nely)
              iel=nelx*(iely)+ielx
              x0=x[iconV[0,iel]]
              y0=y[iconV[0,iel]]
              m_r[im]=-1+2*(xC-x0)/hx
              m_s[im]=-1+2*(yC-y0)/hy
              NNNV[0:mV]=NNV(m_r[im],m_s[im])
              uC=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
              vC=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
              m_u[im]=uC
              m_v[im]=vC
              m_x[im]+=(uA+4*uB+uC)*dt/6
              m_y[im]+=(vA+4*vB+vC)*dt/6
          #end for
       #end if
       filename = 'markers_{:04d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([m_x,m_y,m_u,m_v]).T,header='# x,y')
    #end if

    #####################################################################
    # compute nodal strainrate and heat flux 
    #
    # 3--6--2
    # |  |  |
    # 7--8--5
    # |  |  |
    # 0--4--1
    #####################################################################
    start = time.time()

    rVnodes=[-1,+1,+1,-1,0,+1,0,-1,0]
    sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]
    
    exx_n = np.zeros(NV,dtype=np.float64)  
    eyy_n = np.zeros(NV,dtype=np.float64)  
    exy_n = np.zeros(NV,dtype=np.float64)  
    count = np.zeros(NV,dtype=np.int32)  
    q=np.zeros(NV,dtype=np.float64)
    rho_Q2=np.zeros(NV,dtype=np.float64)
    eta_Q2=np.zeros(NV,dtype=np.float64)
    mats_Q2=np.zeros((nmat,NV),dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*x[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*y[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*x[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*y[iconV[k,iel]]
            #end for
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            e_xx=0.
            e_yy=0.
            e_xy=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            rho_Q2[inode]+=np.dot(rho_nodal[iconP[0:mP,iel]],NNNP[0:mP])
            eta_Q2[inode]+=np.dot(eta_nodal[iconP[0:mP,iel]],NNNP[0:mP])
            for imat in range(0,nmat):
                mats_Q2[imat,inode]+=np.dot(mat_nodal[imat,iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    rho_Q2/=count
    eta_Q2/=count
    for imat in range(0,nmat):
        mats_Q2[imat,:]/=count
    q/=count

    print("     -> exx nodal (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy nodal (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy nodal (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> press nodal (m,M) %.5e %.5e " %(np.min(q),np.max(q)))
    print("     -> rho nodal (m,M) %.5e %.5e " %(np.min(rho_Q2),np.max(rho_Q2)))
    print("     -> eta nodal (m,M) %.5e %.5e " %(np.min(eta_Q2),np.max(eta_Q2)))

    #np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')
    #np.savetxt('rho_Q2.ascii',np.array([x,y,rho_Q2]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute nodal press & sr: %.3f s" % (time.time() - start))

    #####################################################################
    # export solution to vtu files
    #####################################################################

    start = time.time()
    
    if istep%every==0:

       filename = 'mesh_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" %(xP[i],yP[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       if particle_projection > 1:
          vtufile.write("<PointData Scalars='scalars'>\n")
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for i in range(0,NP):
              vtufile.write("%10e \n" %rho_nodal[i])
          vtufile.write("</DataArray>\n")
          vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='nparticle' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (nparticle_in_element[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (eta_elemental[iel]))
       vtufile.write("</DataArray>\n")
       #--
       if particle_projection == 1:
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%10e\n" % (rho_elemental[iel]))
          vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel],iconP[3,iel]))
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

       filename = 'solution_4xQ1_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       if particle_projection > 1:
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          for i in range(0,NV):
              vtufile.write("%10e \n" %rho_Q2[i])
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
          for i in range(0,NV):
              vtufile.write("%10e \n" %eta_Q2[i])
          vtufile.write("</DataArray>\n")
          #--
       for imat in range(0,nmat):
           vtufile.write("<DataArray type='Float32' Name='mat %d' Format='ascii'> \n" %imat)
           for i in range(0,NV):
               vtufile.write("%10e \n" %mats_Q2[imat,i])
           vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       for i in range(0,NV):
           e=np.sqrt(0.5*(exx_n[i]*exx_n[i]+eyy_n[i]*eyy_n[i])+exy_n[i]*exy_n[i])
           vtufile.write("%10e \n" %e)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       filename = 'particles_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%3e \n" %swarm_mat[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%3e \n" %swarm_paint[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nparticle):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nparticle):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nparticle):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       if nmarker>0:
          filename = 'markers_{:04d}.vtu'.format(istep) 
          vtufile=open(filename,"w")
          vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
          vtufile.write("<UnstructuredGrid> \n")
          vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker-1))
          vtufile.write("<Points> \n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
          for i in range(0,nmarker):
              vtufile.write("%.10e %.10e %.10e \n" %(m_x[i],m_y[i],0.))
          vtufile.write("</DataArray>\n")
          vtufile.write("</Points> \n")
          vtufile.write("<PointData Scalars='scalars'>\n")
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
          for i in range(0,nmarker):
              vtufile.write("%10e %10e %10e \n" %(m_u[i],m_v[i],0.))
          vtufile.write("</DataArray>\n")
          vtufile.write("</PointData>\n")
          vtufile.write("<Cells>\n")
          vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
          for i in range(0,nmarker-1):
              vtufile.write("%d %d " % (i,i+1))
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
          for i in range(0,nmarker):
              vtufile.write("%d " % ((i+1)*2))
          vtufile.write("</DataArray>\n")
          vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
          for i in range(0,nmarker):
              vtufile.write("%d " % 3)
          vtufile.write("</DataArray>\n")
          vtufile.write("</Cells>\n")
          vtufile.write("</Piece>\n")
          vtufile.write("</UnstructuredGrid>\n")
          vtufile.write("</VTKFile>\n")
          vtufile.close()

       filename = 'qpts_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))

       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%3e \n" %rhoq[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%3e \n" %etaq[i])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")

       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%10e %10e %10e \n" %(xq[i],yq[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nq):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nq):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nq):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


    print("write vtu files: %.3f s" % (time.time() - start))

    Time+=dt

#end for istep

#--------------------------------------------------------------------------------------------------
# end time stepping loop
#--------------------------------------------------------------------------------------------------

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
