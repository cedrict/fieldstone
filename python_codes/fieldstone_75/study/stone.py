import numpy as np
import sys as sys
import scipy
import time as timing
from numpy.linalg import matrix_rank
from scipy.linalg import null_space

np.set_printoptions(linewidth=220)                                                                                                                                                                               
#------------------------------------------------------------------------------
# only bubble 1 and 2 are real (they come from Lamichhane 2017) 
# but I cannot show they yield a stable element.
# The others are tryouts
#------------------------------------------------------------------------------

def BB(r,s,t):
    return 1 + a1*r + b1*s + c1*t +\
           a2* r**2 + b2*s**2 + c2*t**2 + \
           d2* r*s + e2*s*t + f2*r*t +\
           a3* r**3 + b3*s**3 + c3*t**3 +\
           d3* r**2*s + e3*r**2*t + f3*r*s**2 + g3*s**2*t + h3*r*t**2 + i3*s*t**2 +\
           j3* r*s*t 

def dBBdr(r,s,t):
    return a1 + 2*a2*r + d2*s + f2*t + 3*a3*r**2 + 2*d3*r*s + 2*e3*r*t + f3*s**2 + h3*t**2 + j3*s*t

def dBBds(r,s,t):
    return b1 + 2*b2*s + d2*r + e2*t + 3*b3*s**2 + d3*r**2 + 2*f3*r*s + 2*g3*s*t  + i3*t**2 + j3*r*t  

def dBBdt(r,s,t):
    return c1 + 2*c2*t + e2*s + f2*r + 3*c3*t**2 + e3*r**2  + g3*s**2  + 2*h3*r*t + 2*i3*s*t + j3*r*s

#------------------------------------------------------------------------------

def B(r,s,t):
    if bubble==0:
       return (1-r**2)*(1-s**2)*(1-t**2) 
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1-r)*(1-s)*(1-t)
    if bubble==2:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1+beta*(r+s+t))
    if bubble==4:
       return (1-r**2)*(1-s**2)*(1-t**2)*(1+aa*r+bb*s+cc*t) 
    if bubble==5:
       return (1-r**2)*(1-s**2)*(1-t**2)*(1-aa*r)*(1-bb*s)*(1-cc*t)
    if bubble==6:
       return (1-r**2)**2*(1-s**2)**2*(1-t**2)**2
    if bubble==7:
       return (1-r**2)**3*(1-s**2)**3*(1-t**2)**3
    if bubble==8:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1+r+s+t+r*s+r*t+s*t+r*s*t)
    if bubble==9:
       return (1-r**2)*(1-s**2)*(1-t**2) * (1+aa*r**2+bb*s**2+cc*t**2)
    if bubble==10:
       return (1-r**2)*(1-s**2)*(1-t**2) * (aa*r*s + bb*s*t + cc*r*t)
    if bubble==11:
       return (1-r**2)*(1-s**2)*(1-t**2) * BB(r,s,t) 
    if bubble==12:
       return np.cos(0.5*np.pi*r)*np.cos(0.5*np.pi*s)*np.cos(0.5*np.pi*t)

def dBdr(r,s,t):
    if bubble==0:
       return (-2*r)*(1-s**2)*(1-t**2) 
    if bubble==1:
       return (1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2)
    if bubble==2:
       return (1-s**2)*(1-t**2)*(-beta*(3*r**2+2*r*(s+t)-1)+2*r) 
    if bubble==4:
       return -(1-s**2)*(1-t**2)*(aa*(3*r**2-1)+2*r*(bb*s+cc*t+1))
    if bubble==5:
       return (1-s**2)*(1-t**2)*(aa*(3*r**2-1)-2*r)*(1-bb*s)*(1-cc*t) 
    if bubble==6:
       return -4*r*(1-r**2)*(1-s**2)**2*(1-t**2)**2
    if bubble==7:
       return -6*r*(1-r**2)**2*(1-s**2)**3*(1-t**2)**3
    if bubble==8:
       return (1-s**2)*(1-t**2) * (-3*r**2*(s+t+1)-2*r*(s+1)*(t+1)+s+t+1)
    if bubble==9:
       return -2*r*(1-s**2)*(1-t**2)*(aa*(2*r**2-1)+bb*s**2+cc*t**2+1)
    if bubble==10:
       return -(1-s**2)*(1-t**2)*(aa*(3*r**2-1)*s+2*bb*r*s*t+cc*(3*r**2-1)*t)
    if bubble==11:
       return (1-s**2)*(1-t**2)*(-2*r*BB(r,s,t)+(1-r**2)*dBBdr(r,s,t) )
    if bubble==12:
       return -0.5*np.pi*np.sin(0.5*np.pi*r)*np.cos(0.5*np.pi*s)*np.cos(0.5*np.pi*t)

def dBds(r,s,t):
    if bubble==0:
       return (1-r**2)*(-2*s)*(1-t**2) 
    if bubble==1:
       return (1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2) 
    if bubble==2:
       return (1-r**2)*(1-t**2)*(-beta*(3*s**2+2*s*(r+t)-1)+2*s) 
    if bubble==4:
       return -(1-r**2)*(1-t**2)*(bb*(3*s**2-1)+2*s*(aa*r+cc*t+1))
    if bubble==5:
       return (1-r**2)*(1-t**2)*(bb*(3*s**2-1)-2*s)*(1-aa*r)*(1-cc*t) 
    if bubble==6:
       return -4*s*(1-s**2)*(1-r**2)**2*(1-t**2)**2
    if bubble==7:
       return -6*s*(1-r**2)**3*(1-s**2)**2*(1-t**2)**3
    if bubble==8:
       return (1-r**2)*(1-t**2)*(-(s**2-1)*(r+t+1)-2*s*(r*(s+t+1)+(s+1)*(t+1))  )
    if bubble==9:
       return -2*s*(1-r**2)*(1-t**2)*(aa*r**2+bb*(2*s**2-1)+cc*t**2+1)
    if bubble==10:
       return -(1-r**2)*(1-t**2)*(aa*(3*s**2-1)*r+bb*(3*s**2-1)*t+2*cc*r*s*t)
    if bubble==11:
       return (1-r**2)*(1-t**2)*(-2*s*BB(r,s,t)+(1-s**2)*dBBds(r,s,t) )
    if bubble==12:
       return -0.5*np.pi*np.cos(0.5*np.pi*r)*np.sin(0.5*np.pi*s)*np.cos(0.5*np.pi*t)

def dBdt(r,s,t):
    if bubble==0:
       return (1-r**2)*(1-s**2)*(-2*t) 
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)*(-1-2*t+3*t**2) 
    if bubble==2:
       return (1-r**2)*(1-s**2)*(-beta*(3*t**2+2*t*(r+s)-1)+2*t) 
    if bubble==4:
       return -(1-r**2)*(1-s**2)*(cc*(3*t**2-1)+2*t*(aa*r+bb*s+1))
    if bubble==5:
       return (1-r**2)*(1-s**2)*(cc*(3*t**2-1)-2*t)*(1-aa*r)*(1-bb*s) 
    if bubble==6:
       return -4*t*(1-t**2)*(1-r**2)**2*(1-s**2)**2
    if bubble==7:
       return -6*t*(1-r**2)**3*(1-s**2)**3*(1-t**2)**2
    if bubble==8:
       return (1-r**2)*(1-s**2)*( (t**2-1)*(-(r+s+1))-2*t*(r*(s+t+1)+(s+1)*(t+1)))
    if bubble==9:
       return -2*t*(1-r**2)*(1-s**2)*(aa*r**2+bb*s**2+cc*(2*t**2-1)+1)
    if bubble==10:
       return -(1-r**2)*(1-s**2)*(2*aa*r*s*t+bb*(3*t**2-1)*s+cc*(3*t**2-1)*r)
    if bubble==11:
       return (1-r**2)*(1-s**2)*(-2*t*BB(r,s,t)+(1-t**2)*dBBdt(r,s,t) )
    if bubble==12:
       return -0.5*np.pi*np.cos(0.5*np.pi*r)*np.cos(0.5*np.pi*s)*np.sin(0.5*np.pi*t)

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

if int(len(sys.argv) == 20):
   a1=float(sys.argv[1])
   b1=float(sys.argv[2])
   c1=float(sys.argv[3])
   a2=float(sys.argv[4])
   b2=float(sys.argv[5])
   c2=float(sys.argv[6])
   d2=float(sys.argv[7])
   e2=float(sys.argv[8])
   f2=float(sys.argv[9])
   a3=float(sys.argv[10])
   b3=float(sys.argv[11])
   c3=float(sys.argv[12])
   d3=float(sys.argv[13])
   e3=float(sys.argv[14])
   f3=float(sys.argv[15])
   g3=float(sys.argv[16])
   h3=float(sys.argv[17])
   i3=float(sys.argv[18])
   j3=float(sys.argv[19])
   #print(a1,b1,c1)
   #print(a2,b2,c2,d2,e2,f2)
   #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


nelx =2  # do not exceed 20 
nely =nelx
nelz =nelx
nqperdim=3

bubble=11
    
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

beta=0.125
aa=1.15
bb=2.3
cc=-1.25

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

rVnodes=[-1,+1,+1,-1,-1,+1,+1,-1,0]
sVnodes=[-1,-1,+1,+1,-1,-1,+1,+1,0]
tVnodes=[-1,-1,-1,-1,+1,+1,+1,+1,0]

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

######################################################################
# define boundary conditions
######################################################################
start = timing.time()

bc_fix=np.zeros(Nfem,dtype=bool)    # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps or xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0
    if yV[i]/Ly<eps or yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0
    if zV[i]/Lz<eps or zV[i]/Lz>(1-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0
#end for

print("define b.c.: %.3f s" % (timing.time() - start))

######################################################################
# build FE matrix
######################################################################
start = timing.time()

G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

f_rhs = np.zeros(NfemV,dtype=np.float64)          # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)          # right hand side h 
b_mat = np.zeros((6,ndofV*mV),dtype=np.float64)   # gradient matrix B 
u     = np.zeros(NV,dtype=np.float64)             # x-component velocity
v     = np.zeros(NV,dtype=np.float64)             # y-component velocity
w     = np.zeros(NV,dtype=np.float64)             # z-component velocity
p     = np.zeros(nel,dtype=np.float64)            # pressure 
c_mat = np.zeros((6,6),dtype=np.float64)          # C matrix 
N_mat   = np.zeros((6,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdz = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdt = np.zeros(mV,dtype=np.float64)           # shape functions derivatives


for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)

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

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=NNNP[i]
                    N_mat[3,i]=0.
                    N_mat[4,i]=0.
                    N_mat[5,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob


            #end for kq
        #end for jq
    #end for iq

    #G_el*=1350
    #for i in range(27):
    #    print(" %7f %7f %f %f %.7f %.7f %.7f %.7f" %(G_el[i,0],G_el[i,1],G_el[i,2],G_el[i,3],G_el[i,4],G_el[i,5],G_el[i,6],G_el[i,7]))
    #print(G_el)

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               #K_ref=K_el[ikk,ikk] 
               #for jkk in range(0,mV*ndofV):
               #    f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
               #    K_el[ikk,jkk]=0
               #    K_el[jkk,ikk]=0
               #K_el[ikk,ikk]=K_ref
               #f_el[ikk]=K_ref*bc_val[m1]
               #h_el[:]-=G_el[ikk,:]*bc_val[m1]
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
            #for k2 in range(0,mV):
            #    for i2 in range(0,ndofV):
            #        jkk=ndofV*k2          +i2
            #        m2 =ndofV*iconV[k2,iel]+i2
            #        if sparse:
            #           A_sparse[m1,m2] += K_el[ikk,jkk]
            #        else:
            #           K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]

#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))

######################################################################

for i in range(NfemV):
    for j in range(NfemP):
        if abs(G_mat[i,j])<1e-15:
           G_mat[i,j]=0

######################################################################
#print ("------------------------------------------------------")

#for i in range(NfemV):
#    print(i,G_mat[i,0:NfemP])

#print ("------------------------------------------------------")

#for i in range(NfemV):
#    if max(G_mat[i,:])>1e-15:
#       print(i,G_mat[i,0:NfemP])
#print ("------------------------------------------------------")


#print(G2.astype(int))
#G_mat*=27*50*16

G2 = np.zeros((27,NfemP),dtype=np.float64) 
G2[0:3,:]=G_mat[39:42,:] 
G2[3:,:]=G_mat[81:,:]    
ns = null_space(G2)
#print(ns)
opla=ns.shape
#print(ns.shape)
print('size of nullspace=',opla[1])

#print ("------------------------------------------------------")
#print('beta=',beta,'rank=',matrix_rank(G2))
#print('beta=',beta,'rank=',matrix_rank(G2.T))

exit()










#print('===============================================bubble = ',bubble,'================')
#G2[0:3,:]=np.round(G_mat[39:42,:], decimals=4)
#G2[3:,:]=np.round(G_mat[81:,:], decimals=4)
#print(G2)

#G3 = np.zeros((27,NfemP),dtype=np.int32) 
#for i in range(27):
#    for j in range(NfemP):
#        G3[i,j]=int(round(G2[i,j]))

#print('=========================G3=')
#print(G3)
#ns = null_space(G3)
#print(np.round(ns,decimals=4))
#print(ns.shape)


