import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def bx(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(y*z+3*x**2*y**3*z) + mu * (2+6*x*y) \
        +(2+4*x+2*y+6*x**2*y) * mux \
        +(x+x**3+y+2*x*y**2 ) * muy \
        +(-3*z-10*x*y*z     ) * muz
    return val

def by(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(x*z+3*x**3*y**2*z) + mu * (2 +2*x**2 + 2*y**2) \
       +(x+x**3+y+2*x*y**2   ) * mux \
       +(2+2*x+4*y+4*x**2*y  ) * muy \
       +(-3*z-5*x**2*z       ) * muz 
    return val

def bz(x,y,z,beta):
    mu=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    mux=-beta*(1-2*x)*mu
    muy=-beta*(1-2*y)*mu
    muz=-beta*(1-2*z)*mu
    val=-(x*y+x**3*y**3) + mu * (-10*y*z) \
       +(-3*z-10*x*y*z        ) * mux \
       +(-3*z-5*x**2*z        ) * muy \
       +(-4-6*x-6*y-10*x**2*y ) * muz 
    return val

def mu(x,y,z,beta):
    val=np.exp(1-beta*(x*(1-x)+y*(1-y)+z*(1-z)) )
    return val

def uth(x,y,z):
    val=x+x*x+x*y+x*x*x*y
    return val

def vth(x,y,z):
    val=y+x*y+y*y+x*x*y*y
    return val

def wth(x,y,z):
    val=-2*z-3*x*z-3*y*z-5*x*x*y*z
    return val

def pth(x,y,z):
    val=x*y*z+x*x*x*y*y*y*z-5/32
    return val

def exx_th(x,y,z):
    val=1+2*x+y+3*x*x*y
    return val

def eyy_th(x,y,z):
    val=1+x+2*y+2*x*x*y
    return val

def ezz_th(x,y,z):
    val=-2-3*x-3*y-5*x*x*y
    return val

def exy_th(x,y,z):
    val=(x+y+2*x*y*y+x*x*x)/2
    return val

def exz_th(x,y,z):
    val=(-3*z-10*x*y*z)/2
    return val

def eyz_th(x,y,z):
    val=(-3*z-5*x*x*z)/2
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 17--------")
print("-----------------------------")

m=8      # number of nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 
assert (Lz>0.), "Lz should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx = 13
   nely = 13
   nelz = 13

assert (nelx>0.), "nelx should be positive" 
assert (nely>0.), "nely should be positive" 
assert (nelz>0.), "nelz should be positive" 

visu=1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

sqrt3=np.sqrt(3.)

beta=0

######################################################################
# grid point setup
######################################################################
start = time.time()

x = np.empty(NV, dtype=np.float64)  # x coordinates
y = np.empty(NV, dtype=np.float64)  # y coordinates
z = np.empty(NV, dtype=np.float64)  # z coordinates

counter=0
for i in range(0, nnx):
    for j in range(0, nny):
        for k in range(0, nny):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)
            counter += 1

print("grid points setup: %.3f s" % (time.time() - start))

######################################################################
# connectivity
######################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for i in range(0, nelx):
    for j in range(0, nely):
        for k in range(0, nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1

print("build connectivity: %.3f s" % (time.time() - start))

######################################################################
# define boundary conditions
######################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=float)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
    if y[i]<eps:
       bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
       bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
       bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
    if y[i]>(Ly-eps):
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

print("define b.c.: %.3f s" % (time.time() - start))

######################################################################
# build FE matrix
######################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
w     = np.zeros(NV,dtype=np.float64)          # y-component velocity
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat = np.zeros((6,6),dtype=np.float64) 

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:
            for kq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                wq=1.*1.*1.

                # calculate shape functions
                N[0]=0.125*(1-rq)*(1-sq)*(1-tq)
                N[1]=0.125*(1+rq)*(1-sq)*(1-tq)
                N[2]=0.125*(1+rq)*(1+sq)*(1-tq)
                N[3]=0.125*(1-rq)*(1+sq)*(1-tq)
                N[4]=0.125*(1-rq)*(1-sq)*(1+tq)
                N[5]=0.125*(1+rq)*(1-sq)*(1+tq)
                N[6]=0.125*(1+rq)*(1+sq)*(1+tq)
                N[7]=0.125*(1-rq)*(1+sq)*(1+tq)

                # calculate shape function derivatives
                dNdr[0]=-0.125*(1-sq)*(1-tq) 
                dNdr[1]=+0.125*(1-sq)*(1-tq)
                dNdr[2]=+0.125*(1+sq)*(1-tq)
                dNdr[3]=-0.125*(1+sq)*(1-tq)
                dNdr[4]=-0.125*(1-sq)*(1+tq)
                dNdr[5]=+0.125*(1-sq)*(1+tq)
                dNdr[6]=+0.125*(1+sq)*(1+tq)
                dNdr[7]=-0.125*(1+sq)*(1+tq)

                dNds[0]=-0.125*(1-rq)*(1-tq) 
                dNds[1]=-0.125*(1+rq)*(1-tq)
                dNds[2]=+0.125*(1+rq)*(1-tq)
                dNds[3]=+0.125*(1-rq)*(1-tq)
                dNds[4]=-0.125*(1-rq)*(1+tq)
                dNds[5]=-0.125*(1+rq)*(1+tq)
                dNds[6]=+0.125*(1+rq)*(1+tq)
                dNds[7]=+0.125*(1-rq)*(1+tq)

                dNdt[0]=-0.125*(1-rq)*(1-sq) 
                dNdt[1]=-0.125*(1+rq)*(1-sq)
                dNdt[2]=-0.125*(1+rq)*(1+sq)
                dNdt[3]=-0.125*(1-rq)*(1+sq)
                dNdt[4]=+0.125*(1-rq)*(1-sq)
                dNdt[5]=+0.125*(1+rq)*(1-sq)
                dNdt[6]=+0.125*(1+rq)*(1+sq)
                dNdt[7]=+0.125*(1-rq)*(1+sq)

                # calculate jacobian matrix
                jcb=np.zeros((3,3),dtype=np.float64)
                for k in range(0,m):
                    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                    jcb[0, 2] += dNdr[k]*z[icon[k,iel]]
                    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                    jcb[1, 1] += dNds[k]*y[icon[k,iel]]
                    jcb[1, 2] += dNds[k]*z[icon[k,iel]]
                    jcb[2, 0] += dNdt[k]*x[icon[k,iel]]
                    jcb[2, 1] += dNdt[k]*y[icon[k,iel]]
                    jcb[2, 2] += dNdt[k]*z[icon[k,iel]]

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx, dNdy, dNdz
                xq=0.0
                yq=0.0
                zq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    zq+=N[k]*z[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*mu(xq,yq,zq,beta)*wq*jcob

                for i in range(0, m):
                    f_el[ndofV*i+0]+=N[i]*jcob*wq*bx(xq,yq,zq,beta)
                    f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq,zq,beta)
                    f_el[ndofV*i+2]+=N[i]*jcob*wq*bz(xq,yq,zq,beta)
                    G_el[ndofV*i+0,0]-=dNdx[i]*jcob*wq
                    G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq
                    G_el[ndofV*i+2,0]-=dNdz[i]*jcob*wq

    # impose b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem+1,dtype=np.float64)         # right hand side of Ax=b

a_mat[0:NfemV,0:NfemV]=K_mat
a_mat[0:NfemV,NfemV:Nfem]=G_mat
a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

a_mat[Nfem,NfemV:Nfem]=1
a_mat[NfemV:Nfem,Nfem]=1

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

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

u,v,w=np.reshape(sol[0:NfemV],(NV,3)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.4f %.4f " %(np.min(w),np.max(w)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################
# compute strainrate 
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    wq=2.*2.*2.

    N[0]=0.125*(1.-rq)*(1.-sq)*(1.-tq)
    N[1]=0.125*(1.+rq)*(1.-sq)*(1.-tq)
    N[2]=0.125*(1.+rq)*(1.+sq)*(1.-tq)
    N[3]=0.125*(1.-rq)*(1.+sq)*(1.-tq)
    N[4]=0.125*(1.-rq)*(1.-sq)*(1.+tq)
    N[5]=0.125*(1.+rq)*(1.-sq)*(1.+tq)
    N[6]=0.125*(1.+rq)*(1.+sq)*(1.+tq)
    N[7]=0.125*(1.-rq)*(1.+sq)*(1.+tq)

    dNdr[0]=-0.125*(1.-sq)*(1.-tq) 
    dNdr[1]=+0.125*(1.-sq)*(1.-tq)
    dNdr[2]=+0.125*(1.+sq)*(1.-tq)
    dNdr[3]=-0.125*(1.+sq)*(1.-tq)
    dNdr[4]=-0.125*(1.-sq)*(1.+tq)
    dNdr[5]=+0.125*(1.-sq)*(1.+tq)
    dNdr[6]=+0.125*(1.+sq)*(1.+tq)
    dNdr[7]=-0.125*(1.+sq)*(1.+tq)

    dNds[0]=-0.125*(1.-rq)*(1.-tq) 
    dNds[1]=-0.125*(1.+rq)*(1.-tq)
    dNds[2]=+0.125*(1.+rq)*(1.-tq)
    dNds[3]=+0.125*(1.-rq)*(1.-tq)
    dNds[4]=-0.125*(1.-rq)*(1.+tq)
    dNds[5]=-0.125*(1.+rq)*(1.+tq)
    dNds[6]=+0.125*(1.+rq)*(1.+tq)
    dNds[7]=+0.125*(1.-rq)*(1.+tq)

    dNdt[0]=-0.125*(1.-rq)*(1.-sq) 
    dNdt[1]=-0.125*(1.+rq)*(1.-sq)
    dNdt[2]=-0.125*(1.+rq)*(1.+sq)
    dNdt[3]=-0.125*(1.-rq)*(1.+sq)
    dNdt[4]=+0.125*(1.-rq)*(1.-sq)
    dNdt[5]=+0.125*(1.+rq)*(1.-sq)
    dNdt[6]=+0.125*(1.+rq)*(1.+sq)
    dNdt[7]=+0.125*(1.-rq)*(1.+sq)

    # calculate jacobian matrix
    jcb=np.zeros((3,3),dtype=np.float64)
    for k in range(0,m):
        jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
        jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
        jcb[0, 2] += dNdr[k]*z[icon[k,iel]]
        jcb[1, 0] += dNds[k]*x[icon[k,iel]]
        jcb[1, 1] += dNds[k]*y[icon[k,iel]]
        jcb[1, 2] += dNds[k]*z[icon[k,iel]]
        jcb[2, 0] += dNdt[k]*x[icon[k,iel]]
        jcb[2, 1] += dNdt[k]*y[icon[k,iel]]
        jcb[2, 2] += dNdt[k]*z[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]

    for k in range(0, m):
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        zc[iel]+=N[k]*z[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        ezz[iel]+=dNdz[k]*w[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]
        exz[iel]+=0.5*dNdz[k]*u[icon[k,iel]]+0.5*dNdx[k]*w[icon[k,iel]]
        eyz[iel]+=0.5*dNdz[k]*v[icon[k,iel]]+0.5*dNdy[k]*w[icon[k,iel]]

    visc[iel]=mu(xc[iel],yc[iel],zc[iel],beta)
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                    +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("ezz (m,M) %.4f %.4f " %(np.min(ezz),np.max(ezz)))
print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("exz (m,M) %.4f %.4f " %(np.min(exz),np.max(exz)))
print("eyz (m,M) %.4f %.4f " %(np.min(eyz),np.max(eyz)))
print("visc (m,M) %.4f %.4f " %(np.min(visc),np.max(visc)))

np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute strainrate: %.3f s" % (time.time() - start))

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
   vtufile.write("<DataArray type='Float32' Name='p (A.S.)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pth(xc[iel],yc[iel],zc[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % visc[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f %f %f %f %f %f\n" % (exx[iel], eyy[iel], ezz[iel], exy[iel], eyz[iel], exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (A.S.)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f %f %f %f %f %f\n" % (exx_th(xc[iel],yc[iel],zc[iel]), \
                                              eyy_th(xc[iel],yc[iel],zc[iel]), \
                                              ezz_th(xc[iel],yc[iel],zc[iel]), \
                                              exy_th(xc[iel],yc[iel],zc[iel]), \
                                              eyz_th(xc[iel],yc[iel],zc[iel]), \
                                              exz_th(xc[iel],yc[iel],zc[iel]) ))
   vtufile.write("</DataArray>\n")
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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (A.S.)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(uth(x[i],y[i],z[i]), vth(x[i],y[i],z[i]), wth(x[i],y[i],z[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
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




