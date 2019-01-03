import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import random

#------------------------------------------------------------------------------

def bx(x, y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val
def by(x, y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def uth(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def vth(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pth(x,y):
    val=x*(1.-x)-1./6.
    return val

def exxth(x,y):
    val=4*x*y*(1-x)*(1-2*x)*(1-3*y+2*y**2)
    return val

def eyyth(x,y):
    val=-4*x*y*(1-y)*(1-2*y)*(1-3*x+2*x**2)
    return val

def exyth(x,y):
    val=2*x**2*(1-x)**2*(1-6*y+6*y**2) /2. \
       -2*y**2*(1-y)**2*(1-6*x+6*x**2) /2.
    return val

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

# declare variables
print("variable declaration")

m=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

viscosity=1.  # dynamic viscosity \mu

NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10
sqrt3=np.sqrt(3.)

hx=Lx/nelx
hy=Ly/nely

random_grid=False

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

if random_grid:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           if i>0 and j>0 and i<nnx-1 and j<nny-1:
              dx=random.random()*hx/10
              dy=random.random()*hy/10
           else:
              dx=0
              dy=0
           x[counter]=i*Lx/float(nelx)+dx
           y[counter]=j*Ly/float(nely)+dy
           counter += 1
else:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*Lx/float(nelx)
           y[counter]=j*Ly/float(nely)
           counter += 1

np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
# numbering of faces of the domain
# +---3---+
# |       |
# 0       1
# |       |
# +---2---+

start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
on_bd=np.zeros((nnp,4),dtype=np.bool)  # boundary indicator
 
for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,0]=True
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,1]=True
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,2]=True
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       on_bd[i,3]=True

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)           # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)           # y-component velocity
p     = np.zeros(nel,dtype=np.float64)           # elemental pressure  
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq

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

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

a_mat[0:NfemV,0:NfemV]=K_mat
a_mat[0:NfemV,NfemV:Nfem]=G_mat
a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute elemental strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N[0:m]=NNV(rq,sq)
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure
######################################################################

q=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    q[icon[3,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1

q=q/count

np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')



#####################################################################
# compute nodal strain rate - method 1 : center to node
#####################################################################

exx1=np.zeros(nnp,dtype=np.float64)  
eyy1=np.zeros(nnp,dtype=np.float64)  
exy1=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):

    exx1[icon[0,iel]]+=exx[iel]
    exx1[icon[1,iel]]+=exx[iel]
    exx1[icon[2,iel]]+=exx[iel]
    exx1[icon[3,iel]]+=exx[iel]

    eyy1[icon[0,iel]]+=eyy[iel]
    eyy1[icon[1,iel]]+=eyy[iel]
    eyy1[icon[2,iel]]+=eyy[iel]
    eyy1[icon[3,iel]]+=eyy[iel]

    exy1[icon[0,iel]]+=exy[iel]
    exy1[icon[1,iel]]+=exy[iel]
    exy1[icon[2,iel]]+=exy[iel]
    exy1[icon[3,iel]]+=exy[iel]

    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1

exx1/=count
eyy1/=count
exy1/=count

print("     -> exx1 (m,M) %.4f %.4f " %(np.min(exx1),np.max(exx1)))
print("     -> eyy1 (m,M) %.4f %.4f " %(np.min(eyy1),np.max(eyy1)))
print("     -> exy1 (m,M) %.4f %.4f " %(np.min(exy1),np.max(exy1)))

np.savetxt('srn_1.ascii',np.array([x,y,exx1,eyy1,exy1]).T,header='# x,y,exx1,eyy1,exy1')

#####################################################################
# compute nodal strain rate - method 2 : corners to node
#####################################################################

exx2=np.zeros(nnp,dtype=np.float64)  
eyy2=np.zeros(nnp,dtype=np.float64)  
exy2=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):

    # lower left
    rq=-1.+eps
    sq=-1.+eps
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[0,iel]]+=exx_rs
    eyy2[icon[0,iel]]+=eyy_rs
    exy2[icon[0,iel]]+=exy_rs
    count[icon[0,iel]]+=1

    # lower right
    rq=+1.-eps
    sq=-1.+eps
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[1,iel]]+=exx_rs
    eyy2[icon[1,iel]]+=eyy_rs
    exy2[icon[1,iel]]+=exy_rs
    count[icon[1,iel]]+=1

    # upper right
    rq=+1.-eps
    sq=+1.-eps
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[2,iel]]+=exx_rs
    eyy2[icon[2,iel]]+=eyy_rs
    exy2[icon[2,iel]]+=exy_rs
    count[icon[2,iel]]+=1

    # upper left
    rq=-1.+eps
    sq=+1.-eps
    dNdr[0:m]=dNNVdr(rq,sq)
    dNds[0:m]=dNNVds(rq,sq)
    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    exx_rs=dNdx[0:m].dot(u[icon[0:m,iel]])
    eyy_rs=dNdy[0:m].dot(v[icon[0:m,iel]])
    exy_rs=dNdx[0:m].dot(v[icon[0:m,iel]])*0.5\
          +dNdy[0:m].dot(u[icon[0:m,iel]])*0.5
    exx2[icon[3,iel]]+=exx_rs
    eyy2[icon[3,iel]]+=eyy_rs
    exy2[icon[3,iel]]+=exy_rs
    count[icon[3,iel]]+=1

exx2/=count
eyy2/=count
exy2/=count

print("     -> exx2 (m,M) %.4f %.4f " %(np.min(exx2),np.max(exx2)))
print("     -> eyy2 (m,M) %.4f %.4f " %(np.min(eyy2),np.max(eyy2)))
print("     -> exy2 (m,M) %.4f %.4f " %(np.min(exy2),np.max(exy2)))

np.savetxt('srn_2.ascii',np.array([x,y,exx2,eyy2,exy2]).T,header='# x,y,exx2,eyy2,exy2')

#####################################################################
# compute nodal strain rate - method 3: least squares 
#####################################################################
# numbering of elements inside patch
# -----
# |3|2|
# -----
# |0|1|
# -----
# numbering of nodes of the patch
# 6--7--8
# |  |  |
# 3--4--5
# |  |  |
# 0--1--2

exx3=np.zeros(nnp,dtype=np.float64)  
eyy3=np.zeros(nnp,dtype=np.float64)  
exy3=np.zeros(nnp,dtype=np.float64)  

AA = np.zeros((4,4),dtype=np.float64) 
BBxx = np.zeros(4,dtype=np.float64) 
BByy = np.zeros(4,dtype=np.float64) 
BBxy = np.zeros(4,dtype=np.float64) 

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        if i<nelx-1 and j<nely-1:
           iel0=counter
           iel1=counter+1
           iel2=counter+nelx+1
           iel3=counter+nelx

           AA[0,0]=1.       
           AA[1,0]=1.       
           AA[2,0]=1.       
           AA[3,0]=1.        
           AA[0,1]=xc[iel0] 
           AA[1,1]=xc[iel1] 
           AA[2,1]=xc[iel2] 
           AA[3,1]=xc[iel3] 
           AA[0,2]=yc[iel0] 
           AA[1,2]=yc[iel1] 
           AA[2,2]=yc[iel2] 
           AA[3,2]=yc[iel3] 
           AA[0,3]=xc[iel0]*yc[iel0] 
           AA[1,3]=xc[iel1]*yc[iel1] 
           AA[2,3]=xc[iel2]*yc[iel2] 
           AA[3,3]=xc[iel3]*yc[iel3] 

           BBxx[0]=exx[iel0] 
           BBxx[1]=exx[iel1] 
           BBxx[2]=exx[iel2] 
           BBxx[3]=exx[iel3] 
           solxx=sps.linalg.spsolve(sps.csr_matrix(AA),BBxx)

           BByy[0]=eyy[iel0] 
           BByy[1]=eyy[iel1] 
           BByy[2]=eyy[iel2] 
           BByy[3]=eyy[iel3] 
           solyy=sps.linalg.spsolve(sps.csr_matrix(AA),BByy)

           BBxy[0]=exy[iel0] 
           BBxy[1]=exy[iel1] 
           BBxy[2]=exy[iel2] 
           BBxy[3]=exy[iel3] 
           solxy=sps.linalg.spsolve(sps.csr_matrix(AA),BBxy)
           
           # node 4 of patch
           ip=icon[2,iel0] 
           exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
           eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
           exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # node 1 of patch
           ip=icon[1,iel0] 
           if on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # node 3 of patch
           ip=icon[3,iel0] 
           if on_bd[ip,0]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # node 5 of patch
           ip=icon[2,iel1] 
           if on_bd[ip,1]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # node 7 of patch
           ip=icon[3,iel2] 
           if on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # lower left corner of domain
           ip=icon[0,iel0] 
           if on_bd[ip,0] and on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # lower right corner of domain
           ip=icon[1,iel1] 
           if on_bd[ip,1] and on_bd[ip,2]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # upper right corner of domain
           ip=icon[2,iel2] 
           if on_bd[ip,1] and on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

           # lower right corner of domain
           ip=icon[3,iel3] 
           if on_bd[ip,0] and on_bd[ip,3]:
              exx3[ip]=solxx[0]+solxx[1]*x[ip]+solxx[2]*y[ip]+solxx[3]*x[ip]*y[ip]
              eyy3[ip]=solyy[0]+solyy[1]*x[ip]+solyy[2]*y[ip]+solyy[3]*x[ip]*y[ip]
              exy3[ip]=solxy[0]+solxy[1]*x[ip]+solxy[2]*y[ip]+solxy[3]*x[ip]*y[ip]

        counter+=1

print("     -> exx3 (m,M) %.4f %.4f " %(np.min(exx3),np.max(exx3)))
print("     -> eyy3 (m,M) %.4f %.4f " %(np.min(eyy3),np.max(eyy3)))
print("     -> exy3 (m,M) %.4f %.4f " %(np.min(exy3),np.max(exy3)))

np.savetxt('srn_3.ascii',np.array([x,y,exx3,eyy3,exy3]).T,header='# x,y,exx3,eyy3,exy3')

######################################################################
# compute error
######################################################################
start = time.time()

errv=0.
errp=0.
errexx0=0. ; errexx1=0. ; errexx2=0. ; errexx3=0.
erreyy0=0. ; erreyy1=0. ; erreyy2=0. ; erreyy3=0.
errexy0=0. ; errexy1=0. ; errexy2=0. ; errexy3=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            exx1q=0.0
            exx2q=0.0
            exx3q=0.0
            eyy1q=0.0
            eyy2q=0.0
            eyy3q=0.0
            exy1q=0.0
            exy2q=0.0
            exy3q=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
                exx1q+=N[k]*exx1[icon[k,iel]]
                exx2q+=N[k]*exx2[icon[k,iel]]
                exx3q+=N[k]*exx3[icon[k,iel]]
                eyy1q+=N[k]*eyy1[icon[k,iel]]
                eyy2q+=N[k]*eyy2[icon[k,iel]]
                eyy3q+=N[k]*eyy3[icon[k,iel]]
                exy1q+=N[k]*exy1[icon[k,iel]]
                exy2q+=N[k]*exy2[icon[k,iel]]
                exy3q+=N[k]*exy3[icon[k,iel]]
            exx0q=exx[iel]
            eyy0q=eyy[iel]
            exy0q=exy[iel]
            errv+=((uq-uth(xq,yq))**2+(vq-vth(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pth(xq,yq))**2*weightq*jcob
            errexx0+=(exx0q-exxth(xq,yq))**2*weightq*jcob
            errexx1+=(exx1q-exxth(xq,yq))**2*weightq*jcob
            errexx2+=(exx2q-exxth(xq,yq))**2*weightq*jcob
            errexx3+=(exx3q-exxth(xq,yq))**2*weightq*jcob
            erreyy0+=(eyy0q-eyyth(xq,yq))**2*weightq*jcob
            erreyy1+=(eyy1q-eyyth(xq,yq))**2*weightq*jcob
            erreyy2+=(eyy2q-eyyth(xq,yq))**2*weightq*jcob
            erreyy3+=(eyy3q-eyyth(xq,yq))**2*weightq*jcob
            errexy0+=(exy0q-exyth(xq,yq))**2*weightq*jcob
            errexy1+=(exy1q-exyth(xq,yq))**2*weightq*jcob
            errexy2+=(exy2q-exyth(xq,yq))**2*weightq*jcob
            errexy3+=(exy3q-exyth(xq,yq))**2*weightq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)

errexx0=np.sqrt(errexx0)
errexx1=np.sqrt(errexx1)
errexx2=np.sqrt(errexx2)
errexx3=np.sqrt(errexx3)

erreyy0=np.sqrt(erreyy0)
erreyy1=np.sqrt(erreyy1)
erreyy2=np.sqrt(erreyy2)
erreyy3=np.sqrt(erreyy3)

errexy0=np.sqrt(errexy0)
errexy1=np.sqrt(errexy1)
errexy2=np.sqrt(errexy2)
errexy3=np.sqrt(errexy3)


print("     -> nel= %6d ; errv= %.8e ; errp= %.8e " %(nel,errv,errp))
print("     -> nel= %6d ; errexx1,2,3 %.8e %.8e %.8e %.8e" %(nel,errexx0,errexx1,errexx2,errexx3))
print("     -> nel= %6d ; erreyy1,2,3 %.8e %.8e %.8e %.8e" %(nel,erreyy0,erreyy1,erreyy2,erreyy3))
print("     -> nel= %6d ; errexy1,2,3 %.8e %.8e %.8e %.8e" %(nel,errexy0,errexy1,errexy2,errexy3))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:

       filename = 'solution.vtu'
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (exx[iel]-exxth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (eyy[iel]-eyyth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy0 (err)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (exy[iel]-exyth(xc[iel],yc[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (np.sqrt(exx[iel]**2+eyy[iel]**2+2*exy[iel]**2)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx1[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy1[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy1[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")

       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy2[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx2[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy2[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy2[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")

       #-------------
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exx3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % eyy3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % exy3[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exx3[i]-exxth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='eyy3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (eyy3[i]-eyyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3 (err)' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (exy3[i]-exyth(x[i],y[i])))
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

print("generate vtu: %.3f s" % (time.time() - start))


#####################################################################
# using markers  
#####################################################################
start = time.time()

if random_grid:
   print ("random grid is on!!")

nmarker=100000

xm = np.zeros(nmarker,dtype=np.float64)  # x coordinates
ym = np.zeros(nmarker,dtype=np.float64)  # y coordinates
exx0m = np.zeros(nmarker,dtype=np.float64)  
exx1m = np.zeros(nmarker,dtype=np.float64)  
exx2m = np.zeros(nmarker,dtype=np.float64)  
exx3m = np.zeros(nmarker,dtype=np.float64)  
eyy0m = np.zeros(nmarker,dtype=np.float64)  
eyy1m = np.zeros(nmarker,dtype=np.float64)  
eyy2m = np.zeros(nmarker,dtype=np.float64)  
eyy3m = np.zeros(nmarker,dtype=np.float64)  
exy0m = np.zeros(nmarker,dtype=np.float64)  
exy1m = np.zeros(nmarker,dtype=np.float64)  
exy2m = np.zeros(nmarker,dtype=np.float64)  
exy3m = np.zeros(nmarker,dtype=np.float64)  
    
exx0mT=0. ; exx1mT=0. ; exx2mT=0. ; exx3mT=0.
eyy0mT=0. ; eyy1mT=0. ; eyy2mT=0. ; eyy3mT=0.
exy0mT=0. ; exy1mT=0. ; exy2mT=0. ; exy3mT=0.

for i in range(0,nmarker):
    xm[i]=random.random()
    ym[i]=random.random()
    ielx=int(xm[i]/Lx*nelx)
    iely=int(ym[i]/Ly*nely)
    iel=nelx*iely+ielx
    xmin=x[icon[0,iel]]
    xmax=x[icon[2,iel]]
    rm=((xm[i]-xmin)/(xmax-xmin)-0.5)*2
    ymin=y[icon[0,iel]]
    ymax=y[icon[2,iel]]
    sm=((ym[i]-ymin)/(ymax-ymin)-0.5)*2
    N[0:m]=NNV(rm,sm)

    exx0m[i]=exx[iel]
    exx1m[i]=N[:].dot(exx1[icon[:,iel]])     
    exx2m[i]=N[:].dot(exx2[icon[:,iel]])     
    exx3m[i]=N[:].dot(exx3[icon[:,iel]])     
    eyy0m[i]=eyy[iel]
    eyy1m[i]=N[:].dot(eyy1[icon[:,iel]])     
    eyy2m[i]=N[:].dot(eyy2[icon[:,iel]])     
    eyy3m[i]=N[:].dot(eyy3[icon[:,iel]])     
    exy0m[i]=exy[iel]
    exy1m[i]=N[:].dot(exy1[icon[:,iel]])     
    exy2m[i]=N[:].dot(exy2[icon[:,iel]])     
    exy3m[i]=N[:].dot(exy3[icon[:,iel]])     

    exx0mT+=abs(exx0m[i]-exxth(xm[i],ym[i]))
    exx1mT+=abs(exx1m[i]-exxth(xm[i],ym[i]))
    exx2mT+=abs(exx2m[i]-exxth(xm[i],ym[i]))
    exx3mT+=abs(exx3m[i]-exxth(xm[i],ym[i]))

    eyy0mT+=abs(eyy0m[i]-eyyth(xm[i],ym[i]))
    eyy1mT+=abs(eyy1m[i]-eyyth(xm[i],ym[i]))
    eyy2mT+=abs(eyy2m[i]-eyyth(xm[i],ym[i]))
    eyy3mT+=abs(eyy3m[i]-eyyth(xm[i],ym[i]))

    exy0mT+=abs(exy0m[i]-exyth(xm[i],ym[i]))
    exy1mT+=abs(exy1m[i]-exyth(xm[i],ym[i]))
    exy2mT+=abs(exy2m[i]-exyth(xm[i],ym[i]))
    exy3mT+=abs(exy3m[i]-exyth(xm[i],ym[i]))


exx0mT/=nmarker
exx1mT/=nmarker
exx2mT/=nmarker
exx3mT/=nmarker

eyy0mT/=nmarker
eyy1mT/=nmarker
eyy2mT/=nmarker
eyy3mT/=nmarker

exy0mT/=nmarker
exy1mT/=nmarker
exy2mT/=nmarker
exy3mT/=nmarker

print ('nel ',nel,'avrg exx on markers ',exx0mT,exx1mT,exx2mT,exx3mT)
print ('nel ',nel,'avrg eyy on markers ',eyy0mT,eyy1mT,eyy2mT,eyy3mT)
print ('nel ',nel,'avrg exy on markers ',exy0mT,exy1mT,exy2mT,exy3mT)

print("marker errors: %.3f s" % (time.time() - start))

#np.savetxt('markers.ascii',np.array([xm,ym]).T,header='# x,y')

if visu==1:

   filename = 'markers.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))

   vtufile.write("<PointData Scalars='scalars'>\n")

   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx0' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exx0m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exx1m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exx2m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exx3m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx0 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exx0m[i]-exxth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx1 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exx1m[i]-exxth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx2 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exx2m[i]-exxth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exx3 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exx3m[i]-exxth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy0' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exy0m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exy1m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exy2m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % exy3m[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy0 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exy0m[i]-exyth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy1 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exy1m[i]-exyth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy2 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exy2m[i]-exyth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='exy3 (err)' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%15e \n" % (exy3m[i]-exyth(xm[i],ym[i])))
   vtufile.write("</DataArray>\n")

   vtufile.write("</PointData>\n")

   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nmarker):
       vtufile.write("%10e %10e %10e \n" %(xm[i],ym[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")

   vtufile.write("<Cells>\n")

   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nmarker):
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
