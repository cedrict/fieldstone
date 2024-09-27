import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
import time as time
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

#------------------------------------------------------------------------------

def NNN(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def dNNNdr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t) 
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def dNNNds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t) 
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def dNNNdt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s) 
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

#------------------------------------------------------------------------------

def bx(x,y,z):
    if experiment==0:
       val=0
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       val=0
    if experiment==5:
       val=4*(2*y-1)*(2*z-1)
    return val

def by(x,y,z):
    if experiment==0:
       val=0
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       val=0
    if experiment==5:
       val=4*(2*x-1)*(2*z-1)
    return val

def bz(x,y,z):
    if experiment==0:
       val=-1
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1.01*gz
       else:
          val=1.*gz
    if experiment==5:
       val=-2*(2*x-1)*(2*y-1) 
    return val

#------------------------------------------------------------------------------

def viscosity(x,y,z):
    if experiment==0:
       val=1
    if experiment==1 or experiment==2 or experiment==3 or experiment==4:
       if (x-.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789**2:
          val=1e3
       else:
          val=1.
    if experiment==5:
       val=1.
    return val

#------------------------------------------------------------------------------

def uth(x,y,z):
    if experiment==5:
       val=x*(1-x)*(1-2*y)*(1-2*z)
    elif experiment==0:
       val=0
    return val

def vth(x,y,z):
    if experiment==5:
       val=(1-2*x)*y*(1-y)*(1-2*z)
    elif experiment==0:
       val=0
    return val

def wth(x,y,z):
    if experiment==5:
       val=-2*(1-2*x)*(1-2*y)*z*(1-z)
    elif experiment==0:
       val=0
    return val

def pth(x,y,z):
    if experiment==5:
       val=(2*x-1)*(2*y-1)*(2*z-1)
    elif experiment==0:
       val=0.5-z
    return val

#------------------------------------------------------------------------------

experiment=0

print("-----------------------------")
print("--------- stone 10 ----------")
print("-----------------------------")

m=8     # number of nodes making up an element
ndofV=3  # number of degrees of freedom per node

if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   visu = int(sys.argv[2])
else:
   nelx = 16
   visu = 1

if experiment==0: quarter=False
if experiment==1: quarter=False
if experiment==2: quarter=False
if experiment==3: quarter=False
if experiment==4: quarter=True
if experiment==5: quarter=False

if quarter:
   nely=nelx
   nelz=2*nelx
   Lx=0.5 
   Ly=0.5
   Lz=1.
else: 
   nely=nelx
   nelz=nelx
   Lx=1.
   Ly=1.
   Lz=1.


FS=True
NS=False
OT=False
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

penalty=1.e6  # penalty coefficient value

Nfem=NV*ndofV  # Total number of degrees of freedom

eps=1.e-10

gz=-1.  # gravity vector, z component

sqrt3=np.sqrt(3.)

#################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('Nfem=',Nfem)
print("-----------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.zeros(NV,dtype=np.float64)  # x coordinates
y = np.zeros(NV,dtype=np.float64)  # y coordinates
z = np.zeros(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for
   
print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if experiment==0 or experiment==1 or experiment==2 or experiment==3 or experiment==4:

   if FS or OT:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if x[i]>(Lx-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if y[i]<eps:
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
          if y[i]>(Ly-eps):
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
          if z[i]<eps:
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if not OT and z[i]>(Lz-eps):
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          #end if
      #end for

   if NS:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if x[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if y[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if y[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if z[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if z[i]>(Lz-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= 0.
          if quarter and x[i]>(0.5-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0.
          if quarter and y[i]>(0.5-eps):
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= 0.
      #end for

if experiment==5:
      for i in range(0,NV):
          if x[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if x[i]>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]>(1-eps):
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
      #end for

print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#   /1 1 1 0 0 0\      /2 0 0 0 0 0\ 
#   |1 1 1 0 0 0|      |0 2 0 0 0 0|
# K=|1 1 1 0 0 0|    C=|0 0 2 0 0 0|
#   |0 0 0 0 0 0|      |0 0 0 1 0 0|
#   |0 0 0 0 0 0|      |0 0 0 0 1 0|
#   \0 0 0 0 0 0/      \0 0 0 0 0 1/
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
w     = np.zeros(NV,dtype=np.float64)            # z-component velocity
jcb   = np.zeros((3,3),dtype=np.float64)         # jacobian matrix
k_mat = np.zeros((6,6),dtype=np.float64) 
c_mat = np.zeros((6,6),dtype=np.float64) 

k_mat[0,0]=1. ; k_mat[0,1]=1. ; k_mat[0,2]=1.  
k_mat[1,0]=1. ; k_mat[1,1]=1. ; k_mat[1,2]=1.  
k_mat[2,0]=1. ; k_mat[2,1]=1. ; k_mat[2,2]=1.  

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m*ndofV,dtype=np.float64)
    a_el=np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

    # integrate viscous term at 2*2*2 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.

                # calculate shape functions
                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute coordinates of quadrature points
                xq=N.dot(x[icon[0:m,iel]])
                yq=N.dot(y[icon[0:m,iel]])
                zq=N.dot(z[icon[0:m,iel]])

                # compute dNdx, dNdy, dNdz
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
                    dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
                #end for 

                # construct 3x8 b_mat matrix
                for i in range(0,m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for 

                # compute elemental matrix
                a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,zq)*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,m):
                    b_el[ndofV*i+0]+=N[i]*jcob*weightq*bx(xq,yq,zq)
                    b_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq,zq)
                    b_el[ndofV*i+2]+=N[i]*jcob*weightq*bz(xq,yq,zq)
                #end for 

            #end for kq 
        #end for jq  
    #end for iq  

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    tq=0.
    weightq=2.*2.*2.

    # calculate shape functions
    N[0:m]=NNN(rq,sq,tq)
    dNdr[0:m]=dNNNdr(rq,sq,tq)
    dNds[0:m]=dNNNds(rq,sq,tq)
    dNdt[0:m]=dNNNdt(rq,sq,tq)

    # calculate jacobian matrix
    jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
    jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
    jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
    jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
    jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
    jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
    jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
    jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
    jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
    #end for

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                 [0.     ,dNdy[i],0.     ],
                                 [0.     ,0.     ,dNdz[i]],
                                 [dNdy[i],dNdx[i],0.     ],
                                 [dNdz[i],0.     ,dNdx[i]],
                                 [0.     ,dNdz[i],dNdy[i]]]
    #end for

    # compute elemental matrix
    a_el+=b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndofV*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndofV):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               #end for
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                #end for
            #end for
            rhs[m1]+=b_el[ikk]
        #end for
    #end for

#end for iel

a_mat=csr_matrix(a_mat)

print("build FE system: %.3f s | nel= %d" % (time.time() - start,nel))

#################################################################
# solve system
#################################################################
start = time.time()

sol=sps.linalg.spsolve(a_mat,rhs)

print("solve time: %.3f s | Nfem= %d " % (time.time() - start,Nfem))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u,v,w=np.reshape(sol,(NV,3)).T

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.5e %.5e " %(np.min(w),np.max(w)))

#np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
visc = np.zeros(nel,dtype=np.float64)  
dens = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  
jcb=np.zeros((3,3),dtype=np.float64)

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    weightq=2.*2.*2.

    N[0:m]=NNN(rq,sq,tq)
    dNdr[0:m]=dNNNdr(rq,sq,tq)
    dNds[0:m]=dNNNds(rq,sq,tq)
    dNdt[0:m]=dNNNdt(rq,sq,tq)

    jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
    jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
    jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
    jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
    jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
    jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
    jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
    jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
    jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
    #end for

    for k in range(0,m):
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        zc[iel]+=N[k]*z[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        ezz[iel]+=dNdz[k]*w[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]
        exz[iel]+=0.5*dNdz[k]*u[icon[k,iel]]+0.5*dNdx[k]*w[icon[k,iel]]
        eyz[iel]+=0.5*dNdz[k]*v[icon[k,iel]]+0.5*dNdy[k]*w[icon[k,iel]]
    #end for

    p[iel]=-penalty*(exx[iel]+eyy[iel]+ezz[iel])
    visc[iel]=viscosity(xc[iel],yc[iel],zc[iel])
    dens[iel]=bz(xc[iel],yc[iel],zc[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                        +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])
    
#end for

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.4e %.4e " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyz),np.max(eyz)))
print("     -> visc (m,M) %.4e %.4e " %(np.min(visc),np.max(visc)))
print("     -> dens (m,M) %.4e %.4e " %(np.min(dens),np.max(dens)))

#np.savetxt('pressure.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,zc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p and strainrate: %.3f s" % (time.time() - start))

#####################################################################
# compute vrms
#####################################################################
start = time.time()

errv=0
errp=0
vrms=0.

for iel in range(0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.

                N[0:m]=NNN(rq,sq,tq)
                dNdr[0:m]=dNNNdr(rq,sq,tq)
                dNds[0:m]=dNNNds(rq,sq,tq)
                dNdt[0:m]=dNNNdt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[0,0]=dNdr.dot(x[icon[0:m,iel]])
                jcb[0,1]=dNdr.dot(y[icon[0:m,iel]])
                jcb[0,2]=dNdr.dot(z[icon[0:m,iel]])
                jcb[1,0]=dNds.dot(x[icon[0:m,iel]])
                jcb[1,1]=dNds.dot(y[icon[0:m,iel]])
                jcb[1,2]=dNds.dot(z[icon[0:m,iel]])
                jcb[2,0]=dNdt.dot(x[icon[0:m,iel]])
                jcb[2,1]=dNdt.dot(y[icon[0:m,iel]])
                jcb[2,2]=dNdt.dot(z[icon[0:m,iel]])
                jcob = np.linalg.det(jcb)

                xq=N.dot(x[icon[:,iel]])
                yq=N.dot(y[icon[:,iel]])
                zq=N.dot(z[icon[:,iel]])
                uq=N.dot(u[icon[:,iel]])
                vq=N.dot(v[icon[:,iel]])
                wq=N.dot(w[icon[:,iel]])

                vrms+=(uq**2+vq**2+wq**2)*jcob*weightq

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*weightq*jcob

                errp+=(p[iel]-pth(xq,yq,zq))**2*weightq*jcob

            #end for
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

vrms=np.sqrt(vrms/Lx/Ly/Lz)

print("     -> nel= %6d ; errv: %e ; errp: %e " %(nel,errv,errp))
print("     -> nel= %6d ; vrms: %e" % (nel,vrms))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# export various measurements for stokes sphere benchmark 
#####################################################################

vel=np.sqrt(u**2+v**2+w**2)

print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      np.min(w),np.max(w),\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms)

#####################################################################
# export solution to vtu format
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
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='pressure (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pth(xc[iel],yc[iel],zc[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='density*gz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % dens[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % visc[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='strainrate' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%f\n" % pth(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.20f %.20f %.20f \n" %(uth(x[i],y[i],z[i]),\
                                              vth(x[i],y[i],z[i]),\
                                              wth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
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
