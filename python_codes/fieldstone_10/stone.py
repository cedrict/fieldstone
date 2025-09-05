import numpy as np
import sys as sys
import time as clock 
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

###############################################################################

def basis_functions_V(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t) 
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t) 
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def basis_functions_V_dt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s) 
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

###############################################################################

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

###############################################################################

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

###############################################################################

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

###############################################################################

experiment=0

print("-----------------------------")
print("--------- stone 10 ----------")
print("-----------------------------")

m_V=8     # number of nodes making up an element
ndof_V=3  # number of degrees of freedom per node

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

nn_V=(nelx+1)*(nely+1)*(nelz+1) # total number of nodes
nel=nelx*nely*nelz              # total number of elements
Nfem=nn_V*ndof_V                # total number of degrees of freedom

penalty=1.e6  # penalty coefficient value

gz=-1.  # gravity vector, z component

eps=1.e-10
sqrt3=np.sqrt(3.)

###############################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('Nfem=',Nfem)
print("-----------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nelx+1):
    for j in range(0,nely+1):
        for k in range(0,nelz+1):
            x_V[counter]=i*Lx/float(nelx)
            y_V[counter]=j*Ly/float(nely)
            z_V[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for
   
print("mesh setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

nnx=nelx+1 ; nny=nely+1 ; nnz=nelz+1 

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
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if experiment==0 or experiment==1 or experiment==2 or experiment==3 or experiment==4:

   if FS or OT:
      for i in range(0,nn_V):
          if x_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
          if x_V[i]>(Lx-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
          if y_V[i]<eps:
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
          if y_V[i]>(Ly-eps):
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
          if z_V[i]<eps:
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if not OT and z_V[i]>(Lz-eps):
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          #end if
      #end for

   if NS:
      for i in range(0,nn_V):
          if x_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if x_V[i]>(1-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if y_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if y_V[i]>(1-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if z_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if z_V[i]>(Lz-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0.
          if quarter and x_V[i]>(0.5-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0.
          if quarter and y_V[i]>(0.5-eps):
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0.
      #end for

if experiment==5:
      for i in range(0,nn_V):
          if x_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
          if x_V[i]>(1-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
          if y_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
          if y_V[i]>(1-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
          if z_V[i]<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
          if z_V[i]>(Lz-eps):
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(x[i],y[i],z[i])
      #end for

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
#   /1 1 1 0 0 0\      /2 0 0 0 0 0\ 
#   |1 1 1 0 0 0|      |0 2 0 0 0 0|
# K=|1 1 1 0 0 0|    C=|0 0 2 0 0 0|
#   |0 0 0 0 0 0|      |0 0 0 1 0 0|
#   |0 0 0 0 0 0|      |0 0 0 0 1 0|
#   \0 0 0 0 0 0/      \0 0 0 0 0 1/
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)    # gradient matrix B 
jcb=np.zeros((3,3),dtype=np.float64)           # jacobian matrix

K=np.zeros((6,6),dtype=np.float64) 
K[0,0]=1. ; K[0,1]=1. ; K[0,2]=1.  
K[1,0]=1. ; K[1,1]=1. ; K[1,2]=1.  
K[2,0]=1. ; K[2,1]=1. ; K[2,2]=1.  

C=np.zeros((6,6),dtype=np.float64) 
C[0,0]=2. ; C[1,1]=2. ; C[2,2]=2.
C[3,3]=1. ; C[4,4]=1. ; C[5,5]=1.

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)

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
                N_V=basis_functions_V(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)

                # calculate jacobian matrix
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

                # calculate the determinant of the jacobian
                JxWq=np.linalg.det(jcb)*weightq

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute coordinates of quadrature points
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])

                # compute dNdx, dNdy, dNdz
                dNdx=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]+jcbi[0,2]*dNdt_V[:]
                dNdy=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]+jcbi[1,2]*dNdt_V[:]
                dNdz=jcbi[2,0]*dNdr_V[:]+jcbi[2,1]*dNds_V[:]+jcbi[2,2]*dNdt_V[:]

                # construct 3x8 B matrix
                for i in range(0,m_V):
                    B[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                         [0.     ,dNdy[i],0.     ],
                                         [0.     ,0.     ,dNdz[i]],
                                         [dNdy[i],dNdx[i],0.     ],
                                         [dNdz[i],0.     ,dNdx[i]],
                                         [0.     ,dNdz[i],dNdy[i]]]

                # compute elemental matrix
                A_el+=B.T.dot(C.dot(B))*viscosity(xq,yq,zq)*JxWq

                # compute elemental rhs vector
                for i in range(0,m_V):
                    b_el[ndof_V*i+0]+=N_V[i]*bx(xq,yq,zq)*JxWq
                    b_el[ndof_V*i+1]+=N_V[i]*by(xq,yq,zq)*JxWq
                    b_el[ndof_V*i+2]+=N_V[i]*bz(xq,yq,zq)*JxWq
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
    N=basis_functions_V(rq,sq,tq)
    dNdr=basis_functions_V_dr(rq,sq,tq)
    dNds=basis_functions_V_ds(rq,sq,tq)
    dNdt=basis_functions_V_dt(rq,sq,tq)

    # calculate jacobian matrix
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

    jcbi=np.linalg.inv(jcb)

    JxWq=np.linalg.det(jcb)*weightq

    dNdx=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]+jcbi[0,2]*dNdt[:]
    dNdy=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]+jcbi[1,2]*dNdt[:]
    dNdz=jcbi[2,0]*dNdr[:]+jcbi[2,1]*dNds[:]+jcbi[2,2]*dNdt[:]

    for i in range(0,m_V):
        B[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                             [0.     ,dNdy[i],0.     ],
                             [0.     ,0.     ,dNdz[i]],
                             [dNdy[i],dNdx[i],0.     ],
                             [dNdz[i],0.     ,dNdx[i]],
                             [0.     ,dNdz[i],dNdy[i]]]

    # compute elemental matrix
    A_el+=B.T.dot(K.dot(B))*penalty*JxWq

    # apply boundary conditions
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof_V*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m_V*ndof_V):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               #end for
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix A_fem and right hand side 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for iel

A_fem=csr_matrix(A_fem)

print("build FE system: %.3f s | nel= %d" % (clock.time()-start,nel))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=spsolve(A_fem,b_fem)

print("solve time: %.3f s | Nfem= %d " % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v,w=np.reshape(sol,(nn_V,3)).T

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.5e %.5e " %(np.min(w),np.max(w)))

#np.savetxt('velocity.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure
###############################################################################
start=clock.time()

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

    N=basis_functions_V(rq,sq,tq)
    dNdr=basis_functions_V_dr(rq,sq,tq)
    dNds=basis_functions_V_ds(rq,sq,tq)
    dNdt=basis_functions_V_dt(rq,sq,tq)

    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

    jcbi=np.linalg.inv(jcb)

    dNdx=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]+jcbi[0,2]*dNdt[:]
    dNdy=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]+jcbi[1,2]*dNdt[:]
    dNdz=jcbi[2,0]*dNdr[:]+jcbi[2,1]*dNds[:]+jcbi[2,2]*dNdt[:]

    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    zc[iel]=np.dot(N_V,z_V[icon_V[:,iel]])

    exx[iel]=dNdx[:].dot(u[icon_V[:,iel]])
    eyy[iel]=dNdy[:].dot(v[icon_V[:,iel]])
    ezz[iel]=dNdz[:].dot(w[icon_V[:,iel]])
    exy[iel]=0.5*dNdy[:].dot(u[icon_V[:,iel]])+0.5*dNdx[:].dot(v[icon_V[:,iel]])
    exz[iel]=0.5*dNdz[:].dot(u[icon_V[:,iel]])+0.5*dNdx[:].dot(w[icon_V[:,iel]])
    eyz[iel]=0.5*dNdz[:].dot(v[icon_V[:,iel]])+0.5*dNdy[:].dot(w[icon_V[:,iel]])

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

print("compute p and strainrate: %.3f s" % (clock.time()-start))

###############################################################################
# compute vrms
###############################################################################
start=clock.time()

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

                N=basis_functions_V(rq,sq,tq)
                dNdr=basis_functions_V_dr(rq,sq,tq)
                dNds=basis_functions_V_ds(rq,sq,tq)
                dNdt=basis_functions_V_dt(rq,sq,tq)

                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

                JxWq=np.linalg.det(jcb)*weightq

                xq=N.dot(x_V[icon_V[:,iel]])
                yq=N.dot(y_V[icon_V[:,iel]])
                zq=N.dot(z_V[icon_V[:,iel]])

                uq=N.dot(u[icon_V[:,iel]])
                vq=N.dot(v[icon_V[:,iel]])
                wq=N.dot(w[icon_V[:,iel]])

                vrms+=(uq**2+vq**2+wq**2)*JxWq

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*JxWq

                errp+=(p[iel]-pth(xq,yq,zq))**2*JxWq

            #end for
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

vrms=np.sqrt(vrms/Lx/Ly/Lz)

print("     -> nel= %6d ; errv: %e ; errp: %e " %(nel,errv,errp))
print("     -> nel= %6d ; vrms: %e" % (nel,vrms))

print("compute errors: %.3f s" % (clock.time()-start))

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
# export solution to vtu format
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],z_V[i]))
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
   for i in range (0,nn_V):
       vtufile.write("%f\n" % pth(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.20f %.20f %.20f \n" %(uth(x_V[i],y_V[i],z_V[i]),\
                                              vth(x_V[i],y_V[i],z_V[i]),\
                                              wth(x_V[i],y_V[i],z_V[i])))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],
                                                   icon_V[4,iel],icon_V[5,iel],icon_V[6,iel],icon_V[7,iel]))
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
   print("export to vtu: %.3f s" % (clock.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
