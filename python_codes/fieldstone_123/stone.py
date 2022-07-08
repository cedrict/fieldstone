import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import love
import boussinesq

#------------------------------------------------------------------------------
# experiment=1: Love problem (Becker & Bevis, 2004)
# experiment=2: Boussinesq problem
# experiment=3: fault motion (Savage & Burford, 1973)

experiment=3

#------------------------------------------------------------------------------

def bx(x,y,z):
    val=0
    return val

def by(x,y,z):
    val=0
    return val

def bz(x,y,z):
    val=0
    return val

#------------------------------------------------------------------------------

def uth(x,y,z):
    if experiment==1:
       val=love.u_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.u(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       val=1/np.pi*np.arctan(y/0.25)
    return val

def vth(x,y,z):
    if experiment==1:
       val=love.v_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.v(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       val=0
    return val

def wth(x,y,z):
    if experiment==1:
       val=-love.w_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.w(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       val=0
    return val

def sigmaxx_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxx(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

def sigmayy_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmayy(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

def sigmazz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmazz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

def sigmaxy_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxy(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

def sigmaxz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

def sigmayz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmayz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------- stone 123 ---------")
print("-----------------------------")

m=8     # number of nodes making up an element
ndofV=3  # number of degrees of freedom per node

if int(len(sys.argv) == 2):
   nelx = int(sys.argv[1])
else:
   nelx = 2

if experiment==1 or experiment==2:
   Lx=5e3
   Ly=5e3
   Lz=2.5e3
   nely=nelx
   nelz=int(nelx/2)
   E=0.6e11
   nu=0.25 
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)

if experiment==3:
   Lx=0.5
   Ly=4
   Lz=3
   nely=nelx*16 #int(nelx*Ly/Lx)
   nelz=nelx*12 #int(nelx*Lz/Lx)*2
   nu=0.25
   mu=2
   E=2*mu/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)


hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
   
surf=hx*hy

if experiment==1:
   a=0.5e3
   b=1e3
   pressbc=1000*100*9.82 #rho g h

if experiment==2:
   Pforce=100e9
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

Nfem=NV*ndofV  # Total number of degrees of freedom

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('lambda=',lambdaa/1e9,'GPa')
print('mu=',mu/1e9,'GPa')
print('E=',E/1e9,'GPa')
print('nu=',nu)
print('hx=',hx)
print('hy=',hy)
print('hz=',hz)
print('surf=',surf)
if experiment==1: print('pressbc=',pressbc)
if experiment==2: print('Pforce=',Pforce)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

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
# compute coords of element center
#################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    for k in range(0,m):
        xc[iel]+=x[icon[k,iel]]*0.125
        yc[iel]+=y[icon[k,iel]]*0.125
        zc[iel]+=z[icon[k,iel]]*0.125
    #end for
#end for

print("element center coords: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(Nfem,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=float)  # boundary condition, value

if experiment==1 or experiment==2:
      for i in range(0,NV):
          if x[i]/Lx<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if x[i]/Lx>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]/Ly<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if y[i]/Ly>(1-eps):
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])
          if z[i]/Lz<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= uth(x[i],y[i],z[i])
             bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]= vth(x[i],y[i],z[i])
             bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]= wth(x[i],y[i],z[i])

if experiment==3:
      for i in range(0,NV):
          if y[i]/Ly<eps:
             if z[i]>Lz-0.25:
                bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 0
             else:
                bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]= 1
          if z[i]/Lz<eps:
             bc_fix[i*ndofV+0]=True ; bc_val[i*ndofV+0]=1 

          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0 
          bc_fix[i*ndofV+2]=True ; bc_val[i*ndofV+2]=0


print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#   /1 1 1 0 0 0\      /2 0 0 0 0 0\ 
#   |1 1 1 0 0 0|      |0 2 0 0 0 0|
# K=|1 1 1 0 0 0|    C=|0 0 2 0 0 0|  D=mu*C+lambda*K
#   |0 0 0 0 0 0|      |0 0 0 1 0 0|
#   |0 0 0 0 0 0|      |0 0 0 0 1 0|
#   \0 0 0 0 0 0/      \0 0 0 0 0 1/
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
b_mat = np.zeros((6,ndofV*m),dtype=np.float64)   # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)             # shape functions
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdz  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdt  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component displacement 
v     = np.zeros(NV,dtype=np.float64)            # y-component displacement 
w     = np.zeros(NV,dtype=np.float64)            # z-component displacement 
k_mat = np.zeros((6,6),dtype=np.float64) 
c_mat = np.zeros((6,6),dtype=np.float64) 

k_mat[0,0]=1. ; k_mat[0,1]=1. ; k_mat[0,2]=1.  
k_mat[1,0]=1. ; k_mat[1,1]=1. ; k_mat[1,2]=1.  
k_mat[2,0]=1. ; k_mat[2,1]=1. ; k_mat[2,2]=1.  

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

D_mat=mu*c_mat+lambdaa*k_mat

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m*ndofV,dtype=np.float64)
    a_el=np.zeros((m*ndofV,m*ndofV),dtype=np.float64)

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
                N[0]=0.125*(1.-rq)*(1.-sq)*(1.-tq)
                N[1]=0.125*(1.+rq)*(1.-sq)*(1.-tq)
                N[2]=0.125*(1.+rq)*(1.+sq)*(1.-tq)
                N[3]=0.125*(1.-rq)*(1.+sq)*(1.-tq)
                N[4]=0.125*(1.-rq)*(1.-sq)*(1.+tq)
                N[5]=0.125*(1.+rq)*(1.-sq)*(1.+tq)
                N[6]=0.125*(1.+rq)*(1.+sq)*(1.+tq)
                N[7]=0.125*(1.-rq)*(1.+sq)*(1.+tq)

                # calculate shape function derivatives
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
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[0,2]+=dNdr[k]*z[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                    jcb[1,2]+=dNds[k]*z[icon[k,iel]]
                    jcb[2,0]+=dNdt[k]*x[icon[k,iel]]
                    jcb[2,1]+=dNdt[k]*y[icon[k,iel]]
                    jcb[2,2]+=dNdt[k]*z[icon[k,iel]]
                #end for 
                jcob = np.linalg.det(jcb)
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
                #end for 

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:6, 3*i:3*i+3] = [[dNdx[i],0.     ,0.     ],
                                             [0.     ,dNdy[i],0.     ],
                                             [0.     ,0.     ,dNdz[i]],
                                             [dNdy[i],dNdx[i],0.     ],
                                             [dNdz[i],0.     ,dNdx[i]],
                                             [0.     ,dNdz[i],dNdy[i]]]
                #end for 

                # compute elemental a_mat matrix
                a_el += b_mat.T.dot(D_mat.dot(b_mat))*wq*jcob

                # compute elemental rhs vector
                #for i in range(0, m):
                #    b_el[ndofV*i+0]+=N[i]*jcob*wq*bx(xq,yq,zq)
                #    b_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq,zq)
                #    b_el[ndofV*i+2]+=N[i]*jcob*wq*bz(xq,yq,zq)
                #end for 

            #end for kq 
        #end for jq  
    #end for iq  

    # traction bc on top layer of elts
    if experiment==1 and xc[iel]<a and yc[iel]<b and zc[iel]>Lz-hz:
          if not bc_fix[3*icon[4,iel]+2]:
             b_el[14]-=surf*pressbc*0.25
          if not bc_fix[3*icon[5,iel]+2]:
             b_el[17]-=surf*pressbc*0.25
          if not bc_fix[3*icon[6,iel]+2]:
             b_el[20]-=surf*pressbc*0.25
          if not bc_fix[3*icon[7,iel]+2]:
             b_el[23]-=surf*pressbc*0.25

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
    
if experiment==2:
   for i in range(0,NV):
       if abs(x[i]-Lx/2)/Lx<eps and abs(y[i]-Ly/2)/Ly<eps and abs(z[i]-Lz)/Lz<eps:
          rhs[3*i+2]-=Pforce

a_mat=csr_matrix(a_mat)

print("build FE system: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(a_mat,rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y,z displacement arrays
#####################################################################
start = time.time()

u,v,w=np.reshape(sol,(NV,3)).T

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.5e %.5e " %(np.min(w),np.max(w)))

#np.savetxt('displacement.ascii',np.array([x,y,z,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (time.time() - start))
#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
ezz = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
exz = np.zeros(nel,dtype=np.float64)  
eyz = np.zeros(nel,dtype=np.float64)  
sr = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.
    wq=2.*2.*2.

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
        jcb[0,0] += dNdr[k]*x[icon[k,iel]]
        jcb[0,1] += dNdr[k]*y[icon[k,iel]]
        jcb[0,2] += dNdr[k]*z[icon[k,iel]]
        jcb[1,0] += dNds[k]*x[icon[k,iel]]
        jcb[1,1] += dNds[k]*y[icon[k,iel]]
        jcb[1,2] += dNds[k]*z[icon[k,iel]]
        jcb[2,0] += dNdt[k]*x[icon[k,iel]]
        jcb[2,1] += dNdt[k]*y[icon[k,iel]]
        jcb[2,2] += dNdt[k]*z[icon[k,iel]]
    #end for
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]+jcbi[0,2]*dNdt[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]+jcbi[1,2]*dNdt[k]
        dNdz[k]=jcbi[2,0]*dNdr[k]+jcbi[2,1]*dNds[k]+jcbi[2,2]*dNdt[k]
    #end for

    for k in range(0, m):
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        ezz[iel]+=dNdz[k]*w[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]
        exz[iel]+=0.5*dNdz[k]*u[icon[k,iel]]+0.5*dNdx[k]*w[icon[k,iel]]
        eyz[iel]+=0.5*dNdz[k]*v[icon[k,iel]]+0.5*dNdy[k]*w[icon[k,iel]]
    #end for

    p[iel]=-(lambdaa+2*mu/3)*(exx[iel]+eyy[iel]+ezz[iel])
    sr[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2+ezz[iel]**2)+exy[iel]**2+exz[iel]**2+eyz[iel]**2)
    
#end for

print("     -> p (m,M) %.4f %.4f "   %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.4e %.4e " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyz),np.max(eyz)))

#np.savetxt('pressure.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,zc,p')
#np.savetxt('strain.ascii',np.array([xc,yc,zc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p and strain: %.3f s" % (time.time() - start))

#####################################################################
# compute drms
#####################################################################
start = time.time()

errv=0
drms=0.

for iel in range(0, nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.

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
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[0,2]+=dNdr[k]*z[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                    jcb[1,2]+=dNds[k]*z[icon[k,iel]]
                    jcb[2,0]+=dNdt[k]*x[icon[k,iel]]
                    jcb[2,1]+=dNdt[k]*y[icon[k,iel]]
                    jcb[2,2]+=dNdt[k]*z[icon[k,iel]]
                jcob = np.linalg.det(jcb)

                xq=0.0
                yq=0.0
                zq=0.0
                uq=0.0
                vq=0.0
                wq=0.0
                for k in range(0,m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    zq+=N[k]*z[icon[k,iel]]
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    wq+=N[k]*w[icon[k,iel]]
                #end for 

                drms+=(uq**2+vq**2+wq**2)*jcob*weightq

                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*weightq*jcob

            #end for
        #end for
    #end for
#end for

errv=np.sqrt(errv/Lx/Ly/Lz)

dvrms=np.sqrt(drms/Lx/Ly/Lz)

print("     -> nel= %6d ; errv: %e " %(nel,errv))

print("     -> nel= %6d ; drms: %e" % (nel,drms))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# export profiles
#####################################################################
start = time.time()
   
xprofile=open("xprofile.ascii","w")
yprofile=open("yprofile.ascii","w")
zprofile=open("zprofile.ascii","w")
topfile=open("topfile.ascii","w")

for i in range(0,NV):
    xi=x[i]
    yi=y[i]
    zi=z[i]

    if abs(zi-Lz)/Lz<eps and abs(xi-Lx/2)/Lx<eps:
       xprofile.write("%e %e %e %e %e %e %e %e %e\n" %(x[i],y[i],z[i],u[i],v[i],w[i],\
                                                       uth(xi,yi,zi),vth(xi,yi,zi),wth(xi,yi,zi)))
    if abs(zi-Lz)/Lz<eps and abs(yi-Ly/2)/Ly<eps:
       yprofile.write("%e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,u[i],v[i],w[i],\
                                                        uth(xi,yi,zi),vth(xi,yi,zi),wth(xi,yi,zi)))
    if abs(xi-Lx/2)/Lx<eps and abs(yi-Ly/2)/Ly<eps:
       zprofile.write("%e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,u[i],v[i],w[i],\
                                                        uth(xi,yi,zi),vth(xi,yi,zi),wth(xi,yi,zi)))

for iel in range(0,nel):
    if zc[iel]>Lz-hz:
       topfile.write("%e %e %e %e %e %e %e %e %e\n" %(xc[iel],yc[iel],zc[iel],
                                                      exx[iel],eyy[iel],ezz[iel],
                                                      exy[iel],exz[iel],eyz[iel]))
       

xprofile.close()
yprofile.close()
zprofile.close()
topfile.close()

print("export profiles: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if True:
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
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='strain' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='ezz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % ezz[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='exz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % exz[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='eyz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % eyz[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (lambdaa*(exx[iel]+eyy[iel]+ezz[iel])+2*mu*exx[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (lambdaa*(exx[iel]+eyy[iel]+ezz[iel])+2*mu*eyy[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (lambdaa*(exx[iel]+eyy[iel]+ezz[iel])+2*mu*ezz[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (2*mu*exy[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_xz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (2*mu*exz[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_yz' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (2*mu*eyz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmaxx_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmayy_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_zz (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmazz_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmaxy_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xz (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmaxz_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yz (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (sigmayz_th(xc[iel],yc[iel],zc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.20f %.20f %.20f \n" %(uth(x[i],y[i],z[i]),\
                                              vth(x[i],y[i],z[i]),\
                                              wth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.20f %.20f %.20f \n" %(u[i]-uth(x[i],y[i],z[i]),\
                                              v[i]-vth(x[i],y[i],z[i]),\
                                              w[i]-wth(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmaxx_th(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmayy_th(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_zz (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmazz_th(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmaxy_th(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xz (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmaxz_th(x[i],y[i],z[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yz (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e\n" % (sigmayz_th(x[i],y[i],z[i])))
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
