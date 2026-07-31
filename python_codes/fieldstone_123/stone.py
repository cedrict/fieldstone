import numpy as np
import time as clock
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix
import love
import boussinesq
import numba

###############################################################################
# experiment=1: Love problem (Becker & Bevis, 2004)
# experiment=2: Boussinesq problem
# experiment=3: fault motion (Savage & Burford, 1973)
# experiment=4: magma chamber
# experiment=5: magma chamber (pseudo 2d)

experiment=5

###############################################################################
#def bx(x,y,z):
#    return 0
#def by(x,y,z):
#    return 0
#def bz(x,y,z):
#    return 0

###############################################################################

def uth(x,y,z):
    if experiment==1:
       val=love.u_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.u(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3: 
       val=1/np.pi*np.arctan(y/0.25)
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def vth(x,y,z):
    if experiment==1:
       val=love.v_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.v(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3: val=0
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def wth(x,y,z):
    if experiment==1:
       val=-love.w_Love(x,y,Lz-z,a,b,pressbc,lambdaa,mu)
    if experiment==2:
       val=boussinesq.w(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3: val=0
    if experiment==4: val=0
    if experiment==5: val=0
    return val

def sigmaxx_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxx(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

def sigmayy_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmayy(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

def sigmazz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmazz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

def sigmaxy_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxy(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

def sigmaxz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmaxz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

def sigmayz_th(x,y,z):
    if experiment==1:
       return 0
    if experiment==2:
       return boussinesq.sigmayz(x-Lx/2,y-Ly/2,Lz-z,Pforce,nu,mu) 
    if experiment==3:
       return 0
    if experiment==4:
       return 0
    if experiment==5:
       return 0

###############################################################################

@numba.njit
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

@numba.njit
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

@numba.njit
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

@numba.njit
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

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 123 **********")
print("*******************************")

m_V=8     # number of nodes making up an element
ndof_V=3  # number of degrees of freedom per node

if int(len(sys.argv) == 2):
   nelx = int(sys.argv[1])
else:
   nelx = 200

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

if experiment==4:
   Lx=24e3
   Ly=24e3
   Lz=12e3
   nely=nelx
   nelz=int(nelx/Lx*Lz)
   nu=0.25
   mu=1e10
   E=2*mu/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   K=2*mu*(1+nu)/(3-6*nu)
   D=4e3
   R=1e3
   aaa=2e7

if experiment==5:
   Lx=25e3 #*4
   Ly=1e3
   Lz=25e3 #*4
   nely=2
   nelz=int(nelx/Lx*Lz)
   nu=0.25
   mu=1e10
   E=2*mu/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   K=2*mu*(1+nu)/(3-6*nu)
   D=4e3
   R=1e3
   aaa=2e7 # nu=.25
   #aaa=1.65e7 # nu=.35
   #aaa=1.26e7 # nu=.45

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
   
surf=hx*hy

if experiment==1:
   a=0.5555e3
   b=1.1111e3
   pressbc=1000*100*9.82 #rho g h

if experiment==2:
   Pforce=100e9
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction
nn_V=nnx*nny*nnz  # number of nodes
nel=nelx*nely*nelz  # number of elements, total
Nfem=nn_V*ndof_V  # Total number of degrees of freedom

debug=False

assembly=2

solver=2 # 1: direct ; 2: cg

method=2 # 1: center-> node, 2: more accurate/expensive

#################################################################

print('experiment=',experiment)
print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('Nfem=',Nfem)
print('lambda=',lambdaa/1e9,'GPa')
print('mu=',mu/1e9,'GPa')
print('E=',E/1e9,'GPa')
print('K=',K) 
print('nu=',nu)
print('hx=',hx)
print('hy=',hy)
print('hz=',hz)
print('surf=',surf)
print('assembly=',assembly)
if experiment==1: print('pressbc=',pressbc)
if experiment==2: print('Pforce=',Pforce)
print("*******************************")

#################################################################
# grid point setup
#################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x_V[counter]=i*hx
            y_V[counter]=j*hy
            z_V[counter]=k*hz
            counter += 1
        #end for
    #end for
#end for

if debug: np.savetxt('xyz.ascii',np.array([x_V,y_V,z_V]).T)
   
print("mesh setup: %.3f s" % (clock.time() - start))

#################################################################
# connectivity
#################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
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

#################################################################
# compute coords of element center
#################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
zc=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    for k in range(0,m_V):
        xc[iel]+=x_V[icon_V[k,iel]]*0.125
        yc[iel]+=y_V[icon_V[k,iel]]*0.125
        zc[iel]+=z_V[icon_V[k,iel]]*0.125
    #end for
#end for

if debug: np.savetxt('xyzc.ascii',np.array([xc,yc,zc]).T)

print("element center coords: %.3f s" % (clock.time()-start))

#################################################################
# define boundary conditions
#################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if experiment==1 or experiment==2:
   for i in range(0,nn_V):
       xi=x_V[i]
       yi=y_V[i]
       zi=z_V[i]
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(xi,yi,zi)
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(xi,yi,zi)
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(xi,yi,zi)
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(xi,yi,zi)
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(xi,yi,zi)
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(xi,yi,zi)
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(xi,yi,zi)
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(xi,yi,zi)
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(xi,yi,zi)
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(xi,yi,zi)
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(xi,yi,zi)
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(xi,yi,zi)
       if z_V[i]/Lz<eps:
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= uth(xi,yi,zi)
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= vth(xi,yi,zi)
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= wth(xi,yi,zi)

if experiment==3:
      for i in range(0,nn_V):
          if y_V[i]/Ly<eps:
             if z[i]>Lz-0.25:
                bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0
             else:
                bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 1
          if z_V[i]/Lz<eps:
             bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]=1 

          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]=0 
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]=0

if experiment==4 or experiment==5:
   for i in range(0,nn_V):
       xi=x_V[i]
       yi=y_V[i]
       zi=z_V[i]
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V+0]=True ; bc_val[i*ndof_V+0]= 0
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0
       if y_V[i]/Ly>(1-eps):
          bc_fix[i*ndof_V+1]=True ; bc_val[i*ndof_V+1]= 0
       if z_V[i]/Lz<eps:
          bc_fix[i*ndof_V+2]=True ; bc_val[i*ndof_V+2]= 0

print("define b.c.: %.3f s" % (clock.time()-start))

#################################################################
# this assembly method originates in stone 182 and was later 
# added to this stone
#################################################################
start=clock.time()

if assembly==2:
   ndof_V_el=ndof_V*m_V

   local_to_global=np.zeros((ndof_V_el,nel),dtype=np.int32)
   for iel in range(0,nel):
       for k in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k+i1
               local_to_global[ikk,iel]=ndof_V*icon_V[k,iel]+i1

   bignb=nel*ndof_V_el**2
   II_V=np.zeros(bignb,dtype=np.int32)    
   JJ_V=np.zeros(bignb,dtype=np.int32)    
   VV_V=np.zeros(bignb,dtype=np.float64)    

   counter=0
   for iel in range(0,nel):
       for ikk in range(ndof_V_el):
           m1=local_to_global[ikk,iel]
           for jkk in range(ndof_V_el):
               m2=local_to_global[jkk,iel]
               II_V[counter]=m1
               JJ_V[counter]=m2
               counter+=1

   #print(ndof_V_el)
   #print(bignb)

   print("preparing assembly arrays: %.3f s" % (clock.time()-start))

#################################################################
# build FE matrix
#   /1 1 1 0 0 0\      /2 0 0 0 0 0\ 
#   |1 1 1 0 0 0|      |0 2 0 0 0 0|
# K=|1 1 1 0 0 0|    C=|0 0 2 0 0 0|  D=mu*C+lambda*K
#   |0 0 0 0 0 0|      |0 0 0 1 0 0|
#   |0 0 0 0 0 0|      |0 0 0 0 1 0|
#   \0 0 0 0 0 0/      \0 0 0 0 0 1/
#################################################################
start=clock.time()

if assembly==1: A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)

B = np.zeros((6,ndof_V*m_V),dtype=np.float64)   # gradient matrix B 
k_mat = np.zeros((6,6),dtype=np.float64) 
c_mat = np.zeros((6,6),dtype=np.float64) 
jcb=np.zeros((3,3),dtype=np.float64)

k_mat[0,0]=1. ; k_mat[0,1]=1. ; k_mat[0,2]=1.  
k_mat[1,0]=1. ; k_mat[1,1]=1. ; k_mat[1,2]=1.  
k_mat[2,0]=1. ; k_mat[2,1]=1. ; k_mat[2,2]=1.  

c_mat[0,0]=2. ; c_mat[1,1]=2. ; c_mat[2,2]=2.
c_mat[3,3]=1. ; c_mat[4,4]=1. ; c_mat[5,5]=1.

D_mat=mu*c_mat+lambdaa*k_mat
                
jcob=hx*hy*hz/8
jcbi=np.zeros((3,3),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcbi[2,2]=2/hz

counter=0
for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)

    # integrate viscous term at 8 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:
            for kq in [-1, 1]:

                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.

                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                #jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                #jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                #jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                #jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                #jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                #jcob = np.linalg.det(jcb)
                JxWq=jcob*weightq
                #jcbi = np.linalg.inv(jcb)
                #dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                #dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                #dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                dNdx_V=jcbi[0,0]*dNdr_V
                dNdy_V=jcbi[1,1]*dNds_V
                dNdz_V=jcbi[2,2]*dNdt_V

                if iel==0:
                   # this is a trick: since all elemental matrices 
                   # are identical (constant coefficients in space and 
                   # all elements have the same size) then we can compute 
                   # A-el only once.

                   # construct B matrix
                   for i in range(0,m_V):
                       B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                           [0.       ,dNdy_V[i],0.       ],
                                           [0.       ,0.       ,dNdz_V[i]],
                                           [dNdy_V[i],dNdx_V[i],0.       ],
                                           [dNdz_V[i],0.       ,dNdx_V[i]],
                                           [0.       ,dNdz_V[i],dNdy_V[i]]]
                   # compute elemental matrix
                   A_el+=B.T.dot(D_mat.dot(B))*JxWq
                #end if 

                # compute elemental rhs vector for buoyancy forces
                #N_V=basis_functions_V(rq,sq,tq)
                #xq=np.dot(N_V,x_V[icon_V[:,iel]])
                #yq=np.dot(N_V,y_V[icon_V[:,iel]])
                #zq=np.dot(N_V,z_V[icon_V[:,iel]])
                #for i in range(0, m_V):
                #    b_el[ndof_V*i+0]+=N[i]*jcob*weightq*bx(xq,yq,zq)
                #    b_el[ndof_V*i+1]+=N[i]*jcob*weightq*by(xq,yq,zq)
                #    b_el[ndof_V*i+2]+=N[i]*jcob*weightq*bz(xq,yq,zq)
                #end for 

                if experiment==4 and xc[iel]**2+yc[iel]**2+(zc[iel]-(Lz-D))**2<R**2:
                   for i in range(0,m_V):
                       b_el[ndof_V*i+0]+=dNdx_V[i]*aaa*JxWq
                       b_el[ndof_V*i+1]+=dNdy_V[i]*aaa*JxWq
                       b_el[ndof_V*i+2]+=dNdz_V[i]*aaa*JxWq

                if experiment==5 and xc[iel]**2+(zc[iel]-(Lz-D))**2<R**2:
                   for i in range(0,m_V):
                       b_el[ndof_V*i+0]+=dNdx_V[i]*aaa*JxWq
                       b_el[ndof_V*i+1]+=dNdy_V[i]*aaa*JxWq
                       b_el[ndof_V*i+2]+=dNdz_V[i]*aaa*JxWq

            #end for kq 
        #end for jq  
    #end for iq  

    if iel==0:
       AA_el=np.copy(A_el)
    else:
       A_el=np.copy(AA_el)
 
    # traction bc on top layer of elts
    if experiment==1 and xc[iel]<a and yc[iel]<b and zc[iel]>Lz-hz:
          if not bc_fix[3*icon_V[4,iel]+2]: b_el[14]-=surf*pressbc*0.25
          if not bc_fix[3*icon_V[5,iel]+2]: b_el[17]-=surf*pressbc*0.25
          if not bc_fix[3*icon_V[6,iel]+2]: b_el[20]-=surf*pressbc*0.25
          if not bc_fix[3*icon_V[7,iel]+2]: b_el[23]-=surf*pressbc*0.25

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

    # assemble matrix and right hand side vector 

    if assembly==1:
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

    if assembly==2:
       for i in range(0,ndof_V_el):
           idof=local_to_global[i,iel]
           for j in range(0,ndof_V_el):
               VV_V[counter]=A_el[i,j]
               counter+=1
           b_fem[idof]+=b_el[i]

#end for iel
    
if experiment==2:
   for i in range(0,nn_V):
       if abs(x[i]-Lx/2)/Lx<eps and abs(y[i]-Ly/2)/Ly<eps and abs(z[i]-Lz)/Lz<eps:
          b_fem[3*i+2]-=Pforce

print("build FE system: %.3f s" % (clock.time()-start))

#################################################################
# solve system
#################################################################
start=clock.time()

if assembly==1: 
   A_fem=csr_matrix(A_fem)
if assembly==2: 
   A_fem=sps.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()

if solver==1: #direct solver
   sol = sps.linalg.spsolve(A_fem,b_fem)
if solver==2: #conjugate gradient solver
   sol = sps.linalg.cg(A_fem,b_fem)[0]

print("solve time: %.3f s" % (clock.time()-start))

#####################################################################
# put solution into separate x,y,z displacement arrays
#####################################################################
start=clock.time()

u,v,w=np.reshape(sol,(nn_V,3)).T

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %.5e %.5e " %(np.min(w),np.max(w)))

if debug: np.savetxt('displacement.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

#u[:]=x_V[:]**2
#v[:]=y_V[:]**2
#w[:]=z_V[:]**2

###############################################################################
# retrieve elemental pressure and strain tensor components
# elemental values are then projected onto the nodes.
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
sr=np.zeros(nel,dtype=np.float64)  
e_xx=np.zeros(nel,dtype=np.float64)  
e_yy=np.zeros(nel,dtype=np.float64)  
e_zz=np.zeros(nel,dtype=np.float64)  
e_xy=np.zeros(nel,dtype=np.float64)  
e_xz=np.zeros(nel,dtype=np.float64)  
e_yz=np.zeros(nel,dtype=np.float64)  
sigma_xx=np.zeros(nel,dtype=np.float64)  
sigma_yy=np.zeros(nel,dtype=np.float64)  
sigma_zz=np.zeros(nel,dtype=np.float64)  
sigma_xy=np.zeros(nel,dtype=np.float64)  
sigma_xz=np.zeros(nel,dtype=np.float64)  
sigma_yz=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.
    sq=0.
    tq=0.

    dNdr_V=basis_functions_V_dr(rq,sq,tq)
    dNds_V=basis_functions_V_ds(rq,sq,tq)
    dNdt_V=basis_functions_V_dt(rq,sq,tq)
    #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    #jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    #jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    #jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    #jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    #jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
    #jcbi = np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

    e_xx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    e_yy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    e_zz[iel]=np.dot(dNdz_V[:],w[icon_V[:,iel]])
    e_xy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
             +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e_xz[iel]=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5\
             +np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
    e_yz[iel]=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5\
             +np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5
    
    p[iel]=-(lambdaa+2*mu/3)*(e_xx[iel]+e_yy[iel]+e_zz[iel])

    sigma_xx[iel]=lambdaa*(e_xx[iel]+e_yy[iel]+e_zz[iel])+2*mu*e_xx[iel]
    sigma_yy[iel]=lambdaa*(e_xx[iel]+e_yy[iel]+e_zz[iel])+2*mu*e_yy[iel]
    sigma_zz[iel]=lambdaa*(e_xx[iel]+e_yy[iel]+e_zz[iel])+2*mu*e_zz[iel]
    sigma_xy[iel]=2*mu*e_xy[iel]
    sigma_xz[iel]=2*mu*e_xz[iel]
    sigma_yz[iel]=2*mu*e_yz[iel]

    #end for
#end for

sr[:]=np.sqrt(0.5*(e_xx[:]**2+e_yy[:]**2+e_zz[:]**2)+e_xy[:]**2+e_xz[:]**2+e_yz[:]**2)

print("     -> p (m,M) %.4f %.4f "   %(np.min(p),np.max(p)))
print("     -> e_xx (m,M) %.4e %.4e " %(np.min(e_xx),np.max(e_xx)))
print("     -> e_yy (m,M) %.4e %.4e " %(np.min(e_yy),np.max(e_yy)))
print("     -> e_zz (m,M) %.4e %.4e " %(np.min(e_zz),np.max(e_zz)))
print("     -> e_xy (m,M) %.4e %.4e " %(np.min(e_xy),np.max(e_xy)))
print("     -> e_xz (m,M) %.4e %.4e " %(np.min(e_xz),np.max(e_xz)))
print("     -> e_yz (m,M) %.4e %.4e " %(np.min(e_yz),np.max(e_yz)))
print("     -> sigma_xx (m,M) %.4e %.4e " %(np.min(sigma_xx),np.max(sigma_xx)))
print("     -> sigma_yy (m,M) %.4e %.4e " %(np.min(sigma_yy),np.max(sigma_yy)))
print("     -> sigma_zz (m,M) %.4e %.4e " %(np.min(sigma_zz),np.max(sigma_zz)))
print("     -> sigma_xy (m,M) %.4e %.4e " %(np.min(sigma_xy),np.max(sigma_xy)))
print("     -> sigma_xz (m,M) %.4e %.4e " %(np.min(sigma_xz),np.max(sigma_xz)))
print("     -> sigma_yz (m,M) %.4e %.4e " %(np.min(sigma_yz),np.max(sigma_yz)))

if debug:
   np.savetxt('p.ascii',np.array([xc,yc,zc,p]).T,header='# xc,yc,zc,p')
   np.savetxt('strain_e.ascii',np.array([xc,yc,zc,e_xx,e_yy,e_xy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p, strain & stress: %.3f s" % (clock.time()-start))

#####################################################################
# project p, strain & stress onto nodes using basis fcts at nodes
#####################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  
e_xx_n=np.zeros(nn_V,dtype=np.float64)  
e_yy_n=np.zeros(nn_V,dtype=np.float64)  
e_zz_n=np.zeros(nn_V,dtype=np.float64)  
e_xy_n=np.zeros(nn_V,dtype=np.float64)  
e_xz_n=np.zeros(nn_V,dtype=np.float64)  
e_yz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xx_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_zz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yz_n=np.zeros(nn_V,dtype=np.float64)  

r_V=np.array([-1,  1,  1, -1, -1,  1, 1 ,-1],np.float64)
s_V=np.array([-1, -1,  1,  1, -1, -1, 1 , 1],np.float64)
t_V=np.array([-1, -1, -1, -1,  1,  1, 1 , 1],np.float64)

if method==1: # center to node

   for iel in range(0,nel):
       for k in range(0,m_V):
           inode=icon_V[k,iel]
           e_xx_n[inode]+=e_xx[iel]
           e_yy_n[inode]+=e_yy[iel]
           e_zz_n[inode]+=e_zz[iel]
           e_xy_n[inode]+=e_xy[iel]
           e_xz_n[inode]+=e_xz[iel]
           e_yz_n[inode]+=e_yz[iel]
           sigma_xx_n[inode]+=sigma_xx[iel]
           sigma_yy_n[inode]+=sigma_yy[iel]
           sigma_zz_n[inode]+=sigma_zz[iel]
           sigma_xy_n[inode]+=sigma_xy[iel]
           sigma_xz_n[inode]+=sigma_xz[iel]
           sigma_yz_n[inode]+=sigma_yz[iel]
           q[inode]+=p[iel]
           count[inode]+=1
       #end for
   #end for
   q[:]/=count[:]
   e_xx_n[:]/=count[:]
   e_yy_n[:]/=count[:]
   e_zz_n[:]/=count[:]
   e_xy_n[:]/=count[:]
   e_xz_n[:]/=count[:]
   e_yz_n[:]/=count[:]
   sigma_xx_n[:]/=count[:]
   sigma_yy_n[:]/=count[:]
   sigma_zz_n[:]/=count[:]
   sigma_xy_n[:]/=count[:]
   sigma_xz_n[:]/=count[:]
   sigma_yz_n[:]/=count[:]

else:

   for iel in range(0,nel):
       for k in range(0,m_V):
           inode=icon_V[k,iel]
           rq=r_V[k]
           sq=s_V[k]
           tq=t_V[k]

           dNdr_V=basis_functions_V_dr(rq,sq,tq)
           dNds_V=basis_functions_V_ds(rq,sq,tq)
           dNdt_V=basis_functions_V_dt(rq,sq,tq)
           #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
           #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
           #jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
           #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
           #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
           #jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
           #jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
           #jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
           #jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
           #jcbi = np.linalg.inv(jcb)
           dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
           dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
           dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

           e_xx_n[inode]+=np.dot(dNdx_V[:],u[icon_V[:,iel]])
           e_yy_n[inode]+=np.dot(dNdy_V[:],v[icon_V[:,iel]])
           e_zz_n[inode]+=np.dot(dNdz_V[:],w[icon_V[:,iel]])
           e_xy_n[inode]+=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                         +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
           e_xz_n[inode]+=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5\
                         +np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
           e_yz_n[inode]+=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5\
                         +np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5

           count[inode]+=1
       #end for
   #end for iel
   e_xx_n[:]/=count[:]
   e_yy_n[:]/=count[:]
   e_zz_n[:]/=count[:]
   e_xy_n[:]/=count[:]
   e_xz_n[:]/=count[:]
   e_yz_n[:]/=count[:]

   print("     -> exx (m,M) %.4e %.4e " %(np.min(e_xx_n),np.max(e_xx_n)))
   print("     -> eyy (m,M) %.4e %.4e " %(np.min(e_yy_n),np.max(e_yy_n)))
   print("     -> ezz (m,M) %.4e %.4e " %(np.min(e_zz_n),np.max(e_zz_n)))
   print("     -> exy (m,M) %.4e %.4e " %(np.min(e_xy_n),np.max(e_xy_n)))
   print("     -> exz (m,M) %.4e %.4e " %(np.min(e_xz_n),np.max(e_xz_n)))
   print("     -> eyz (m,M) %.4e %.4e " %(np.min(e_yz_n),np.max(e_yz_n)))

   divv_n=e_xx_n[:]+e_yy_n[:]+e_zz_n[:]       
   q[:]=-(lambdaa+2*mu/3)*divv_n[:]
   sigma_xx_n=lambdaa*divv_n[:]+2*mu*e_xx_n[:]
   sigma_yy_n=lambdaa*divv_n[:]+2*mu*e_yy_n[:]
   sigma_zz_n=lambdaa*divv_n[:]+2*mu*e_zz_n[:]
   sigma_xy_n=2*mu*e_xy_n[:]
   sigma_xz_n=2*mu*e_xz_n[:]
   sigma_yz_n=2*mu*e_yz_n[:]

if debug:
   np.savetxt('strain_n.ascii',np.array([x_V,y_V,z_V,e_xx_n,e_yy_n,e_zz_n]).T)

print("compute nodal stress: %.3f s" % (clock.time()-start))

#####################################################################
# compute drms (root mean square displacement)
#####################################################################
start=clock.time()

errv=0
drms=0.
jcob=hx*hy*hz/8

if experiment>10: 

   for iel in range(0,nel):
       for iq in [-1,1]:
           for jq in [-1,1]:
               for kq in [-1,1]:
                   rq=iq/sqrt3
                   sq=jq/sqrt3
                   tq=kq/sqrt3
                   weightq=1.*1.*1.

                   N_V=basis_functions_V(rq,sq,tq)
                   #dNdr_V=basis_functions_V_dr(rq,sq,tq)
                   #dNds_V=basis_functions_V_ds(rq,sq,tq)
                   #dNdt_V=basis_functions_V_dt(rq,sq,tq)
                   #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                   #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                   #jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                   #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                   #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                   #jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                   #jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                   #jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                   #jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                   #jcob = np.linalg.det(jcb)

                   xq=np.dot(N_V,x_V[icon_V[:,iel]])
                   yq=np.dot(N_V,y_V[icon_V[:,iel]])
                   zq=np.dot(N_V,z_V[icon_V[:,iel]])
                   uq=np.dot(N_V,u[icon_V[:,iel]])
                   vq=np.dot(N_V,v[icon_V[:,iel]])
                   wq=np.dot(N_V,w[icon_V[:,iel]])

                   drms+=(uq**2+vq**2+wq**2)*jcob*weightq

                   errv+=((uq-uth(xq,yq,zq))**2+\
                          (vq-vth(xq,yq,zq))**2+\
                          (wq-wth(xq,yq,zq))**2)*weightq*jcob

               #end for
           #end for
       #end for
   #end for

   errv=np.sqrt(errv/Lx/Ly/Lz)

   drms=np.sqrt(drms/Lx/Ly/Lz)

   print("     -> nel= %6d ; errv: %e " %(nel,errv))

   print("     -> nel= %6d ; drms: %e" % (nel,drms))

print("compute errors & drms: %.3f s" % (clock.time() - start))

#####################################################################
# export profiles
#####################################################################
start=clock.time()

if experiment==1 or experiment==2 or experiment==3:
   
   xprofile=open("xprofile.ascii","w")
   yprofile=open("yprofile.ascii","w")
   zprofile=open("zprofile.ascii","w")
   topfile_e=open("topfile_strain.ascii","w")
   topfile_s=open("topfile_stress.ascii","w")
       
   zprofile.write("x,y,z,u,v,w,uth,vth,wth,q \n")

   for i in range(0,nn_V):
       xi=x_V[i]
       yi=y_V[i]
       zi=z_V[i]
       ui=u[i]
       vi=v[i]
       wi=w[i]
       qi=q[i]
       uthi=uth(xi,yi,zi)
       vthi=vth(xi,yi,zi)
       wthi=wth(xi,yi,zi)

       if abs(zi-Lz)/Lz<eps and abs(xi-Lx/2)/Lx<eps:
          xprofile.write("%e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,ui,vi,wi,uthi,vthi,wthi))
       if abs(zi-Lz)/Lz<eps and abs(yi-Ly/2)/Ly<eps:
          yprofile.write("%e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,ui,vi,wi,uthi,vthi,wthi))
       if abs(xi-Lx/2)/Lx<eps and abs(yi-Ly/2)/Ly<eps:
          zprofile.write("%e %e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,ui,vi,wi,uthi,vthi,wthi,qi))

   for iel in range(0,nel):
       if zc[iel]>Lz-hz:
          topfile_e.write("%e %e %e %e %e %e %e %e %e \n" %(\
                         xc[iel],yc[iel],zc[iel],\
                         e_xx[iel],e_yy[iel],e_zz[iel],\
                         e_xy[iel],e_xz[iel],e_yz[iel]))
   
       if zc[iel]>Lz-hz:
          topfile_s.write("%e %e %e %e %e %e %e %e %e \n" %(\
                         xc[iel],yc[iel],zc[iel],\
                         sigma_xx[iel],sigma_yy[iel],sigma_zz[iel],\
                         sigma_xy[iel],sigma_xz[iel],sigma_yz[iel]))

   xprofile.close()
   yprofile.close()
   zprofile.close()
   topfile_e.close()
   topfile_s.close()

if experiment==4 or experiment==5:

   #sectionm_file=open("section_mid.ascii","w")
   #sections_file=open("section_surf.ascii","w")
   #sectionm_file.write("#1 2 3 4 5 6 7  8  9 10 11 12 13\n")
   #sectionm_file.write("#x y z u v w p xx yy zz xy xz yz\n")
   #sections_file.write("#1 2 3 4 5 6 7  8  9 10 11 12 13\n")
   #sections_file.write("#x y z u v w p xx yy zz xy xz yz\n")
   #for i in range(0,nn_V):
   #    xi=x_V[i] ; yi=y_V[i] ; zi=z_V[i]
   #    ui=u[i]   ; vi=v[i]   ; wi=w[i]
   #    if abs(zi-(Lz-D/2.))/Lz<eps:
   #       sectionm_file.write("%e %e %e %e %e %e %e %e %e %e %e %e %e \n" %(
   #                          xi,yi,zi,ui,vi,wi,q[i],\
   #                          sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i], 
   #                          sigma_xy_n[i],sigma_xz_n[i],sigma_yz_n[i]))
   #    if abs(zi-Lz)/Lz<eps:
   #       sections_file.write("%e %e %e %e %e %e %e %e %e %e %e %e %e \n" %(
   #                          xi,yi,zi,ui,vi,wi,q[i],\
   #                          sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i], 
   #                          sigma_xy_n[i],sigma_xz_n[i],sigma_yz_n[i]))
   #sectionm_file.close()
   #sections_file.close()

   xprofile=open("xprofile.ascii","w")
   xprofile.write("#1 2 3 4 5 6 7  8  9 10 11 12 13\n")
   xprofile.write("#x y z u v w p xx yy zz xy xz yz\n")
   for i in range(0,nn_V):
       xi=x_V[i] ; yi=y_V[i] ; zi=z_V[i]
       ui=u[i]   ; vi=v[i]   ; wi=w[i]   ; qi=q[i]
       if abs(zi-Lz)/Lz<eps and abs(yi)/Ly<eps:
          xprofile.write("%e %e %e %e %e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,ui,vi,wi,qi,\
          sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i],sigma_xy_n[i],sigma_xz_n[i],sigma_yz_n[i]))
   xprofile.close()

   xprofile2=open("xprofile2.ascii","w") # at 2km depth
   xprofile2.write("#1 2 3 4 5 6 7  8  9 10 11 12 13\n")
   xprofile2.write("#x y z u v w p xx yy zz xy xz yz\n")
   for i in range(0,nn_V):
       xi=x_V[i] ; yi=y_V[i] ; zi=z_V[i]
       ui=u[i]   ; vi=v[i]   ; wi=w[i]   ; qi=q[i]
       if abs(zi-(Lz-D/2))/Lz<eps and abs(yi)/Ly<eps:
          xprofile2.write("%e %e %e %e %e %e %e %e %e %e %e %e %e \n" %(xi,yi,zi,ui,vi,wi,qi,\
          sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i],sigma_xy_n[i],sigma_xz_n[i],sigma_yz_n[i]))
   xprofile2.close()

   zprofile=open("zprofile.ascii","w")
   zprofile.write("#1 2 3  4 \n")
   zprofile.write("#z w p zz \n")
   for i in range(0,nn_V):
       xi=x_V[i] ; yi=y_V[i] ; zi=z_V[i] ; wi=w[i]   ; qi=q[i]
       if abs(xi)/Lx<eps and abs(yi)/Ly<eps:
          zprofile.write("%e %e %e %e \n" %(zi,wi,qi,sigma_zz_n[i]))
   zprofile.close()

print("export profiles: %.3f s" % (clock.time()-start))

#####################################################################
# plot of solution
#####################################################################
start=clock.time()

if True:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   if experiment==5:
      for i in range(0,nn_V):
          vtufile.write("%e %e %e \n" %(x_V[i],z_V[i],0))
   else:
      for i in range(0,nn_V):
          vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   p.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='strain' Format='ascii'> \n")
   sr.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
   e_xx.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
   e_yy.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_zz' Format='ascii'> \n")
   e_zz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
   e_xy.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_xz' Format='ascii'> \n")
   e_xz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='e_yz' Format='ascii'> \n")
   e_yz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
   sigma_xx.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
   sigma_yy.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
   sigma_zz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
   sigma_xy.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_xz' Format='ascii'> \n")
   sigma_xz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='sigma_yz' Format='ascii'> \n")
   sigma_yz.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   match experiment:
    case 2:
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
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   match experiment:
    case 2:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
      for i in range(0,nn_V):
          vtufile.write("%.20f %.20f %.20f \n" %(uth(x_V[i],y_V[i],z_V[i]),\
                                                 vth(x_V[i],y_V[i],z_V[i]),\
                                                 wth(x_V[i],y_V[i],z_V[i])))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (error)' Format='ascii'> \n")
      for i in range(0,nn_V):
          vtufile.write("%.20f %.20f %.20f \n" %(u[i]-uth(x_V[i],y_V[i],z_V[i]),\
                                                 v[i]-vth(x_V[i],y_V[i],z_V[i]),\
                                                 w[i]-wth(x_V[i],y_V[i],z_V[i])))
      vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmaxx_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmayy_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_zz (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmazz_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmaxy_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_xz (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmaxz_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='sigma_yz (th)' Format='ascii'> \n")
   #for i in range (0,nn_V):
   #    vtufile.write("%e\n" % (sigmayz_th(x_V[i],y_V[i],z_V[i])))
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   q.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
   sigma_xx_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
   sigma_yy_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
   sigma_zz_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
   sigma_xy_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_xz' Format='ascii'> \n")
   sigma_xz_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sigma_yz' Format='ascii'> \n")
   sigma_yz_n.tofile(vtufile, sep=" ", format="%.4e")
   vtufile.write("</DataArray>\n")

   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],
                                                   icon_V[4,iel],icon_V[5,iel],icon_V[6,iel],icon_V[7,iel]))
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
   print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")
