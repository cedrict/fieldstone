import sys as sys
import numpy as np
import time as timing
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import random
from numpy import linalg 

#------------------------------------------------------------------------------

def bx(x,y):
    if bench==1:
       return 0
    if bench==3:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
           (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
           (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
           1.-4.*y+12.*y*y-8.*y*y*y)
       return val

def by(x,y):
    if bench==1:
       return 0
    if bench==3:
       val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
       return val

#------------------------------------------------------------------------------

def eta(x,y):
    return 1e0 

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if bench==1:
       if x<0.0001:#  and y>0 and y<1: #left
          return 0
       if x>1-.0001:#  and y>0 and y<1: #right
          return 0
       if y<0.0001 and x>0 and x<1:
          return -1
       if y>1-.0001 and x>0 and x<1:
          return 1
    if bench==3:
       return x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)

def velocity_y(x,y):
    if bench==1:
       return 0
    if bench==3:
       return -y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)

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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(r,s):
    NP_0=1-r-s
    NP_1=r
    NP_2=s
    return np.array([NP_0,NP_1,NP_2],dtype=np.float64)

#------------------------------------------------------------------------------

def paint(x,y,exp):
    val=0
    #-----------
    if exp==1:
       if x<0.5:
          val=1
    #-----------
    if exp==2:
       if x<0.75:
          val=1
    #-----------
    if exp==3:
       if abs(x-0.5)<0.125:
          val=1
    #-----------
    if exp==4:
       if abs(y-0.5)<0.125:
          val=1
    return val

#------------------------------------------------------------------------------

ndim=2
ndofV=2
ndofP=1
mV=9
mP=3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

# bench=1: cakm06
# bench=3: Donea & Huerta

bench=3

Lx=1
Ly=1

nmarker_per_dim=3
mdistribution=3
    
mx=10
my=10
M=mx*my # nb of bins
C=2

nstep=10000

tfinal=20000

if int(len(sys.argv) == 6):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   CFL_nb=float(sys.argv[3])
   rk=int(sys.argv[4])
   exp=int(sys.argv[5])
   print('read stuff')
else:
   nelx = 100
   nely = 100
   CFL_nb=0.05
   rk=2
   exp=1

nnx=2*nelx+1
nny=2*nely+1
nel=nelx*nely
NV=(2*nelx+1)*(2*nely+1)
NP=3*nel
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP
hx=Lx/nelx
hy=Ly/nely

meth  = 2

print('bench=',bench)
print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('method=',meth)
print('CFL_nb=',CFL_nb)
print("-----------------------------")

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

eps=1e-8

eta_ref=1.
pnormalise=False
sparse=True

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

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

#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

for iel in range(nel):
    iconP[0,iel]=3*iel
    iconP[1,iel]=3*iel+1
    iconP[2,iel]=3*iel+2

counter=0
for iel in range(nel):
    xP[counter]=xV[iconV[8,iel]]
    yP[counter]=yV[iconV[8,iel]]
    counter+=1
    xP[counter]=xV[iconV[8,iel]]+hx/2
    yP[counter]=yV[iconV[8,iel]]
    counter+=1
    xP[counter]=xV[iconV[8,iel]]
    yP[counter]=yV[iconV[8,iel]]+hy/2
    counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(NfemV, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

if sparse:
   if pnormalise:
      A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   else:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

constr  = np.zeros(NfemP,dtype=np.float64)        # constraint matrix/vector
f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    if meth==2:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)           # int of shape functions P

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            if meth==1:
               NNNP[0:mP]=NNP(rq,sq)
            else:
               NNNP[0]=(m11+m21*xq+m31*yq)
               NNNP[1]=(m12+m22*xq+m32*yq)
               NNNP[2]=(m13+m23*xq+m33*yq)
            #print(NNNP[0],NNNP[1],NNNP[2])

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNNP[:]+=NNNP[:]*jcob*weightq

        # end for jq
    # end for iq

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
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
        if sparse and pnormalise:
           A_sparse[Nfem,NfemV+m2]=constr[m2]
           A_sparse[NfemV+m2,Nfem]=constr[m2]

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = timing.time()

if not sparse:
   if pnormalise:
      a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
      a_mat[Nfem,NfemV:Nfem]=constr
      a_mat[NfemV:Nfem,Nfem]=constr
   else:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   #end if
else:
   if pnormalise:
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   else:
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
#else:

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# solve system
######################################################################
start = timing.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (timing.time() - start))

profile=open("profile.ascii","w")
for i in range(0,NV):
    if abs(xV[i]-0.5)<1e-4:
       profile.write("%e %e\n" %(yV[i],u[i]))
profile.close()

###############################################################################
# now that we have solved the Stokes equations in the domain we can 
# use the velocity field to advect the passive particles
###############################################################################

###########################################################
# marker setup
###########################################################
start = timing.time()

nmarker_per_element=nmarker_per_dim*nmarker_per_dim

if mdistribution==1: # pure random
   nmarker=nel*nmarker_per_element
   swarm_x=np.empty(nmarker,dtype=np.float64)
   swarm_y=np.empty(nmarker,dtype=np.float64)
   swarm_c=np.empty(nmarker,dtype=np.int32)
   counter=0
   for iel in range(0,nel):
       x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
       x2=xV[iconV[1,iel]] ; y2=yV[iconV[1,iel]]
       x3=xV[iconV[2,iel]] ; y3=yV[iconV[2,iel]]
       x4=xV[iconV[3,iel]] ; y4=yV[iconV[3,iel]]
       for im in range(0,nmarker_per_element):
           # generate random numbers r,s between 0 and 1
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N1=0.25*(1-r)*(1-s)
           N2=0.25*(1+r)*(1-s)
           N3=0.25*(1+r)*(1+s)
           N4=0.25*(1-r)*(1+s)
           swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
           swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
           counter+=1

elif mdistribution==2: # regular
   nmarker=nel*nmarker_per_element
   swarm_x=np.empty(nmarker,dtype=np.float64)
   swarm_y=np.empty(nmarker,dtype=np.float64)
   swarm_c=np.empty(nmarker,dtype=np.int32)
   counter=0
   for iel in range(0,nel):
       x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
       x2=xV[iconV[1,iel]] ; y2=yV[iconV[1,iel]]
       x3=xV[iconV[2,iel]] ; y3=yV[iconV[2,iel]]
       x4=xV[iconV[3,iel]] ; y4=yV[iconV[3,iel]]
       for j in range(0,nmarker_per_dim):
           for i in range(0,nmarker_per_dim):
               r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
               s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               counter+=1

if mdistribution==3: # pure random whole domain
   nmarker=20000
   swarm_x=np.empty(nmarker,dtype=np.float64)
   swarm_y=np.empty(nmarker,dtype=np.float64)
   swarm_c=np.empty(nmarker,dtype=np.int32)
   for im in range(0,nmarker):
       swarm_x[im]=random.uniform(0,1)
       swarm_y[im]=random.uniform(0,1)

swarm_x0=np.empty(nmarker,dtype=np.float64)
swarm_y0=np.empty(nmarker,dtype=np.float64)
swarm_x0[:]=swarm_x[:]
swarm_y0[:]=swarm_y[:]

print("nmarker=",nmarker)

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("swarm setup: %.3f s" % (timing.time() - start))

#################################################################
# material layout
#################################################################
start = timing.time()

for im in range(0,nmarker):
    swarm_c[im] = paint(swarm_x[im],swarm_y[im],exp)

print("     -> swarm_c (m,M) %.4f %.4f " %(np.min(swarm_c),np.max(swarm_c)))

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_c]).T,header='# x,y,mat')
    
P=np.zeros(C,dtype=np.float64)
for im in range(0,nmarker):
    P[swarm_c[im]]+=1
P/=M    
print('     -> Pc=',P)

print("swarm paint: %.3f s" % (timing.time() - start))

######################################################################
# compute timestep 
######################################################################

hx=Lx/nelx
hy=Ly/nely

dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

print("dt= %.3e" %(dt))

#################################################################
# advecting markers and computing entropies
#################################################################

SfileC=open("S_Cedric.ascii","w")
SfileE=open("S_Erik.ascii","w")
mfile=open("marker0.ascii","w")
numberfile=open("number.ascii","w")
nfile=open("n.ascii","w")

total_time=0

S_entropy = np.zeros((nstep,6),float)
particles_0 = np.zeros((nmarker,9),float) 

particles_0[:,4]=swarm_x0
particles_0[:,5]=swarm_y0
particles_0[:,1]=swarm_x0
particles_0[:,2]=swarm_y0

particles_0[:,7]=swarm_c


for istep in range(0,nstep):

    print ('---------------------------------------')
    print ('----------------istep= %i -------------' %istep)
    print ('---------------------------------------')
    print ('time = %e' % (total_time))

    swarm_r=np.empty(nmarker,dtype=np.float64)
    swarm_s=np.empty(nmarker,dtype=np.float64)
    swarm_u=np.empty(nmarker,dtype=np.float64)
    swarm_v=np.empty(nmarker,dtype=np.float64)
    swarm_ielx=np.empty(nmarker,dtype=np.int)
    swarm_iely=np.empty(nmarker,dtype=np.int)
    number=np.zeros(nel,dtype=np.int32)
    n=np.zeros((M,C),dtype=np.int32)

    ###########################################################################
    # compute nb of species c in each element

    for im in range(0,nmarker):
        ielx=int(swarm_x[im]/Lx*mx)
        iely=int(swarm_y[im]/Ly*my)
        iel=mx*(iely)+ielx
        n[iel,swarm_c[im]]+=1

    for c in range(0,C):
        print('c=',c,'n(j,c) (m/M): ',np.min(n[:,c]),np.max(n[:,c]))

    nfile.write('%d %d %d %d %d \n'  %(istep,np.min(n[:,0]),np.max(n[:,0]),np.min(n[:,1]),np.max(n[:,1])))
    nfile.flush()

    ###########################################################################
    # compute total entropy 

    #compute denom for p_{j,c} 
    denom1=0
    for i in range(0,M):
        for c in range(0,C):
            denom1+=n[i,c]/P[c]
    #print('denom1=',denom1)

    #computing p_{j,c}
    pp1=np.zeros((M,C),dtype=np.float64)
    for j in range(0,M):
        for c in range(0,C):
            pp1[j,c]=n[j,c]/P[c]/denom1    

    S1=0
    for j in range(0,M):
        for c in range(0,C):
            if pp1[j,c]>0:
               S1-=pp1[j,c]*np.log(pp1[j,c])

    #print('C: S full= %e ' %S1)

    ###########################################################################
    # compute S_location entropy 

    #computing p_{j}
    pp3=np.zeros(M,dtype=np.float64)
    for j in range(0,M):
        for c in range(0,C):
            pp3[j]+=n[j,c]/P[c]/denom1    

    S2=0
    for j in range(0,M):
          if pp3[j]>0:
             S2-=pp3[j]*np.log(pp3[j])

    #print('C: S_location=',S2)


    ###########################################################################
    # compute S_location(species) entropy 

    #compute denom array for p_{c|j} 
    denom2=np.zeros(M,dtype=np.float64)
    for j in range(0,M):
        for c in range(0,C):
            denom2[j]+=n[j,c]/P[c]

    #print(denom2)

    #computing p_{c|j}
    pp2=np.zeros((M,C),dtype=np.float64)
    for j in range(0,M):
        for c in range(0,C):
            pp2[j,c]=n[j,c]/P[c]/denom2[j]    

    SS3=np.zeros(M,dtype=np.float64)
    for j in range(0,M):
        for c in range(0,C):
            if pp2[j,c]>0:
               SS3[j]-=pp2[j,c]*np.log(pp2[j,c]) #S_j(species)

    S3=0
    for j in range(0,M):
        S3+=pp3[j]*SS3[j]    
    
    #print('C: S_location(species)=',S3)

    S1b=S1/np.log(M)
    S2b=S2/np.log(C)
    S3b=S3/np.log(C)

    print('C: S full             : %e | normalised: %e ' %(S1,S1b))
    print('C: S_location         : %e | normalised: %e ' %(S2,S2b))
    print('C: S_location(species): %e | normalised: %e ' %(S3,S3b))

    SfileC.write("%e %e %e %e %e %e %e \n" %(istep*dt,S1,S2,S3,S1b,S2b,S3b))
    SfileC.flush()

    ###########################################################################
    # Erik vd Wiel approach
    ###########################################################################

    c = np.zeros(C,float)
    P_c = np.zeros(C,float) 
    bins = np.zeros((M,C),float)
    bins2 = np.zeros((M,C),float)
    bins3= np.zeros(M,float)
    bins4 = np.zeros((M,C),float)
    S_entropy_array = np.zeros(M,float)  

    l_x = 1.0 / mx
    for j in range (0,nmarker):
        for l in range(0,my):
            cx =  l * l_x
            cx2 = (l+1) * l_x
            for k in range(0,mx):
                cy = k * l_x
                cy2 = (k+1) * l_x
                counter = k + (mx*l)
                if particles_0[j,1] >= cx and particles_0[j,1] <= cx2 and\
                   particles_0[j,2] >= cy and particles_0[j,2] <= cy2:
                    particles_0[j,8] = counter
                    for m in range(0,C):
                        if particles_0[j,7] == (m):
                            bins[counter,m] +=   1

    #print (bins[:,0])
    #print (bins[:,1])

    for j in range(0,M):
        for k in range(0,C):
            c[k] += bins[j][k] 

    #print(c)

    total = sum(c)
    for k in range(0,C):
        P_c[k] = c[k] / M 

    #print(P_c)

    sum_Njc_Pc =0
    for j in range(0,M):
        for k in range(0,C):
            sum_Njc_Pc += bins[j][k] / P_c[k]  #denom1

    for j in range(0,M):
        for k in range(0,C):
            #   Ni,c / Pc (i: 1 tot bins, c: 1 tot species) voor Pj,c
            bins2[j][k] = bins[j][k] / P_c[k] / sum_Njc_Pc                  #p_{j,c}
            #   Nj,c / Pc (c: 1 tot species gesommeerd -> voor Pj
            bins3[j] +=  bins[j][k] / P_c[k]                                #denom2=\sum_c n_j,c P_c
        for k in range(0,C):
            #   Nj,c / Pc ) / all part/all classes -> voor Pc,j
            bins4[j,k] = bins[j,k] / P_c[k] / bins3[j]              #p_c|j 
        bins3[j] /= sum_Njc_Pc


    #---------------------------
    #compute S_full
    #---------------------------
    for j in range(0,M):
        for k in range(0,C):         
            if bins2[j][k] != 0:
               S_entropy[istep][0] -= bins2[j][k] * np.log(bins2[j][k]) ## S(full)=-\sum_j\sum_c p_{j,c} log p_{j,c}

    #---------------------------
    #compute S_location
    #---------------------------
    for j in range(0,M):
        if bins3[j] !=0:
           S_entropy[istep,1] -= bins3[j] * np.log(bins3[j])   ## S(location)    

    #---------------------------
    #compute S_location(species)
    #---------------------------
    for j in range(0,M):
        for k in range(0,C):
            if bins4[j,k] !=0:
               S_entropy_array[j] -=  bins4[j][k] * np.log(bins4[j][k]) #S_j(species)=-\sum_c p_{c|j} log p_{c|j}
    for j in range(0,M):
        S_entropy[istep,2] +=  S_entropy_array[j] * bins3[j] ## S_location(species) = \sum_j p_j S_j(species)

    S_entropy[istep,3] = S_entropy[istep,0]/np.log(M)
    S_entropy[istep,4] = S_entropy[istep,1]/np.log(C)
    S_entropy[istep,5] = S_entropy[istep,2]/np.log(C)

    print('E: S full             : %e | normalised: %e ' %(S_entropy[istep,0],S_entropy[istep,3]))
    print('E: S_location         : %e | normalised: %e ' %(S_entropy[istep,1],S_entropy[istep,4]))
    print('E: S_location(species): %e | normalised: %e ' %(S_entropy[istep,2],S_entropy[istep,5]))

    SfileE.write("%e %e %e %e %e %e %e \n" %(istep*dt,S_entropy[istep,0],S_entropy[istep,1],\
                                                      S_entropy[istep,2],S_entropy[istep,3],\
                                                      S_entropy[istep,4],S_entropy[istep,5]))
    SfileE.flush()

    ###########################################################################
    start = timing.time()

    if rk==1:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           number[iel]+=1
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
           swarm_r[im]=-1+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_u[im]=um
           swarm_v[im]=vm
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
           swarm_ielx[im]=ielx
           swarm_iely[im]=iely
       #end for

    if rk==2:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
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
           number[iel]+=1
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
           swarm_r[im]=-1+2*(xm-x0)/hx
           swarm_s[im]=-1+2*(ym-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_u[im]=um
           swarm_v[im]=vm
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
           swarm_ielx[im]=ielx
           swarm_iely[im]=iely
       #end for

    if rk==3:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
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
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
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
           number[iel]+=1
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
           swarm_r[im]=-1+2*(xC-x0)/hx
           swarm_s[im]=-1+2*(yC-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           uC=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vC=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_u[im]=uA+4*uB+uC
           swarm_v[im]=vA+4*vB+vC
           swarm_x[im]+=swarm_u[im]*dt/6
           swarm_y[im]+=swarm_v[im]*dt/6
           swarm_ielx[im]=ielx
           swarm_iely[im]=iely
       #end for


    particles_0[:,1]=swarm_x
    particles_0[:,2]=swarm_y

    numberfile.write("%e %e \n" %(np.min(number),np.max(number)))
    numberfile.flush()
    mfile.write("%e %e \n" %(swarm_x[0],swarm_y[0]))
    mfile.flush()

    print("advect particles: %.3f s" % (timing.time() - start))

    ###########################################################################

    start = timing.time()

    if istep%1==0:

       filename = 'swarm_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%.10e %.10e %.10e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[i],swarm_v[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32'  Name='c' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%d \n" %(swarm_c[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32'  Name='x0' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%e \n" %(swarm_x0[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32'  Name='y0' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%e \n" %(swarm_y0[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32'  Name='ielx' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%e \n" %(swarm_ielx[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32'  Name='iely' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%e \n" %(swarm_iely[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
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

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" %(p[iconP[0,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='nmarker' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" %(number[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
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
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       np.savetxt('swarm_{:04d}.ascii'.format(istep),np.array([swarm_x,swarm_y,swarm_c]).T,header='# x,y,c')

    print("export to vtu: %.3f s" % (timing.time() - start))

    total_time+=dt

    if total_time>tfinal:
       exit()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
