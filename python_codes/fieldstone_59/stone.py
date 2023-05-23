import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as timing
from scipy.sparse import lil_matrix
from numpy import linalg as LA

#------------------------------------------------------------------------------
def gx(x,y,theta):
    return 9.81*np.sin(theta)

def gy(x,y,theta):
    return -9.81*np.cos(theta)

#------------------------------------------------------------------------------

def viscosity(exx,eyy,exy,pq,iter,x,y,T,d):
    if iter==0:
       e2=1e-10
    else:
       e2=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)

    #glen's flow law
    if T<263:
       A=3.61e5
       n=3
       Q=60e3
    else:
       A=1.73e21
       n=3
       Q=139e3
    #end if
    eta_glen=0.5*A**(-1./n)*e2**(1./n-1)*np.exp(Q/n/8.314/T) *1e6

    #dislocation (indep of d)
    if T<262:
       A=5e5
       n=4
       Q=64.e3
    else:
       A=6.96e23
       n=4
       Q=155.e3
    #end if
    eta_disl=0.5*A**(-1./n)*e2**(1./n-1)*np.exp(Q/n/8.314/T) *1e6

    #GBS-limited creep
    if T<262:
       A=1.1e2
       n=1.8
       Q=70.e3
       p=1.4
    else:
       A=8.5e37
       n=1.8
       Q=250.e3
       p=1.4
    #end if
    eta_gbs=0.5*A**(-1./n)*d**(p/n) *e2**(1./n-1)*np.exp(Q/n/8.314/T) *1e6

    if rheology==1:
       val=eta_glen
    if rheology==2:
       val=eta_disl
    if rheology==3:
       val=eta_gbs
    if rheology==4:
       val=1./(1./eta_gbs+1./eta_disl)

    val=min(1.e17,val)
    val=max(1.e11,val)
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

cm=0.01
year=3600*24*365.

print("-----------------------------")
print("---------- stone 59 ---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   solver = int(sys.argv[4])
   benchmark = int(sys.argv[5])
else:
   nelx = 100 # multiple of 4
   nely = 25
   visu = 1
   solver = 2 
   benchmark=3

Lx=125e3 
Ly=2.500e3
g=9.81
theta=0.1/180.*np.pi

#rheology=1: glen
#rheology=2: disl
#rheology=3: gbs
#rheology=4: disl+gbs

rheology=4

nnx=2*nelx+1                  # number of elements, x direction
nny=2*nely+1                  # number of elements, y direction
NV=nnx*nny                    # number of nodes
nel=nelx*nely                 # number of elements, total
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs

eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

rho=917

tol_nl=1e-6

if solver==1:
   use_SchurComplementApproach=True
   use_preconditioner=True
   niter_stokes=250
   solver_tolerance=1e-6
else:
   use_SchurComplementApproach=False

eta_ref=1.e13      # scaling of G blocks
scaling_coeff=eta_ref/Ly

niter_min=1
niter=100

if use_SchurComplementApproach:
   ls_conv_file=open("linear_solver_convergence.ascii","w")
   ls_niter_file=open("linear_solver_niter.ascii","w")

sparse=True

rVnodes=[-1,+1,1,-1, 0,1,0,-1,0]
sVnodes=[-1,-1,1,+1,-1,0,1, 0,0]

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("Nfem=",Nfem)
print("sparse",sparse)
print("hx",hx)
print("hy",hy)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

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

start = timing.time()

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

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

u     =np.zeros(NV,dtype=np.float64)    # x-component velocity
v     =np.zeros(NV,dtype=np.float64)    # y-component velocity
bc_fix=np.zeros(NfemV,dtype=np.bool)    # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(xV[i],yV[i])
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = rho*g*np.sin(theta)/2/1e15*yV[i]*(2*Ly-yV[i]) 
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# prescribe temperature in the ice sheet 
#################################################################
T  =np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    if yV[i]<1000:
       T[i]=-(yV[i])*(271-245)/1000. + 271.
    else:
       T[i]=-(yV[i]-1000)/1500.*(245-244)+245.
    #end if
#end for

#################################################################
# prescribe grain size in the ice sheet 
#################################################################
d  =np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    if yV[i]>1100:
       d[i]=0.0035
    elif yV[i]>300:
       d[i]=0.0025
    else:
       d[i]=0.007
    #end if
#end for

#------------------------------------------------------------------------------
# non-linear iterations
#------------------------------------------------------------------------------
method=1
if method==1:
   c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
elif method==2:
   c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
p       = np.zeros(NfemP,dtype=np.float64)        # pressure field 
pold    = np.zeros(NfemP,dtype=np.float64)        # pressure field 
Res     = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol     = np.zeros(Nfem,dtype=np.float64)         # solution vector 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
conv_inf= np.zeros(niter,dtype=np.float64)        
conv_two= np.zeros(niter,dtype=np.float64)        
conv_inf_Ru = np.zeros(niter,dtype=np.float64)        
conv_inf_Rv = np.zeros(niter,dtype=np.float64)        
conv_inf_Rp = np.zeros(niter,dtype=np.float64)        
solP    = np.zeros(NfemP,dtype=np.float64)  
solV    = np.zeros(NfemV,dtype=np.float64)  
a_mat   = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs     = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

for iter in range(0,niter):

   print("--------------------------")
   print("iter=", iter)
   print("--------------------------")

   #################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################

   if sparse:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
   else:   
      K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
      G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

   M_mat = np.zeros((NfemP,NfemP),dtype=np.float64) # schur precond
   f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
   h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
   xq    = np.zeros(9*nel,dtype=np.float64)        # x coords of q points 
   yq    = np.zeros(9*nel,dtype=np.float64)        # y coords of q points 
   etaq  = np.zeros(9*nel,dtype=np.float64)        # viscosity of q points 
   pq    = np.zeros(9*nel,dtype=np.float64)        # pressure of q points 
   srq   = np.zeros(9*nel,dtype=np.float64)        # strain rate of q points 

   counter=0
   for iel in range(0,nel):

       # set arrays to 0 every loop
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)
       M_el=np.zeros((mP,mP),dtype=np.float64)  

       # integrate viscous term at 4 quadrature points
       for jq in [0,1,2]:
           for iq in [0,1,2]:

               # position & weight of quad. point
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               NNNV[0:9]=NNV(rq,sq)
               dNNNVdr[0:9]=dNNVdr(rq,sq)
               dNNNVds[0:9]=dNNVds(rq,sq)
               NNNP[0:4]=NNP(rq,sq)

               # calculate jacobian matrix
               jcb=np.zeros((ndim,ndim),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb)

               # compute dNdx & dNdy & strainrate
               exxq=0.0
               eyyq=0.0
               exyq=0.0
               Tq=0.0
               dq=0.0
               for k in range(0,mV):
                   xq[counter]+=NNNV[k]*xV[iconV[k,iel]]
                   yq[counter]+=NNNV[k]*yV[iconV[k,iel]]
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                   exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                   eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                   exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
                   Tq+=NNNV[k]*T[iconV[k,iel]]
                   dq+=NNNV[k]*d[iconV[k,iel]]
                
               # compute pressure
               for k in range(0,mP):
                   pq[counter]+=NNNP[k]*p[iconP[k,iel]]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute effective plastic viscosity
               etaq[counter]=viscosity(exxq,eyyq,exyq,pq[counter],\
                                                   iter,xq[counter],yq[counter],Tq,dq)
               srq[counter]=np.sqrt(0.5*(exxq*exxq+eyyq*eyyq)+exyq*exyq)

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*gx(xq,yq,theta)*rho
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,theta)*rho

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob

               for i in range(0,mP):
                   for j in range(0,mP):
                       M_el[i,j]+=NNNP[i]*NNNP[j]*weightq*jcob/etaq[counter]
                   # end for j
               # end for i

               counter+=1
           # end for iq 
       # end for jq 

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
                  #end for jkk
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
               # end if 
           # end for i1 
       #end for k1 

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
                       # end if
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   if sparse:
                      A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*scaling_coeff
                      A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
                   else:
                      G_mat[m1,m2]+=G_el[ikk,jkk]*scaling_coeff
                   #end if
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,mP):
           m1=iconP[k1,iel]
           h_rhs[m1]+=h_el[k1]*scaling_coeff
           for k2 in range(0,mP):
               m2=iconP[k2,iel]
               M_mat[m1,m2]+=M_el[k1,k2]
           #end for k2 
       #end for k1

   # end for iel 

   if not sparse:
      print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
      print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
   print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
   print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))


   print("     -> srq  (m,M) %.4e %.4e " %(np.min(srq),np.max(srq)))
   print("     -> etaq (m,M) %.4e %.4e " %(np.min(etaq),np.max(etaq)))

   print("build FE matrix: %.3f s" % (timing.time() - start))

   ######################################################################
   # assemble K, G, GT, f, h into A and rhs
   ######################################################################
   start = timing.time()

   if use_SchurComplementApproach:

      # convert matrices to CSR format
      G_mat=sps.csr_matrix(G_mat)
      K_mat=sps.csr_matrix(K_mat)
      M_mat=sps.csr_matrix(M_mat)

      Res[0:NfemV]=K_mat.dot(solV)+G_mat.dot(solP)-f_rhs
      Res[NfemV:Nfem]=G_mat.T.dot(solV)-h_rhs

      # declare necessary arrays
      rvect_k=np.zeros(NfemP,dtype=np.float64) 
      pvect_k=np.zeros(NfemP,dtype=np.float64) 
      zvect_k=np.zeros(NfemP,dtype=np.float64) 
      ptildevect_k=np.zeros(NfemV,dtype=np.float64) 
      dvect_k=np.zeros(NfemV,dtype=np.float64) 
   
      # carry out solve
      solP[:]=0.
      solV=sps.linalg.spsolve(K_mat,f_rhs)
      rvect_k=G_mat.T.dot(solV)-h_rhs
      rvect_0=np.linalg.norm(rvect_k)
      if use_preconditioner:
         zvect_k=sps.linalg.spsolve(M_mat,rvect_k)
      else:
         zvect_k=rvect_k
      pvect_k=zvect_k
      for k in range (0,niter_stokes):
          ptildevect_k=G_mat.dot(pvect_k)
          dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)
          alpha=(rvect_k.dot(zvect_k))/(ptildevect_k.dot(dvect_k))
          solP+=alpha*pvect_k
          solV-=alpha*dvect_k
          rvect_kp1=rvect_k-alpha*G_mat.T.dot(dvect_k)
          if use_preconditioner:
              zvect_kp1=sps.linalg.spsolve(M_mat,rvect_kp1)
          else:
              zvect_kp1=rvect_kp1
          beta=(zvect_kp1.dot(rvect_kp1))/(zvect_k.dot(rvect_k))
          pvect_kp1=zvect_kp1+beta*pvect_k
          rvect_k=rvect_kp1
          pvect_k=pvect_kp1
          zvect_k=zvect_kp1
          xi=np.linalg.norm(rvect_k)/rvect_0
          ls_conv_file.write("%d %6e \n"  %(k,xi))
          print("lin.solver: %d %6e" % (k,xi))
          if xi<solver_tolerance:
             ls_niter_file.write("%d \n"  %(k))
             break 
      u,v=np.reshape(solV[0:NfemV],(NV,2)).T
      p=solP[0:NfemP]*scaling_coeff
   else:
      rhs[0:NfemV]=f_rhs
      rhs[NfemV:Nfem]=h_rhs
      if not sparse:
         a_mat[:,:]=0
         a_mat[0:NfemV,0:NfemV]=K_mat
         a_mat[0:NfemV,NfemV:Nfem]=G_mat
         a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
         Res=a_mat.dot(sol)-rhs
         sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
      else:
         sparse_matrix=A_sparse.tocsr()
         Res=sparse_matrix.dot(sol)-rhs
         sol=sps.linalg.spsolve(sparse_matrix,rhs)

      u,v=np.reshape(sol[0:NfemV],(NV,2)).T
      p=sol[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("solve system: %.3f s - Nfem %d" % (timing.time() - start, Nfem))

   #################################################################
   # compute non-linear residual
   #################################################################
   start = timing.time()

   if iter==0:
      Res0_inf=LA.norm(Res,np.inf)
      Res0_two=LA.norm(Res,2)

   Res_inf=LA.norm(Res,np.inf)
   Res_two=LA.norm(Res,2)

   print("Nonlinear residual (inf. norm) %.7e" % (Res_inf/Res0_inf))
   print("Nonlinear residual (two  norm) %.7e" % (Res_two/Res0_two))

   conv_inf[iter]=Res_inf/Res0_inf
   conv_two[iter]=Res_two/Res0_two

   if Res_inf/Res0_inf<tol_nl and iter>niter_min:
      break

   np.savetxt('nonlinear_conv_inf.ascii',np.array(conv_inf[0:niter]).T)
   np.savetxt('nonlinear_conv_two.ascii',np.array(conv_two[0:niter]).T)

   Res_u,Res_v=np.reshape(Res[0:NfemV],(NV,2)).T
   Res_p=Res[NfemV:Nfem]
   
   conv_inf_Ru[iter]=LA.norm(Res_u,np.inf)
   conv_inf_Rv[iter]=LA.norm(Res_v,np.inf)
   conv_inf_Rp[iter]=LA.norm(Res_p,np.inf)

   np.savetxt('nonlinear_conv_inf_Ru.ascii',np.array(conv_inf_Ru[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rv.ascii',np.array(conv_inf_Rv[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rp.ascii',np.array(conv_inf_Rp[0:niter]).T)

   #np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
   #np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x,y,u,v]).T,header='# x,y,u,v')
   #np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')
   #np.savetxt('srq_{:04d}.ascii'.format(iter),np.array([xq,yq,srq]).T,header='# x,y,sr')

   print("computing res norms: %.3f s" % (timing.time() - start))

   #####################################################################
   # interpolate pressure onto velocity grid points
   #####################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)
   Res_q=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):
       q[iconV[0,iel]]=p[iconP[0,iel]]
       q[iconV[1,iel]]=p[iconP[1,iel]]
       q[iconV[2,iel]]=p[iconP[2,iel]]
       q[iconV[3,iel]]=p[iconP[3,iel]]
       q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
       q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
       q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
       q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
       q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+\
                        p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

   for iel in range(0,nel):
       Res_q[iconV[0,iel]]=Res_p[iconP[0,iel]]
       Res_q[iconV[1,iel]]=Res_p[iconP[1,iel]]
       Res_q[iconV[2,iel]]=Res_p[iconP[2,iel]]
       Res_q[iconV[3,iel]]=Res_p[iconP[3,iel]]
       Res_q[iconV[4,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]])*0.5
       Res_q[iconV[5,iel]]=(Res_p[iconP[1,iel]]+Res_p[iconP[2,iel]])*0.5
       Res_q[iconV[6,iel]]=(Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.5
       Res_q[iconV[7,iel]]=(Res_p[iconP[3,iel]]+Res_p[iconP[0,iel]])*0.5
       Res_q[iconV[8,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]]+\
                            Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.25

   #np.savetxt('q_{:04d}.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute strainrate 
   ######################################################################
   start = timing.time()

   xc = np.zeros(nel,dtype=np.float64)  
   yc = np.zeros(nel,dtype=np.float64)  
   pc = np.zeros(nel,dtype=np.float64)  
   Tc = np.zeros(nel,dtype=np.float64)  
   dc = np.zeros(nel,dtype=np.float64)  
   exx = np.zeros(nel,dtype=np.float64)  
   eyy = np.zeros(nel,dtype=np.float64)  
   exy = np.zeros(nel,dtype=np.float64)  
   sr  = np.zeros(nel,dtype=np.float64)  

   for iel in range(0,nel):

       rq = 0.0
       sq = 0.0
       weightq = 2.0 * 2.0

       NNNV[0:9]=NNV(rq,sq)
       dNNNVdr[0:9]=dNNVdr(rq,sq)
       dNNNVds[0:9]=dNNVds(rq,sq)

       jcb=np.zeros((ndim,ndim),dtype=np.float64)
       for k in range(0,mV):
           jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
           jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
           jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
           jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
       jcob=np.linalg.det(jcb)
       jcbi=np.linalg.inv(jcb)

       for k in range(0,mV):
           dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
           dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

       for k in range(0,mV):
           xc[iel] += NNNV[k]*xV[iconV[k,iel]]
           yc[iel] += NNNV[k]*yV[iconV[k,iel]]
           Tc[iel] += NNNV[k]*T[iconV[k,iel]]
           dc[iel] += NNNV[k]*d[iconV[k,iel]]
           exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
           eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
           exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

       sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

       for k in range(0,mP):
           pc[iel] += NNNP[k]*p[iconP[k,iel]]

   print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
   print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
   print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
   print("     -> sr  (m,M) %.5e %.5e " %(np.min(sr),np.max(sr)))
   print("     -> pc  (m,M) %.5e %.5e " %(np.min(pc),np.max(pc)))
   print("     -> Tc  (m,M) %.5e %.5e " %(np.min(Tc),np.max(Tc)))
   print("     -> dc  (m,M) %.5e %.5e " %(np.min(dc),np.max(dc)))

   print("compute press & sr: %.3f s" % (timing.time() - start))

   #####################################################################

   avrg_press=np.sum(pc)/nel

   print ("     -> avrg press. %.5e" % avrg_press)

   #####################################################################
   # project strainrate onto velocity grid
   #####################################################################
   start = timing.time()

   exxn=np.zeros(NV,dtype=np.float64)
   eyyn=np.zeros(NV,dtype=np.float64)
   exyn=np.zeros(NV,dtype=np.float64)
   srn=np.zeros(NV,dtype=np.float64)
   c=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):
       for i in range(0,mV):
           NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i])
           dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i])
           dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i])
           jcb=np.zeros((ndim,ndim),dtype=np.float64)
           for k in range(0,mV):
               jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
               jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
               jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
               jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
           jcob=np.linalg.det(jcb)
           jcbi=np.linalg.inv(jcb)
           for k in range(0,mV):
               dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
               dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
           e_xx=0.
           e_yy=0.
           e_xy=0.
           for k in range(0,mV):
               e_xx += dNNNVdx[k]*u[iconV[k,iel]]
               e_yy += dNNNVdy[k]*v[iconV[k,iel]]
               e_xy += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                       0.5*dNNNVdx[k]*v[iconV[k,iel]]
           exxn[iconV[i,iel]]+=e_xx
           eyyn[iconV[i,iel]]+=e_yy
           exyn[iconV[i,iel]]+=e_xy
           c[iconV[i,iel]]+=1.
       # end for i
   # end for iel

   exxn/=c
   eyyn/=c
   exyn/=c

   srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

   print("     -> exx (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
   print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
   print("     -> exy (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
   print("     -> sr  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

   print("compute nod strain rate: %.3f s" % (timing.time() - start))
 
   ######################################################################
   etan=np.zeros(NV,dtype=np.float64)

   for i in range (0,NV):
       etan[i]=viscosity(exxn[i],eyyn[i],exyn[i],q[i],iter,xV[i],yV[i],T[i],d[i])

   ######################################################################
   # generate vtu output at every nonlinear iteration
   ######################################################################

   filename = 'solution_nl_{:04d}.vtu'.format(iter)
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
   vtufile.write("<DataArray type='Float32' Name='sr (middle) (x10^-15)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (sr[iel]*1e15))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_u[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_v[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_q[i])
   vtufile.write("</DataArray>\n")
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

#------------------------------------------------------------------------------
# end of non-linear iterations
#------------------------------------------------------------------------------

######################################################################
# compute averaged elemental strainrate 
# I use a 5 point quadrature rule (per dimension) and compute the 
# average strain rate tensor components per element. 
######################################################################
start = timing.time()

exx_avrg = np.zeros(nel,dtype=np.float64)  
eyy_avrg = np.zeros(nel,dtype=np.float64)  
exy_avrg = np.zeros(nel,dtype=np.float64)  
sr_avrg  = np.zeros(nel,dtype=np.float64)  

qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.  
qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.  
qc5c=0.    
qw5a=(322.-13.*np.sqrt(70.))/900.
qw5b=(322.+13.*np.sqrt(70.))/900.
qw5c=128./225.
qcoords5=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
qweights5=[qw5a,qw5b,qw5c,qw5b,qw5a]

for iel in range(0,nel):
    for jq in [0,1,2,3,4]:
        for iq in [0,1,2,3,4]:
            # position & weight of quad. point
            rq=qcoords5[iq]
            sq=qcoords5[jq]
            weightq=qweights5[iq]*qweights5[jq]
            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)
            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            # end for
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)
            # compute dNdx & dNdy & strainrate
            exxq=0.0
            eyyq=0.0
            exyq=0.0
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
            # end for
            exx_avrg[iel] += exxq*jcob*weightq
            eyy_avrg[iel] += eyyq*jcob*weightq
            exy_avrg[iel] += exyq*jcob*weightq
        # end for
    # end for
    exx_avrg[iel] /= (hx*hy) 
    eyy_avrg[iel] /= (hx*hy) 
    exy_avrg[iel] /= (hx*hy) 
    sr_avrg[iel]=np.sqrt(0.5*(exx_avrg[iel]**2+eyy_avrg[iel]**2)+exy_avrg[iel]**2)
#end for

print("     -> exx_avrg (m,M) %.6e %.6e " %(np.min(exx_avrg),np.max(exx_avrg)))
print("     -> eyy_avrg (m,M) %.6e %.6e " %(np.min(eyy_avrg),np.max(eyy_avrg)))
print("     -> exy_avrg (m,M) %.6e %.6e " %(np.min(exy_avrg),np.max(exy_avrg)))
print("     -> sr_avrg  (m,M) %.6e %.6e " %(np.min(sr_avrg),np.max(sr_avrg)))

print("compute avrg elemental strain rate: %.3f s" % (timing.time() - start))

#np.savetxt('sr_avrg.ascii',np.array([xc,yc,exx_avrg,eyy_avrg,exy_avrg]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################

core1file=open('core1.ascii',"w")
core2file=open('core2.ascii',"w")
for i in range(0,NV):
    if abs(xV[i]-Lx/2)/Lx<eps:
       core1file.write("%5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e\n"\
                    %(xV[i],yV[i],u[i],v[i],q[i],T[i],d[i],exxn[i],eyyn[i],exyn[i],srn[i],etan[i]))
    if abs(xV[i]-Lx/4)/Lx<eps:
       core2file.write("%5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e\n"\
                    %(xV[i],yV[i],u[i],v[i],q[i],T[i],d[i],exxn[i],eyyn[i],exyn[i],srn[i],etan[i]))

core1file.close()
core2file.close()

#midfile=open('nueffs.ascii',"w")
#for i in range(0,NV):
#    if abs(xV[i]-Lx/2)/Lx<eps:
#       eta10= viscosity(0.,0.,1e-10,0.,1,0.,0.,T[i],d[i])
#       eta11= viscosity(0.,0.,1e-11,0.,1,0.,0.,T[i],d[i])
#       eta12= viscosity(0.,0.,1e-12,0.,1,0.,0.,T[i],d[i])
#       eta13= viscosity(0.,0.,1e-13,0.,1,0.,0.,T[i],d[i])
#       midfile.write("%5e %5e %5e %5e %5e %5e %5e\n"\
#                      %(yV[i],T[i],d[i],eta10,eta11,eta12,eta13))
#midfile.close()


#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

filename = 'solution.vtu'
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
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xx (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xx (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx_avrg[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xy (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xy (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy_avrg[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (middle)(x10^-15)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (sr[iel]*1e15))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (avrg)(x10^-15)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (sr_avrg[iel]*1e15))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],iter,xc[iel],yc[iel],Tc[iel],dc[iel])
    vtufile.write("%10e\n" %eta) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity (log)' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],iter,xc[iel],yc[iel],Tc[iel],dc[iel])
    vtufile.write("%10e\n" %(np.log10(eta))) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],iter,xc[iel],yc[iel],Tc[iel],dc[iel])
    vtufile.write("%10e\n" %(2.*eta*exx_avrg[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],iter,xc[iel],yc[iel],Tc[iel],dc[iel])
    vtufile.write("%10e\n" %(2.*eta*eyy_avrg[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],iter,xc[iel],yc[iel],Tc[iel],dc[iel])
    vtufile.write("%10e\n" %(2.*eta*exy_avrg[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %etan[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_u[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_v[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exxn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %srn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (x10^-15)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(srn[i]*1e15))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(T[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='d' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(d[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
