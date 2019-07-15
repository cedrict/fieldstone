import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time

#------------------------------------------------------------------------------
def gx(x,y):
    return 0

def gy(x,y):
    return -9.81

def ubc(x,y):
    vaal=-1.e-15*(Lx/2.0)
    if x<Lx/2:
       val=vaal
    elif x>Lx/2:
       val=-vaal
    else:
       val=0
    return val

def vbc(x,y):
    return 0

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

def viscosity(exx,eyy,exy,pq,c,phi,iter):
    if iter==0:
       val=1e25
    else:
       try: 
          e2=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)
          e2=max(1e-25,e2)
          Y=pq*np.sin(phi)+c*np.cos(phi)
          val=Y/(2.*e2)
          val=min(1.e25,val)
          val=max(1.e20,val)
       except Exception as e: 
          print (e)  
          pass
    return val

#------------------------------------------------------------------------------

cm=0.01
year=3600*24*365.

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=100000.  # horizontal extent of the domain 
Ly=10000.  # vertical extent of the domain 

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   solver = int(sys.argv[4])
else:
   nelx = 120
   nely = 12
   visu = 1
   solver = 2 
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV               # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs

eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

rho=2800
cohesion=1e7
phi=30./180*np.pi
psi=30./180*np.pi
tol_nl=1e-8

if solver==1:
   use_SchurComplementApproach=True
   use_preconditioner=True
   niter_stokes=250
   solver_tolerance=1e-6
else:
   use_SchurComplementApproach=False

method=2

eta_ref=1.e23      # scaling of G blocks
scaling_coeff=eta_ref/Ly

niter_min=1
niter=50

if use_SchurComplementApproach:
   ls_conv_file=open("linear_solver_convergence.ascii","w")
   ls_niter_file=open("linear_solver_niter.ascii","w")
   
shear_band_L_file=open("shear_band_L.ascii","w")
shear_band_R_file=open("shear_band_R.ascii","w")

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnp=",nnp)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("Nfem=",Nfem)
print("------------------------------")

two_sin_psi=2.*np.sin(psi)

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(nnp,dtype=np.float64)  # x coordinates
y=np.empty(nnp,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx/2.
        y[counter]=j*hy/2.
        counter += 1

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

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

iconV=np.zeros((mV,nel),dtype=np.int16)
iconP=np.zeros((mP,nel),dtype=np.int16)

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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(x[i],y[i])
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(x[i],y[i])
       u[i]=ubc(x[i],y[i])
    if x[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(x[i],y[i])
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(x[i],y[i])
       u[i] = ubc(x[i],y[i])
    if y[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(x[i],y[i])
       u[i] = ubc(x[i],y[i])
       v[i] = vbc(x[i],y[i])

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#------------------------------------------------------------------------------
# non-linear iterations
#------------------------------------------------------------------------------
if method==1:
   c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
elif method==2:
   c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,3]],dtype=np.float64) 

dNVdx = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdy = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
p     = np.zeros(NfemP,dtype=np.float64)        # pressure field 
Res   = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol   = np.zeros(Nfem,dtype=np.float64)         # solution vector 
b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NP    = np.zeros(mP,dtype=np.float64)           # shape functions P
conv  = np.zeros(niter,dtype=np.float64)        
solP  = np.zeros(NfemP,dtype=np.float64)  
solV  = np.zeros(NfemV,dtype=np.float64)  
a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

for iter in range(0,niter):

   print("--------------------------")
   print("iter=", iter)
   print("--------------------------")

   #################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################

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

               NV[0:9]=NNV(rq,sq)
               dNVdr[0:9]=dNNVdr(rq,sq)
               dNVds[0:9]=dNNVds(rq,sq)
               NP[0:4]=NNP(rq,sq)

               # calculate jacobian matrix
               jcb=np.zeros((2,2),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
                   jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
                   jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
                   jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb)

               # compute dNdx & dNdy & strainrate
               exxq=0.0
               eyyq=0.0
               exyq=0.0
               for k in range(0,mV):
                   xq[counter]+=NV[k]*x[iconV[k,iel]]
                   yq[counter]+=NV[k]*y[iconV[k,iel]]
                   dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                   dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
                   exxq+=dNVdx[k]*u[iconV[k,iel]]
                   eyyq+=dNVdy[k]*v[iconV[k,iel]]
                   exyq+=0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]
                
               # compute pressure
               for k in range(0,mP):
                   pq[counter]+=NP[k]*p[iconP[k,iel]]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.     ],
                                            [0.      ,dNVdy[i]],
                                            [dNVdy[i],dNVdx[i]]]

               # compute effective plastic viscosity
               etaq[counter]=viscosity(exxq,eyyq,exyq,pq[counter],cohesion,phi,iter)
               #eta_eff=1e25
               srq[counter]=np.sqrt(0.5*(exxq*exxq+eyyq*eyyq)+exyq*exyq)
               dilation_rate=two_sin_psi*srq[counter]

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+0]+=NV[i]*jcob*weightq*gx(xq,yq)*rho
                   f_el[ndofV*i+1]+=NV[i]*jcob*weightq*gy(xq,yq)*rho

               if method==1:
                  for i in range(0,mV):
                      f_el[ndofV*i+0]-=2./3.*dNVdx[i]*jcob*weightq*eta_eff*dilation_rate
                      f_el[ndofV*i+1]-=2./3.*dNVdy[i]*jcob*weightq*eta_eff*dilation_rate

               for i in range(0,mP):
                   N_mat[0,i]=NP[i]
                   N_mat[1,i]=NP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                
               h_el[:]-=NP[:]*dilation_rate*weightq*jcob

               for i in range(0,mP):
                   for j in range(0,mP):
                       M_el[i,j]+=NP[i]*NP[j]*weightq*jcob/etaq[counter]
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
                       K_mat[m1,m2]+=K_el[ikk,jkk]
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   G_mat[m1,m2]+=G_el[ikk,jkk]
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,mP):
           m1=iconP[k1,iel]
           h_rhs[m1]+=h_el[k1]
           for k2 in range(0,mP):
               m2=iconP[k2,iel]
               M_mat[m1,m2]+=M_el[k1,k2]
           #end for k2 
       #end for k1

   # end for iel 
        
   G_mat*=scaling_coeff
   h_rhs*=scaling_coeff

   print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
   print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
   print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

   print("build FE matrix: %.3f s" % (time.time() - start))

   ######################################################################
   # assemble K, G, GT, f, h into A and rhs
   ######################################################################
   start = time.time()

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
      u,v=np.reshape(solV[0:NfemV],(nnp,2)).T
      p=solP[0:NfemP]*scaling_coeff
   else:
      a_mat[:,:]=0
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
      rhs[0:NfemV]=f_rhs
      rhs[NfemV:Nfem]=h_rhs
      Res=a_mat.dot(sol)-rhs
      sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
      u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
      p=sol[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("solve system: %.3f s - Nfem %d" % (time.time() - start, Nfem))

   #################################################################
   # compute non-linear residual
   #################################################################

   #np.savetxt('Res.ascii',Res,header='# x,y,u,v')

   if iter==0:
      Res0=np.max(abs(Res))

   #print (iter,Res0,(np.max(abs(Res))))

   print("Nonlinear residual (inf. norm) %.7e" % (np.max(abs(Res))/Res0))

   conv[iter]=np.max(abs(Res))/Res0

   if np.max(abs(Res))/Res0<tol_nl and iter>niter_min:
      break

   np.savetxt('nonlinear_conv.ascii',np.array(conv[0:niter]).T)

   np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
   np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x,y,u,v]).T,header='# x,y,u,v')
   np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')
   np.savetxt('srq_{:04d}.ascii'.format(iter),np.array([xq,yq,srq]).T,header='# x,y,sr')

   #####################################################################
   # interpolate pressure onto velocity grid points
   #####################################################################

   q=np.zeros(nnp,dtype=np.float64)

   for iel in range(0,nel):
       q[iconV[0,iel]]=p[iconP[0,iel]]
       q[iconV[1,iel]]=p[iconP[1,iel]]
       q[iconV[2,iel]]=p[iconP[2,iel]]
       q[iconV[3,iel]]=p[iconP[3,iel]]
       q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
       q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
       q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
       q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
       q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

   np.savetxt('q_{:04d}.ascii',np.array([x,y,q]).T,header='# x,y,q')

   ######################################################################
   # compute strainrate 
   ######################################################################
   start = time.time()

   xc = np.zeros(nel,dtype=np.float64)  
   yc = np.zeros(nel,dtype=np.float64)  
   pc = np.zeros(nel,dtype=np.float64)  
   exx = np.zeros(nel,dtype=np.float64)  
   eyy = np.zeros(nel,dtype=np.float64)  
   exy = np.zeros(nel,dtype=np.float64)  
   e   = np.zeros(nel,dtype=np.float64)  

   for iel in range(0,nel):

       rq = 0.0
       sq = 0.0
       weightq = 2.0 * 2.0

       NV[0:9]=NNV(rq,sq)
       dNVdr[0:9]=dNNVdr(rq,sq)
       dNVds[0:9]=dNNVds(rq,sq)

       jcb=np.zeros((2,2),dtype=np.float64)
       for k in range(0,mV):
           jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
           jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
           jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
           jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
       jcob=np.linalg.det(jcb)
       jcbi=np.linalg.inv(jcb)

       for k in range(0,mV):
           dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
           dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

       for k in range(0,mV):
           xc[iel] += NV[k]*x[iconV[k,iel]]
           yc[iel] += NV[k]*y[iconV[k,iel]]
           exx[iel] += dNVdx[k]*u[iconV[k,iel]]
           eyy[iel] += dNVdy[k]*v[iconV[k,iel]]
           exy[iel] += 0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]

       e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

       for k in range(0,mP):
           pc[iel] += NP[k]*p[iconP[k,iel]]

   print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
   print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
   print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
   print("     -> pc  (m,M) %.5e %.5e " %(np.min(pc),np.max(pc)))

   print("compute press & sr: %.3f s" % (time.time() - start))

#------------------------------------------------------------------------------
# end of non-linear iterations
#------------------------------------------------------------------------------

np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################
# extracting shear bands 
#####################################################################

counter = 0
for j in range(0,nely):
    emaxL=0.
    emaxR=0.
    for i in range(0,nelx):
        if i<=nelx/2 and e[counter]>emaxL:
           emaxL=e[counter]
           ilocL=counter
        # end if
        if i>=nelx/2 and e[counter]>emaxR:
           emaxR=e[counter]
           ilocR=counter
        # end if
        counter += 1
    # end for i
    shear_band_L_file.write("%6e %6e %6e \n"  % (xc[ilocL],yc[ilocL],e[ilocL]) )
    shear_band_R_file.write("%6e %6e %6e \n"  % (xc[ilocR],yc[ilocR],e[ilocR]) )
# end for j



#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 
    

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
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % ( e[iel] ))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (  viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],cohesion,phi,iter)))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='dilation rate (R)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (  two_sin_psi*e[iel] ))
vtufile.write("</DataArray>\n")

#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
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
