import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix, lil_matrix
import time as clock
from numpy import linalg as LA

###############################################################################
  
Lx=1e5 
Ly=1e5  
p_left=1e9
p_right=0.
Pi=(p_right-p_left)/Lx
Pi2=Pi/2.
nnn=0  # 0 (poiseuille) or 1 
eta0=1e25 
eps0=1e-17
tau0=9e7
K=(eta0*eps0-tau0)/eps0**nnn
delta=2*eps0*eta0/abs(Pi)
y1=0.5*Ly-delta
y2=0.5*Ly+delta

###############################################################################

def viscosity(exx,eyy,exy):
    if nnn==0:
       val=1e25
    if nnn==1:
       ee=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)
       if ee<eps0:
          val=1e25
       else:
          val=K+tau0/ee
       #end if
       #print (iter,val)
    return val

def velocity_th(x,y):
    if nnn>0:
       u1=2.*nnn/(nnn+1)*K/Pi2*(( Pi2/K*(y-y1)+ eps0**nnn)**(1.+1./nnn) - (-Pi2/K*y1 + eps0**nnn )**(1.+1./nnn) )
       u2=Pi2/eta0*(y**2-y*Ly)+2.*nnn/(nnn+1)*K/Pi2*( eps0**(nnn+1)-(eps0**nnn-Pi2/K*y1)**(1+1./nnn) ) \
         - Pi2/eta0*y1*(y1-Ly)
       u3=2.*nnn/(nnn+1)*K/Pi2*( (-Pi2/K*(y-y2)+ eps0**nnn )**(1.+1./nnn)- (-Pi2/K*(Ly-y2)+eps0**nnn )**(1.+1./nnn) )
       if y<y1:
          val=u1
       elif y<y2: 
          val=u2
       else:
          val=u3
    else:
       val=0
    return val

def exy_th(x,y):
    if nnn>0:
       exy1=(Pi2/K*(y-y1)+ eps0**nnn )**(1./nnn)
       exy2=0.5*Pi/eta0*(y-Ly/2.)
       exy3=-(-Pi2/K*(y-y2)+ eps0**nnn ) **(1./nnn)
       if y<y1:
          val=exy1
       elif y<y2: 
          val=exy2
       else:
          val=exy3
    else:
       val=0
    return val

###############################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

cm=0.01
year=3600*24*365.

print("*******************************")
print("********** stone 061 **********")
print("*******************************")

ndim=2
m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   solver = int(sys.argv[4])
   benchmark = int(sys.argv[5])
else:
   nelx = 16
   nely = 64
   visu = 1
   solver = 2 

gx=0.
gy=0.
rho=0.
 
nnx=2*nelx+1             # number of elements, x direction
nny=2*nely+1             # number of elements, y direction
nn_V=nnx*nny             # number of nodes
nel=nelx*nely            # number of elements, total
Nfem_V=nn_V*ndof_V       # number of velocity dofs
Nfem_P=(nelx+1)*(nely+1) # number of pressure dofs
Nfem=Nfem_V+Nfem_P       # total number of dofs

eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

rel_tol_nl=1e-6
abs_tol_nl=1e-10

if solver==1:
   use_SchurComplementApproach=True
   use_preconditioner=True
   niter_stokes=250
   solver_tolerance=1e-6
else:
   use_SchurComplementApproach=False

eta_ref=1.e25      # scaling of G blocks
scaling_coeff=eta_ref/Ly

niter_min=0
niter=50

if use_SchurComplementApproach:
   ls_conv_file=open("linear_solver_convergence.ascii","w")
   ls_niter_file=open("linear_solver_niter.ascii","w")

sparse=True

r_V=[-1,+1,1,-1, 0,1,0,-1,0]
s_V=[-1,-1,1,+1,-1,0,1, 0,0]

###############################################################################
###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("sparse",sparse)
print("hx",hx)
print("hy",hy)
print("Pi",Pi)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)    # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    if x_V[i]/Lx>1-eps:
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    if y_V[i]/Ly<eps:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0 
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    if y_V[i]/Ly>1-eps:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0# 1e-9 
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# non-linear iterations
###############################################################################
###############################################################################
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

u=np.zeros(nn_V,dtype=np.float64)    # x-component velocity
v=np.zeros(nn_V,dtype=np.float64)    # y-component velocity
p=np.zeros(Nfem_P,dtype=np.float64)        # pressure field 
pold=np.zeros(Nfem_P,dtype=np.float64)        # pressure field 
Res=np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol=np.zeros(Nfem,dtype=np.float64)         # solution vector 
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
conv_inf=np.zeros(niter,dtype=np.float64)        
conv_two=np.zeros(niter,dtype=np.float64)        
conv_inf_Ru=np.zeros(niter,dtype=np.float64)        
conv_inf_Rv=np.zeros(niter,dtype=np.float64)        
conv_inf_Rp=np.zeros(niter,dtype=np.float64)        
solP=np.zeros(Nfem_P,dtype=np.float64)  
solV=np.zeros(Nfem_V,dtype=np.float64)  
a_mat=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iter in range(0,niter):

   print("--------------------------")
   print("iter=", iter)
   print("--------------------------")

   ############################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   ############################################################################

   if sparse:
      A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
   else:   
      K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
      G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

   M_mat=np.zeros((Nfem_P,Nfem_P),dtype=np.float64) # schur precond
   f_rhs=np.zeros(Nfem_V,dtype=np.float64)          # right hand side f 
   h_rhs=np.zeros(Nfem_P,dtype=np.float64)          # right hand side h 
   xq   =np.zeros(9*nel,dtype=np.float64)           # x coords of q points 
   yq   =np.zeros(9*nel,dtype=np.float64)           # y coords of q points 
   etaq =np.zeros(9*nel,dtype=np.float64)           # viscosity of q points 
   pq   =np.zeros(9*nel,dtype=np.float64)           # pressure of q points 
   srq  =np.zeros(9*nel,dtype=np.float64)           # strain rate of q points 

   counter=0
   for iel in range(0,nel):

       f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
       K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
       G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
       h_el=np.zeros((m_P),dtype=np.float64)
       M_el=np.zeros((m_P,m_P),dtype=np.float64)  

       for jq in [0,1,2]:
           for iq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_V=basis_functions_V(rq,sq)
               N_P=basis_functions_P(rq,sq)
               dNdr_V=basis_functions_V_dr(rq,sq)
               dNds_V=basis_functions_V_ds(rq,sq)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq
               dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
               dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
               pq[counter]=np.dot(N_P,p[icon_P[:,iel]])
               exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
               eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
               exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                    np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
               xq[counter]=np.dot(N_V,x_V[icon_V[:,iel]])
               yq[counter]=np.dot(N_V,y_V[icon_V[:,iel]])
               for i in range(0,m_V):
                   B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                     [0.       ,dNdy_V[i]],
                                     [dNdy_V[i],dNdx_V[i]]]

               # compute effective plastic viscosity
               etaq[counter]=viscosity(exxq,eyyq,exyq)
               srq[counter]=np.sqrt(0.5*(exxq*exxq+eyyq*eyyq)+exyq*exyq)

               K_el+=B.T.dot(C.dot(B))*etaq[counter]*JxWq

               for i in range(0,m_V):
                   f_el[ndof_V*i+0]+=N_V[i]*gx*rho*JxWq
                   f_el[ndof_V*i+1]+=N_V[i]*gy*rho*JxWq
               #end for

               for i in range(0,m_P):
                   N_mat[0,i]=N_P[i]
                   N_mat[1,i]=N_P[i]
                   N_mat[2,i]=0.
               #end for

               G_el-=B.T.dot(N_mat)*JxWq

               for i in range(0,m_P):
                   for j in range(0,m_P):
                       M_el[i,j]+=N_P[i]*N_P[j]*JxWq/etaq[counter]
                   # end for j
               # end for i

               counter+=1
           # end for iq 
       # end for jq 

       if x_V[icon_V[0,iel]]<1e-6:
          f_el[ 0]+=p_left*hy/6.
          f_el[ 6]+=p_left*hy/6.
          f_el[14]+=p_left*hy/6.*4.

       # impose b.c. 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               if bc_fix[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
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

       # assemble matrix and right hand side 
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               for k2 in range(0,m_V):
                   for i2 in range(0,ndof_V):
                       jkk=ndof_V*k2          +i2
                       m2 =ndof_V*icon_V[k2,iel]+i2
                       if sparse:
                          A_sparse[m1,m2] += K_el[ikk,jkk]
                       else:
                          K_mat[m1,m2]+=K_el[ikk,jkk]
                       # end if
                   #end for i2
               #end for k2
               for k2 in range(0,m_P):
                   jkk=k2
                   m2 =icon_P[k2,iel]
                   if sparse:
                      A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]*scaling_coeff
                      A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
                   else:
                      G_mat[m1,m2]+=G_el[ikk,jkk]*scaling_coeff
                   #end if
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,m_P):
           m1=icon_P[k1,iel]
           h_rhs[m1]+=h_el[k1]*scaling_coeff
           for k2 in range(0,m_P):
               m2=icon_P[k2,iel]
               M_mat[m1,m2]+=M_el[k1,k2]
           #end for k2 
       #end for k1

   # end for iel 

   print("build FE matrix: %.3f s" % (clock.time()-start))

   ############################################################################
   # assemble K, G, GT, f, h into A and rhs
   ############################################################################
   start=clock.time()

   if use_SchurComplementApproach:

      # convert matrices to CSR format
      G_mat=sps.csr_matrix(G_mat)
      K_mat=sps.csr_matrix(K_mat)
      M_mat=sps.csr_matrix(M_mat)

      Res[0:Nfem_V]=K_mat.dot(solV)+G_mat.dot(solP)-f_rhs
      Res[Nfem_V:Nfem]=G_mat.T.dot(solV)-h_rhs

      # declare necessary arrays
      rvect_k=np.zeros(Nfem_P,dtype=np.float64) 
      pvect_k=np.zeros(Nfem_P,dtype=np.float64) 
      zvect_k=np.zeros(Nfem_P,dtype=np.float64) 
      ptildevect_k=np.zeros(Nfem_V,dtype=np.float64) 
      dvect_k=np.zeros(Nfem_V,dtype=np.float64) 
   
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
          ls_conv_file.write("%d %.6e \n"  %(k,xi))
          print("lin.solver: %d %6e" % (k,xi))
          if xi<solver_tolerance:
             ls_niter_file.write("%d \n"  %(k))
             break 
      u,v=np.reshape(solV[0:Nfem_V],(nn_V,2)).T
      p=solP[0:Nfem_P]*scaling_coeff
   else:
      rhs[0:Nfem_V]=f_rhs
      rhs[Nfem_V:Nfem]=h_rhs
      if not sparse:
         a_mat[:,:]=0
         a_mat[0:Nfem_V,0:Nfem_V]=K_mat
         a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
         a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
         Res=a_mat.dot(sol)-rhs
         sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
      else:
         sparse_matrix=A_sparse.tocsr()
         Res=sparse_matrix.dot(sol)-rhs
         sol=sps.linalg.spsolve(sparse_matrix,rhs)

      u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
      p=sol[Nfem_V:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("solve system: %.3f s - Nfem %d" % (clock.time()-start, Nfem))

   ############################################################################
   # compute non-linear residual
   ############################################################################
   start=clock.time()

   if iter==0:
      Res0_inf=LA.norm(Res,np.inf)
      Res0_two=LA.norm(Res,2)

   Res_inf=LA.norm(Res,np.inf)
   Res_two=LA.norm(Res,2)

   print("Nonlinear residual (inf. norm) %.7e" % (Res_inf/Res0_inf))
   print("Nonlinear residual (two  norm) %.7e" % (Res_two/Res0_two))

   conv_inf[iter]=Res_inf/Res0_inf
   conv_two[iter]=Res_two/Res0_two

   if Res_inf/Res0_inf<rel_tol_nl and iter>niter_min:
      print('***** converged*****')
      break

   if Res_inf<abs_tol_nl and iter>niter_min:
      print('***** converged*****')
      break

   np.savetxt('nonlinear_conv_inf.ascii',np.array(conv_inf[0:niter]).T)
   np.savetxt('nonlinear_conv_two.ascii',np.array(conv_two[0:niter]).T)

   Res_u,Res_v=np.reshape(Res[0:Nfem_V],(nn_V,2)).T
   Res_p=Res[Nfem_V:Nfem]
   
   conv_inf_Ru[iter]=LA.norm(Res_u,np.inf)
   conv_inf_Rv[iter]=LA.norm(Res_v,np.inf)
   conv_inf_Rp[iter]=LA.norm(Res_p,np.inf)

   np.savetxt('nonlinear_conv_inf_Ru.ascii',np.array(conv_inf_Ru[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rv.ascii',np.array(conv_inf_Rv[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rp.ascii',np.array(conv_inf_Rp[0:niter]).T)

   if debug:
      np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
      np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
      np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')
      np.savetxt('srq_{:04d}.ascii'.format(iter),np.array([xq,yq,srq]).T,header='# x,y,sr')

   print("computing res norms: %.3f s" % (clock.time()-start))

   ############################################################################
   # interpolate pressure onto velocity grid points
   ############################################################################
   start=clock.time()

   q=np.zeros(nn_V,dtype=np.float64)
   Res_q=np.zeros(nn_V,dtype=np.float64)

   for iel in range(0,nel):
       q[icon_V[0,iel]]=p[icon_P[0,iel]]
       q[icon_V[1,iel]]=p[icon_P[1,iel]]
       q[icon_V[2,iel]]=p[icon_P[2,iel]]
       q[icon_V[3,iel]]=p[icon_P[3,iel]]
       q[icon_V[4,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
       q[icon_V[5,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
       q[icon_V[6,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
       q[icon_V[7,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
       q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+\
                        p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25
       Res_q[icon_V[0,iel]]=Res_p[icon_P[0,iel]]
       Res_q[icon_V[1,iel]]=Res_p[icon_P[1,iel]]
       Res_q[icon_V[2,iel]]=Res_p[icon_P[2,iel]]
       Res_q[icon_V[3,iel]]=Res_p[icon_P[3,iel]]
       Res_q[icon_V[4,iel]]=(Res_p[icon_P[0,iel]]+Res_p[icon_P[1,iel]])*0.5
       Res_q[icon_V[5,iel]]=(Res_p[icon_P[1,iel]]+Res_p[icon_P[2,iel]])*0.5
       Res_q[icon_V[6,iel]]=(Res_p[icon_P[2,iel]]+Res_p[icon_P[3,iel]])*0.5
       Res_q[icon_V[7,iel]]=(Res_p[icon_P[3,iel]]+Res_p[icon_P[0,iel]])*0.5
       Res_q[icon_V[8,iel]]=(Res_p[icon_P[0,iel]]+Res_p[icon_P[1,iel]]+\
                            Res_p[icon_P[2,iel]]+Res_p[icon_P[3,iel]])*0.25

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (clock.time()-start))

   ############################################################################
   # compute strainrate 
   ############################################################################
   start=clock.time()

   x_e=np.zeros(nel,dtype=np.float64)  
   y_e=np.zeros(nel,dtype=np.float64)  
   p_e=np.zeros(nel,dtype=np.float64)  
   exx=np.zeros(nel,dtype=np.float64)  
   eyy=np.zeros(nel,dtype=np.float64)  
   exy=np.zeros(nel,dtype=np.float64)  
   sr=np.zeros(nel,dtype=np.float64)  

   rq = 0.0
   sq = 0.0
   for iel in range(0,nel):
       N_V=basis_functions_V(rq,sq)
       N_P=basis_functions_P(rq,sq)
       dNdr_V=basis_functions_V_dr(rq,sq)
       dNds_V=basis_functions_V_ds(rq,sq)
       jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
       jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
       jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
       jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
       jcbi=np.linalg.inv(jcb)
       dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
       dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
       x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
       y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
       exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
       eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
       exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
               +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
       sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
       p_e[iel]=np.dot(N_P,p[icon_P[:,iel]])

   print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
   print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
   print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
   print("     -> sr  (m,M) %.5e %.5e " %(np.min(sr),np.max(sr)))
   print("     -> p_e (m,M) %.5e %.5e " %(np.min(p_e),np.max(p_e)))

   print("compute press & sr: %.3f s" % (clock.time()-start))

   ############################################################################

   avrg_press=np.sum(p_e)/nel

   print ("     -> avrg press. %.5e" % avrg_press)

   ############################################################################
   # project strainrate onto velocity grid
   ############################################################################
   start=clock.time()

   exxn=np.zeros(nn_V,dtype=np.float64)
   eyyn=np.zeros(nn_V,dtype=np.float64)
   exyn=np.zeros(nn_V,dtype=np.float64)
   srn=np.zeros(nn_V,dtype=np.float64)
   c=np.zeros(nn_V,dtype=np.float64)

   for iel in range(0,nel):
       for i in range(0,m_V):
           N_V=basis_functions_V(r_V[i],s_V[i])
           dNdr_V=basis_functions_V_dr(r_V[i],s_V[i])
           dNds_V=basis_functions_V_ds(r_V[i],s_V[i])
           jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
           jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
           jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
           jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
           jcbi=np.linalg.inv(jcb)
           JxWq=np.linalg.det(jcb)*weightq
           dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
           dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
           exxn[icon_V[i,iel]]+=np.dot(dNdx_V[:],u[icon_V[:,iel]])
           eyyn[icon_V[i,iel]]+=np.dot(dNdy_V[:],v[icon_V[:,iel]])
           exyn[icon_V[i,iel]]+=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                               +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
           c[icon_V[i,iel]]+=1.
       # end for i
   # end for iel
   exxn/=c
   eyyn/=c
   exyn/=c

   srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

   print("     -> exx (m,M) %.4e %.4e " %(np.min(exxn),np.max(exxn)))
   print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyyn),np.max(eyyn)))
   print("     -> exy (m,M) %.4e %.4e " %(np.min(exyn),np.max(exyn)))
   print("     -> sr  (m,M) %.4e %.4e " %(np.min(srn),np.max(srn)))

   print("compute nod strain rate: %.3f s" % (clock.time()-start))

   ############################################################################
   # generate vtu output at every nonlinear iteration
   ############################################################################

   filename = 'solution_nl_{:04d}.vtu'.format(iter) ; print('     -> creating '+filename)
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.4e %.4e %.4e \n" %(x_V[i],y_V[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='sr (middle) (x10^-15)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%.4e\n" % (sr[iel]*1e15))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.4e \n" %Res_u[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.4e \n" %Res_v[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.4e \n" %Res_q[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],\
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
       vtufile.write("%d \n" %23)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

# end of non-linear iterations

###############################################################################
# compute averaged elemental strainrate 
# I use a 5 point quadrature rule (per dimension) and compute the 
# average strain rate tensor components per element. 
###############################################################################
start=clock.time()

exx_avrg=np.zeros(nel,dtype=np.float64)  
eyy_avrg=np.zeros(nel,dtype=np.float64)  
exy_avrg=np.zeros(nel,dtype=np.float64)  
sr_avrg=np.zeros(nel,dtype=np.float64)  

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
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
            exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            exx_avrg[iel]+=exxq*JxWq
            eyy_avrg[iel]+=eyyq*JxWq
            exy_avrg[iel]+=exyq*JxWq
        # end for
    # end for
    exx_avrg[iel]/=(hx*hy) 
    eyy_avrg[iel]/=(hx*hy) 
    exy_avrg[iel]/=(hx*hy) 
    sr_avrg[iel]=np.sqrt(0.5*(exx_avrg[iel]**2+eyy_avrg[iel]**2)+exy_avrg[iel]**2)
#end for

print("     -> exx_avrg (m,M) %.4e %.4e " %(np.min(exx_avrg),np.max(exx_avrg)))
print("     -> eyy_avrg (m,M) %.4e %.4e " %(np.min(eyy_avrg),np.max(eyy_avrg)))
print("     -> exy_avrg (m,M) %.4e %.4e " %(np.min(exy_avrg),np.max(exy_avrg)))
print("     -> sr_avrg  (m,M) %.4e %.4e " %(np.min(sr_avrg),np.max(sr_avrg)))

print("compute avrg elemental strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e %.4e %.4e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % exx_avrg[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % exy_avrg[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (middle)(x10^-15)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % (sr[iel]*1e15))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (avrg)(x10^-15)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%.4e\n" % (sr_avrg[iel]*1e15))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    eta=viscosity(exx[iel],eyy[iel],exy[iel])
    vtufile.write("%.4e\n" %eta) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity (log)' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel])
    vtufile.write("%.4e\n" %(np.log10(eta))) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e %.4e %.4e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e %.4e %.4e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (analytical)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e %.4e %.4e \n" %(0.,velocity_th(x_V[i],y_V[i]),0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %Res_u[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %Res_v[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %Res_q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %exxn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %exyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy (analytical)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" % exy_th(x_V[i],y_V[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %srn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (x10^-15)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e \n" %(srn[i]*1e15))
vtufile.write("</DataArray>\n")

vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                icon_V[6,iel],icon_V[7,iel]))
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

###############################################################################

np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,p',fmt='%.4e')
np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v',fmt='%.4e')
np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta',fmt='%.4e')
np.savetxt('pq.ascii',np.array([xq,yq,pq]).T,header='# x,y,p',fmt='%.4e')
np.savetxt('sr.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy',fmt='%.4e')
np.savetxt('sr_avrg.ascii',np.array([x_e,y_e,exx_avrg,eyy_avrg,exy_avrg]).T,header='# x,y,exx,eyy,exy',fmt='%.4e')
np.savetxt('srq.ascii',np.array([xq,yq,srq]).T,header='# x,y,sr',fmt='%.4e')

sol_file=open("velocity_th.ascii","w")
for i in range(0,nn_V):
    sol_file.write("%.4e %.4e %.4e %.4e \n" %(x_V[i],y_V[i],velocity_th(x_V[i],y_V[i]),0.))
sol_file.close()

sol_file=open("exy_th.ascii","w")
for i in range(0,nn_V):
    sol_file.write("%.4e %.4e %.4e \n" %(x_V[i],y_V[i],exy_th(x_V[i],y_V[i])))
sol_file.close()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
###############################################################################
