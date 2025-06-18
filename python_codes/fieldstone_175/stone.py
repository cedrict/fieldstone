import numpy as np
import sys as sys
import time as timing
from numpy import linalg as LA
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

#------------------------------------------------------------------------------

def gx(x,y):
    return 0

def gy(x,y):
    return 0

def ud(x,y,Lx,Ly):
    return x/2-Lx/4/np.pi/Lx*np.sin(2*np.pi*x/Lx)-Lx/4

def duddx(x,y,Lx,Ly):
    return (1-np.cos(2*np.pi*x/Lx))/2

def d2uddx2(x,y,Lx,Ly):
    return np.pi/Lx*np.cos(2*np.pi*x/Lx)

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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,\
                     NV_6,NV_7,NV_8],dtype=np.float64)

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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,\
                     dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,\
                     dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

#------------------------------------------------------------------------------

eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

nelx=32
nely=16

Lx=2.
Ly=1.

eta=1.
rho=1.

#################################################################

nnx=2*nelx+1                  # number of nodes, x direction
nny=2*nely+1                  # number of nodes, y direction
NV=nnx*nny                    # total number of nodes
nel=nelx*nely                 # total number of elements
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
hx=Lx/nelx                    # mesh size in x direction
hy=Ly/nely                    # mesh size in y direction

scaling_coeff=1

#################################################################
# quadrature parameters

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

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
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter+=1

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# build connectivity arrays for velocity and pressure
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

bc_fix=np.zeros(NfemV,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value

for i in range(0,NV):
    if abs(yV[i]-Ly)/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

    if abs(yV[i])/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       # fix 1 u dof to remove translational nullspace 
       if abs(xV[i]-Lx/2)<eps:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0 

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix A and rhs 
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################

c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

#c_mat = np.array([[4,-2,0],[-2,4,0],[0,0,3]],dtype=np.float64) 
#c_mat/=3

if True:

   A_sparse= lil_matrix((Nfem,Nfem),dtype=np.float64) # FEM stokes matrix 
   dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
   b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
   rhs     = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
   N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # N matrix  
   f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
   h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
               
   #only valid for rectangular elements!
   jcbi=np.zeros((ndim,ndim),dtype=np.float64)
   jcob=hx*hy/4
   jcbi[0,0] = 2/hx 
   jcbi[1,1] = 2/hy

   counter=0
   for iel in range(0,nel):

       # set arrays to 0 for each element 
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)

       # integrate viscous term at quadrature points
       for jq in [0,1,2]:
           for iq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV=NNV(rq,sq)
               dNNNVdr=dNNVdr(rq,sq)
               dNNNVds=dNNVds(rq,sq)
               NNNP=NNP(rq,sq)

               # calculate jacobian matrix
               #jcb=np.zeros((ndim,ndim),dtype=np.float64)
               #for k in range(0,mV):
               #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
               #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
               #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
               #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               #jcob = np.linalg.det(jcb)
               #jcbi = np.linalg.inv(jcb)

               xq=NNNV.dot(xV[iconV[:,iel]])
               yq=NNNV.dot(yV[iconV[:,iel]])

               # compute dNdx & dNdy & strainrate
               for k in range(0,mV):
                   #dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   #dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                   dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*(rho*gx(xq,yq)-d2uddx2(xq,yq,Lx,Ly)*(2-2./ndim))
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*(rho*gy(xq,yq)-0)

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                
               h_el[:]-=NNNP[:]*duddx(xq,yq,Lx,Ly)*weightq*jcob

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
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*scaling_coeff
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,mP):
           m1=iconP[k1,iel]
           h_rhs[m1]+=h_el[k1]*scaling_coeff
       #end for k1

   # end for iel 

   print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
   print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

   print("build FE matrix: %.3f s" % (timing.time() - start))

   ######################################################################
   # assemble f, h into rhs and solve
   ######################################################################
   start = timing.time()

   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   sparse_matrix=A_sparse.tocsr()
   sol=sps.linalg.spsolve(sparse_matrix,rhs)

   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   np.savetxt('solution.ascii',np.array([xV,yV,u,v]).T)

   print("solve system: %.3f s - Nfem %d" % (timing.time() - start, Nfem))

   #################################################################
   #normalise pressure
   #################################################################
   start = timing.time()

   int_p=0
   for iel in range(0,nel):
       for jq in [0,1,2]:
           for iq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNP[0:mP]=NNP(rq,sq)
               jcob=hx*hy/4
               p_q=NNNP[0:mP].dot(p[iconP[0:mP,iel]])
               int_p+=p_q*weightq*jcob
           #end for
       #end for
   #end for

   avrg_p=int_p/Lx/Ly

   print("     -> int_p %e " %(int_p))
   print("     -> avrg_p %e " %(avrg_p))

   p[:]-=avrg_p

   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("normalise pressure: %.3f s" % (timing.time() - start))

   #####################################################################
   # interpolate pressure onto velocity grid points (for plotting)
   #####################################################################
   # velocity    pressure
   # 3---6---2   3-------2
   # |       |   |       |
   # 7   8   5   |       |
   # |       |   |       |
   # 0---4---1   0-------1
   #################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)

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

   print("project p onto vel nodes: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute strainrate at center of element 
   ######################################################################
   start = timing.time()

   xc = np.zeros(nel,dtype=np.float64)  
   yc = np.zeros(nel,dtype=np.float64)  
   exx = np.zeros(nel,dtype=np.float64)  
   eyy = np.zeros(nel,dtype=np.float64)  
   exy = np.zeros(nel,dtype=np.float64)  
   sr  = np.zeros(nel,dtype=np.float64)  
   pc  = np.zeros(nel,dtype=np.float64)  

   for iel in range(0,nel):

       rq = 0.
       sq = 0.

       NNNV=NNV(rq,sq)
       dNNNVdr=dNNVdr(rq,sq)
       dNNNVds=dNNVds(rq,sq)
       NNNP=NNP(rq,sq)

       #jcb=np.zeros((ndim,ndim),dtype=np.float64)
       #for k in range(0,mV):
       #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
       #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
       #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
       #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
       #jcob=np.linalg.det(jcb)
       #jcbi=np.linalg.inv(jcb)

       for k in range(0,mV):
           dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
           dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

       for k in range(0,mV):
           xc[iel]+=NNNV[k]*xV[iconV[k,iel]]
           yc[iel]+=NNNV[k]*yV[iconV[k,iel]]
           exx[iel]+=dNNNVdx[k]*u[iconV[k,iel]]
           eyy[iel]+=dNNNVdy[k]*v[iconV[k,iel]]
           exy[iel]+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

       sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

       for k in range(0,mP):
           pc[iel] += NNNP[k]*p[iconP[k,iel]]

   #end if

   print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
   print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
   print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
   print("     -> sr  (m,M) %.5e %.5e " %(np.min(sr),np.max(sr)))
   print("     -> pc  (m,M) %.5e %.5e " %(np.min(pc),np.max(pc)))

   np.savetxt('solution_c.ascii',np.array([xc,yc,pc]).T)

   print("compute press & sr: %.3f s" % (timing.time() - start))

   #####################################################################
   # compute strainrate on velocity grid
   #####################################################################
   start = timing.time()

   exxn=np.zeros(NV,dtype=np.float64)
   eyyn=np.zeros(NV,dtype=np.float64)
   exyn=np.zeros(NV,dtype=np.float64)
   srn=np.zeros(NV,dtype=np.float64)
   divvn=np.zeros(NV,dtype=np.float64)
   c=np.zeros(NV,dtype=np.float64)

   rVnodes=[-1,+1,1,-1, 0,1,0,-1,0]
   sVnodes=[-1,-1,1,+1,-1,0,1, 0,0]

   for iel in range(0,nel):
       for i in range(0,mV):
           NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i])
           dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i])
           dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i])
           #jcb=np.zeros((ndim,ndim),dtype=np.float64)
           #for k in range(0,mV):
           #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
           #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
           #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
           #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
           #jcob=np.linalg.det(jcb)
           #jcbi=np.linalg.inv(jcb)
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

   divvn[:]=exxn[:]+eyyn[:]

   srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

   print("     -> exxn  (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
   print("     -> eyyn  (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
   print("     -> exyn  (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
   print("     -> srn   (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))
   print("     -> divvn (m,M) %.6e %.6e " %(np.min(divvn),np.max(divvn)))

   print("compute nod strain rate: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute vrms
   ######################################################################
   start = timing.time()

   vrms=0.
   counterq=0
   for iel in range(0,nel):
       for iq in [0,1,2]:
           for jq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               jcob=hx*hy/4 #only for rect elements!
               uq=0.
               vq=0.
               for k in range(0,mV):
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
               #end for
               vrms+=(uq**2+vq**2)*weightq*jcob
               counterq+=1
           #end for
       #end for
   #end for

   vrms=np.sqrt(vrms/(Lx*Ly))

   print("     -> vrms= %.7e " %vrms)

   print("compute vrms: %.3f s" % (timing.time() - start))

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
            #jcb=np.zeros((ndim,ndim),dtype=np.float64)
            #for k in range(0,mV):
            #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            # end for
            #jcob = np.linalg.det(jcb)
            #jcbi = np.linalg.inv(jcb)
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
# plot of solution
#####################################################################

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
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='ud' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e %e %e \n" %(ud(xV[i],yV[i],Lx,Ly),0,0.))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='duddx' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%e \n" %(duddx(xV[i],yV[i],Lx,Ly)))
vtufile.write("</DataArray>\n")



vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                   iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                   iconV[6,iel],iconV[7,iel],iconV[8,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*9))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()


print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
