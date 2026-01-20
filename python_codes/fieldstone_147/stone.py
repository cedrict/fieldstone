import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix
from schur_complement_cg_solver import *
from uzawa1_solver import *
from uzawa2_solver import *
from uzawa3_solver import *
from uzawa1_solver_L2 import *
from uzawa2_solver_L2 import *
from uzawa3_solver_L2 import *
from uzawa1_solver_L2b import *
from projection_solver import *
from basis_functions import *

###############################################################################
# bench=1: donea & huerta
# bench=2: stokes sphere
# bench=3: block
# bench=4: ??
###############################################################################

def bx(x,y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    else: 
       val=0 
    return val

def by(x,y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2:
       if (x-.5)**2+(y-0.5)**2<0.123456789**2:
          val=-1.01
       else:
          val=-1.
    if bench==3:
       if abs(x-.5)<0.0625 and abs(y-0.5)<0.0625:
          val=-1.01
       else:
          val=-1.
    if bench==4:
       val=0 
    return val

###############################################################################

def eta(x,y):
    if bench==1: val=1
    if bench==2:
       if (x-.5)**2+(y-0.5)**2<0.123456789**2:
          val=1000.
       else:
          val=1.
    if bench==3:
       if abs(x-.5)<0.0625 and abs(y-0.5)<0.0625:
          val=1000
       else:
          val=1
    if bench==4: val=1
    return val

###############################################################################

def u_analytical(x,y):
    if bench==1: val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=20*x*y**3
    return val

def v_analytical(x,y):
    if bench==1: val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=5*x**4-5*y**4
    return val

def p_analytical(x,y):
    if bench==1: val=x*(1.-x)-1./6.
    if bench==2: val=0
    if bench==3: val=0
    if bench==4: val=60*x**2*y-20*y**3-5
    return val

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 147 **********")
print("*******************************")

m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 6):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   nq_per_dim=int(sys.argv[4])
   solver=int(sys.argv[5])
else:
   nelx=32
   nely=nelx
   visu=1
   nq_per_dim=3
   solver=1
    
nnx=2*nelx+1           # number of V nodes, x direction
nny=2*nely+1           # number of V nodes, y direction
nn_V=nnx*nny           # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely          # total number of elements
Nfem_V=nn_V*ndof_V     # number of velocity dofs
Nfem_P=nn_P            # number of pressure dofs
Nfem=Nfem_V+Nfem_P     # total number of dofs

hx=Lx/nelx
hy=Ly/nely

bench=3
   
omega15=1000 # parameter for solver 15
omega18=1    # parameter for solver 18
omega19=1    # parameter for solver 19

use_precond=False
precond_type=0
tolerance=1e-7
niter_max=500

projection=False

debug=False

###########################################################
# boundary conditions
# NS: analytical velocity prescribed on all sides
# FS: free slip on all sides
# OT: open top, free slip sides and bottom
# BO: no slip sides, free slip bottom & top

FS=False
NS=True   
OT=False
BO=False

if bench==1: NS=True
if bench==4: NS=True

###############################################################################

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nq_per_dim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
if solver==1:  print("solver=",solver,' | direct solver on Stokes matrix')
if solver==2:  print("solver=",solver,' | gmres from linalg')
if solver==3:  print("solver=",solver,' | lgmres from linalg')
if solver==4:  print("solver=",solver,' | Schur complement CG solver using direct solver for inner solve')
if solver==5:  print("solver=",solver,' | minres from linalg')
if solver==6:  print("solver=",solver,' | qmr from linalg')
if solver==7:  print("solver=",solver,' | tfqmr from linalg')
if solver==8:  print("solver=",solver,' | gcrotmk from linalg')
if solver==9:  print("solver=",solver,' | bicg from linalg')
if solver==10: print("solver=",solver,' | bicgstab from linalg')
if solver==11: print("solver=",solver,' | direct solver on Stokes matrix using umfpack')
if solver==12: print("solver=",solver,' | cgs from linalg') 
if solver==13: print("solver=",solver,' | Schur complement CG solver using cg for inner solve') 
if solver==14: print("solver=",solver,' | Schur complement CG solver using splu') 
if solver==15: print("solver=",solver,' | uzawa 1')
if solver==16: print("solver=",solver,' | uzawa 2')
if solver==17: print("solver=",solver,' | projection solver')
if solver==18: print("solver=",solver,' | Uzawa 1 + L2 projection')
if solver==19: print("solver=",solver,' | Uzawa 1b + L2 projection')
if solver==20: print("solver=",solver,' | Uzawa 2 + L2 projection')
if solver==21: print("solver=",solver,' | Uzawa 3 (CG on Schur) projection')
if solver==22: print("solver=",solver,' | Uzawa 3 (CG on Schur) + L2 projection')
if solver==23: print("solver=",solver,' | Uzawa 3 (CG on Schur), LU inner, L2 projection')
 
print("------------------------------")

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

if debug: np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

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
# compute list of V-nodes seen by each P-node
# it is worth acknowedging that the lists are automatically sorted
###############################################################################
start=clock.time()

nb_of_Vnodes_seen_by_Pnode=np.zeros(nn_P,dtype=np.int32)
list_of_Vnodes_seen_by_Pnode=np.zeros((nn_P,25),dtype=np.int32)

counter=0
for jp in range(0,nely+1):
    for ip in range(0,nelx+1): # loop over pressure nodes
        iv=2*ip
        jv=2*jp
        kv=nnx*jv+iv
        #print('P-node',ip,jp,counter,' -> V-node',iv,jv,kv)
        Nv=0
        for n in (-2,-1,0,1,2):
            for m in (-2,-1,0,1,2):
                iiv=iv+m
                jjv=jv+n
                if iiv>=0 and iiv<nnx and jjv>=0 and jjv<nny: #if V-node exists
                   kkv=nnx*jjv+iiv
                   #print('m=',m,'n=',n,'iiv=',iiv,'jjv=',jjv,'kkv=',kkv)
                   list_of_Vnodes_seen_by_Pnode[counter,Nv]=kkv
                   Nv+=1
                #end if
            #end for
        #end for
        nb_of_Vnodes_seen_by_Pnode[counter]=Nv
        counter += 1
    #end for
#end for

print("link P-node V-node arrays: %.3f s" % (clock.time()-start))

###############################################################################
# compute element center coords and viscosity
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
eta_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    x_e[iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]])/2
    y_e[iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]])/2
    eta_e[iel]=eta(x_e[iel],y_e[iel])

print("element centers: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

if NS:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_analytical(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_analytical(x_V[i],y_V[i])
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_analytical(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_analytical(x_V[i],y_V[i])
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_analytical(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_analytical(x_V[i],y_V[i])
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_analytical(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_analytical(x_V[i],y_V[i])
       #end if
   #end for

if FS:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       #end if
   #end for

if OT:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       #end if
   #end for

if BO:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
          bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
       #end if
   #end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)      
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
A_fem   = lil_matrix((Nfem,Nfem),dtype=np.float64)     # matrix of Ax=b
K_mat   = lil_matrix((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat   = lil_matrix((Nfem_V,Nfem_P),dtype=np.float64) # matrix G
H_mat   = lil_matrix((Nfem_P,Nfem_V),dtype=np.float64) # matrix H 
L_mat   = lil_matrix((Nfem_P,Nfem_P),dtype=np.float64) # matrix GT*G
MP_mat  = lil_matrix((Nfem_P,Nfem_P),dtype=np.float64) # pressure mass matrix
MV_mat  = lil_matrix((Nfem_V,Nfem_V),dtype=np.float64) # velocity mass matrix
f_rhs   = np.zeros(Nfem_V,dtype=np.float64)            # right hand side f 
h_rhs   = np.zeros(Nfem_P,dtype=np.float64)            # right hand side h 
b_fem   = np.zeros(Nfem,dtype=np.float64)              # rhs of fem linear system
aa_mat  = np.zeros((m_P,2),dtype=np.float64)      
bb_mat  = np.zeros((2,ndof_V*m_V),dtype=np.float64)     
N_mat   = np.zeros((3,m_P),dtype=np.float64)          # matrix  
dNNNVdx = np.zeros(m_V,dtype=np.float64)              # shape functions derivatives
dNNNVdy = np.zeros(m_V,dtype=np.float64)              # shape functions derivatives
dNNNPdx = np.zeros(m_P,dtype=np.float64)              # shape functions derivatives
dNNNPdy = np.zeros(m_P,dtype=np.float64)              # shape functions derivatives

for iel in range(0,nel):

    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    H_el=np.zeros((m_P,m_V*ndof_V),dtype=np.float64)
    L_el=np.zeros((m_P,m_P),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)
    MP_el=np.zeros((m_P,m_P),dtype=np.float64)
    MV_el=np.zeros((m_V,m_V),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            dNdr_P=basis_functions_P_dr(rq,sq)
            dNds_P=basis_functions_P_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            dNdx_P=jcbi[0,0]*dNdr_P+jcbi[0,1]*dNds_P
            dNdy_P=jcbi[1,0]*dNdr_P+jcbi[1,1]*dNds_P
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq

            aa_mat[:,0]=dNdx_P[:]
            aa_mat[:,1]=dNdy_P[:]

            for i in range(0,m_V):
                bb_mat[0,2*i  ]=N_V[i] 
                bb_mat[1,2*i+1]=N_V[i] 
            H_el+=aa_mat@bb_mat*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*JxWq*bx(xq,yq)
                f_el[ndof_V*i+1]+=N_V[i]*JxWq*by(xq,yq)
            #end for 

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=B.T.dot(N_mat)*JxWq

            MP_el+=np.outer(N_P,N_P)*JxWq

        #end for jq
    #end for iq

    #print(MP_el*9/hx/hy) ok! identical to appendix A.2.1
    
    L_el=G_el.T.dot(G_el) # before b.c.!

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                    A_fem[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
                A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

    #assemble pressure mass matrix
    for k1 in range(0,m_P):
        m1=icon_P[k1,iel]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            L_mat[m1,m2]+=L_el[k1,k2]
            MP_mat[m1,m2]+=MP_el[k1,k2]
        #end for 
    #end for 

    #assemble other gradient matrix H
    for k1 in range(0,m_P):
        m1=icon_P[k1,iel]
        for k2 in range(0,m_V):
            for i2 in range(0,ndof_V):
                jkk=ndof_V*k2          +i2
                m2 =ndof_V*icon_V[k2,iel]+i2
                H_mat[m1,m2]+=H_el[k1,jkk]
            #end for 
        #end for 
    #end for 

#end for iel

b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# compute S1, S2 matrices to scale blocks K,G
###############################################################################

S1=lil_matrix((Nfem_V,Nfem_V),dtype=np.float64)
S2=lil_matrix((Nfem_P,Nfem_P),dtype=np.float64)

#for i in range(0,Nfem_V):
#    S1[i,i]=np.sqrt(K_mat[i,i])
#S1=csr_matrix(S1)

print('UNFINISHED SCALING BLOCKS ') 

###############################################################################
# compute preconditioner
###############################################################################
#Mprec=lil_matrix((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
#for i in range(0,Nfem):
#    Mprec[i,i]=10
#Mprec=csr_matrix(Mprec)

###############################################################################
# compute Schur preconditioner
# 0: identity matrix (for tests/reference)
# 1: GT.D_K^-1.G using dot on assembled matrices
# 2: GT.D_K^-1.G using clever algorithm
###############################################################################
start=clock.time()

M_mat=lil_matrix((Nfem_P,Nfem_P),dtype=np.float64)
   
if precond_type==0:
   for i in range(0,Nfem_P):
       M_mat[i,i]=1

if precond_type==1:
   Km1=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) 
   for i in range(0,Nfem_V):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))

#plt.spy(M_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build Schur matrix precond: %e s, nel= %d" % (clock.time()-start,nel))

###############################################################################
# convert lil matrices to csr
###############################################################################
start=clock.time()

if solver==14 or solver==23: # LU requires CSC
   K_mat=csc_matrix(K_mat)
else:
   K_mat=csr_matrix(K_mat)
G_mat=csr_matrix(G_mat)
H_mat=csr_matrix(H_mat)
M_mat=csr_matrix(M_mat)
MP_mat=csr_matrix(MP_mat)
L_mat=csr_matrix(L_mat)
sparse_matrix=sps.csc_matrix(A_fem)

print("convert to CSR: %.3f s, nel= %d" % (clock.time()-start, nel))



###############################################################################
# solve system
###############################################################################
start=clock.time()

niter=0

if solver==1: sol=sps.linalg.spsolve(sparse_matrix,b_fem,use_umfpack=False)

elif solver==2:
   sol,info=sps.linalg.gmres(sparse_matrix,b_fem,restart=2000,tol=tolerance)
   if info!=0: exit('gmres did not converge')

elif solver==3: sol=sps.linalg.lgmres(sparse_matrix,b_fem,atol=1e-16,tol=tolerance)[0]

elif solver==4: solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                             Nfem_V,Nfem_P,niter_max,tolerance,use_precond,'direct')
elif solver==5: sol=sps.linalg.minres(sparse_matrix,b_fem,tol=1e-12)[0]

elif solver==6: sol=sps.linalg.qmr(sparse_matrix,b_fem,tol=tolerance)[0]

elif solver==7: sol=sps.linalg.tfqmr(sparse_matrix,b_fem, tol=tolerance)[0]

elif solver==8: sol=sps.linalg.gcrotmk(sparse_matrix,b_fem,atol=1e-16, tol=tolerance)[0]

elif solver==9: sol=sps.linalg.bicg(sparse_matrix,b_fem, tol=tolerance)[0]

elif solver==10: sol=sps.linalg.bicgstab(sparse_matrix,b_fem, tol=tolerance)[0]

elif solver==11: sol=sps.linalg.spsolve(sparse_matrix,b_fem,use_umfpack=True)

elif solver==12: sol=sps.linalg.cgs(sparse_matrix,b_fem,tol=tolerance)[0]

elif solver==13: solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                              Nfem_V,Nfem_P,niter_max,tolerance,use_precond,'cg')
elif solver==14: solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                              Nfem_V,Nfem_P,niter_max,tolerance,use_precond,'splu')
elif solver==15: solV,p,niter=uzawa1_solver(K_mat,G_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance,omega15)

elif solver==16: solV,p,niter=uzawa2_solver(K_mat,G_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance)

elif solver==17: solV,p,niter=projection_solver(K_mat,G_mat,L_mat,f_rhs,h_rhs,Nfem_V,Nfem_P,niter_max,tolerance)

elif solver==18: solV,p,niter=uzawa1_solver_L2(K_mat,G_mat,MP_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance,omega18)

elif solver==19: solV,p,niter=uzawa1_solver_L2b(K_mat,G_mat,MP_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance,omega19)

elif solver==20: solV,p,niter=uzawa2_solver_L2(K_mat,G_mat,MP_mat,H_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance)

elif solver==21: solV,p,niter=uzawa3_solver(K_mat,G_mat,f_rhs,h_rhs,Nfem_P,niter_max,tolerance)

elif solver==22: solV,p,niter=uzawa3_solver_L2(K_mat,G_mat,MP_mat,H_mat,f_rhs,h_rhs,\
                              Nfem_P,niter_max,tolerance,'direct')

elif solver==23: solV,p,niter=uzawa3_solver_L2(K_mat,G_mat,MP_mat,H_mat,f_rhs,h_rhs,\
                              Nfem_P,niter_max,tolerance,'splu')
else:
   exit('solver unknown')

print("solve time: %.3f s, nel= %d, niter= %d" % (clock.time()-start,nel,niter))

###############################################################################
# Poisson correction 
# L_mat has been assembled above, as well as h_rhs and G_mat
###############################################################################
start=clock.time()

if projection and (solver ==15 or solver==16):

   #L_mat=G_mat.T.dot(G_mat) not good coz G contains zero lines
   b_rhs=h_rhs-G_mat.T.dot(solV)        # right hand side b 
   qq=sps.linalg.spsolve(L_mat,b_rhs)

   print("     -> projection: qq (m,M) %.4e %.4e " %(np.min(qq),np.max(qq)))
   print("     -> projection: u,v corr (m,M) %.4e %.4e " %(np.min(G_mat.dot(qq)),np.max(G_mat.dot(qq))))

   solV+=G_mat.dot(qq)

   print("post projection: %.3f s, nel= %d" % (clock.time()-start,nel))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

if solver==4 or solver==13 or solver==14 or solver==15 or solver==16 or\
   solver==17 or solver==18 or solver==19 or solver==20 or solver==21 or\
   solver==22 or solver==23: 
   u,v=np.reshape(solV,(nn_V,2)).T
else:
   u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
   p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
#normalise pressure
###############################################################################
start=clock.time()

pavrg=0.
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            pq=N_P[:].dot(p[icon_P[:,iel]])
            pavrg+=pq*JxWq
        # end for jq
    # end for iq
# end for iel

p[:]-=pavrg/Lx/Ly

print("     -> pavrg %e " %(pavrg))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel,nodes in enumerate(icon_V.T):
    rq=0. ; sq=0. ; weightq=2.*2.
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

vrms=0.
errv=0.
divv=0.
errp=0.
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
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
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
            errv+=((uq-u_analytical(xq,yq))**2+(vq-v_analytical(xq,yq))**2)*JxWq
            vrms+=(uq**2+vq**2)*JxWq
            divv+=(exxq+eyyq)**2*JxWq
            pq=N_P.dot(p[icon_P[:,iel]])
            errp+=(pq-p_analytical(xq,yq))**2*JxWq
        #end for jq
    #end for iq
#end for iel

divv=np.sqrt(divv/Lx/Ly)
errv=np.sqrt(errv/Lx/Ly)
errp=np.sqrt(errp/Lx/Ly)
vrms=np.sqrt(vrms/Lx/Ly)

print("     -> nel= %6d ; errv= %e ; errp= %e ; divv= %e ; solver= %d" %(nel,errv,errp,divv,solver))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel,nodes in enumerate(icon_P.T):
    q[icon_V[0,iel]]=p[nodes[0]]
    q[icon_V[1,iel]]=p[nodes[1]]
    q[icon_V[2,iel]]=p[nodes[2]]
    q[icon_V[3,iel]]=p[nodes[3]]
    q[icon_V[4,iel]]=(p[nodes[0]]+p[nodes[1]])*0.5
    q[icon_V[5,iel]]=(p[nodes[1]]+p[nodes[2]])*0.5
    q[icon_V[6,iel]]=(p[nodes[2]]+p[nodes[3]])*0.5
    q[icon_V[7,iel]]=(p[nodes[3]]+p[nodes[0]])*0.5
    q[icon_V[8,iel]]=(p[nodes[0]]+p[nodes[1]]+p[nodes[2]]+p[nodes[3]])*0.25

if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("from p to q: %.3f s" % (clock.time()-start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################
start=clock.time()

for i in range(0,nn_V):
    if abs(x_V[i]-0.5)<eps and abs(y_V[i]-0.5)<eps:
       uc=u[i]
       vc=abs(v[i])

vel=np.sqrt(u**2+v**2)
print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms,uc,vc)

#profile=open('profile.ascii',"w")
#for i in range(0,NV):
#    if abs(x[i]-0.5)<1e-6:
#       profile.write("%10e %10e %10e %10e \n" %(y[i],u[i],v[i],q[i]))

print("export measurements: %.3f s" % (clock.time()-start))
    
###############################################################################
# plot of solution
###############################################################################
start=clock.time()

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
exx.tofile(vtufile,sep=' ',format='%.5e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
eyy.tofile(vtufile,sep=' ',format='%.5e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
exy.tofile(vtufile,sep=' ',format='%.5e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
eta_e.tofile(vtufile,sep=' ',format='%.5e')
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.5e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %p_analytical(x_V[i],y_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                   icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                   icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*9))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export solution to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
