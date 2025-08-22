import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix
from schur_complement_cg_solver import *
import scipy.sparse.linalg as sla
from uzawa1_solver import *
from uzawa1_solver_L2 import *
from uzawa1_solver_L2b import *
from uzawa2_solver import *
from projection_solver import *
from basis_functions import *

###############################################################################
# bench=1: donea & huerta
# bench=2: stokes sphere
# bench=3: block
###############################################################################

def bx(x,y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2 or bench==3 or bench==4:
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
    if bench==1:
       val=1
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
    if bench==4:
       val=1
    return val

###############################################################################

def uth(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2 or bench==3:
       val=0
    if bench==4:
       val=20*x*y**3
    return val

def vth(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2 or bench==3:
       val=0
    if bench==4:
       val=5*x**4-5*y**4
    return val

def pth(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    if bench==2 or bench==3:
       val=0
    if bench==4:
       val=60*x**2*y-20*y**3-5
    return val

###############################################################################

eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 6):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   nqperdim = int(sys.argv[4])
   solver = int(sys.argv[5])
else:
   nelx = 32
   nely = nelx
   visu = 1
   nqperdim=3
   solver=19
    
nnx=2*nelx+1         # number of V nodes, x direction
nny=2*nely+1         # number of V nodes, y direction
NV=nnx*nny           # number of V nodes
NP=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely        # total number of elements
NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total number of dofs

hx=Lx/nelx
hy=Ly/nely

bench=1
   
omega=1

use_precond=False
precond_type=0
tolerance=1e-7
niter_max=5000

projection=False

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

if bench==1:
   NS=True

if bench==4:
   NS=True

###############################################################################

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
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
print("NV=",NV)
print("NP=",NP)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("Nfem=",Nfem)
print("solver=",solver)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

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
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if NS:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=uth(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(xV[i],yV[i])
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=uth(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(xV[i],yV[i])
       if yV[i]<eps:
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=uth(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(xV[i],yV[i])
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=uth(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=vth(xV[i],yV[i])
       #end if
   #end for

if FS:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
   #end for

if OT:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
   #end for

if BO:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #end if
   #end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

A_fem   = lil_matrix((Nfem,Nfem),dtype=np.float64)   # matrix of Ax=b
K_mat   = lil_matrix((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat   = lil_matrix((NfemV,NfemP),dtype=np.float64) # matrix GT
L_mat   = lil_matrix((NfemP,NfemP),dtype=np.float64) # matrix GT*G
MP_mat  = lil_matrix((NfemP,NfemP),dtype=np.float64) # pressure mass matrix
MV_mat  = lil_matrix((NfemV,NfemV),dtype=np.float64) # velocity mass matrix
f_rhs   = np.zeros(NfemV,dtype=np.float64)           # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)           # right hand side h 
b_fem   = np.zeros(Nfem,dtype=np.float64)            # rhs of fem linear system
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)    # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)    # matrix  
dNNNVdx = np.zeros(mV,dtype=np.float64)              # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)              # shape functions derivatives
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb     = np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):

    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    L_el=np.zeros((mP*ndofP,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    MP_el=np.zeros((mP,mP),dtype=np.float64)
    MV_el=np.zeros((mV,mV),dtype=np.float64)

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            NNNP=NNP(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=dNNNVdr[:].dot(xV[iconV[:,iel]])
            jcb[0,1]=dNNNVdr[:].dot(yV[iconV[:,iel]])
            jcb[1,0]=dNNNVds[:].dot(xV[iconV[:,iel]])
            jcb[1,1]=dNNNVds[:].dot(yV[iconV[:,iel]])
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=NNNV[:].dot(xV[iconV[:,iel]])
            yq=NNNV[:].dot(yV[iconV[:,iel]])
            dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]+jcbi[0,1]*dNNNVds[:]
            dNNNVdy[:]=jcbi[1,0]*dNNNVdr[:]+jcbi[1,1]*dNNNVds[:]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]
            #end for 

            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)
            #end for 

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            MP_el+=np.outer(NNNP,NNNP)*weightq*jcob

        #end for jq
    #end for iq

    #print(MP_el*9/hx/hy) ok! identical to appendix A.2.1
    
    L_el=G_el.T.dot(G_el) # before b.c.!

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
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

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
                    A_fem[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
                A_fem[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_fem[NfemV+m2,m1]+=G_el[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

    #assemble pressure mass matrix
    for k1 in range(0,mP):
        m1=iconP[k1,iel]
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            L_mat[m1,m2]+=L_el[k1,k2]
            MP_mat[m1,m2]+=MP_el[k1,k2]
        #end for 
    #end for 

#end for iel

b_fem[0:NfemV]=f_rhs
b_fem[NfemV:Nfem]=h_rhs

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# compute preconditioner
###############################################################################

Mprec = lil_matrix((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
for i in range(0,Nfem):
    Mprec[i,i]=10
Mprec=csr_matrix(Mprec)

###############################################################################
# compute Schur preconditioner
###############################################################################
start=clock.time()

M_mat = lil_matrix((NfemP,NfemP),dtype=np.float64)  # matrix of Ax=b
   
#if precond_type==0:
#   for i in range(0,NfemP):
#       M_mat[i,i]=1

#if precond_type==1:
#   for iel in range(0,nel):
#       M_mat[iel,iel]=hx*hy/eta(xc[iel],yc[iel])

if precond_type==2:
   Km1 = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))

if precond_type==3:
   Km1 = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,NfemP):
       for j in range(0,NfemP):
           if i!=j:
              M_mat[i,j]=0.

if precond_type==4:
   Km1 = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,NfemP):
       for j in range(0,NfemP):
           if i!=j:
              M_mat[i,i]+=np.abs(M_mat[i,j])
              M_mat[i,j]=0.

#plt.spy(M_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build Schur matrix precond: %e s, nel= %d" % (clock.time()-start,nel))

###############################################################################
# convert lil matrices to csr
###############################################################################
start=clock.time()

if solver==14:
   K_mat=csc_matrix(K_mat)
else:
   K_mat=csr_matrix(K_mat)
G_mat=csr_matrix(G_mat)
M_mat=csr_matrix(M_mat)
MP_mat=csr_matrix(MP_mat)
L_mat=csr_matrix(L_mat)
sparse_matrix=sps.csc_matrix(A_fem)

print("convert to CSR: %.3f s, nel= %d" % (clock.time()-start, nel))

###############################################################################
#start = clock.time()

#ILUfact = sla.spilu(sparse_matrix)
#M = sla.LinearOperator(
#    shape = sparse_matrix.shape,
#    matvec = lambda b: ILUfact.solve(b)
#)

#other option from 
#https://stackoverflow.com/questions/58895934/how-to-implement-ilu-precondioner-in-scipy
#sA_iLU = sparse.linalg.spilu(sA)
#M = sparse.linalg.LinearOperator((nrows,ncols), sA_iLU.solve)
#also does nto work

print("generate ILU precond: %.3f s, nel= %d" % (clock.time()-start,nel))

###############################################################################
# solve system
###############################################################################
start=clock.time()

if solver==1:
   sol=sps.linalg.spsolve(sparse_matrix,b_fem,use_umfpack=False)
elif solver==2:
   sol,info = scipy.sparse.linalg.gmres(sparse_matrix,b_fem,restart=2000,tol=tolerance,M=Mprec)
   if info!=0: exit('gmres did not converge')
elif solver==3:
   sol = scipy.sparse.linalg.lgmres(sparse_matrix,b_fem,atol=1e-16,tol=tolerance)[0]
elif solver==4:
   solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                                           NfemV,NfemP,niter_max,tolerance,use_precond,'direct')
elif solver==5:
   sol = scipy.sparse.linalg.minres(sparse_matrix,b_fem, tol=1e-10)[0]
elif solver==6:
   sol = scipy.sparse.linalg.qmr(sparse_matrix,b_fem,tol=tolerance)[0]
elif solver==7:
   sol = scipy.sparse.linalg.tfqmr(sparse_matrix,b_fem, tol=1e-10)[0]
elif solver==8:
   sol = scipy.sparse.linalg.gcrotmk(sparse_matrix,b_fem,atol=1e-16, tol=1e-10)[0]
elif solver==9:
   sol = scipy.sparse.linalg.bicg(sparse_matrix,b_fem, tol=tolerance)[0]
elif solver==10:
   sol = scipy.sparse.linalg.bicgstab(sparse_matrix,b_fem, tol=tolerance)[0]
elif solver==11:
   sol=sps.linalg.spsolve(sparse_matrix,b_fem,use_umfpack=True)
elif solver==12:
   sol = scipy.sparse.linalg.cgs(sparse_matrix,b_fem, tol=tolerance)[0]
elif solver==13:
   solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                                           NfemV,NfemP,niter_max,tolerance,use_precond,'cg')
elif solver==14:
   solV,p,niter=schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                                           NfemV,NfemP,niter_max,tolerance,use_precond,'splu')
elif solver==15:
   solV,p,niter=uzawa1_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                       NfemV,NfemP,niter_max,tolerance,use_precond,'direct',omega)
elif solver==16:
   solV,p,niter=uzawa2_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                       NfemV,NfemP,niter_max,tolerance,use_precond,'direct')
elif solver==17:
   solV,p,niter=projection_solver(K_mat,G_mat,L_mat,f_rhs,h_rhs,\
                                  NfemV,NfemP,niter_max,tolerance)
elif solver==18:
   solV,p,niter=uzawa1_solver_L2(K_mat,G_mat,MP_mat,f_rhs,h_rhs,\
                                 NfemP,niter_max,tolerance,omega)
elif solver==19:
   solV,p,niter=uzawa1_solver_L2b(K_mat,G_mat,MP_mat,f_rhs,h_rhs,\
                                 NfemP,niter_max,tolerance,omega)

else:
   exit('solver unknown')

print("solve time: %.3f s, nel= %d" % (clock.time()-start,nel))

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
   solver==17 or solver==18 or solver==19: 
   u,v=np.reshape(solV,(NV,2)).T
else:
   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
#normalise pressure
###############################################################################
start=clock.time()

pavrg=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            NNNP=NNP(rq,sq)
            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            pavrg+=pq*weightq*jcob
        # end for jq
    # end for iq
# end for iel

p[:]-=pavrg/Lx/Ly

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0. ; sq=0. ; weightq=2.*2.

    NNNV=NNV(rq,sq)
    dNNNVdr=dNNVdr(rq,sq)
    dNNNVds=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    #end for
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    #end for

    for k in range(0,mV):
        xc[iel]+=NNNV[k]*xV[iconV[k,iel]]
        yc[iel]+=NNNV[k]*yV[iconV[k,iel]]
        exx[iel]+=dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel]+=dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel]+=0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
    #end for

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

vrms=0.
errv=0.
divv=0.
errp=0.
avrgp=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            NNNP=NNP(rq,sq)

            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)

            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            exxq=0.0
            eyyq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
            #end for
            errv+=((uq-uth(xq,yq))**2+(vq-vth(xq,yq))**2)*weightq*jcob
            vrms+=(uq**2+vq**2)*weightq*jcob
            divv+=(exxq+eyyq)**2*weightq*jcob

            pq=NNNP.dot(p[iconP[:,iel]])
            errp+=(pq-pth(xq,yq))**2*weightq*jcob

            avrgp+=pq*weightq*jcob

        #end for jq
    #end for iq
#end for iel

divv=np.sqrt(divv/Lx/Ly)
errv=np.sqrt(errv/Lx/Ly)
errp=np.sqrt(errp/Lx/Ly)
vrms=np.sqrt(vrms/Lx/Ly)
avrgp=avrgp/Lx/Ly

print("     -> nel= %6d ; errv= %e ; errp= %e ; divv= %e" %(nel,errv,errp,divv))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

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

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("from p to q: %.3f s" % (clock.time()-start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################
start=clock.time()

for i in range(0,NV):
    if abs(xV[i]-0.5)<eps and abs(yV[i]-0.5)<eps:
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
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
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
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %pth(xV[i],yV[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                   iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                   iconV[6,iel],iconV[7,iel],iconV[8,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
