import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix,lil_matrix
import time as time

#------------------------------------------------------------------------------

def rho(x,y):
    if (x)**2+(y-Ly/2)**2<0.123456789**2:
       val=1.01
    else:
       val=1.
    return val

def eta(x,y):
    if (x)**2+(y-Ly/2)**2<0.123456789**2:
       val=1000.
    else:
       val=1.
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

eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
   
Lx=0.5 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx     = int(sys.argv[1])
   visu     = int(sys.argv[2])
   nqperdim = int(sys.argv[3])
   Ly       = float(sys.argv[4])
else:
   nelx = 256
   visu = 1
   nqperdim=3
   Ly=1.  

nely=int(nelx*Ly/Lx)

axisymmetric=True
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

NV=nnx*nny           # number of V nodes
NP=(nelx+1)*(nely+1) # number of P nodes

nel=nelx*nely  # number of elements, total

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

hx=Lx/nelx
hy=Ly/nely

pnormalise=True

gx=0
gy=-1

sparse=True

#################################################################

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
if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

#################################################################
#################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("NP=",NP)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

if sparse:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
NNNV    = np.zeros(mV,dtype=np.float64)          # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)          # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)          # x-component velocity
v       = np.zeros(NV,dtype=np.float64)          # y-component velocity

if axisymmetric:
   c_mat   = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
   b_mat   = np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
   N_mat   = np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
else:
   c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
   N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

mass=0.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)
            NNNP[0:4]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for 
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
            #end for 

            if axisymmetric: #-----------------------

               for i in range(0,mV):
                   b_mat[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [NNNV[i]/xq,0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob * 2*np.pi*xq
               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=NNNP[i]
                   N_mat[3,i]=0.
               G_el-=b_mat.T.dot(N_mat)*weightq*jcob * 2*np.pi*xq

               mass+=jcob*weightq*rho(xq,yq) * 2*np.pi*xq

               for i in range(0,mV):
                   f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rho(xq,yq) * 2*np.pi*xq
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho(xq,yq) * 2*np.pi*xq
               #end for 

               NNNNP[:]+=NNNP[:]*jcob*weightq * 2*np.pi*xq

            else : #----------------------------------

               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]
               #end for 
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob
               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.
               G_el-=b_mat.T.dot(N_mat)*weightq*jcob 

               mass+=jcob*weightq*rho(xq,yq)

               for i in range(0,mV):
                   f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rho(xq,yq)
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho(xq,yq)
               #end for 

               NNNNP[:]+=NNNP[:]*jcob*weightq

        #end for jq
    #end for iq

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
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
                    #end if
                #end for
            #end for
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
                #end if
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
    #end for 

#end for iel

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

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
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# if pressure normalisation is not on, then a pressure dof 
# must be fixed to remove the nullspace.
######################################################################

#if not pnormalise:
#   for i in range(0,Nfem):
#       a_mat[Nfem-1,i]=0
#   a_mat[Nfem-1,Nfem-1]=1
#   rhs[Nfem-1]=-1/6

######################################################################
# solve system
######################################################################
start = time.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    NNNV[0:9]=NNV(rq,sq)
    dNNNVdr[0:9]=dNNVdr(rq,sq)
    dNNNVds[0:9]=dNNVds(rq,sq)

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
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
    #end for

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute vrms 
######################################################################
start = time.time()

vrms=0.
avrgp=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)
            NNNP[0:4]=NNP(rq,sq)

            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)

            uq=0.0
            vq=0.0
            for k in range(0,mV):
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            #end for
            vrms+=(uq**2+vq**2)*weightq*jcob

            pq=0.0
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            #end for
            avrgp+=pq*weightq*jcob

        #end for jq
    #end for iq
#end for iel

vrms=np.sqrt(vrms/Lx/Ly)
avrgp=avrgp/Lx/Ly

if pnormalise:
   p-=avrgp

print("     -> nel= %6d ; vrms= %e " %(nel,vrms))

print("compute vrms: %.3f s" % (time.time() - start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################

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
    q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25
#end for

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

#####################################################################

if axisymmetric:

   massT=np.pi*Lx**2*Ly*1 + 4./3.*np.pi*0.123456789**3*0.01

else:

   massT=Lx*Ly*1 + 0.5*np.pi*0.123456789**2*0.01

print('mass system (anal)', massT)
print('mass system (meas)', mass)


#####################################################################
# export various measurements for stokes sphere benchmark 
#####################################################################

for i in range(0,NV):
    if abs(xV[i])<eps and abs(yV[i]-Ly/2)<eps:
       uc=u[i]
       vc=abs(v[i])

vel=np.sqrt(u**2+v**2)

vs=2./9.*0.01/1*1*0.123456789**2

fh=-2.1050*(0.123456789/Lx)  \
   +2.0865*(0.123456789/Lx)**3  \
   -1.7068*(0.123456789/Lx)**5\
   +0.72603*(0.123456789/Lx)**6

gamma_h = (1-0.75857*(0.123456789/Lx)**5)/(1+fh)

vh=vs/gamma_h

ff=-2.10444*(0.123456789/Lx)\
   +2.08877*(0.123456789/Lx)**3\
   -0.94813*(0.123456789/Lx)**5\
   -1.372*(0.123456789/Lx)**6\
   +3.87*(0.123456789/Lx)**8\
   -4.19*(0.123456789/Lx)**10

gamma_f=1./(1+ff)

vf=vs/gamma_f

print('vs=',vs)
print('vh=',vh)
print('vf=',vf)

print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      0,0,\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),\
      vrms,avrgp,mass,uc,vc,
      vs,vh,vf,massT)

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
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %(rho(xV[i],yV[i])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %(eta(xV[i],yV[i])))
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
