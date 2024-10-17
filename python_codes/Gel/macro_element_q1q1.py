import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix 
import time as time
from scipy.linalg import null_space

#------------------------------------------------------------------------------

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4      # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom per node

if int(len(sys.argv) == 8):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   rho1 = float(sys.argv[4])
   drho = float(sys.argv[5])
   eta1 = float(sys.argv[6])
   eta2 = float(sys.argv[7])
else:
   nelx = 2
   nely = 2
   visu = 1
   rho1 = 3200
   drho = 32
   eta1 = 1e21
   eta2 = 1e23

rho2=rho1+drho
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nnp*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

Lx=4
Ly=4


pnormalise=True

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################

print('nelx= %d ' %nelx)
print('nely= %d ' %nely)
print('Lx=   %e ' %Lx)
print('Ly=   %e ' %Lx)
print('rho1= %e ' %rho1)
print('rho2= %e ' %rho2)
print('eta1= %e ' %eta1)
print('eta2= %e ' %eta2)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp,dtype=np.float64)  # x coordinates
y = np.empty(nnp,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter]=i+j*(nelx+1)
        icon[1,counter]=i+1+j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx+1)
        icon[3,counter]=i+(j+1)*(nelx+1)
        counter+=1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,nnp):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if y[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0


print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT -C][p] [h]
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64)             # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64)             # matrix G
C_mat = np.zeros((NfemP,NfemP),dtype=np.float64)             # matrix C
f_rhs = np.zeros(NfemV,dtype=np.float64)                     # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)                     # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)                     # constraint matrix/vector
b_mat = np.zeros((3,ndofV*m),dtype=np.float64)               # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)                         # shape functions
dNdx  = np.zeros(m,dtype=np.float64)                         # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)                         # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)                         # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)                         # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)                       # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)                       # y-component velocity
p     = np.zeros(nnp,dtype=np.float64)                       # pressure 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

Nvect = np.zeros((1,m),dtype=np.float64)
N_mat = np.zeros((3,m),dtype=np.float64)
    
Navrg = np.zeros(m,dtype=np.float64)
Navrg[0]=0.25
Navrg[1]=0.25
Navrg[2]=0.25
Navrg[3]=0.25

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,m*ndofP),dtype=np.float64)
    C_el=np.zeros((m*ndofP,m*ndofP),dtype=np.float64)
    h_el=np.zeros((m*ndofP),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            #K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq,case)*weightq*jcob

            # compute elemental rhs vector
            #for i in range(0,m):
            #    f_el[ndofV*i  ]+=N[i]*jcob*weightq*bx(xq,yq,case)
            #    f_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq,case)

            # compute G_el matrix
            for i in range(0,m):
                N_mat[0,i]=N[i]
                N_mat[1,i]=N[i]
                N_mat[2,i]=0
            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            # compute C_el matrix
            #Nvect[0,0:m]=N[0:m]-Navrg[0:m]
            #C_el+=Nvect.T.dot(Nvect)*jcob*weightq/viscosity(xq,yq,case)

    G_el*=6
    print(G_el)

    # impose b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1  # from 0 to 7 
            m1 =ndofV*icon[k1,iel]+i1
            # assemble K block
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2  # from 0 to 7 
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            # assemble f vector 
            f_rhs[m1]+=f_el[ikk]
            # assemble G block
            for k2 in range(0,m):
                m2 = icon[k2,iel]
                jkk=k2                       # from 0 to 3
                G_mat[m1,m2]+=G_el[ikk,jkk]
        for k2 in range(0,m):
            C_mat[icon[k1,iel],icon[k2,iel]]+=C_el[k1,k2] 

    for k2 in range(0,m): # assemble h
        m2=icon[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=N[k2]


for i in range (NfemV):
    print ("%3i  & %3i & %3i & %3i & %3i &  %3i & %3i & %3i & %3i \\\\ " \
          %(int(round(G_mat[i,0])),int(round(G_mat[i,1])),int(round(G_mat[i,2])),int(round(G_mat[i,3])),int(round(G_mat[i,4])),int(round(G_mat[i,5])),int(round(G_mat[i,6])),int(round(G_mat[i,7])),int(round(G_mat[i,8]))))


#print (G_mat)
G2 = np.zeros((2,NfemP),dtype=np.float64) # matrix GT

print("----------------------------------------------")

G2[0,:]=G_mat[8,:]
G2[1,:]=G_mat[9,:]

#for i in range (10):
#    print ("%3f %3f %3f %3f %3f %3f %3f %3f %3f " %(G2[i,0],G2[i,1],G2[i,2],G2[i,3],G2[i,4],G2[i,5],G2[i,6],G2[i,7],G2[i,8]))

ns = null_space(G2)

print(ns)









