import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
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

m=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

nel=5
NV=8
    
NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

viscosity=1.  # dynamic viscosity \mu

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV, dtype=np.float64)  # x coordinates
y = np.empty(NV, dtype=np.float64)  # y coordinates


x[0]=0
y[0]=0

x[1]=1
y[1]=0

x[2]=1
y[2]=1

x[3]=0
y[3]=1

x[4]=0.25
y[4]=0.25

x[5]=0.75
y[5]=0.25

x[6]=0.75
y[6]=0.75

x[7]=0.25
y[7]=0.75

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)

icon[0, 0] = 0 
icon[1, 0] = 1
icon[2, 0] = 5
icon[3, 0] = 4

icon[0, 1] = 5
icon[1, 1] = 1
icon[2, 1] = 2
icon[3, 1] = 6

icon[0, 2] = 7
icon[1, 2] = 6
icon[2, 2] = 2
icon[3, 2] = 3

icon[0, 3] = 0
icon[1, 3] = 4
icon[2, 3] = 7
icon[3, 3] = 3

icon[0, 4] = 4
icon[1, 4] = 5
icon[2, 4] = 6
icon[3, 4] = 7


#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0, NV):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((3,ndofV*m),dtype=np.float64)  # gradient matrix B 
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                #f_el[ndofV*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                #f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq


    #print('elt',iel,'->',G_el)

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
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]




for i in range (NfemV):
    print (G_mat[i,0], G_mat[i,2], G_mat[i,3], G_mat[i,4], G_mat[i,4]) 
    #print ("%3i  & %3i & %3i & %3i  \\\\ " %(int(round(G_mat[i,0])),int(round(G_mat[i,1])),int(round(G_mat[i,2])),int(round(G_mat[i,3])) ))


#print (G_mat)
G2 = np.zeros((8,NfemP),dtype=np.float64) # matrix GT

print("----------------------------------------------")

G2[0,:]=G_mat[8,:]
G2[1,:]=G_mat[9,:]
G2[2,:]=G_mat[10,:]
G2[3,:]=G_mat[11,:]
G2[4,:]=G_mat[12,:]
G2[5,:]=G_mat[13,:]
G2[6,:]=G_mat[14,:]
G2[7,:]=G_mat[15,:]



#for i in range (10):
#    print ("%3f %3f %3f %3f %3f %3f %3f %3f %3f " %(G2[i,0],G2[i,1],G2[i,2],G2[i,3],G2[i,4],G2[i,5],G2[i,6],G2[i,7],G2[i,8]))

ns = null_space(G2)

print(ns)






