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

Lx=2.  # horizontal extent of the domain 
Ly=2.  # vertical extent of the domain 

nel=5*4
NV=25
    
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

x[2]=2
y[2]=0

x[3]=0
y[3]=1

x[4]=1
y[4]=1

x[5]=2
y[5]=1

x[6]=0
y[6]=2

x[7]=1
y[7]=2

x[8]=2
y[8]=2

x[ 9]=0.22
y[ 9]=0.22

x[10]=0.78
y[10]=0.22

x[11]=1.22
y[11]=0.22

x[12]=1.78
y[12]=0.22

x[13]=0.22
y[13]=0.78

x[14]=0.78
y[14]=0.78

x[15]=1.22
y[15]=0.78

x[16]=1.78
y[16]=0.78


x[17]=0.22
y[17]=1.22

x[18]=0.78
y[18]=1.22

x[19]=1.22
y[19]=1.22

x[20]=1.78
y[20]=1.22

x[21]=0.22
y[21]=1.78

x[22]=0.78
y[22]=1.78

x[23]=1.22
y[23]=1.78

x[24]=1.78
y[24]=1.78


np.savetxt('velocity.ascii',np.array([x,y]).T)




#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)

icon[0, 0] = 0 
icon[1, 0] = 1
icon[2, 0] = 10
icon[3, 0] = 9

icon[0, 1] = 10
icon[1, 1] = 1
icon[2, 1] = 4
icon[3, 1] = 14

icon[0, 2] = 13
icon[1, 2] = 14
icon[2, 2] = 4
icon[3, 2] = 3

icon[0, 3] = 0
icon[1, 3] = 1
icon[2, 3] = 13
icon[3, 3] = 3

icon[0, 4] = 9
icon[1, 4] = 10
icon[2, 4] = 14
icon[3, 4] = 13

icon[0,5]=1
icon[1,5]=2 
icon[2,5]=12
icon[3,5]=11

icon[0,6]=12
icon[1,6]=2
icon[2,6]=5
icon[3,6]=16

icon[0,7]=15
icon[1,7]=16
icon[2,7]=5
icon[3,7]=4

icon[0,8]=1
icon[1,8]=11
icon[2,8]=15
icon[3,8]=4

icon[0,9]=11
icon[1,9]=12
icon[2,9]=16
icon[3,9]=15


icon[0,10]=3
icon[1,10]=4
icon[2,10]=18
icon[3,10]=17

icon[0,11]=18
icon[1,11]=4
icon[2,11]=7
icon[3,11]=22

icon[0,12]=21
icon[1,12]=22
icon[2,12]=7
icon[3,12]=6

icon[0,13]=3
icon[1,13]=17
icon[2,13]=21
icon[3,13]=6

icon[0,14]=17
icon[1,14]=18
icon[2,14]=22
icon[3,14]=21



icon[0,15]=4
icon[1,15]=5
icon[2,15]=20
icon[3,15]=19

icon[0,16]=20
icon[1,16]=5
icon[2,16]=8
icon[3,16]=24

icon[0,17]=23
icon[1,17]=24
icon[2,17]=8
icon[3,17]=7

icon[0,18]=4
icon[1,18]=19
icon[2,18]=23
icon[3,18]=7

icon[0,19]=19
icon[1,19]=20
icon[2,19]=24
icon[3,19]=23

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
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
    print (G_mat[i,:]) 
    #print ("%3i  & %3i & %3i & %3i  \\\\ " %(int(round(G_mat[i,0])),int(round(G_mat[i,1])),int(round(G_mat[i,2])),int(round(G_mat[i,3])) ))


#print (G_mat)
G2 = np.zeros((34,NfemP),dtype=np.float64) # matrix GT

print("----------------------------------------------")

G2[0,:] =G_mat[8,:] #4
G2[1,:] =G_mat[9,:]
G2[2,:] =G_mat[18,:] #9
G2[3,:] =G_mat[19,:]
G2[4,:] =G_mat[20,:] #10
G2[5,:] =G_mat[21,:]
G2[6,:] =G_mat[22,:] #11
G2[7,:] =G_mat[23,:]
G2[8,:] =G_mat[24,:] #12
G2[9,:] =G_mat[25,:]
G2[10,:]=G_mat[26,:] #13
G2[11,:]=G_mat[27,:]
G2[12,:]=G_mat[28,:] #14
G2[13,:]=G_mat[29,:]

G2[14,:]=G_mat[30,:] #15
G2[15,:]=G_mat[31,:]
G2[16,:]=G_mat[32,:] #16
G2[17,:]=G_mat[33,:]
G2[18,:]=G_mat[34,:] #17
G2[19,:]=G_mat[35,:]
G2[20,:]=G_mat[36,:] #18
G2[21,:]=G_mat[37,:]
G2[22,:]=G_mat[38,:] #19
G2[23,:]=G_mat[39,:]

G2[24,:]=G_mat[40,:] #20
G2[25,:]=G_mat[41,:]
G2[26,:]=G_mat[42,:] #21
G2[27,:]=G_mat[43,:]
G2[28,:]=G_mat[44,:] #22
G2[29,:]=G_mat[45,:]
G2[30,:]=G_mat[46,:] #23
G2[31,:]=G_mat[47,:]
G2[32,:]=G_mat[48,:] #24
G2[33,:]=G_mat[49,:]





#for i in range (10):
#    print ("%3f %3f %3f %3f %3f %3f %3f %3f %3f " %(G2[i,0],G2[i,1],G2[i,2],G2[i,3],G2[i,4],G2[i,5],G2[i,6],G2[i,7],G2[i,8]))

ns = null_space(G2)

print(ns)






