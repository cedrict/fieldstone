import sys as sys
import numpy as np
import time as timing
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import random
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
from numpy import linalg 
np.set_printoptions(linewidth=220)                                                                                      
###############################################################################

def B(r,s):
    if bubble==0:
       return (1-r**2)*(1-s**2)
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    if bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))

def dBdr(r,s):
    if bubble==0:
       return (-2*r)*(1-s**2)
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    if bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))

def dBds(r,s):
    if bubble==0:
       return (1-r**2)*(-2*s)
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    if bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)

#------------------------------------------------------------------------------

def NNV(r,s):
    NV_0= 0.25*(1-r)*(1-s) - 0.25*B(r,s)
    NV_1= 0.25*(1+r)*(1-s) - 0.25*B(r,s)
    NV_2= 0.25*(1+r)*(1+s) - 0.25*B(r,s)
    NV_3= 0.25*(1-r)*(1+s) - 0.25*B(r,s)
    NV_4= B(r,s)
    return NV_0,NV_1,NV_2,NV_3,NV_4

def dNNVdr(r,s):
    dNVdr_0=-0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_1=+0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_2=+0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_3=-0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_4=dBdr(r,s) 
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4

def dNNVds(r,s):
    dNVds_0=-0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_1=-0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_2=+0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_3=+0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_4=dBds(r,s) 
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4

def NNP(r,s):
    NP_0= 0.25*(1-r)*(1-s)
    NP_1= 0.25*(1+r)*(1-s)
    NP_2= 0.25*(1+r)*(1+s)
    NP_3= 0.25*(1-r)*(1+s)
    return NP_0,NP_1,NP_2,NP_3 

#------------------------------------------------------------------------------

ndim=2
ndofV=2
ndofP=1
mV=5
mP=4

Lx=2
Ly=2

bubble=2

nelx = 2
nely = 2
visu = 1
nqperdim=4

if int(len(sys.argv) == 2):
   beta=float(sys.argv[1])
else:
   beta=0.25

nel=nelx*nely
NV=(nelx+1)*(nely+1)+nel
NP=(nelx+1)*(nely+1)
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP
hx=Lx/nelx
hy=Ly/nely

print('-----------------------')
print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('bubble=',bubble)
print('beta=',beta)
print('-----------------------')

nqperdim=2

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

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


eps=1e-8

sparse=True

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        xV[counter]=i*hx
        yV[counter]=j*hy
        counter += 1

for j in range(0,nely):
    for i in range(0,nelx):
        xV[counter]=i*hx+1/2.*hx
        yV[counter]=j*hy+1/2.*hy
        counter += 1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        iconV[0, counter] = i + j * (nelx + 1)
        iconV[1, counter] = i + 1 + j * (nelx + 1)
        iconV[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        iconV[3, counter] = i + (j + 1) * (nelx + 1)
        iconV[4, counter] = (nelx+1)*(nely+1)+counter
        counter += 1

for iel in range (0,nel):
    print ("iel=",iel)
    print ("node 1",iconV[0,iel],"at pos.",xV[iconV[0,iel]], yV[iconV[0,iel]])
    print ("node 2",iconV[1,iel],"at pos.",xV[iconV[1,iel]], yV[iconV[1,iel]])
    print ("node 3",iconV[2,iel],"at pos.",xV[iconV[2,iel]], yV[iconV[2,iel]])
    print ("node 4",iconV[3,iel],"at pos.",xV[iconV[3,iel]], yV[iconV[3,iel]])
    print ("node 5",iconV[4,iel],"at pos.",xV[iconV[4,iel]], yV[iconV[4,iel]])


#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

xP[0:NP]=xV[0:NP]
yP[0:NP]=yV[0:NP]

iconP[0:mP,0:nel]=iconV[0:mP,0:nel]

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################

bc_fix = np.zeros(NfemV, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = timing.time()

G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity

for iel in range(0,nel):

    # set arrays to 0 every loop
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)
            #print(jcob)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]


            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]


            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        # end for jq
    # end for iq



    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            if bc_fix[m1]:
               G_el[ikk,:]=0

    if iel==0: print(G_el)

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            #for k2 in range(0,mV):
            #    for i2 in range(0,ndofV):
            #        jkk=ndofV*k2          +i2
            #        m2 =ndofV*iconV[k2,iel]+i2
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

######################################################################

for i in range(NfemV):
    for j in range(NfemP):
        if abs(G_mat[i,j])<1e-15:
           G_mat[i,j]=0

######################################################################

# my numbering of Vnodes     his
#
# 6    7    8              x---x---x
#   11   12                | 4   3
# 3    4    5              x   5
#   9    10                | 1   2
# 0    1    2              x---x---x

#print(G_mat)

#print('matrix G has size',np.shape(G_mat))

#for i in range(NV):
#    if not bc_fix[2*i]:
#       print(G_mat[2*i,:])
#    if not bc_fix[2*i+1]:
#       print(G_mat[2*i+1,:])

print('-----------------------')

if bubble==0:
   G_mat*=-9

if bubble==1:
   G_mat*=-9*3/4

if bubble==2:
   G_mat*=-18*3

#for i in range(0,NfemP):
    #print(G_mat[18,i],G_mat[19,i],G_mat[20,i],G_mat[21,i],\
    #      G_mat[24,i],G_mat[25,i],G_mat[22,i],G_mat[23,i],\
    #      G_mat[ 8,i],G_mat[ 9,i])
    #print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" \
    #     %(G_mat[18,i],G_mat[19,i],G_mat[20,i],G_mat[21,i],
    #       G_mat[24,i],G_mat[25,i],G_mat[22,i],G_mat[23,i],\
    #       G_mat[ 8,i],G_mat[ 9,i]))
    #print(G_mat[18:22,i],G_mat[24:26,i],G_mat[22:24],G_mat[8:10,i])

D = np.zeros((NfemP,10),dtype=np.float64) 

for i in range(0,NfemP):
    D[i,0]=G_mat[18,i]
    D[i,1]=G_mat[19,i]
    D[i,2]=G_mat[20,i]
    D[i,3]=G_mat[21,i]
    D[i,4]=G_mat[24,i]
    D[i,5]=G_mat[25,i]
    D[i,6]=G_mat[22,i]
    D[i,7]=G_mat[23,i]
    D[i,8]=G_mat[ 8,i]
    D[i,9]=G_mat[ 9,i]

print(D)
print('-----------------------')

ns = null_space(D)
opla=ns.shape
print('size of nullspace=',opla[1])

print('beta=',beta,'rank=',matrix_rank(D))

######################################################################
