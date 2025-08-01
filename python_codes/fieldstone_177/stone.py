import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix


Lx=1
Ly=1
Lz=1

nelx=8
nely=9
nelz=10

nel=nelx*nely*nelz

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
he=max(hx,hy,hz)

m=8

nnx=nelx+1
nny=nely+1
nnz=nelz+1

NT=nnx*nny*nnz
Nfem=nnx*nny*nnz

hcond=1

dt=1e-3

###############################################################################
# local coordinates of elemental nodes
###############################################################################

rnodes=np.array([-1, 1, 1,-1,-1, 1, 1 ,-1],np.float64)
snodes=np.array([-1,-1, 1, 1,-1,-1, 1 , 1],np.float64)
tnodes=np.array([-1,-1,-1,-1, 1, 1, 1 , 1],np.float64)

###############################################################################
# setup quadrature points and weights
###############################################################################

a=1/np.sqrt(3)
quadrature_points = [(-a,-a,-a ,1),
                     ( a,-a,-a ,1) , 
                     ( a, a,-a ,1) ,
                     (-a, a,-a ,1) ,
                     (-a,-a, a ,1) , 
                     ( a,-a, a ,1) , 
                     ( a, a, a ,1) ,
                     (-a, a, a ,1) ]

###############################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('Nfem=',Nfem)
print("-----------------------------")

###############################################################################
# grid point setup
###############################################################################
start = time.time()

x = np.zeros(NT,dtype=np.float64)
y = np.zeros(NT,dtype=np.float64)
z = np.zeros(NT,dtype=np.float64)

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)
            counter += 1
        #end for
    #end for
#end for

print("mesh setup: %.3f s" % (time.time() - start))

###############################################################################
# build connectivity array (python is row major)
###############################################################################
start = time.time()

icon=np.zeros((nel,m),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[counter,0]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[counter,1]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[counter,2]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[counter,3]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[counter,4]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[counter,5]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[counter,6]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[counter,7]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (time.time() - start))

###############################################################################
# prescribe velocity on mesh
###############################################################################
start = time.time()

u = np.zeros(NT,dtype=np.float64)
v = np.zeros(NT,dtype=np.float64)
w = np.zeros(NT,dtype=np.float64)

######################################################################
# define boundary conditions temperature
######################################################################
start = timing.time()

bc_fix=np.zeros(Nfem,dtype=bool) 
bc_val=np.zeros(Nfem,dtype=np.float64)

for i in range(0,NT):
    if z[i]<eps:
       bc_fix[i] = True ; bc_val[i] = Temperature1
    if z[i]/Lz>1-eps:
       bc_fix[i] = True ; bc_val[i] = Temperature2
# end for

print("boundary conditions T: %.3f s" % (timing.time() - start))

###############################################################################
# build matrix
###############################################################################
start = time.time()

Amat=lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs=np.zeros(Nfem,dtype=np.float64)
Me=np.zeros((m,m),dtype=np.float64)
Kde=np.zeros((m,m),dtype=np.float64)
Kae=np.zeros((m,m),dtype=np.float64)

for e,nodes in enumerate(icon):
    xe,ye,ze=x[nodes],y[nodes],z[nodes]
    ue,ve,we=u[nodes],v[nodes],w[nodes]

    Me[:,:]=0
    Kae[:,:]=0
    Kde[:,:]=0

    for rq,sq,tq,weightq in quadrature_points:

        N=0.125*(1+rnodes*rq)*(1+snodes*sq)*(1+tnodes*tq)

        dNdr=0.125*rnodes*(1+snodes*sq)*(1+tnodes*tq)
        dNds=0.125*snodes*(1+rnodes*rq)*(1+tnodes*tq)
        dNdt=0.125*tnodes*(1+rnodes*rq)*(1+snodes*sq)

        invJ=np.diag([2/hx,2/hy,2/hz])
        jcob=hx*hy*hz/8

        B=(invJ@np.vstack((dNdr,dNds,dNdt))).T  # (8x3) shape

        velq=np.dot(N,np.vstack((ue,ve,we)).T) # (3,) shape
        #print(np.shape(velq))
      
        advN=B@velq # (8,) shape

        Me+=np.outer(N,N)*jcob*weightq
        Kae+=np.outer(N,advN)*jcob*weightq
        Kde+=B@B.T*hcond*jcob*weightq

    #end for quad points

    Ae=Me+(Kae+Kde)*dt
    be=0

    #impose boundary conditions


    #assemble
    Amat[np.ix_(nodes,nodes)]+=Ae

#end for elements

print("build matrix: %.3f s" % (time.time() - start))

