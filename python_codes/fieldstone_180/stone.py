import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

nnx=5                         # number of nodes
x=np.array([0,0.3,0.4,0.9,1]) # coordinates of nodes
hcond=np.ones(nnx)                # nodal heat conductivity

###############################################################################
# compute spacing between nodes 

h=np.zeros(nnx-1,dtype=np.float64)
for i in range(0,nnx-1):
    h[i]=x[i+1]-x[i]

print('x=',x)
print('h=',h)

###############################################################################
# build linear system

b=np.zeros(nnx,dtype=np.float64)
A=lil_matrix((nnx,nnx),dtype=np.float64)

A[0,0]=1         ; b[0]=0     # left boundary condition
A[nnx-1,nnx-1]=1 ; b[nnx-1]=1 # right boundary condition

for i in range(1,nnx-1):
    A[i,i-1]=-(hcond[i-1]+hcond[i])/h[i-1]/(h[i-1]+h[i])
    A[i,i]  = (hcond[i+1]+hcond[i])/h[i]  /(h[i-1]+h[i]) \
            + (hcond[i-1]+hcond[i])/h[i-1]/(h[i-1]+h[i]) 
    A[i,i+1]=-(hcond[i+1]+hcond[i])/h[i]  /(h[i-1]+h[i]) 

#print(A)

###############################################################################
# solve linear system A.x=b

T=sps.linalg.spsolve(sps.csr_matrix(A),b)

print('T=',T)

np.savetxt('T.ascii',np.array([x,T]).T,header='# x,T')

###############################################################################
