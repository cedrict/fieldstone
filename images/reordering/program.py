import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

n=8

# declare matrix
A_sparse = lil_matrix((n,n),dtype=np.float64)

# fill matrix
A_sparse[0,0]=1
A_sparse[0,4]=2

A_sparse[1,1]=3
A_sparse[1,2]=4
A_sparse[1,5]=5
A_sparse[1,7]=6

A_sparse[2,1]=7
A_sparse[2,2]=8
A_sparse[2,4]=9

A_sparse[3,3]=10
A_sparse[3,6]=11

A_sparse[4,0]=12
A_sparse[4,2]=13
A_sparse[4,4]=14

A_sparse[5,1]=15
A_sparse[5,5]=16
A_sparse[5,7]=17

A_sparse[6,3]=18
A_sparse[6,6]=19

A_sparse[7,1]=20
A_sparse[7,5]=21
A_sparse[7,7]=22

#convert to CSR format
sparse_matrix=A_sparse.tocsr()

#--------------------------------------------------------------
#create rhs

rhs=np.empty(n,dtype=np.float64)
for i in range(0,n):
    rhs[i]=i

print('rhs',rhs)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print(sol)

#--------------------------------------------------------------
# export matrix structure to png image

plt.spy(A_sparse, markersize=10)
plt.savefig('matrix_bef.png', bbox_inches='tight')
plt.clf()

#--------------------------------------------------------------
# found how to use the obtained permutation online 

perm = reverse_cuthill_mckee(sparse_matrix,symmetric_mode=True)

print ('perm=',perm)

#sparse_matrix=A_sparse[np.ix_(perm,perm)]
sparse_matrix=sparse_matrix[np.ix_(perm,perm)]

#--------------------------------------------------------------
# export matrix structure to png image

plt.spy(sparse_matrix, markersize=10)
plt.savefig('matrix_aft.png', bbox_inches='tight')
plt.clf()


#-----------------------------------
# reorder rhs 

rhs=rhs[np.ix_(perm)]

print('reordered rhs',rhs)

#-----------------------------------
# solve reordered system

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print(sol)

#-----------------------------------

perm_inv=np.empty(n,dtype=np.int32)
for i in range(0,n):
    perm_inv[perm[i]]=i

print(perm_inv)

sol=sol[np.ix_(perm_inv)]
print(sol)




