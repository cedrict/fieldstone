import numpy as np

A = np.array([[-2, -4,  2],
              [-2,  1,  2],
              [4,   2,  5]])
print (A)
print ('-------------------------------------------------------')

eigenvalues, eigenvectors = np.linalg.eig(A) # must use this line of code exactly 

print(f'eigenvalues of matrix A are: \n {eigenvalues}')
print ('-------------------------------------------------------')
print(f'eigenvectors of matrix A are:\n {eigenvectors}')

# sort eigenvalues from smallest to largest

print(np.argsort(eigenvalues),eigenvalues,np.flip(np.argsort(eigenvalues)))
idx=np.flip(np.argsort(eigenvalues))

