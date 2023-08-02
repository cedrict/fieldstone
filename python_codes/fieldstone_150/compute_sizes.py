from scipy import sparse
from numpy import array

I = array([0,0,1,3,1,0,0])
J = array([0,2,1,3,1,0,0])
V = array([1,1,1,1,1,1,1])

B = sparse.coo_matrix((V,(I,J)),shape=(4,4)).tocsr()


nel=256**2
ndofV=2
mV=9
mP=4
size=nel*( (mV*ndofV)**2 + 2*(mV*ndofV*mP) )
print('array size=',size)
print('--> arrays I,J are int32')
print('    ',size*32,'bits')
print('    ',size*32/8,'bytes')
print('    ',size*32/8/1024,'kbytes')
print('    ',size*32/8/1024/1024,'Mbytes')
print('    ',size*32/8/1024/1024/1024,'Gbytes each')

print('--> array V is float64')
print('    ',size*64,'bits')
print('    ',size*64/8,'bytes')
print('    ',size*64/8/1024,'kbytes')
print('    ',size*64/8/1024/1024,'Mbytes')
print('    ',size*64/8/1024/1024/1024,'Gbytes')

