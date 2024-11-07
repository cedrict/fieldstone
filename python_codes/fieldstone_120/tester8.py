import numpy as np
import FEquadrature as Q
import FEbasis2D as FE

print('**********************************************************')
print('QUADRILATERALS')
print('**********************************************************')
for nqpts in 1,2,3,4,5,6,7,8,9,10:
    weights=Q.qweights_1D(nqpts)
    print('nqpts=',nqpts,'| sum(weights) 1D=',np.sum(weights))
    
    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature('Q1',nqpts)
    print('nqpts=',nqpts,'| sum(weights) 2D=',np.sum(qweights))

print('**********************************************************')
print('TRIANGLES')
print('**********************************************************')
for nqpts in 1,3,6,7,12,13,16:
    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature('P1',nqpts)
    print('nqpts=',nqpts,'| sum(weights) 2D=',np.sum(qweights))

print('**********************************************************')


