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
for nqpts in 1,3,4,6,7,12,13,16:
    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature('P1',nqpts)
    print('nqpts=',nqpts,'| sum(weights) 2D=',np.sum(qweights))


print('**********************************************************')

for space in 'Q1','Q1+','Q2','Q2s','Q3','Q4','P1','P2','P1+','P2+','P3','DSSY1','DSSY2','RT1','RT2':
    print('*****-> '+space)
    FE.visualise_nodes(space)

    for nqpts in 2,3,4,5,6,7,8,9,10:
        nq,rq,sq,wq=Q.quadrature(space,nqpts)
        Q.visualise_quadrature_points(space,nqpts)


    FE.visualise_basis_functions(space)


