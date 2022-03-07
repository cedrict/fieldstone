import numpy as np
import FEquadrature as Q
import FEbasis2D as FE

print('**********************************************************')

for nqperdim in 1,2,3,4,5,6,7,8,9,10:
    weights=Q.qweights_1D(nqperdim)
    print('nqperdim=',nqperdim,'| sum(weights)=',np.sum(weights))

print('**********************************************************')

for space in 'Q1','Q1+','Q2','Q3','Q4','P1','P2','P1+','P2+','P3':
    print('*****-> '+space)
    FE.visualise_nodes(space)

    #for nqpts in 2,3,4,5,6,7,8,9,10:
    #    nq,rq,sq,wq=Q.quadrature(space,nqpts)
    #    Q.visualise_quadrature_points(space,nqpts)


    FE.visualise_basis_functions(space)


