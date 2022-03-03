import numpy as np
import FEquadrature as Q
import FEbasis2D as FE

print('**********************************************************')

for nqperdim in 1,2,3,4,5,6:
    weights=Q.qweights_1D(nqperdim)
    print('nqperdim=',nqperdim,'| sum(weights)=',np.sum(weights))

print('**********************************************************')

for space in 'Q1','Q2','Q3','Q4','P1','P2','P1+','P2+':
    print('*****-> '+space)
    FE.visualise_nodes(space)
    print('     -> generated '+space+'nodes.pdf')

    for nqpts in 2,3,4:
        nq,rq,sq,wq=Q.quadrature(space,nqpts)
        Q.visualise_quadrature_points(space,nqpts)




