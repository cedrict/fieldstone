import numpy as np
import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 

Lx=3
Ly=2

nelx=3
nely=2

nqpts=7

print('=========================================')
print(' tester 7: 0th order consistency')
print('=========================================')

for Vspace in ['Q1','Q1+','Q2','Q3','Q2s','DSSY1','DSSY2','RT1','RT2','Han','Chen',\
               'P1','P1+','P1NC','P2','P2+','P3','P4']:

    print('=========================================')
    print(' '+Vspace)
    print('=========================================')
    mV=FE.NNN_m(Vspace)
    rnodes=FE.NNN_r(Vspace)
    snodes=FE.NNN_s(Vspace)

    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

    print('nqel=',nqel)

    NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace)

    for iel in range(0,nel):
        for iq in range(0,nqel):
            rq=qcoords_r[iq]
            sq=qcoords_s[iq]
            weightq=qweights[iq]
            NNNV=FE.NNN(rq,sq,Vspace)
            print(np.sum(NNNV))
        #end for
    #end for
#end for
