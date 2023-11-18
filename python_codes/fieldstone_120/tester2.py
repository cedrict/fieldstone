import numpy as np
import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 

Lx=3
Ly=2

nelx=3
nely=2

nqpts=3

print('=========================================')
print(' tester 2: areas')
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

    area=np.zeros(nel,dtype=np.float64) 
    for iel in range(0,nel):
        for iq in range(0,nqel):
            rq=qcoords_r[iq]
            sq=qcoords_s[iq]
            weightq=qweights[iq]
            NNNV=FE.NNN(rq,sq,Vspace)
            xq=NNNV.dot(xV[iconV[:,iel]]) 
            yq=NNNV.dot(yV[iconV[:,iel]]) 
            dNNNVdr=FE.dNNNdr(rq,sq,Vspace)
            dNNNVds=FE.dNNNds(rq,sq,Vspace)
            jcob,jcbi=Tools.J(mV,dNNNVdr,dNNNVds,xV[iconV[0:mV,iel]],yV[iconV[0:mV,iel]])
            print('iel= %d | iq= %d | jcob= %.2f %.2f %.2f %.2f %.2f' %(iel,iq,jcbi[0,0],jcbi[0,1],jcbi[1,0],jcbi[1,1],jcob))
            area[iel]+=jcob*weightq
        #end for
    #end for
    print('areas=',area)

print('=========================================')
