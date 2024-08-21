import numpy as np
import FEbasis2D as FE
import FEquadrature as Q
import FEtools as Tools 

Lx=3
Ly=2

nelx=3
nely=2

nqpts=3

eps=1e-12
    
print('=========================================')
print(' tester 3: linear fields')
print('=========================================')


for Vspace in ['Q1','Q1+','Q2','Q3','Q2s','DSSY1','DSSY2','RT1','RT2','Han','Chen',\
               'P1','P1+','P1NC','P2','P2+','P3','P4']:

    mV=FE.NNN_m(Vspace)
    rnodes=FE.NNN_r(Vspace)
    snodes=FE.NNN_s(Vspace)

    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

    NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace,mtype)

    u=xV
    v=yV

    dNNNVdx= np.zeros(mV,dtype=np.float64)
    dNNNVdy= np.zeros(mV,dtype=np.float64)
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
            dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]+jcbi[0,1]*dNNNVds[:]
            dNNNVdy[:]=jcbi[1,0]*dNNNVdr[:]+jcbi[1,1]*dNNNVds[:]
            exxq=dNNNVdx.dot(u[iconV[:,iel]]) 
            eyyq=dNNNVdy.dot(v[iconV[:,iel]]) 
            
            if abs(exxq-1)>eps or abs(eyyq-1)>eps:
               exit('pb')
    #end for
    print(Vspace+' passed')

print('=========================================')
