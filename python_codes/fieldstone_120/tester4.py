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
print(' tester 4: quadratic fields')
print('=========================================')

for Vspace in ['Q1','Q1+','Q2','Q3','Q2s','DSSY1','DSSY2','RT1','RT2','Han','Chen',\
               'P1','P1+','P1NC','P2','P2+','P3','P4']:

    pb=False

    mV=FE.NNN_m(Vspace)
    rnodes=FE.NNN_r(Vspace)
    snodes=FE.NNN_s(Vspace)

    nqel,qcoords_r,qcoords_s,qweights=Q.quadrature(Vspace,nqpts)

    NV,nel,xV,yV,iconV=Tools.cartesian_mesh(Lx,Ly,nelx,nely,Vspace,mtype)

    u=xV**2/2
    v=yV**2/2

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
            
            if abs(exxq-xq)>eps or abs(eyyq-yq)>eps:
               #print(exxq,eyyq)
               pb=True
               #exit('pb')
    #end for
    if not pb: print(Vspace+' passed')

exit()

print('=========================================')






















































exit()


















jcb=np.zeros((2,2),dtype=np.float64)

#------------------------------------------------------------------------------
# testing Q0
#------------------------------------------------------------------------------

Vspace='Q0'

m=FE.NNN_m(Vspace)

rnodes=FE.NNN_r(Vspace)
snodes=FE.NNN_s(Vspace)

print('Q0')
print('m=',m)

#for i in range(0,m):
#   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],Vspace))

nqperdim=FE.NNN_nqperdim(Vspace)
print('nqperdim=',nqperdim)

#------------------------------------------------------------------------------
# testing Q1
#------------------------------------------------------------------------------

Vspace='Q1'

m=FE.NNN_m(Vspace)

rnodes=FE.NNN_r(Vspace)
snodes=FE.NNN_s(Vspace)

print('Q1')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],Vspace))

nqperdim=FE.NNN_nqperdim(Vspace)
print('nqperdim=',nqperdim)

qcoords=Q.qcoords(nqperdim)
qweights=Q.qweights(nqperdim)

print(qcoords)
print(qweights)

for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        rq=qcoords[iq]
        sq=qcoords[jq]
        weightq=qweights[iq]*qweights[jq]
        NNN=FE.NNN(rq,sq,Vspace)
        dNNNdr=FE.dNNNdr(rq,sq,Vspace)
        dNNNds=FE.dNNNds(rq,sq,Vspace)


#------------------------------------------------------------------------------
# testing Q2
#------------------------------------------------------------------------------

Vspace='Q2'

m=FE.NNN_m(Vspace)

rnodes=FE.NNN_r(Vspace)
snodes=FE.NNN_s(Vspace)

print('Q2')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],Vspace))

#------------------------------------------------------------------------------
# testing Q3
#------------------------------------------------------------------------------

Vspace='Q3'

m=FE.NNN_m(Vspace)

rnodes=FE.NNN_r(Vspace)
snodes=FE.NNN_s(Vspace)

print('Q3')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],Vspace))

#------------------------------------------------------------------------------
# testing Q3
#------------------------------------------------------------------------------

Vspace='Q3'

m=FE.NNN_m(Vspace)

rnodes=FE.NNN_r(Vspace)
snodes=FE.NNN_s(Vspace)

print('Q3')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],Vspace))



