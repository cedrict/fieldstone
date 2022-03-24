import numpy as np
import FEbasis2D as FE
import FEquadrature as Q

for elt in ['Q0','Q1','Q1+','Q2','Q3','Q2s','P0','P1','P1+','P1NC','P2','P3','DSSY1','DSSY2','RT1','RT2','Han']:
    print('=========================================')
    print(' '+elt)
    print('=========================================')
    m=FE.NNN_m(elt)
    rnodes=FE.NNN_r(elt)
    snodes=FE.NNN_s(elt)
    print('m=',m)
    for i in range(0,m):
        print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))

exit()























































exit()


















jcb=np.zeros((2,2),dtype=np.float64)

#------------------------------------------------------------------------------
# testing Q0
#------------------------------------------------------------------------------

elt='Q0'

m=FE.NNN_m(elt)

rnodes=FE.NNN_r(elt)
snodes=FE.NNN_s(elt)

print('Q0')
print('m=',m)

#for i in range(0,m):
#   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))

nqperdim=FE.NNN_nqperdim(elt)
print('nqperdim=',nqperdim)

#------------------------------------------------------------------------------
# testing Q1
#------------------------------------------------------------------------------

elt='Q1'

m=FE.NNN_m(elt)

rnodes=FE.NNN_r(elt)
snodes=FE.NNN_s(elt)

print('Q1')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))

nqperdim=FE.NNN_nqperdim(elt)
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
        NNN=FE.NNN(rq,sq,elt)
        dNNNdr=FE.dNNNdr(rq,sq,elt)
        dNNNds=FE.dNNNds(rq,sq,elt)


#------------------------------------------------------------------------------
# testing Q2
#------------------------------------------------------------------------------

elt='Q2'

m=FE.NNN_m(elt)

rnodes=FE.NNN_r(elt)
snodes=FE.NNN_s(elt)

print('Q2')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))

#------------------------------------------------------------------------------
# testing Q3
#------------------------------------------------------------------------------

elt='Q3'

m=FE.NNN_m(elt)

rnodes=FE.NNN_r(elt)
snodes=FE.NNN_s(elt)

print('Q3')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))

#------------------------------------------------------------------------------
# testing Q3
#------------------------------------------------------------------------------

elt='Q3'

m=FE.NNN_m(elt)

rnodes=FE.NNN_r(elt)
snodes=FE.NNN_s(elt)

print('Q3')
print('m=',m)

for i in range(0,m):
   print ('node',i,':',FE.NNN(rnodes[i],snodes[i],elt))



