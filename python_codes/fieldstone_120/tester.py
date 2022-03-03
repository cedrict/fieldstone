import FEbasis2D as FE
import FEquadrature as Q

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







