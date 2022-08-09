import time as time
import numpy as np
import random

dmin=0.
dmax=2
nbin=100
delta=(dmax-dmin)/nbin

###########################################################

nvo=10000

xvo = np.empty(nvo,dtype=np.float64)  
yvo = np.empty(nvo,dtype=np.float64)
zvo = np.empty(nvo,dtype=np.float64)
for i in range(0,nvo):
    xvo[i]=random.uniform(0.001,0.999)
    yvo[i]=random.uniform(0.001,0.999)
    zvo[i]=random.uniform(0.001,0.999)

###########################################################
Lx=1
Ly=1
Lz=1
rho=nvo/Lx/Ly/Lz # average density

rdf=np.zeros((2,nbin),dtype=np.float64)
norm=np.zeros(nbin,dtype=np.float64)

for i in range(nbin):
    rdf[0,i]=(i+0.5)*delta
    norm[i]=rho*4*np.pi*rdf[0,i]**2*delta

for i in range(0,nvo):
    for j in range(0,i):
        xij=xvo[i]-xvo[j]
        yij=yvo[i]-yvo[j]
        zij=zvo[i]-zvo[j]
        dij=np.sqrt(xij**2+yij**2+zij**2)
        index=int(dij/delta)
        rdf[1,index]+=1
    #end for
#end for

np.savetxt('rdf.ascii',rdf.T)

for i in range(nbin):
    rdf[1,i]/=norm[i]

np.savetxt('rdf_normalised.ascii',rdf.T)
