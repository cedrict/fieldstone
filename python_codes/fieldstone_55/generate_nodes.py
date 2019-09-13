
import numpy as np
from parameters import *

#--------------------------------------------

filename='subd.node'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d \n" %(640+281+4*np_box,2))

counter=0

#--------------------------------1
Nshape=640

x=np.empty(Nshape,dtype=np.float64)     # x coordinates
y=np.empty(Nshape,dtype=np.float64)     # y coordinates
x[0:Nshape],y[0:Nshape]=np.loadtxt('cedric1_shape.dat',unpack=True,usecols=[0,1])

for i in range (0,Nshape):
    x[i]=x[i]*1e5+xL-L
    y[i]=y[i]*1e5+Ly
    nodesfile.write("%5d %10e %10e \n" %(counter+1,x[i],y[i]))
    counter+=1

#--------------------------------2
Nshape=281

x=np.empty(Nshape,dtype=np.float64)     # x coordinates
y=np.empty(Nshape,dtype=np.float64)     # y coordinates
x[0:Nshape],y[0:Nshape]=np.loadtxt('cedric1_xmid.dat',unpack=True,usecols=[0,1])

for i in range (0,Nshape):
    x[i]=x[i]*1e5+xL-L
    y[i]=y[i]*1e5+Ly
    nodesfile.write("%5d %10e %10e \n" %(counter+1,x[i],y[i]))
    counter+=1

#--------------------------------7
for i in range (0,np_box):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,i*Lx/(np_box),0.))
    counter+=1

#--------------------------------8
for i in range (1,np_box+1):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,i*Lx/(np_box),Ly))
    counter+=1

#--------------------------------9
for i in range (1,np_box+1):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,0.,i*Ly/(np_box)))
    counter+=1

#--------------------------------10
for i in range (0,np_box):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,Lx,i*Ly/(np_box)))
    counter+=1

