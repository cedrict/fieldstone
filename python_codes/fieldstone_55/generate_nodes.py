
import numpy as np
from parameters import *

#--------------------------------------------

filename='subd.node'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d \n" %(3*np_plate+3*np_slab+4*np_box+3*np_plate2+2*np_circle,2))

counter=0

#--------------------------------1
for i in range (0,np_plate):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xK+i*L/(np_plate-1),yL))
    counter+=1

#--------------------------------2
for i in range (0,np_slab):
    t=i*theta/(np_slab-1)
    nodesfile.write("%5d %10e %10e \n"  %(counter+1,xM+rad*np.cos(np.pi/2-t),yM+rad*np.sin(np.pi/2-t)) )
    counter+=1

#--------------------------------3
for i in range (0,np_plate):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xK+i*L/(np_plate-1),yL-h/2))
    counter+=1

#--------------------------------4
for i in range (0,np_slab):
    t=i*theta/(np_slab-1)
    nodesfile.write("%5d %10e %10e \n"  %(counter+1,xM+(rad-h/2)*np.cos(np.pi/2-t),yM+(rad-h/2)*np.sin(np.pi/2-t)) )
    counter+=1

#--------------------------------5
for i in range (0,np_plate):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xK+i*L/(np_plate-1),yL+h/2))
    counter+=1

#--------------------------------6
for i in range (0,np_slab):
    t=i*theta/(np_slab-1)
    nodesfile.write("%5d %10e %10e \n"  %(counter+1,xM+(rad+h/2)*np.cos(np.pi/2-t),yM+(rad+h/2)*np.sin(np.pi/2-t)) )
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

#--------------------------------11
for i in range (0,np_plate2):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xK,yK-h/2+i*h/(np_plate2-1)))
    counter+=1

#--------------------------------12
for i in range (0,np_plate2):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xL,yL-h/2+i*h/(np_plate2-1)))
    counter+=1

#--------------------------------13
for i in range (0,np_plate2):
    nodesfile.write("%5d %10e %10e \n" %(counter+1,xN+i*(xP-xN)/(np_plate2-1),yN+i*(yP-yN)/(np_plate2-1)))
    counter+=1

#--------------------------------14
for i in range (0,np_circle):
    t=i*np.pi/(np_circle-1)+np.pi/2
    nodesfile.write("%5d %10e %10e \n"  %(counter+1,xK+h/2*np.cos(t),yK+h/2*np.sin(t)) )
    counter+=1

#--------------------------------15
for i in range (0,np_circle):
    t=i*np.pi/(np_circle-1)+np.pi/2-theta+np.pi
    nodesfile.write("%5d %10e %10e \n"  %(counter+1,xQ+h/2*np.cos(t),yQ+h/2*np.sin(t)) )
    counter+=1



#nodesfile.write("%5d %10e %10e \n" %(counter+1,xM,yM))
#nodesfile.write("%5d %10e %10e \n" %(counter+1,xN,yN))
#nodesfile.write("%5d %10e %10e \n" %(counter+1,xP,yP))


