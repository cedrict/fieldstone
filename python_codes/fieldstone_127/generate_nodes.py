import numpy as np
from parameters import *

#------------------------------------------------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d %3d %3d\n" %(np_left+np_right+np_top+np_bottom+np_object+1,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

#------------------------------------------------------------------------------
# left: 
#------------------------------------------------------------------------------
for i in range (1,np_left+1):
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0.,i*Ly/(np_left),0))
    counter+=1
    if i<np_left-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#------------------------------------------------------------------------------
# bottom
#------------------------------------------------------------------------------
for i in range (0,np_bottom):
    nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,i*Lx/(np_bottom),0., 0))
    counter+=1
    if i<np_bottom-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#------------------------------------------------------------------------------
# top
#------------------------------------------------------------------------------
for i in range (1,np_top+1):
    nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,i*Lx/(np_top),Ly, 0))
    counter+=1
    if i<np_top-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#------------------------------------------------------------------------------
# right
#------------------------------------------------------------------------------
for i in range (0,np_right):
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,Lx,i*Ly/(np_right),0))
    counter+=1
    if i<np_right-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#------------------------------------------------------------------------------
# object: 
#------------------------------------------------------------------------------

for i in range (0,np_object):
       angle=2*np.pi/(np_object-1)*i
       x=xobject+rad*np.cos(angle)
       y=yobject+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<np_object-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


#middle of object
nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xobject,yobject, 0))
counter+=1

nodesfile.write("%5d %3d \n" %(counter_segment,1))

segmentfile.write("%5d \n" %(0))
segmentfile.write("%5d %e %e \n" %(1,0.1,0.1))
