import numpy as np
from parameters import *

#------------------------------------------------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d %3d %3d\n" %(np_left+np_right+np_top+np_bottom+np_sphere,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

#------------------------------------------------------------------------------
# left: # making sure two points are exavtly where the 
# circle intersects the left boundary
#------------------------------------------------------------------------------
for i in range (1,np_left+1):

    if (i-1)*Ly/(np_left) < Ly/2-rad and (i+1)*Ly/(np_left) > Ly/2-rad:
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0.,Ly/2-rad,0))
    elif (i-1)*Ly/(np_left) < Ly/2+rad and (i+1)*Ly/(np_left) > Ly/2+rad:
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,0.,Ly/2+rad,0))
    else:
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
xmin=0
xmax=Lx
ddx=(xmax-xmin)/(np_top-1)
for i in range (0,np_top):
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,xmin+i*ddx,Ly,0))
    counter+=1

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
# sphere: making sure circle points do not end up on left boundary 
#------------------------------------------------------------------------------
for i in range (0,np_sphere):
    angle=-np.pi/2*0.995+np.pi*0.995/(np_sphere-1)*i
    x=rad*np.cos(angle)
    y=Ly/2+rad*np.sin(angle)
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    counter+=1
    if i<np_sphere-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


