import numpy as np
from parameters import *

#------------------------------------------------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d %3d %3d\n" %(np_inner+np_outer+2*np_vert+np_blob+np_moho+np_trans,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

#------------------------------------------------------------------------------
# inner boundary counterclockwise
#------------------------------------------------------------------------------

for i in range (0,np_inner):
    angle=-np.pi/2+np.pi/np_inner*i
    xi=R_inner*np.cos(angle)
    zi=R_inner*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 1))
    counter+=1
    if i<np_inner-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,1))

#------------------------------------------------------------------------------
# top vertical wall 
#------------------------------------------------------------------------------

for i in range (0,np_vert):
    xi=0
    zi=R_inner+(R_outer-R_inner)/np_vert*i
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 2))
    counter+=1
    if i<np_vert-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,2))

#------------------------------------------------------------------------------
# outer boundary clockwise
#------------------------------------------------------------------------------

for i in range (0,np_outer):
    angle=np.pi/2-np.pi/np_outer*i
    xi=R_outer*np.cos(angle)
    zi=R_outer*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 3))
    counter+=1
    if i<np_outer-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,3))

#------------------------------------------------------------------------------
# bottom vertical wall
#------------------------------------------------------------------------------

for i in range (0,np_vert):
    xi=0
    zi=-R_outer+(R_outer-R_inner)/np_vert*i
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 4))
    counter+=1
    if i<np_vert-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,4))

#------------------------------------------------------------------------------
# blob 
#------------------------------------------------------------------------------

for i in range (0,np_blob):

    angle=-np.pi/2+np.pi/np_blob*i
    xi=R_blob*np.cos(angle)
    zi=z_blob+R_blob*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 5))
    counter+=1
    if i<np_blob-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,5))

#------------------------------------------------------------------------------
# moho 
#------------------------------------------------------------------------------

for i in range (0,np_moho):
    angle=np.pi/2-np.pi/np_moho*i
    xi=R_moho*np.cos(angle)
    zi=R_moho*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 3))
    counter+=1
    if i<np_moho-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,3))

#------------------------------------------------------------------------------
# transition in the mantle 
#------------------------------------------------------------------------------

for i in range (0,np_trans):
    angle=np.pi/2-np.pi/np_trans*i
    xi=R_trans*np.cos(angle)
    zi=R_trans*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 3))
    counter+=1
    if i<np_trans-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,3))



#------------------------------------------------------------------------------

nodesfile.write("%5d %3d \n" %(counter_segment,1))

segmentfile.write("%5d \n" %(1))
segmentfile.write("%5d %e %e \n" %(1,0.1,0.1))

