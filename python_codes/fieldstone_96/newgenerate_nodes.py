import numpy as np
from parameters import *

nnt2=2*(nnt-1)+1
nnr2=4*res+1

nnt3=2*(nnt-1)+1
nnr3=2*res+1

#------------------------------------------------------------------------------
filename='mypoints'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d %3d %3d\n" %(np_blob+nnr*nnt+ nnt2*nnr2 +nnt3*nnr3   ,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

#------------------------------------------------------------------------------
# background matrix of nodes
#------------------------------------------------------------------------------

grid1file=open('grid_background.ascii',"w")
for j in range(0,nnr):
    for i in range(0,nnt):
        rad=R_inner+(R_outer-R_inner)/(nnr-1)*j
        angle=np.pi/2-np.pi/(nnt-1)*i
        xi=rad*np.cos(angle)
        zi=rad*np.sin(angle)
        nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 1))
        grid1file.write("%f %f \n" %(xi,zi))
        counter+=1

#------------------------------------------------------------------------------
# higher resolution in top 200km
#------------------------------------------------------------------------------

grid2file=open('grid_top.ascii',"w")
for j in range(0,nnr2):
    for i in range(0,nnt2):
        rad=R_outer-200e3+(200e3)/(nnr2-1)*j
        angle=np.pi/2-np.pi/(nnt2-1)*i
        xi=max(0,rad*np.cos(angle))
        zi=rad*np.sin(angle)
        nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 1))
        grid2file.write("%f %f \n" %(xi,zi))
        counter+=1
        if j==nnr2-1 and i<nnt2-1:
           counter_segment+=1
           segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,5))

#------------------------------------------------------------------------------
# higher resolution in bottom 100km
#------------------------------------------------------------------------------

grid2file=open('grid_bottom.ascii',"w")
for j in range(0,nnr3):
    for i in range(0,nnt3):
        rad=R_inner+(100e3)/(nnr3-1)*j
        angle=np.pi/2-np.pi/(nnt3-1)*i
        xi=max(0,rad*np.cos(angle))
        zi=rad*np.sin(angle)
        nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 1))
        grid2file.write("%f %f \n" %(xi,zi))
        counter+=1
        if j==0 and i<nnt3-1:
           counter_segment+=1
           segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,5))

#------------------------------------------------------------------------------
# blob 
#------------------------------------------------------------------------------

for i in range (0,np_blob):
    angle=-np.pi/2+np.pi/(np_blob-1)*i
    xi=R_blob*np.cos(angle)
    zi=z_blob+R_blob*np.sin(angle)
    nodesfile.write("%5d %f %f %3d \n" %(counter+1,xi,zi, 5))
    counter+=1
    if i<np_blob-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,5))

#------------------------------------------------------------------------------

nodesfile.write("%5d %3d \n" %(counter_segment,1))

segmentfile.write("%5d \n" %(1))
segmentfile.write("%5d %e %e \n" %(1,100000,0))


