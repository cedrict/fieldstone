import numpy as np
from parameters import *

#------------------------------------------------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")

nodesfile.write("%5d %5d %3d %3d\n" %(np_left+np_right+np_top+np_bottom+np_object+np_surf+1,2,0,1))

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

if experiment==1 or experiment==2:

   for i in range (0,np_object):
       angle=2*np.pi/(np_object-1)*i
       x=xobject+rad*np.cos(angle)
       y=yobject+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<np_object-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

if experiment==3:

   for i in range (0,n_p):
       x=xobject-size/2+i*size/n_p
       y=yobject-size/2
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,n_p):
       x=xobject+size/2
       y=yobject-size/2+i*size/n_p
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (n_p,0,-1):
       x=xobject-size/2+i*size/n_p
       y=yobject+size/2
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (n_p,0,-1):
       x=xobject-size/2
       y=yobject-size/2+i*size/n_p
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

if experiment==5:

   for i in range (0,2*n_p):
       x=xA+i*(xB-xA)/(2*n_p)
       y=yA-delta
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<2*n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,n_p):
       angle=np.pi/2-np.pi/2*i/n_p
       rad=0.123-delta
       x=xD+rad*np.cos(angle)
       y=yD+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (0,n_p):
       angle=-np.pi+np.pi*i/n_p
       rad=delta
       x=xC+rad*np.cos(angle)
       y=yC+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (0,2*n_p):
       angle=np.pi/2*i/(2*n_p)
       rad=0.123+delta
       x=xD+rad*np.cos(angle)
       y=yD+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<2*n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (0,2*n_p):
       x=xB-i*(xB-xA)/(2*n_p)
       y=yB+delta
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<2*n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,n_p):
       angle=np.pi/2+np.pi*i/n_p
       rad=delta
       x=xA+rad*np.cos(angle)
       y=yA+rad*np.sin(angle)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


if experiment==6:

   xA=1000e3
   yA=650e3
   xB=xA
   yB=450e3
   xC=1100e3
   yC=yB
   xD=xC
   yD=550e3
   xE=Lx
   yE=yD

   for i in range (0,n_p):
       x=xA
       y=yA+i*(yB-yA)/(n_p)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<2*n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,n_p//2):
       x=xB+i*(xC-xB)/(n_p/2)
       y=yB
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p/2-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,n_p//2):
       x=xC
       y=yC+i*(yD-yC)/(n_p/2)
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<n_p/2-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       
   for i in range (0,3*n_p):
       x=xD+i*(xE-xD)/(3*n_p-1)
       y=yD
       nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
       counter+=1
       if i<3*n_p-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))
       





#middle of object
nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,xobject,yobject, 0))
counter+=1

#------------------------------------------------------------------------------
# surface
#------------------------------------------------------------------------------

if experiment==1:
   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,i*0.37/n_p,0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,0.38+(i-1)*0.24/(n_p-1),0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,0.63+(i-1)*0.36/(n_p-1),0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


if experiment==2:
   for i in range (1,np_surf+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,i*Lx/(np_surf+1),0.75, 0))
       counter+=1
       if i<np_surf-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


if experiment==3:
   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,i*0.43/n_p,0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,0.44+(i-1)*0.12/(n_p-1),0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

   for i in range (1,n_p+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,0.5,0.57+(i-1)*0.43/n_p,0))
       counter+=1
       if i<n_p:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

if experiment==4:
   for i in range (0,np_surf):
       x=i*Lx/(np_surf-1)
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,x,0.2+0.02*np.cos(np.pi*x/Lx), 0))
       counter+=1
       if i<np_surf-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


if experiment==5:
   for i in range (1,np_surf+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,i*Lx/(np_surf+1),0.75, 0))
       counter+=1
       if i<np_surf-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

if experiment==6:
   for i in range (1,np_surf+1):
       nodesfile.write("%5d %10e %10e %3d \n" %(counter+1,(i-1)*Lx/(np_surf-1),650e3, 0))
       counter+=1
       if i<np_surf-1:
          counter_segment+=1
          segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


   nodesfile.write("%5d %3d \n" %(counter_segment,1))

   segmentfile.write("%5d \n" %(0))
   segmentfile.write("%5d %e %e \n" %(1,0.1,0.1))



