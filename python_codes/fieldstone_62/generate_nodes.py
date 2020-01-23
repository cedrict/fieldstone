from points import *

#---------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")
nodesfile.write("%5d %5d %3d %3d\n" %(2351,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

#print ("#AC#############")
npts=165
dx=(xA-xC)/(npts-1)
dy=0.5e3
for i in range(0,npts):
    counter+=1
    x=xC+i*dx
    y=yC+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#DB#############")
npts=165
dx=(xB-xD)/(npts-1)
dy=0.5e3
for i in range(0,npts):
    counter+=1
    x=xD+i*dx
    y=yD+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#56#############")
npts=165
dx=(x5-x6)/(npts-1)
dy=(y5-y6)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x6+i*dx
    y=y6+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#56#############")
npts=165
dx=(x7-x8)/(npts-1)
dy=(y7-y8)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x8+i*dx
    y=y8+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))




#print ("#CD#############")
npts=25
dx=(xD-xC)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xC+i*dx
    y=yD
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#DE#############")
npts=148
dx=(xE-xD)/(npts-1)
dy=(yE-yD)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xD+i*dx
    y=yD+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))

#print ("#EJ#############")
npts=21
dx=(xJ-xE)/(npts-1)
dy=(yJ-yE)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xE+i*dx
    y=yE+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#IK#############")
npts=70
dx=(xK-xI)/(npts-1)
dy=(yK-yI)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xI+i*dx
    y=yI+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#GL#############")
npts=151
dx=(xL-xG)/(npts-1)
dy=(yL-yG)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xG+i*dx
    y=yG+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#BM#############")
npts=151
dx=(xM-xB)/(npts-1)
dy=(yM-yB)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xB+i*dx
    y=yB+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#AB#############")
npts=25
dx=(xB-xA)/(npts-1)
dy=(yB-yA)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xA+i*dx
    y=yA+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#QA#############")
npts=125
dx=(xA-xQ)/(npts-1)
dy=(yA-yQ)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xQ+i*dx
    y=yQ+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#PF#############")
npts=115
dx=(xF-xP)/(npts-1)
dy=(yF-yP)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xP+i*dx
    y=yP+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#OH#############")
npts=80
dx=(xH-xO)/(npts-1)
dy=(yH-yO)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xO+i*dx
    y=yO+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#NC#############")
npts=10
dx=(xC-xN)/(npts-1)
dy=(yC-yN)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xN+i*dx
    y=yN+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#RQ#############")
npts=50
dx=(xQ-x3)/(npts-1)
dy=(yQ-y3)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x3+i*dx
    y=y3+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#SP#############")
npts=50
dx=(xP-xS)/(npts-1)
dy=(yP-yS)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xS+i*dx
    y=yS+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#TO#############")
npts=50
dx=(xO-xT)/(npts-1)
dy=(yO-yT)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xT+i*dx
    y=yT+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#UN#############")
npts=50
dx=(xN-xU)/(npts-1)
dy=(yN-yU)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xU+i*dx
    y=yU+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


#print ("#MW#############")
npts=50
dx=(x2-xM)/(npts-1)
dy=(y2-yM)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xM+i*dx
    y=yM+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#LX#############")
npts=50
dx=(xX-xL)/(npts-1)
dy=(yX-yL)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xL+i*dx
    y=yL+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#KY#############")
npts=50
dx=(xY-xK)/(npts-1)
dy=(yY-yK)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xK+i*dx
    y=yK+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#JZ#############")
npts=50
dx=(xZ-xJ)/(npts-1)
dy=(yZ-yJ)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=xJ+i*dx
    y=yJ+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#01#############")
npts=100
dx=(x1-x0)/(npts-1)
dy=(y1-y0)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x0+i*dx
    y=y0+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#12#############")
npts=135
dx=(x2-x1)/(npts-1)
dy=(y2-y1)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x1+i*dx
    y=y1+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

#print ("#03#############")
npts=135
dx=(x0-x3)/(npts-1)
dy=(y0-y3)/(npts-1)
for i in range(0,npts):
    counter+=1
    x=x3+i*dx
    y=y3+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))




nodesfile.write("%5d %5d\n" %(counter_segment, 1))

#holes
segmentfile.write("%5d \n" %(0))

print(counter,'points')
print(counter_segment,'segments')





























