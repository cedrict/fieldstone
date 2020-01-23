from points import *

#---------------------------------------
# count how many points we feed triangle

npts_AC=165
npts_DB=165
npts_CD=35
npts_DE=148
npts_EJ=21
npts_RQ=50
npts_IK=70
npts_GL=201
npts_AB=25
npts_JZ=150
npts_KY=150
npts_LX=300
npts_BM=201
npts_QA=125
npts_PF=115
npts_OH=80
npts_NC=20
npts_UN=50
npts_MW=300
npts_TO=50
npts_SP=50
npts_01=100
npts_12=135
npts_03=135
npts_56=201
npts_78=201

npts=\
npts_AC+\
npts_DB+\
npts_CD+\
npts_DE+\
npts_EJ+\
npts_RQ+\
npts_IK+\
npts_GL+\
npts_AB+\
npts_JZ+\
npts_KY+\
npts_LX+\
npts_BM+\
npts_QA+\
npts_PF+\
npts_OH+\
npts_NC+\
npts_UN+\
npts_MW+\
npts_TO+\
npts_SP+\
npts_01+\
npts_12+\
npts_03+\
npts_56+\
npts_78

#---------------------------------------

filename='mypoints'
nodesfile=open(filename,"w")
#nodesfile.write("%5d %5d %3d %3d\n" %(2351,2,0,1))
nodesfile.write("%5d %5d %3d %3d\n" %(npts,2,0,1))

counter=0
counter_segment=0

segmentfile=open("mysegments","w")

##AC#############
dx=(xA-xC)/(npts_AC-1)
dy=0.5e3
for i in range(0,npts_AC):
    counter+=1
    x=xC+i*dx
    y=yC+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_AC-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##DB#############
dx=(xB-xD)/(npts_DB-1)
dy=0.5e3
for i in range(0,npts_DB):
    counter+=1
    x=xD+i*dx
    y=yD+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_DB-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##56#############
dx=(x5-x6)/(npts_56-1)
dy=(y5-y6)/(npts_56-1)
for i in range(0,npts_56):
    counter+=1
    x=x6+i*dx
    y=y6+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_56-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##78#############
dx=(x7-x8)/(npts_78-1)
dy=(y7-y8)/(npts_78-1)
for i in range(0,npts_78):
    counter+=1
    x=x8+i*dx
    y=y8+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_78-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##CD#############
dx=(xD-xC)/(npts_CD-1)
for i in range(0,npts_CD):
    counter+=1
    x=xC+i*dx
    y=yD
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_CD-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##DE#############
dx=(xE-xD)/(npts_DE-1)
dy=(yE-yD)/(npts_DE-1)
for i in range(0,npts_DE):
    counter+=1
    x=xD+i*dx
    y=yD+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_DE-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##EJ#############
dx=(xJ-xE)/(npts_EJ-1)
dy=(yJ-yE)/(npts_EJ-1)
for i in range(0,npts_EJ):
    counter+=1
    x=xE+i*dx
    y=yE+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_EJ-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##IK#############
dx=(xK-xI)/(npts_IK-1)
dy=(yK-yI)/(npts_IK-1)
for i in range(0,npts_IK):
    counter+=1
    x=xI+i*dx
    y=yI+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_IK-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##GL#############
dx=(xL-xG)/(npts_GL-1)
dy=(yL-yG)/(npts_GL-1)
for i in range(0,npts_GL):
    counter+=1
    x=xG+i*dx
    y=yG+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_GL-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##BM#############
dx=(xM-xB)/(npts_BM-1)
dy=(yM-yB)/(npts_BM-1)
for i in range(0,npts_BM):
    counter+=1
    x=xB+i*dx
    y=yB+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_BM-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##AB#############
dx=(xB-xA)/(npts_AB-1)
dy=(yB-yA)/(npts_AB-1)
for i in range(0,npts_AB):
    counter+=1
    x=xA+i*dx
    y=yA+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_AB-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##QA#############
dx=(xA-xQ)/(npts_QA-1)
dy=(yA-yQ)/(npts_QA-1)
for i in range(0,npts_QA):
    counter+=1
    x=xQ+i*dx
    y=yQ+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_QA-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##PF#############
dx=(xF-xP)/(npts_PF-1)
dy=(yF-yP)/(npts_PF-1)
for i in range(0,npts_PF):
    counter+=1
    x=xP+i*dx
    y=yP+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_PF-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##OH#############
dx=(xH-xO)/(npts_OH-1)
dy=(yH-yO)/(npts_OH-1)
for i in range(0,npts_OH):
    counter+=1
    x=xO+i*dx
    y=yO+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_OH-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##NC#############
dx=(xC-xN)/(npts_NC-1)
dy=(yC-yN)/(npts_NC-1)
for i in range(0,npts_NC):
    counter+=1
    x=xN+i*dx
    y=yN+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_NC-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##RQ#############
dx=(xQ-x3)/(npts_RQ-1)
dy=(yQ-y3)/(npts_RQ-1)
for i in range(0,npts_RQ):
    counter+=1
    x=x3+i*dx
    y=y3+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_RQ-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##SP#############
dx=(xP-xS)/(npts_SP-1)
dy=(yP-yS)/(npts_SP-1)
for i in range(0,npts_SP):
    counter+=1
    x=xS+i*dx
    y=yS+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_SP-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##TO#############
dx=(xO-xT)/(npts_TO-1)
dy=(yO-yT)/(npts_TO-1)
for i in range(0,npts_TO):
    counter+=1
    x=xT+i*dx
    y=yT+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_TO-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##UN#############
dx=(xN-xU)/(npts_UN-1)
dy=(yN-yU)/(npts_UN-1)
for i in range(0,npts_UN):
    counter+=1
    x=xU+i*dx
    y=yU+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_UN-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))


##MW#############
dx=(x2-xM)/(npts_MW-1)
dy=(y2-yM)/(npts_MW-1)
for i in range(0,npts_MW):
    counter+=1
    x=xM+i*dx
    y=yM+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_MW-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##LX#############
dx=(xX-xL)/(npts_LX-1)
dy=(yX-yL)/(npts_LX-1)
for i in range(0,npts_LX):
    counter+=1
    x=xL+i*dx
    y=yL+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_LX-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##KY#############
dx=(xY-xK)/(npts_KY-1)
dy=(yY-yK)/(npts_KY-1)
for i in range(0,npts_KY):
    counter+=1
    x=xK+i*dx
    y=yK+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_KY-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##JZ#############
dx=(xZ-xJ)/(npts_JZ-1)
dy=(yZ-yJ)/(npts_JZ-1)
for i in range(0,npts_JZ):
    counter+=1
    x=xJ+i*dx
    y=yJ+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_JZ-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##01#############
dx=(x1-x0)/(npts_01-1)
dy=(y1-y0)/(npts_01-1)
for i in range(0,npts_01):
    counter+=1
    x=x0+i*dx
    y=y0+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_01-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##12#############
dx=(x2-x1)/(npts_12-1)
dy=(y2-y1)/(npts_12-1)
for i in range(0,npts_12):
    counter+=1
    x=x1+i*dx
    y=y1+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_12-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))

##03#############
dx=(x0-x3)/(npts_03-1)
dy=(y0-y3)/(npts_03-1)
for i in range(0,npts_03):
    counter+=1
    x=x3+i*dx
    y=y3+i*dy
    nodesfile.write("%5d %10e %10e %3d\n" %(counter+1,x,y,0))
    if i<npts_03-1:
       counter_segment+=1
       segmentfile.write("%5d %5d %5d %5d \n" %(counter_segment,counter,counter+1,0))




nodesfile.write("%5d %5d\n" %(counter_segment, 1))

#holes
segmentfile.write("%5d \n" %(0))

print(counter,'points')
print(counter_segment,'segments')

