import numpy as np

###############################################################################
# Q1 basis functions derivatives
###############################################################################

def dNdr(r,s):
       dNdr_0=-0.25*(1.-s)
       dNdr_1=+0.25*(1.-s)
       dNdr_2=+0.25*(1.+s)
       dNdr_3=-0.25*(1.+s)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def dNds(r,s):
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.+r)
       dNds_3=+0.25*(1.-r)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###############################################################################

def mesher(Lx,Ly,nelx,nely,nel,NV,mV):

    x = np.zeros(NV,dtype=np.float64)  # x coordinates
    y = np.zeros(NV,dtype=np.float64)  # y coordinates
    icon =np.zeros((mV,nel),dtype=np.int32)
  
    hx=Lx/nelx
    hy=Ly/nely

    N1=(nelx+1)*(nely+1) # corners of Q1 mesh 
    N2=nelx*nely*11      # inside nodes
    N3=3*(nelx+1)*nely   # vertical sides nodes
    N4=3*(nely+1)*nelx   # horizontal sides nodes

    rA=0.2 ; sA=0.2
    rB=0.6 ; sB=0.6
    rC=0.4 ; sC=0.7
    rD=sC  ; sD=rC

    rM=0.1  ; sM=0 ; rS=sM ; sS=rM
    rL=rM/2 ; sL=0 ; rT=sL ; sT=rL
    rN=0.55 ; sN=0 ; rR=sN ; sR=rN

    rI=(rS+rA)/2 ; sI=(sS+sA)/2 ; rJ=sI ; sJ=rI
    rK=(rI+rL)/2 ; sK=(sI+sL)/2 

    rF=(rA+1)/2 ; sF=(sA+sS)/2 ; rE=sF ; sE=rF
    rH=(rJ+1)/2 ; sH=(sJ+sT)/2 ; rG=sH ; sG=rH

    # nodes 1,2,3,4
    counter=0    
    for j in range(0,nely+1):
        for i in range(0,nelx+1):
            x[counter]=i*hx
            y[counter]=j*hy
            counter+=1    

    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rA)*hx ; y[counter]=(j+sA)*hy ; counter+=1 # A
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rB)*hx ; y[counter]=(j+sB)*hy ; counter+=1 # B 
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rC)*hx ; y[counter]=(j+sC)*hy ; counter+=1 # C
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rD)*hx ; y[counter]=(j+sD)*hy ; counter+=1 # D
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rE)*hx ; y[counter]=(j+sE)*hy ; counter+=1 # E
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rF)*hx ; y[counter]=(j+sF)*hy ; counter+=1 # F
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rG)*hx ; y[counter]=(j+sG)*hy ; counter+=1 # G
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rH)*hx ; y[counter]=(j+sH)*hy ; counter+=1 # H
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rI)*hx ; y[counter]=(j+sI)*hy ; counter+=1 # I
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rJ)*hx ; y[counter]=(j+sJ)*hy ; counter+=1 # J
    for j in range(0,nely):
        for i in range (0,nelx):
            x[counter]=(i+rK)*hx ; y[counter]=(j+sK)*hy ; counter+=1 # K

    for j in range(0,nely+1):
        for i in range (0,nelx):
            x[counter]=(i+rL)*hx ; y[counter]=(j+sL)*hy ; counter+=1 # L
            x[counter]=(i+rM)*hx ; y[counter]=(j+sM)*hy ; counter+=1 # M
            x[counter]=(i+rN)*hx ; y[counter]=(j+sN)*hy ; counter+=1 # N

    for j in range(0,nely):
        for i in range (0,nelx+1):
            x[counter]=(i+rT)*hx ; y[counter]=(j+sT)*hy ; counter+=1 # T
            x[counter]=(i+rS)*hx ; y[counter]=(j+sS)*hy ; counter+=1 # S
            x[counter]=(i+rR)*hx ; y[counter]=(j+sR)*hy ; counter+=1 # R

    iel=0    
    cell=0
    for j in range(0,nely):
        for i in range(0,nelx):

            nodeA=N1+ 0*nelx*nely+cell
            nodeB=N1+ 1*nelx*nely+cell
            nodeC=N1+ 2*nelx*nely+cell
            nodeD=N1+ 3*nelx*nely+cell
            nodeE=N1+ 4*nelx*nely+cell
            nodeF=N1+ 5*nelx*nely+cell
            nodeG=N1+ 6*nelx*nely+cell
            nodeH=N1+ 7*nelx*nely+cell
            nodeI=N1+ 8*nelx*nely+cell
            nodeJ=N1+ 9*nelx*nely+cell
            nodeK=N1+10*nelx*nely+cell

            nodeL=N1+N2+3*i+j*3*nelx
            nodeM=nodeL+1
            nodeN=nodeM+1
            #print(nodeL,nodeM,nodeN)

            nodeO=N1+N2+3*i+(j+1)*3*nelx
            nodeP=nodeO+1
            nodeQ=nodeP+1
            #print('OPQ-->',nodeO,nodeP,nodeQ)

            nodeT=N1+N2+N4+3*j*(nelx+1)+i*3
            nodeS=nodeT+1
            nodeR=nodeS+1
            #print('TSR->',nodeT,nodeS,nodeR)

            nodeW=N1+N2+N4+3*j*(nelx+1)+(i+1)*3
            nodeV=nodeW+1
            nodeU=nodeV+1
            #print('WVU->',nodeW,nodeV,nodeU)

            nodeSW= i + j * (nelx + 1)
            nodeSE= i + 1 + j * (nelx + 1)
            nodeNE= i + 1 + (j + 1) * (nelx + 1)
            nodeNW= i + (j + 1) * (nelx + 1)

            #sub element 0 
            icon[0,iel]=nodeSW
            icon[1,iel]=nodeL
            icon[2,iel]=nodeK
            icon[3,iel]=nodeT 
            iel+=1    

            #sub element 1 
            icon[0,iel]=nodeL
            icon[1,iel]=nodeM
            icon[2,iel]=nodeJ
            icon[3,iel]=nodeK
            iel+=1    

            #sub element 2
            icon[0,iel]=nodeM
            icon[1,iel]=nodeN
            icon[2,iel]=nodeH
            icon[3,iel]=nodeJ
            iel+=1    

            #sub element 3
            icon[0,iel]=nodeN
            icon[1,iel]=nodeSE
            icon[2,iel]=nodeW
            icon[3,iel]=nodeH
            iel+=1    

            #sub element 4
            icon[0,iel]=nodeT
            icon[1,iel]=nodeK
            icon[2,iel]=nodeI
            icon[3,iel]=nodeS
            iel+=1    

            #sub element 5
            icon[0,iel]=nodeK
            icon[1,iel]=nodeJ
            icon[2,iel]=nodeA
            icon[3,iel]=nodeI
            iel+=1    

            #sub element 6
            icon[0,iel]=nodeJ
            icon[1,iel]=nodeH
            icon[2,iel]=nodeF
            icon[3,iel]=nodeA
            iel+=1    

            #sub element 7
            icon[0,iel]=nodeH
            icon[1,iel]=nodeW
            icon[2,iel]=nodeV
            icon[3,iel]=nodeF
            iel+=1    

            #sub element 8
            icon[0,iel]=nodeS
            icon[1,iel]=nodeI
            icon[2,iel]=nodeG
            icon[3,iel]=nodeR
            iel+=1    

            #sub element 9
            icon[0,iel]=nodeI
            icon[1,iel]=nodeA
            icon[2,iel]=nodeE
            icon[3,iel]=nodeG
            iel+=1    

            #sub element 10
            icon[0,iel]=nodeA
            icon[1,iel]=nodeB
            icon[2,iel]=nodeC
            icon[3,iel]=nodeE
            iel+=1    

            #sub element 11
            icon[0,iel]=nodeA
            icon[1,iel]=nodeF
            icon[2,iel]=nodeD
            icon[3,iel]=nodeB
            iel+=1    

            #sub element 12
            icon[0,iel]=nodeF
            icon[1,iel]=nodeV
            icon[2,iel]=nodeU
            icon[3,iel]=nodeD
            iel+=1    

            #sub element 13
            icon[0,iel]=nodeR
            icon[1,iel]=nodeG
            icon[2,iel]=nodeO
            icon[3,iel]=nodeNW
            iel+=1    

            #sub element 14
            icon[0,iel]=nodeG
            icon[1,iel]=nodeE
            icon[2,iel]=nodeP
            icon[3,iel]=nodeO
            iel+=1    

            #sub element 15
            icon[0,iel]=nodeE
            icon[1,iel]=nodeC
            icon[2,iel]=nodeQ
            icon[3,iel]=nodeP
            iel+=1    

            #sub element 16
            icon[0,iel]=nodeC
            icon[1,iel]=nodeB
            icon[2,iel]=nodeNE
            icon[3,iel]=nodeQ
            iel+=1    

            #sub element 17
            icon[0,iel]=nodeD
            icon[1,iel]=nodeU
            icon[2,iel]=nodeNE
            icon[3,iel]=nodeB
            iel+=1    

            cell+=1

        #end for i
    #end for j 

    return x,y,icon

###############################################################################

nelx=6
nely=4
nel=nelx*nely*18

m=4
Lx=3
Ly=2

N=(nelx+1)*(nely+1) # corners of Q1 mesh 
N+=nelx*nely*11     # inside nodes
N+=3*(nelx+1)*nely  # vertical sides nodes
N+=3*(nely+1)*nelx  # horizontal sides nodes

print('nel=',nel)
print('N=',N)

x,y,icon=mesher(Lx,Ly,nelx,nely,nel,N,m)

#for iel in range (54,71):
#    print ("---------------------------")
#    print ("iel=",iel,icon[:,iel])

###############################################################################
# compute area of elements
###############################################################################
   
nqperdim=2
qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
qweights=[1.,1.]

area  = np.zeros(nel,dtype=np.float64) 
dNNdr = np.zeros(m,dtype=np.float64)    
dNNds = np.zeros(m,dtype=np.float64)     

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNdr[0:m]=dNdr(rq,sq)
            dNNds[0:m]=dNds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0] += dNNdr[k]*x[icon[k,iel]]
                jcb[0,1] += dNNdr[k]*y[icon[k,iel]]
                jcb[1,0] += dNNds[k]*x[icon[k,iel]]
                jcb[1,1] += dNNds[k]*y[icon[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            area[iel]+=jcob*weightq
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))

###############################################################################
# export to vtu
###############################################################################

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%e %e %e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e \n" % area[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

###############################################################################
