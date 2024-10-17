import numpy as np
import random

###############################################################################
# Q2 basis functions in 2D

def NNN(r,s):
    val = np.zeros(9,dtype=np.float64)
    val[0]= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    val[1]= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    val[2]= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    val[3]= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    val[4]=    (1.-r**2) * 0.5*s*(s-1.)
    val[5]= 0.5*r*(r+1.) *    (1.-s**2)
    val[6]=    (1.-r**2) * 0.5*s*(s+1.)
    val[7]= 0.5*r*(r-1.) *    (1.-s**2)
    val[8]=    (1.-r**2) *    (1.-s**2)
    return val

###############################################################################
# B2 basis functions in 2D

def BBB(r,s):
    val = np.zeros(9,dtype=np.float64)
    val[0]= 0.25*(1-r)**2 * 0.25*(1-s)**2  
    val[1]= 0.25*(1+r)**2 * 0.25*(1-s)**2  
    val[2]= 0.25*(1+r)**2 * 0.25*(1+s)**2  
    val[3]= 0.25*(1-r)**2 * 0.25*(1+s)**2  
    val[4]= 0.5*(1-r**2)  * 0.25*(1-s)**2  
    val[5]= 0.25*(1+r)**2 * 0.5*(1-s**2) 
    val[6]= 0.5*(1-r**2)  * 0.25*(1+s)**2  
    val[7]= 0.25*(1-r)**2 * 0.5*(1-s**2) 
    val[8]= 0.5*(1-r**2)  * 0.5*(1-s**2) 
    return val

###############################################################################

def export_swarm_to_vtu(name,n,x,y,f,g):

   vtufile=open(name,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(n,n))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for k in range(0,n):
       vtufile.write("%e %e %e \n" %(x[k],y[k],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='density (Q_2)' Format='ascii'> \n")
   for k in range(0,n):
       vtufile.write("%e \n" %(f[k]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='density (B_2)' Format='ascii'> \n")
   for k in range(0,n):
       vtufile.write("%e \n" %(g[k]))
   vtufile.write("</DataArray>\n")

   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #-- 
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,n):
       vtufile.write("%d \n" %(iel))
   vtufile.write("</DataArray>\n")
   #-- 
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,n):
       vtufile.write("%d \n" %((iel+1)*1))
   vtufile.write("</DataArray>\n")
   #-- 
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,n):
       vtufile.write("%d \n" %1) 
   vtufile.write("</DataArray>\n")
   #-- 
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################

rnodes=[-1,+1,1,-1, 0,1,0,-1,0]
snodes=[-1,-1,1,+1,-1,0,1, 0,0] 

for icase in range(0,9):
    if icase==0:
       # 000
       # 000
       # 100 
       field=np.array([1,0,0,0,0,0,0,0,0],dtype=np.float64)
    if icase==1:
       # 000
       # 100
       # 000 
       field=np.array([0,0,0,0,0,0,0,1,0],dtype=np.float64)
    if icase==2:
       # 000
       # 100
       # 100 
       field=np.array([1,0,0,0,0,0,0,1,0],dtype=np.float64)
    if icase==3:
       # 100
       # 100
       # 100 
       field=np.array([1,0,0,1,0,0,0,1,0],dtype=np.float64)
    if icase==4:
       # 000
       # 100
       # 110 
       field=np.array([1,0,0,0,1,0,0,1,0],dtype=np.float64)
    if icase==5:
       # 000
       # 110
       # 110 
       field=np.array([1,0,0,0,1,0,0,1,1],dtype=np.float64)
    if icase==6:
       # 100
       # 100
       # 110 
       field=np.array([1,0,0,1,1,0,0,1,0],dtype=np.float64)
    if icase==7:
       # 100
       # 110
       # 110 
       field=np.array([1,0,0,1,1,0,0,1,1],dtype=np.float64)
    if icase==8:
       # 111
       # 111
       # 111 
       field=np.array([1,1,1,1,1,1,1,1,1],dtype=np.float64)

    nmarker=300000
    swarm_r=np.empty(nmarker,dtype=np.float64)
    swarm_s=np.empty(nmarker,dtype=np.float64)
    swarm_f=np.empty(nmarker,dtype=np.float64)
    swarm_g=np.empty(nmarker,dtype=np.float64)
    for im in range(0,nmarker):
        r=random.uniform(-1.,+1)
        s=random.uniform(-1.,+1)
        swarm_r[im]=r
        swarm_s[im]=s
        N=NNN(r,s)
        swarm_f[im]=np.dot(N[:],field[:])
        B=BBB(r,s)
        swarm_g[im]=np.dot(B[:],field[:])

    print('%d %f %f %f %f '   %(icase,np.min(swarm_f),np.max(swarm_f),np.min(swarm_g),np.max(swarm_g)))

    export_swarm_to_vtu('field_'+str(icase)+'.vtu',nmarker,swarm_r,swarm_s,swarm_f,swarm_g)

    export_swarm_to_vtu('nodes_'+str(icase)+'.vtu',9,rnodes,snodes,field,field)

#end for
