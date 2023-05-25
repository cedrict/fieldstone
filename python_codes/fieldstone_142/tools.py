import numpy as np
import time as timing

##################################################################################################

def export_elements_to_vtu(x,y,icon,filename):
    N=np.size(x)
    m,nel=np.shape(icon)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #-- 
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
    vtufile.write("</DataArray>\n")
    #-- 
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #-- 
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5) 
    vtufile.write("</DataArray>\n")
    #-- 
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

##################################################################################################

def export_elements_to_vtuP2(x,y,icon,filename):
    N=np.size(x)
    m,nel=np.shape(icon)

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%f %f %f \n" %(x[i],y[i],0.))
        #print(x[i],y[i],0.)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],icon[4,iel],icon[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

##################################################################################################
# convert P1 mesh into P2 mesh -> originates in stone 131
##################################################################################################

def mesh_P1_to_P2(x,y,icon):
    NV=np.size(x)
    m,nel=np.shape(icon)

    iconP2 =np.zeros((6,nel),dtype=np.int32)
    matrix =np.zeros((NV,NV),dtype=np.int32)

    counter=NV
    for iel in range(0,nel): #loop over elements
        iconP2[0,iel]=icon[0,iel]
        iconP2[1,iel]=icon[1,iel]
        iconP2[2,iel]=icon[2,iel]
        for k in range(0,m): # loop over faces
            noode1=icon[k,iel]
            noode2=icon[(k+1)%3,iel]
            node1=min(noode1,noode2)
            node2=max(noode1,noode2)
            if matrix[node1,node2]==0:
               matrix[node1,node2]=counter
               counter+=1
            iconP2[k+3,iel]=matrix[node1,node2]
        #end for
    #end for
    NVnew=counter

    xV = np.zeros(NVnew,dtype=np.float64)  
    yV = np.zeros(NVnew,dtype=np.float64)  
    xV[0:NV]=x[0:NV]
    yV[0:NV]=y[0:NV]

    for iel in range(0,nel): #loop over elements
        xV[iconP2[3,iel]]=0.5*(xV[iconP2[0,iel]]+xV[iconP2[1,iel]])
        xV[iconP2[4,iel]]=0.5*(xV[iconP2[1,iel]]+xV[iconP2[2,iel]])
        xV[iconP2[5,iel]]=0.5*(xV[iconP2[2,iel]]+xV[iconP2[0,iel]])
        yV[iconP2[3,iel]]=0.5*(yV[iconP2[0,iel]]+yV[iconP2[1,iel]])
        yV[iconP2[4,iel]]=0.5*(yV[iconP2[1,iel]]+yV[iconP2[2,iel]])
        yV[iconP2[5,iel]]=0.5*(yV[iconP2[2,iel]]+yV[iconP2[0,iel]])

    #print('---------------------smart------------------------')
    #print('NV=',NVnew)
    #print(iconP2)
    #print(xV)
    #print(yV)

    return NVnew,xV,yV,iconP2


#------------------------------------------------------------------------------

def compute_triangles_center_coordinates(coords,nodesArray):
    tx = coords[:,0]
    ty = coords[:,1]
    xc = (tx[nodesArray[:,0]] + tx[nodesArray[:,1]] + tx[nodesArray[:,2]]) / 3 
    yc = (ty[nodesArray[:,0]] + ty[nodesArray[:,1]] + ty[nodesArray[:,2]]) / 3 
    center = np.stack([xc, yc],axis=1)
    return center 

#------------------------------------------------------------------------------

def export_elements_to_vtu(x,y,icon,filename,area):
    N=np.size(x)
    m,nel=np.shape(icon)

    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()



#------------------------------------------------------------------------------

def compute_triangles_area(coords,nodesArray):
    
    tx = coords[:,0]
    ty = coords[:,1]
    
    # Triangle Area is calculated via Heron's formula, see wikipedia
    
    a = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,1]])**2)
    b = np.sqrt((tx[nodesArray[:,2]]-tx[nodesArray[:,1]])**2 + (ty[nodesArray[:,2]]-ty[nodesArray[:,1]])**2)
    c = np.sqrt((tx[nodesArray[:,0]]-tx[nodesArray[:,2]])**2 + (ty[nodesArray[:,0]]-ty[nodesArray[:,2]])**2)
    
    area = 0.5 * np.sqrt(a**2 * c**2 - (( a**2 + c**2 - b**2) / 2)**2)
    area = area.reshape(-1,1) #Transposing the 1xN matrix into Nx1 shape
    
    return area


#------------------------------------------------------------------------------

def compute_segs(InputCoords):
    segs = np.stack([np.arange(len(InputCoords)),np.arange(len(InputCoords))+1],axis=1)%len(InputCoords)
    return segs
