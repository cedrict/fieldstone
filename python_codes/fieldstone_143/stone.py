import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import time as timing

#------------------------------------------------------------------------------

def compute_segs(InputCoords):
    segs = np.stack([np.arange(len(InputCoords)),np.arange(len(InputCoords))+1],axis=1)%len(InputCoords)
    return segs

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

def compute_triangles_center_coordinates(coords,nodesArray):

    tx = coords[:,0]
    ty = coords[:,1]

    xc = (tx[nodesArray[:,0]] + tx[nodesArray[:,1]] + tx[nodesArray[:,2]]) / 3 
    yc = (ty[nodesArray[:,0]] + ty[nodesArray[:,1]] + ty[nodesArray[:,2]]) / 3 

    center = np.stack([xc, yc],axis=1)

    return center 

#------------------------------------------------------------------------------

def viscosity(x,y,Ly,eta_um,eta_c,eta_o,xA,yA,xB,yB,xC,yC,xD,yD,xE,yE,xF,yF,xG,yG,xI,yI):

    if y<yI:
       val=1e21
    elif y<yG:
       val5e22
    else:
       val=eta_um

    if x>xA and x<xC and y>yB:
       val=eta_c

    if x>xC and x<xE and y>yK:
       val=eta_o
    if x>xE and y>(yF-yE)/(xF-xE)*(x-xE)+yE:
       val=eta_o

    return val

#------------------------------------------------------------------------------
Lx=6000
Ly=3000 

h=20

xA=1000
yA=3000

xB=1000
yB=2800

xC=xA+2000
yC=3000

xD=xB+2000
yD=yB

xE=6000-500 #Lx-lr
yE=3000-100

xF=6000-h
yF=3000

xG=0
yG=3000-660

xH=6000
yH=3000-660

xI=0
yI=350

xJ=6000
yJ=350

xK=xD
yK=yE


if True:

   square_vertices = np.array([[0,0],[0,Ly],[Lx,Ly],[Lx,0]])
   square_edges = compute_segs(square_vertices)
   offset=4

   #segment AB 
   nptsAB=int((yA-yB)/h)
   x1=np.zeros(nptsAB) ; x1[:]=xA
   y1=np.linspace(yA,yB, nptsAB, endpoint=True)
   pointsAB = np.stack([x1,y1], axis = 1)
   segsAB = np.stack([np.arange(nptsAB-1) +offset , np.arange(nptsAB-1) + 1 +offset], axis=1) 
   offset+=nptsAB

   #segment BD
   nptsBD=int((xD-xB)/h)
   x1=np.linspace(xB,xD, nptsBD, endpoint=True)
   y1=np.zeros(nptsBD) ; y1[:]=yB
   pointsBD = np.stack([x1,y1], axis = 1)
   segsBD = np.stack([np.arange(nptsBD-1) +offset , np.arange(nptsBD-1) + 1 +offset], axis=1) 
   offset+=nptsBD

   #segment CD 
   nptsCD=int((yC-yD)/h)
   x1=np.zeros(nptsCD) ; x1[:]=xC
   y1=np.linspace(yC,yD, nptsCD, endpoint=True)
   pointsCD = np.stack([x1,y1], axis = 1)
   segsCD = np.stack([np.arange(nptsCD-1) +offset , np.arange(nptsCD-1) + 1 +offset], axis=1) 
   offset+=nptsCD

   #segment KE
   nptsKE=int((xE-xK)/h)
   x1=np.linspace(xK,xE, nptsKE, endpoint=True)
   y1=np.zeros(nptsKE) ; y1[:]=yK
   pointsKE = np.stack([x1,y1], axis = 1)
   segsKE = np.stack([np.arange(nptsKE-1) +offset , np.arange(nptsKE-1) + 1 +offset], axis=1) 
   offset+=nptsKE

   #segment EF
   nptsEF=int((xF-xE)/h)
   x1=np.linspace(xE,xF, nptsEF, endpoint=True)
   y1=np.linspace(yE,yF, nptsEF, endpoint=True)
   pointsEF = np.stack([x1,y1], axis = 1)
   segsEF = np.stack([np.arange(nptsEF-1) +offset , np.arange(nptsEF-1) + 1 +offset], axis=1) 
   offset+=nptsEF

   #segment GH
   nptsGH=300
   x1=np.linspace(xG,xH, nptsGH, endpoint=True)
   y1=np.zeros(nptsGH) ; y1[:]=yG
   pointsGH = np.stack([x1,y1], axis = 1)
   segsGH = np.stack([np.arange(nptsGH-1) +offset , np.arange(nptsGH-1) + 1 +offset], axis=1) 
   offset+=nptsGH

   #segment IJ
   nptsIJ=200
   x1=np.linspace(xI,xJ, nptsIJ, endpoint=True)
   y1=np.zeros(nptsIJ) ; y1[:]=yI
   pointsIJ = np.stack([x1,y1], axis = 1)
   segsIJ = np.stack([np.arange(nptsIJ-1) +offset , np.arange(nptsIJ-1) + 1 +offset], axis=1) 
   offset+=nptsIJ

   #assemble all coordinate arrays
   points = np.vstack([square_vertices, pointsAB, pointsBD, pointsCD, pointsKE, pointsEF, pointsGH, pointsIJ])

   #assemble all segments arrays
   SEGS = np.vstack([square_edges, segsAB, segsBD, segsCD, segsKE, segsEF, segsGH, segsIJ])

   O1 = {'vertices' : points, 'segments' : SEGS}
   T1 = tr.triangulate(O1, 'pqa20000') # tr.triangulate() computes the main dictionary 

   area=compute_triangles_area(T1['vertices'], T1['triangles'])
   icon=T1['triangles'] ; icon=icon.T
   x=T1['vertices'][:,0] 
   y=T1['vertices'][:,1] 
   export_elements_to_vtu(x,y,icon,'example1.vtu',area)

   m,nel=np.shape(icon)

   print(nel)

######################################################################
# compute element center coordinates
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]= (x[icon[0,iel]]+x[icon[1,iel]]+x[icon[2,iel]])/3
    yc[iel]= (y[icon[0,iel]]+y[icon[1,iel]]+y[icon[2,iel]])/3

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> yc (m,M) %.6e %.6e " %(np.min(yc),np.max(yc)))

print("compute element center coords: %.3f s" % (timing.time() - start))






