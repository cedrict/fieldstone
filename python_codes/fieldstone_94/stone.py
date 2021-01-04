import numpy as np

###############################################################################
# declare parameters
###############################################################################

ncellx    = 720 #number of points in longitudinal direction
ncelly    = 360 #number of points in latitudinal direction
ncellz    = 29  #number of points in radial direction
resx      = 0.5 #resolution in longitudinal direction (deg)
resy      = 0.5 #resolution in latitudinal direction (deg)

###############################################################################

geometry  = 1 #0 for cartesian projection, 1 for spherical projection
Rearth    = 6371e3 #radius of the earth

###############################################################################

ncell = ncellx*ncelly*ncellz
nnx = ncellx+1
nny = ncelly+1
nnz = ncellz+1
nnp = nnx*nny*nnz

###############################################################################
# declare array
###############################################################################

theta  = np.zeros(ncell,dtype=np.float64)  
phi  = np.zeros(ncell,dtype=np.float64)  
depth = np.zeros(ncell,dtype=np.float64)
ano = np.zeros(ncell,dtype=np.float64)
x   = np.zeros(nnp,dtype=np.float64)  
y   = np.zeros(nnp,dtype=np.float64)  
z   = np.zeros(nnp,dtype=np.float64)  
dVp = np.zeros(ncell,dtype=np.float64)
d=np.array([0,10,20,35,70,110,160,210,260,310,360,410,470,530,595,660,760,860,\
            980,110,1250,1400,1600,1800,2000,2200,2400,2560,2740,2890], dtype=np.float64) 
d*=1e3

###############################################################################
# generate node coordinates
# miny and minx are needed to rotate frame to fit data set with coastlines
###############################################################################

minx=-180
miny=90
counter = 0
for k in range(0,nnz):
    for j in range(0,nny):
        for i in range(0,nnx):
            x[counter] = minx + resx*i
            y[counter] = miny - resy*j
            z[counter] = -d[k]
            counter += 1

if geometry == 1:
   counter = 0
   for k in range(0,nnz):
       for j in range(0,nny):
           for i in range(0,nnx):
               r=Rearth-d[k]
               theta=np.radians(90-y[counter])
               phi=np.radians(x[counter])
               x[counter]=r*np.sin(theta)*np.cos(phi)
               y[counter]=r*np.sin(theta)*np.sin(phi)
               z[counter]=r*np.cos(theta)
               counter += 1
            
###############################################################################
# generate connectivity array
###############################################################################

icon = np.zeros((ncell,8),dtype=np.int32) 

counter = 0 
for k in range(0,ncellz): 
    for j in range(0,ncelly): 
        for i in range(0,ncellx): 
            icon[counter,0] = j*nnx     + i   + k*nnx*nny
            icon[counter,1] = j*nnx     + i+1 + k*nnx*nny
            icon[counter,2] = (j+1)*nnx + i+1 + k*nnx*nny
            icon[counter,3] = (j+1)*nnx + i   + k*nnx*nny
            icon[counter,4] = j*nnx     + i   + (k+1)*nnx*nny
            icon[counter,5] = j*nnx     + i+1 + (k+1)*nnx*nny
            icon[counter,6] = (j+1)*nnx + i+1 + (k+1)*nnx*nny
            icon[counter,7] = (j+1)*nnx + i   + (k+1)*nnx*nny
            counter += 1
    #print("creating icon array",int(((k+1)/29)*100),"% completed")

###############################################################################
# read data file and determine dVp value for every cell    
###############################################################################

f = open('UU-P07_lon_lat_depth_%dVp_cell_depth_midpoint','r')
lines = f.readlines()
f.close       
for i in range(0,ncell):  
    vals=lines[i].strip().split()
    dVp[i]=vals[3]

###############################################################################
#write data to vtu-file
###############################################################################

if geometry==0:
   factor=50000
else:
   factor=1

vtufile=open("solution.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,ncell))
#POINTS
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%f %f %f \n" %(x[i],y[i],z[i]/factor))
vtufile.write("</DataArray> \n")
vtufile.write("</Points> \n")
#SCALARS
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='dVp' Format='ascii'> \n")
for i in range(0,ncell):
        vtufile.write(" %f \n" %(-dVp[i])) 
vtufile.write("</DataArray> \n")
vtufile.write("</CellData> \n")
#CONNECTIVITY
vtufile.write("<Cells> \n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'>\n")
for iel in range(0,ncell):
    vtufile.write("%d %d %d %d %d %d %d %d \n" %(icon[iel,0],icon[iel,1],icon[iel,2],icon[iel,3],\
                                                 icon[iel,4],icon[iel,5],icon[iel,6],icon[iel,7]))
vtufile.write("</DataArray> \n")
#OFFSETS
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for i in range (0,ncell):
        vtufile.write("%d \n" %((i+1)*8))
vtufile.write("</DataArray> \n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for i in range (0,ncell):
        vtufile.write("%d \n" %(12)) 
vtufile.write("</DataArray> \n")
vtufile.write("</Cells> \n")
vtufile.write("</Piece> \n")
vtufile.write("</UnstructuredGrid> \n")
vtufile.write("</VTKFile> \n")
vtufile.close()
