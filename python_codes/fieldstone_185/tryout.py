import numpy as np
from mesher_cube import * 
from analytical_solution import *

Lx=0.1
Ly=0.1
Lz=0.1

nelx=16
nely=nelx
nelz=nelx

x_V,y_V,z_V,icon_V,nn_V,nel=mesher_cube(Lx,Ly,Lz,nelx,nely,nelz)

# examples of fields on nodes:

field=np.zeros(nn_V,dtype=np.float64) 

vx=np.zeros(nn_V,dtype=np.float64) 
vy=np.zeros(nn_V,dtype=np.float64) 
vz=np.zeros(nn_V,dtype=np.float64) 

t_xx=np.zeros(nn_V,dtype=np.float64) 
t_yy=np.zeros(nn_V,dtype=np.float64) 
t_zz=np.zeros(nn_V,dtype=np.float64) 
t_xy=np.zeros(nn_V,dtype=np.float64) 
t_xz=np.zeros(nn_V,dtype=np.float64) 
t_yz=np.zeros(nn_V,dtype=np.float64) 

#compute displacement, stresses, strain on mesh nodes and store in arrays











vtufile=open('mesh3d.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
#####*************************
vtufile.write("<PointData Scalars='scalars'>\n")

#--exporting a field
vtufile.write("<DataArray type='Float32' Name='field' Format='ascii'> \n")
field.tofile(vtufile,sep=' ',format='%.4e') 
vtufile.write("</DataArray>\n")

#--exporting a vector
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vector' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(vx[i],vy[i],vz[i]))
vtufile.write("</DataArray>\n")

#--exporting a tensor
vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='tensor' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e %e %e %e\n" %(t_xx[i],t_yy[i],t_zz[i],\
                                          t_xy[i],t_yz[i],t_xz[i]))
vtufile.write("</DataArray>\n")

#--
vtufile.write("</PointData>\n")
#####**************************
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],\
                                                   icon_V[2,iel],icon_V[3,iel],\
                                                   icon_V[4,iel],icon_V[5,iel],\
                                                   icon_V[6,iel],icon_V[7,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
       vtufile.write("%d \n" %12)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

