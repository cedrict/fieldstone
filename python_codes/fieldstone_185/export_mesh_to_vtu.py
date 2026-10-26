import numpy as np

def export_mesh_to_vtu(nn_V,nel,x_V,y_V,z_V,icon_V,volume,\
                      face1,face2,face3,face4,face5,face6):

   vtufile=open('mesh3d.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
   #for i in range(0,nn_V):
   #    vtufile.write("%e %e %e \n" %(ui,vi,wi))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % volume[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='face1' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face1[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='face2' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face2[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='face3' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face3[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='face4' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face4[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='face5' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face5[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='face6' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % face6[iel])
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
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
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

