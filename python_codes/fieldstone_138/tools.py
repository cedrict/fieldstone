import numpy as np

def export_mesh_3D(NV,nel,x,y,z,icon,filename,Mx,My,Mz,nnx,nny,nnz):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")

   counter=0
   zmin=1e50
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               if k==nnz-1:
                  zmin=min(zmin,z[counter])
               counter += 1
   #print(zmin)

   vtufile.write("<DataArray type='Float32' Name='z' Format='ascii'> \n")
   for i in range (0,NV):
        vtufile.write("%10e\n" % max(z[i],zmin))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   ##### 
   ##### TODO: check why I removed 3 lines below before ?

   if nel==nel:
   #abs(np.max(Mx)-np.min(Mx))>1e-6 or\
   #   abs(np.max(My)-np.min(My))>1e-6 or\
   #   abs(np.max(Mz)-np.min(Mz))>1e-6 : 

      vtufile.write("<CellData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='M vector' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%f %f %f \n" % (Mx[iel],My[iel],Mz[iel]))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='Mx' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%f \n" % (Mx[iel]))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='My' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%f \n" % (My[iel]))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='Mz' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%f \n" % (Mz[iel]))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='M' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%f \n" % (np.sqrt(Mx[iel]**2+My[iel]**2+Mz[iel]**2)))
      vtufile.write("</DataArray>\n")

      vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %12)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################

def export_mesh_2D(NV,nel,x,y,z,icon,filename):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #--
   vtufile.write("<Cells>\n")
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

def export_plane_measurements(NV,nel,x,y,z,icon,filename,B_vi,B_si,B_th):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (vol int)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(B_vi[0,i],B_vi[1,i],B_vi[2,i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (surf int)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(B_si[0,i],B_si[1,i],B_si[2,i]))
   vtufile.write("</DataArray>\n")
   #--

   if abs(np.max(B_th)-np.min(B_th))>1e-12:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (analytical)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e %e %e \n" %(B_th[0,i],B_th[1,i],B_th[2,i]))
      vtufile.write("</DataArray>\n")

   #--
   vtufile.write("</PointData>\n")
   #####
   #--
   vtufile.write("<Cells>\n")
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

def export_line_measurements(N,x,y,z,filename,B_vi,B_si,B_th):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N-1))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (vol int)' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%e %e %e \n" %(B_vi[0,i],B_vi[1,i],B_vi[2,i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (surf int)' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%e %e %e \n" %(B_si[0,i],B_si[1,i],B_si[2,i]))
   vtufile.write("</DataArray>\n")
   #--
   if abs(np.max(B_th)-np.min(B_th))>1e-12:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (analytical)' Format='ascii'> \n")
      for i in range(0,N):
          vtufile.write("%e %e %e \n" %(B_th[0,i],B_th[1,i],B_th[2,i]))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####

   #--
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range (0,N-1):
       vtufile.write("%d %d \n" %(i,i+1))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range (0,N-1):
       vtufile.write("%d \n" %((i+1)*2))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,N-1):
       vtufile.write("%d \n" %3)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")

   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()


###############################################################################

def export_path_measurements(npath,xpath,ypath,zpath,filename):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npath,npath))
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,npath):
       vtufile.write("%10e %10e %10e \n" %(xpath[i],ypath[i],zpath[i]))
#       vtufile.write("%.10e %.10e %.10e \n" %(xpath[i],ypath[i],zpath[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #vtufile.write("<PointData Scalars='scalars'>\n")
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
   #for i in range(0,npath):
   #    vtufile.write("%10e %10e %10e \n" %(swarm_u[i],swarm_v[i],0.))
   #vtufile.write("</DataArray>\n")
   #vtufile.write("</PointData>\n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,npath):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,npath):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,npath):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   print('produced path.vtu')
###################################################################################################

def export_spiral_measurements(N,x,y,z,filename,B_vi,B_si,B_th):

   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,N))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")

   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (vol int)' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%e %e %e \n" %(B_vi[0,i],B_vi[1,i],B_vi[2,i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (surf int)' Format='ascii'> \n")
   for i in range(0,N):
       vtufile.write("%e %e %e \n" %(B_si[0,i],B_si[1,i],B_si[2,i]))
   vtufile.write("</DataArray>\n")
   #--
   if abs(np.max(B_th)-np.min(B_th))>1e-12:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='B (analytical)' Format='ascii'> \n")
      for i in range(0,N):
          vtufile.write("%e %e %e \n" %(B_th[0,i],B_th[1,i],B_th[2,i]))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####

   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range (0,N):
       vtufile.write("%d \n" % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range (0,N):
       vtufile.write("%d \n" %(i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,N):
       vtufile.write("%d \n" % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()












