import numpy as np
import time as time
from compute_gravity_hexahedron_faces import *
from compute_gravity_hexahedron_quadrature import *
from compute_gravity_hexahedron_mascons import *
from compute_gravity_hexahedron_mascons2 import *

#----------------------------------------------------------------------------------------

def export_vector_to_vtu(pt,vec,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %2d ' NumberOfCells=' %2d '> \n" %(1,1))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt[0],pt[1],pt[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vector' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(vec[0],vec[1],vec[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d\n" %(0))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    vtufile.write("%d \n" %(1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    vtufile.write("%d \n" %1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#----------------------------------------------------------------------------------------

def export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8,1))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt_1[0],pt_1[1],pt_1[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_2[0],pt_2[1],pt_2[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_3[0],pt_3[1],pt_3[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_4[0],pt_4[1],pt_4[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_5[0],pt_5[1],pt_5[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_6[0],pt_6[1],pt_6[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_7[0],pt_7[1],pt_7[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_8[0],pt_8[1],pt_8[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(0,1,2,3,4,5,6,7))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    vtufile.write("%d \n" %(8))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    vtufile.write("%d \n" %12)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#----------------------------------------------------------------------------------------

def export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,name):
    vtufile=open(name+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8,12))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    vtufile.write("%10e %10e %10e \n" %(pt_1[0],pt_1[1],pt_1[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_2[0],pt_2[1],pt_2[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_3[0],pt_3[1],pt_3[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_4[0],pt_4[1],pt_4[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_5[0],pt_5[1],pt_5[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_6[0],pt_6[1],pt_6[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_7[0],pt_7[1],pt_7[2]))
    vtufile.write("%10e %10e %10e \n" %(pt_8[0],pt_8[1],pt_8[2]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Int32'  Name='node nb' Format='ascii'> \n")
    for i in range(0,8):
        vtufile.write("%d \n" %(i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    vtufile.write("%d %d %d \n" %( 4, 0, 5))
    vtufile.write("%d %d %d \n" %( 1, 5, 0))
    vtufile.write("%d %d %d \n" %( 5, 1, 6))
    vtufile.write("%d %d %d \n" %( 2, 6, 1))
    vtufile.write("%d %d %d \n" %( 2, 3, 6))
    vtufile.write("%d %d %d \n" %( 7, 6, 3))
    vtufile.write("%d %d %d \n" %( 7, 3, 4))
    vtufile.write("%d %d %d \n" %( 0, 4, 3))
    vtufile.write("%d %d %d \n" %( 6, 7, 5))
    vtufile.write("%d %d %d \n" %( 4, 5, 7))
    vtufile.write("%d %d %d \n" %( 3, 2, 0))
    vtufile.write("%d %d %d \n" %( 1, 0, 2))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,12):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,12):
        vtufile.write("%d \n" %5)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

#----------------------------------------------------------------------------------------

def export_gravity_on_plane(x,y,z,icon,nelx,nely,Np,\
                             grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons,\
                             grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad,\
                             grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces):

   filename = 'solution.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(Np,nelx*nely))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,Np):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],z[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity anomaly (mascons)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e %10e %10e \n" %(grid_gx_mascons[i],grid_gy_mascons[i],grid_gz_mascons[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity anomaly (quadrature)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e %10e %10e \n" %(grid_gx_quad[i],grid_gy_quad[i],grid_gz_quad[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity anomaly (faces)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e %10e %10e \n" %(grid_gx_faces[i],grid_gy_faces[i],grid_gz_faces[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='gravity potential (mascons)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e  \n" %(grid_U_mascons[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='gravity potential (quadrature)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e  \n" %(grid_U_quad[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='gravity potential (faces)' Format='ascii'> \n")
   for i in range(0,Np):
       vtufile.write("%10e  \n" %(grid_U_faces[i]))
   vtufile.write("</DataArray>\n")
   #-------------
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nelx*nely):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nelx*nely):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nelx*nely):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

test=1

print('test=',test)

if test==1:

   Ggrav=1
    
   mascons_file=open('mascons.ascii',"w")
   faces_file=open('faces.ascii',"w")
   quadrature_file=open('quadrature.ascii',"w")

   pt_1=np.array([0,0,0],dtype=np.float64)
   pt_2=np.array([1,0,0],dtype=np.float64)
   pt_3=np.array([1,1,0],dtype=np.float64)
   pt_4=np.array([0,1,0],dtype=np.float64)
   pt_5=np.array([0,0,1],dtype=np.float64)
   pt_6=np.array([1,0,1],dtype=np.float64)
   pt_7=np.array([1,1,1],dtype=np.float64)
   pt_8=np.array([0,1,1],dtype=np.float64)

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces')

   rho0=1
   pt_M=np.array([2,0.5,0.5],dtype=np.float64)

   for n_per_dim in range(2,48):

       print('n_per_dim',n_per_dim)
       
       start = time.time()
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,n_per_dim)
       mascons_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))

       start = time.time()
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       faces_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))

       start = time.time()
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,n_per_dim)
       if max(abs(g))>1e-10:
          quadrature_file.write("%d %e %e %e %e %e \n" %(n_per_dim,g[0],g[1],g[2],U,time.time() - start))
 
#--------------------------------------------------------------------------------------------------

if test==2:

   km=1e3
   mGal=1e-5
   Ggrav=6.67e-11

   pt_1=np.array([0  ,-0.1,-0.2],dtype=np.float64) ;  pt_1*=km
   pt_2=np.array([0.2,-0.1,-0.2],dtype=np.float64) ;  pt_2*=km
   pt_3=np.array([0.2, 0.1,-0.2],dtype=np.float64) ;  pt_3*=km
   pt_4=np.array([0  , 0.1,-0.2],dtype=np.float64) ;  pt_4*=km
   pt_5=np.array([-0.2,-0.1,-0.9],dtype=np.float64) ;  pt_5*=km
   pt_6=np.array([0   ,-0.1,-0.9],dtype=np.float64) ;  pt_6*=km
   pt_7=np.array([0   , 0.1,-0.9],dtype=np.float64) ;  pt_7*=km
   pt_8=np.array([-0.2, 0.1,-0.9],dtype=np.float64) ;  pt_8*=km

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces')

   rho0=-170

   Lx=2*km
   Ly=2*km
   nnx=41 ; nelx=nnx-1
   nny=41 ; nely=nny-1
   Np=nnx*nny

   x=np.zeros(Np,dtype=np.float64)
   y=np.zeros(Np,dtype=np.float64)
   z=np.zeros(Np,dtype=np.float64)
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nnx-1)-Lx/2
           y[counter]=j*Ly/float(nny-1)-Ly/2
           counter += 1

   icon=np.zeros((4,nelx*nely),dtype=np.int32)
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter] = i + j * (nelx + 1)
           icon[1,counter] = i + 1 + j * (nelx + 1)
           icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3,counter] = i + (j + 1) * (nelx + 1)
           counter += 1

   pt_M=np.zeros(3,dtype=np.float64)
   grid_gx_mascons=np.zeros(Np,dtype=np.float64)
   grid_gy_mascons=np.zeros(Np,dtype=np.float64)
   grid_gz_mascons=np.zeros(Np,dtype=np.float64)
   grid_gx_quad=np.zeros(Np,dtype=np.float64)
   grid_gy_quad=np.zeros(Np,dtype=np.float64)
   grid_gz_quad=np.zeros(Np,dtype=np.float64)
   grid_gx_faces=np.zeros(Np,dtype=np.float64)
   grid_gy_faces=np.zeros(Np,dtype=np.float64)
   grid_gz_faces=np.zeros(Np,dtype=np.float64)
   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons[i]=g[0]
       grid_gy_mascons[i]=g[1]
       grid_gz_mascons[i]=g[2]
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,5)
       grid_gx_quad[i]=g[0]
       grid_gy_quad[i]=g[1]
       grid_gz_quad[i]=g[2]
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]=g[0]
       grid_gy_faces[i]=g[1]
       grid_gz_faces[i]=g[2]

   grid_gx_quad/=mGal
   grid_gy_quad/=mGal
   grid_gz_quad/=mGal
   grid_gx_mascons/=mGal
   grid_gy_mascons/=mGal
   grid_gz_mascons/=mGal
   grid_gx_faces/=mGal
   grid_gy_faces/=mGal
   grid_gz_faces/=mGal

#--------------------------------------------------------------------------------------------------

if test==3:

   km=1e3
   mGal=1e-5
   Ggrav=6.67e-11

   #surface
   Lx=7*km
   Ly=7*km
   nnx=71 ; nelx=nnx-1
   nny=71 ; nely=nny-1
   Np=nnx*nny

   x=np.zeros(Np,dtype=np.float64)
   y=np.zeros(Np,dtype=np.float64)
   z=np.zeros(Np,dtype=np.float64)
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nnx-1)+2*km
           y[counter]=j*Ly/float(nny-1)+2*km
           counter += 1

   icon=np.zeros((4,nelx*nely),dtype=np.int32)
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter] = i + j * (nelx + 1)
           icon[1,counter] = i + 1 + j * (nelx + 1)
           icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3,counter] = i + (j + 1) * (nelx + 1)
           counter += 1

   pt_M=np.zeros(3,dtype=np.float64)
   grid_gx_mascons=np.zeros(Np,dtype=np.float64)
   grid_gy_mascons=np.zeros(Np,dtype=np.float64)
   grid_gz_mascons=np.zeros(Np,dtype=np.float64)
   grid_gx_mascons2=np.zeros(Np,dtype=np.float64)
   grid_gy_mascons2=np.zeros(Np,dtype=np.float64)
   grid_gz_mascons2=np.zeros(Np,dtype=np.float64)
   grid_gx_quad=np.zeros(Np,dtype=np.float64)
   grid_gy_quad=np.zeros(Np,dtype=np.float64)
   grid_gz_quad=np.zeros(Np,dtype=np.float64)
   grid_gx_faces=np.zeros(Np,dtype=np.float64)
   grid_gy_faces=np.zeros(Np,dtype=np.float64)
   grid_gz_faces=np.zeros(Np,dtype=np.float64)
   grid_U_mascons=np.zeros(Np,dtype=np.float64)
   grid_U_mascons2=np.zeros(Np,dtype=np.float64)
   grid_U_quad=np.zeros(Np,dtype=np.float64)
   grid_U_faces=np.zeros(Np,dtype=np.float64)

   #blue block
   rho0=-400
   pt_1=np.array([3.5,  3,-3],dtype=np.float64) ; pt_1*=km
   pt_2=np.array([5.5,  3,-3],dtype=np.float64) ; pt_2*=km
   pt_3=np.array([5.3,  5,-3],dtype=np.float64) ; pt_3*=km
   pt_4=np.array([3.3,4.9,-3],dtype=np.float64) ; pt_4*=km
   pt_5=np.array([3.5,  3,-7],dtype=np.float64) ; pt_5*=km
   pt_6=np.array([5.5,  3,-7],dtype=np.float64) ; pt_6*=km
   pt_7=np.array([5.5,  5,-7],dtype=np.float64) ; pt_7*=km
   pt_8=np.array([3.5,  5,-7],dtype=np.float64) ; pt_8*=km

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron1')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces1')

   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons[i]+=g[0]
       grid_gy_mascons[i]+=g[1]
       grid_gz_mascons[i]+=g[2]
       grid_U_mascons[i]=U
       g,U=compute_gravity_hexahedron_mascons2(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons2[i]+=g[0]
       grid_gy_mascons2[i]+=g[1]
       grid_gz_mascons2[i]+=g[2]
       grid_U_mascons2[i]=U
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,7)
       grid_gx_quad[i]+=g[0]
       grid_gy_quad[i]+=g[1]
       grid_gz_quad[i]+=g[2]
       grid_U_quad[i]=U
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]+=g[0]
       grid_gy_faces[i]+=g[1]
       grid_gz_faces[i]+=g[2]
       grid_U_faces[i]=U

   #red block
   rho0=400
   pt_1=np.array([6.5,5.5,-2],dtype=np.float64) ; pt_1*=km
   pt_2=np.array([7.5,5.5,-2],dtype=np.float64) ; pt_2*=km
   pt_3=np.array([7.5,7.5,-2],dtype=np.float64) ; pt_3*=km
   pt_4=np.array([6.5,7.5,-2],dtype=np.float64) ; pt_4*=km
   pt_5=np.array([6.5,5.5,-5],dtype=np.float64) ; pt_5*=km
   pt_6=np.array([8  ,5.5,-5],dtype=np.float64) ; pt_6*=km
   pt_7=np.array([8  ,7.5,-5],dtype=np.float64) ; pt_7*=km
   pt_8=np.array([6.5,7.5,-5],dtype=np.float64) ; pt_8*=km

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron2')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces2')

   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons[i]+=g[0]
       grid_gy_mascons[i]+=g[1]
       grid_gz_mascons[i]+=g[2]
       grid_U_mascons[i]=U
       g,U=compute_gravity_hexahedron_mascons2(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons2[i]+=g[0]
       grid_gy_mascons2[i]+=g[1]
       grid_gz_mascons2[i]+=g[2]
       grid_U_mascons2[i]=U
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,7)
       grid_gx_quad[i]+=g[0]
       grid_gy_quad[i]+=g[1]
       grid_gz_quad[i]+=g[2]
       grid_U_quad[i]=U
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]+=g[0]
       grid_gy_faces[i]+=g[1]
       grid_gz_faces[i]+=g[2]
       grid_U_faces[i]=U

   grid_gx_quad/=mGal
   grid_gy_quad/=mGal
   grid_gz_quad/=mGal
   grid_gx_mascons/=mGal
   grid_gy_mascons/=mGal
   grid_gz_mascons/=mGal
   grid_gx_mascons2/=mGal
   grid_gy_mascons2/=mGal
   grid_gz_mascons2/=mGal
   grid_gx_faces/=mGal
   grid_gy_faces/=mGal
   grid_gz_faces/=mGal

   np.savetxt('gravity_mascons.ascii',np.array([x,y,grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons]).T)
   np.savetxt('gravity_mascons2.ascii',np.array([x,y,grid_gx_mascons2,grid_gy_mascons2,grid_gz_mascons2,grid_U_mascons2]).T)
   np.savetxt('gravity_quadrature.ascii',np.array([x,y,grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad]).T)
   np.savetxt('gravity_faces.ascii',np.array([x,y,grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces]).T)

   mascons_file=open('line_mascons.ascii',"w")
   mascons2_file=open('line_mascons2.ascii',"w")
   faces_file=open('line_faces.ascii',"w")
   quadrature_file=open('line_quadrature.ascii',"w")
   for i in range(0,Np):
       if abs(y[i]-x[i])<1e-6:
          quadrature_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_quad[i],grid_gy_quad[i],grid_gz_quad[i],grid_U_quad[i]))
          mascons_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_mascons[i],grid_gy_mascons[i],grid_gz_mascons[i],grid_U_mascons[i]))
          mascons2_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_mascons2[i],grid_gy_mascons2[i],grid_gz_mascons2[i],grid_U_mascons2[i]))
          faces_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_faces[i],grid_gy_faces[i],grid_gz_faces[i],grid_U_faces[i]))
   mascons_file.close()
   mascons2_file.close()
   faces_file.close()
   quadrature_file.close()

   export_gravity_on_plane(x,y,z,icon,nelx,nely,Np,\
                            grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons,\
                            grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad,\
                            grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces)

#--------------------------------------------------------------------------------------------------

if test==4:

   Ggrav=1

   pt_1=np.array([-0.1,-0.16,0-0.01],dtype=np.float64)
   pt_2=np.array([1.2,-0.05,0.02],dtype=np.float64)
   pt_3=np.array([1.02,1-0.04,0],dtype=np.float64)
   pt_4=np.array([-0.03,1,-0.11],dtype=np.float64)
   pt_5=np.array([0+0.02,0,1],dtype=np.float64)
   pt_6=np.array([1,0,1.02],dtype=np.float64)
   pt_7=np.array([1.01,0.98,1.07],dtype=np.float64)
   pt_8=np.array([0,1.03,1],dtype=np.float64)

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces')

   rho0=1

   Lx=4
   Ly=4
   nnx=31 ; nelx=nnx-1
   nny=31 ; nely=nny-1
   Np=nnx*nny

   x=np.zeros(Np,dtype=np.float64)
   y=np.zeros(Np,dtype=np.float64)
   z=np.zeros(Np,dtype=np.float64)
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nnx-1)-Lx/3
           y[counter]=j*Ly/float(nny-1)-Ly/3
           z[counter]=1.25
           counter += 1

   icon=np.zeros((4,nelx*nely),dtype=np.int32)
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter] = i + j * (nelx + 1)
           icon[1,counter] = i + 1 + j * (nelx + 1)
           icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3,counter] = i + (j + 1) * (nelx + 1)
           counter += 1

   pt_M=np.zeros(3,dtype=np.float64)
   grid_gx_mascons=np.zeros(Np,dtype=np.float64)
   grid_gy_mascons=np.zeros(Np,dtype=np.float64)
   grid_gz_mascons=np.zeros(Np,dtype=np.float64)
   grid_gx_quad=np.zeros(Np,dtype=np.float64)
   grid_gy_quad=np.zeros(Np,dtype=np.float64)
   grid_gz_quad=np.zeros(Np,dtype=np.float64)
   grid_gx_faces=np.zeros(Np,dtype=np.float64)
   grid_gy_faces=np.zeros(Np,dtype=np.float64)
   grid_gz_faces=np.zeros(Np,dtype=np.float64)
   grid_U_mascons=np.zeros(Np,dtype=np.float64)
   grid_U_quad=np.zeros(Np,dtype=np.float64)
   grid_U_faces=np.zeros(Np,dtype=np.float64)
   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,8)
       grid_gx_mascons[i]=g[0]
       grid_gy_mascons[i]=g[1]
       grid_gz_mascons[i]=g[2]
       grid_U_mascons[i]=U
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,5)
       grid_gx_quad[i]=g[0]
       grid_gy_quad[i]=g[1]
       grid_gz_quad[i]=g[2]
       grid_U_quad[i]=U
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]=g[0]
       grid_gy_faces[i]=g[1]
       grid_gz_faces[i]=g[2]
       grid_U_faces[i]=U

   np.savetxt('gravity_mascons.ascii',np.array([x,y,grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons]).T)
   np.savetxt('gravity_quadrature.ascii',np.array([x,y,grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad]).T)
   np.savetxt('gravity_faces.ascii',np.array([x,y,grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces]).T)

   export_gravity_on_plane(x,y,z,icon,nelx,nely,Np,\
                            grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons,\
                            grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad,\
                            grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces)

#--------------------------------------------------------------------------------------------------

if test==5:

   km=1e3
   mGal=1e-5
   Ggrav=6.67e-11

   #surface
   Lx=20*km
   Ly=20*km
   nnx=41 ; nelx=nnx-1
   nny=41 ; nely=nny-1
   Np=nnx*nny

   x=np.zeros(Np,dtype=np.float64)
   y=np.zeros(Np,dtype=np.float64)
   z=np.zeros(Np,dtype=np.float64)
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nnx-1)-10*km
           y[counter]=j*Ly/float(nny-1)-10*km
           counter += 1

   icon=np.zeros((4,nelx*nely),dtype=np.int32)
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter] = i + j * (nelx + 1)
           icon[1,counter] = i + 1 + j * (nelx + 1)
           icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
           icon[3,counter] = i + (j + 1) * (nelx + 1)
           counter += 1

   pt_M=np.zeros(3,dtype=np.float64)
   grid_gx_mascons=np.zeros(Np,dtype=np.float64)
   grid_gy_mascons=np.zeros(Np,dtype=np.float64)
   grid_gz_mascons=np.zeros(Np,dtype=np.float64)
   grid_gx_quad=np.zeros(Np,dtype=np.float64)
   grid_gy_quad=np.zeros(Np,dtype=np.float64)
   grid_gz_quad=np.zeros(Np,dtype=np.float64)
   grid_gx_faces=np.zeros(Np,dtype=np.float64)
   grid_gy_faces=np.zeros(Np,dtype=np.float64)
   grid_gz_faces=np.zeros(Np,dtype=np.float64)
   grid_U_mascons=np.zeros(Np,dtype=np.float64)
   grid_U_quad=np.zeros(Np,dtype=np.float64)
   grid_U_faces=np.zeros(Np,dtype=np.float64)

   #blue block
   rho0=-500
   pt_1=np.array([-3.5,-4.5,-2],dtype=np.float64) ; pt_1*=km
   pt_2=np.array([ 2.5,-4.5,-2],dtype=np.float64) ; pt_2*=km
   pt_3=np.array([ 2.5, 4.5,-2],dtype=np.float64) ; pt_3*=km
   pt_4=np.array([-3.5, 4.5,-2],dtype=np.float64) ; pt_4*=km
   pt_5=np.array([-3.5,-4.5,-6],dtype=np.float64) ; pt_5*=km
   pt_6=np.array([ 0.5,-4.5,-6],dtype=np.float64) ; pt_6*=km
   pt_7=np.array([ 0.5, 4.5,-6],dtype=np.float64) ; pt_7*=km
   pt_8=np.array([-3.5, 4.5,-6],dtype=np.float64) ; pt_8*=km

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron1')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces1')

   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons[i]+=g[0]
       grid_gy_mascons[i]+=g[1]
       grid_gz_mascons[i]+=g[2]
       grid_U_mascons[i]=U
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,7)
       grid_gx_quad[i]+=g[0]
       grid_gy_quad[i]+=g[1]
       grid_gz_quad[i]+=g[2]
       grid_U_quad[i]=U
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]+=g[0]
       grid_gy_faces[i]+=g[1]
       grid_gz_faces[i]+=g[2]
       grid_U_faces[i]=U

   #red block
   rho0=-500
   pt_1=np.array([2  ,-4.5,-3],dtype=np.float64) ; pt_1*=km
   pt_2=np.array([6.5,-4.5,-3],dtype=np.float64) ; pt_2*=km
   pt_3=np.array([6.5, 4.5,-3],dtype=np.float64) ; pt_3*=km
   pt_4=np.array([2  , 4.5,-3],dtype=np.float64) ; pt_4*=km
   pt_5=np.array([0.23,-4.5,-6.5],dtype=np.float64) ; pt_5*=km
   pt_6=np.array([6.5 ,-4.5,-6.5],dtype=np.float64) ; pt_6*=km
   pt_7=np.array([6.5 , 4.5,-6.5],dtype=np.float64) ; pt_7*=km
   pt_8=np.array([0.23, 4.5,-6.5],dtype=np.float64) ; pt_8*=km

   export_hexahedron_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'hexahedron2')
   export_faces_to_vtu(pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,'faces2')

   for i in range(0,Np):
       pt_M[0]=x[i]
       pt_M[1]=y[i]
       pt_M[2]=z[i]
       g,U=compute_gravity_hexahedron_mascons(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,10)
       grid_gx_mascons[i]+=g[0]
       grid_gy_mascons[i]+=g[1]
       grid_gz_mascons[i]+=g[2]
       grid_U_mascons[i]=U
       g,U=compute_gravity_hexahedron_quadrature(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0,7)
       grid_gx_quad[i]+=g[0]
       grid_gy_quad[i]+=g[1]
       grid_gz_quad[i]+=g[2]
       grid_U_quad[i]=U
       g,U=compute_gravity_hexahedron_faces(Ggrav,pt_1,pt_2,pt_3,pt_4,pt_5,pt_6,pt_7,pt_8,pt_M,rho0)
       grid_gx_faces[i]+=g[0]
       grid_gy_faces[i]+=g[1]
       grid_gz_faces[i]+=g[2]
       grid_U_faces[i]=U

   grid_gx_quad/=mGal
   grid_gy_quad/=mGal
   grid_gz_quad/=mGal
   grid_gx_mascons/=mGal
   grid_gy_mascons/=mGal
   grid_gz_mascons/=mGal
   grid_gx_faces/=mGal
   grid_gy_faces/=mGal
   grid_gz_faces/=mGal

   np.savetxt('gravity_mascons.ascii',np.array([x,y,grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons]).T)
   np.savetxt('gravity_quadrature.ascii',np.array([x,y,grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad]).T)
   np.savetxt('gravity_faces.ascii',np.array([x,y,grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces]).T)

   mascons_file=open('line_mascons.ascii',"w")
   faces_file=open('line_faces.ascii',"w")
   quadrature_file=open('line_quadrature.ascii',"w")
   for i in range(0,Np):
       if abs(y[i]-x[i])<1e-6:
          quadrature_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_quad[i],grid_gy_quad[i],grid_gz_quad[i],grid_U_quad[i]))
          mascons_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_mascons[i],grid_gy_mascons[i],grid_gz_mascons[i],grid_U_mascons[i]))
          faces_file.write("%e %e %e %e %e %e\n" %(x[i],y[i],grid_gx_faces[i],grid_gy_faces[i],grid_gz_faces[i],grid_U_faces[i]))
   mascons_file.close()
   faces_file.close()
   quadrature_file.close()

   export_gravity_on_plane(x,y,z,icon,nelx,nely,Np,\
                            grid_gx_mascons,grid_gy_mascons,grid_gz_mascons,grid_U_mascons,\
                            grid_gx_quad,grid_gy_quad,grid_gz_quad,grid_U_quad,\
                            grid_gx_faces,grid_gy_faces,grid_gz_faces,grid_U_faces)

#--------------------------------------------------------------------------------------------------

