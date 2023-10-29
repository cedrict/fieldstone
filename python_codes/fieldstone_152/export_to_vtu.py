###############################################################################

import numpy as np
from basis_functions import *
from analytical_solution import *
from gravity_vector import *
from density import *
from viscosity import *

###############################################################################

def export_solutionQ1_to_vtu(istep,exp,NV,nel,xV,yV,vel_unit,u,v,vr,vt,rad,theta,\
                             exx2,eyy2,exy2,sr2,e_rr2,e_tt2,e_rt2,q,viscosity_nodal,\
                             density_nodal,iconQ1,g0,R1,R2,kk,rho_m):

   compute_sr1=False
   compute_sr3=False

   vtufile=open("solutionQ1_"+str(istep)+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]/vel_unit,v[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e %e %e \n" %(velocity_x(xV[i],yV[i],R1,R2,kk,exp),velocity_y(xV[i],yV[i],R1,R2,kk,exp),0.))
      vtufile.write("</DataArray>\n")
      #--
      #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
      #for i in range(0,NV):
      #    vtufile.write("%e %e %e \n" %(u_err[i],v_err[i],0.))
      #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(vr[i]/vel_unit,vt[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %rad[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta (co-latitude)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %density_nodal[i])
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      #--
      vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_xx(xV[i],yV[i],R1,R2,kk,exp)))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_yy(xV[i],yV[i],R1,R2,kk,exp)))
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %(sr_xy(xV[i],yV[i],R1,R2,kk,exp)))
      vtufile.write("</DataArray>\n")
   #--
   if compute_sr1:
      vtufile.write("<DataArray type='Float32' Name='sr1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %sr1[i])
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='sr2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %sr2[i])
   vtufile.write("</DataArray>\n")
   #--
   if compute_sr1:
      vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exx1[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %eyy1[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exy1[i])
      vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %exy2[i])
   vtufile.write("</DataArray>\n")
   #--
   if compute_sr3:
      vtufile.write("<DataArray type='Float32' Name='sr3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %sr3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exx3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %eyy3[i])
      vtufile.write("</DataArray>\n")
      #--
      vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%e \n" %exy3[i])
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='err' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rr2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='ett' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_tt2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='ert' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rt2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %viscosity_nodal[i])
   vtufile.write("</DataArray>\n")
   #--
   if exp==0:
      vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%e \n" % pressure(xV[i],yV[i],R1,R2,kk,rho_m,g0,exp))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,4*nel):
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

def export_solutionQ2_to_vtu(istep,NV,nel,xV,yV,vel_unit,u,v,viscosity_nodal,\
                             hull,surfaceV,cmbV,bc_fix,area,viscosity_elemental,\
                             density_elemental,surface_element,cmb_element,iconV):

   vtufile=open("solutionQ2_"+str(istep)+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]/vel_unit,v[i]/vel_unit,0.))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal1' Format='ascii'> \n")
   #for i in range(0,NV):       
   #    vtufile.write("%e %e %e \n" %(nx1[i],ny1[i],0.))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal2' Format='ascii'> \n")
   #for i in range(0,NV):
   #    vtufile.write("%e %e %e \n" %(nx2[i],ny2[i],0.))
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal diff' Format='ascii'> \n")
   #for i in range(0,NV):
   #    vtufile.write("%e %e %e \n" %(nx1[i]-nx2[i],ny1[i]-ny2[i],0.))
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %viscosity_nodal[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='hull' Format='ascii'> \n")
   for i in range(0,NV):
       if hull[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='surfaceV' Format='ascii'> \n")
   for i in range(0,NV):
       if surfaceV[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='cmbV' Format='ascii'> \n")
   for i in range(0,NV):
       if cmbV[i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='surfaceQ1' Format='ascii'> \n")
   #for i in range(0,NV):
   #    if surfaceQ1[i]:
   #       vtufile.write("%e \n" % 1)
   #    else:
   #       vtufile.write("%e \n" % 0)
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix(u)' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[2*i]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix(v)' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[2*i+1]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %area[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %viscosity_elemental[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e \n" %density_elemental[iel])
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='vol' Format='ascii'> \n")
   #for iel in range(0,nel):
   #    vtufile.write("%e \n" %vol[iel])
   #vtufile.write("</DataArray>\n")
   #
   vtufile.write("<DataArray type='Float32' Name='surface_element' Format='ascii'> \n")
   for iel in range(0,nel):
       if surface_element[iel]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #
   vtufile.write("<DataArray type='Float32' Name='cmb_element' Format='ascii'> \n")
   for iel in range(0,nel):
       if cmb_element[iel]:
          vtufile.write("%e \n" % 1)
       else:
          vtufile.write("%e \n" % 0)
   vtufile.write("</DataArray>\n")
   #
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                   iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %23)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################

def export_quadrature_points_to_vtu(nqperdim,nqel,qcoords_r,qcoords_s,mapping,xmapping,ymapping):

   vtufile=open("quadrature_points_"+str(nqperdim)+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(nqel,nqel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for k in range(0,nqel):
       rq=qcoords_r[k]
       sq=qcoords_s[k]
       NNNV=NNN(rq,sq,mapping)
       xq=np.dot(NNNV[:],xmapping[:,0])
       yq=np.dot(NNNV[:],ymapping[:,0])
       vtufile.write("%e %e %e \n" %(xq,yq,0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nqel):
       vtufile.write("%d \n" %(iel))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nqel):
       vtufile.write("%d \n" %((iel+1)*1))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nqel):
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

def export_mapping_points_to_vtu(mapping,mmapping,xmapping,ymapping):

   vtufile=open("mapping_points_"+mapping+".vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(mmapping,mmapping))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,mmapping):
       vtufile.write("%e %e %e \n" %(xmapping[i,0],ymapping[i,0],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,mmapping):
       vtufile.write("%d \n" %(iel))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,mmapping):
       vtufile.write("%d \n" %((iel+1)*1))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,mmapping):
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

def export_Q1_mesh_to_vtu(NV,nel,xV,yV,iconQ1):

   vtufile=open("mesh_Q1.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
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

def export_gravity_to_vtu(istep,np_grav,xM,zM,gvect_x,gvect_z):

   filename = 'gravity_{:04d}.vtu'.format(istep)
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints='%5d' NumberOfCells='%5d'> \n" %(np_grav,np_grav))
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,np_grav):
       vtufile.write("%e %e %e \n" %(xM[i],zM[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,np_grav):
       vtufile.write("%e %e %e \n" %(gvect_x[i],gvect_z[i],0))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range (0,np_grav):
       vtufile.write("%d \n" %(i))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range (0,np_grav):
       vtufile.write("%d \n" %(i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,np_grav):
       vtufile.write("%d \n" %1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################

def export_slices(nel_phi,NV,nel,r,theta,iconV,rho):
    dphi=2*np.pi/nel_phi
    
    phi=np.zeros(NV,dtype=np.float64) 
    for jel in range(0,nel_phi):

        phi[:]=jel*dphi

        xxx=r[:]*np.sin(theta[:])*np.cos(phi[:])
        yyy=r[:]*np.sin(theta[:])*np.sin(phi[:])
        zzz=r[:]*np.cos(theta[:])

        vtufile=open("slice_"+str(jel)+".vtu","w")
        vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
        vtufile.write("<UnstructuredGrid> \n")
        vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
        #####
        vtufile.write("<Points> \n")
        vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
        for i in range(0,NV):
            vtufile.write("%e %e %e \n" %(xxx[i],yyy[i],zzz[i]))
        vtufile.write("</DataArray>\n")
        vtufile.write("</Points> \n")
        #####
        vtufile.write("<CellData Scalars='scalars'>\n")
        vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
        for iel in range(0,nel):
            vtufile.write("%e \n" % rho[iel])
        vtufile.write("</DataArray>\n")
        vtufile.write("</CellData>\n")
        #####
        vtufile.write("<Cells>\n")
        #--
        vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
        for iel in range (0,nel):
            vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                        iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
        vtufile.write("</DataArray>\n")
        #--
        vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
        for iel in range (0,nel):
            vtufile.write("%d \n" %((iel+1)*8))
        vtufile.write("</DataArray>\n")
        #--
        vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
        for iel in range (0,nel):
            vtufile.write("%d \n" %23)
        vtufile.write("</DataArray>\n")
        #--
        vtufile.write("</Cells>\n")
        #####
        vtufile.write("</Piece>\n")
        vtufile.write("</UnstructuredGrid>\n")
        vtufile.write("</VTKFile>\n")
        vtufile.close()

###############################################################################
