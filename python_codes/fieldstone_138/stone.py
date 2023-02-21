import numpy as np
from magnetostatics import *
from tools import *
import random
import time as time
from set_measurement_parameters import *

#------------------------------------------------------------------------------
# this function returns a topography value at each point x,y passed as argument

def topography(x,y,A,llambda,cos_dir,sin_dir,slopex,slopey):
    pert1=A*np.sin(2*np.pi/llambda*(x*cos_dir+y*sin_dir))
    pert2=slopex*x+slopey*y 
    return pert1+pert2

#------------------------------------------------------------------------------
# returns analytical solution (vector B) 

def compute_analytical_solution(x,y,z,R,Mx,My,Mz,xcenter,ycenter,zcenter,benchmark):

   #-----------------------------------------------------------------
   if benchmark=='1': 
      mu0=4*np.pi #*1e-7
      V=4/3*np.pi*R**3
      r=np.sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)
      Bx=0
      By=0
      Bz=2*mu0*V/4/np.pi/r**3*Mz0

   #-----------------------------------------------------------------
   if benchmark=='2a' or benchmark=='2b' or benchmark=='4':
      Bx=0
      By=0
      Bz=0

   #-----------------------------------------------------------------
   if benchmark=='3': 
      r=np.sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)
      theta=np.arccos((z-zcenter)/r)
      phi=np.arctan2((y-ycenter),(x-xcenter))
      mu0=4*np.pi #*1e-7
      Q=(R/r)**3*mu0/3
      rux=np.sin(theta)*np.cos(phi)
      ruy=np.sin(theta)*np.sin(phi)
      ruz=np.cos(theta)
      thux=np.cos(theta)*np.cos(phi)
      thuy=np.cos(theta)*np.sin(phi)
      thuz=-np.sin(theta)
      Bx=Q*Mz*(2*(rux*np.cos(theta))+thux*np.sin(theta))
      By=Q*Mz*(2*(ruy*np.cos(theta))+thuy*np.sin(theta))
      Bz=Q*Mz*(2*(ruz*np.cos(theta))+thuz*np.sin(theta))

   return np.array([Bx,By,Bz],dtype=np.float64)

print('========================================')
print('=             ETNA project             =')
print('========================================')

###############################################################################
#benchmark:
#1: dipole (small sphere, far away), line measurement
#2a: random perturbation internal nodes cubic-> checks cancellation of internal faces 
#2b: random perturbation internal nodes pancake-> checks cancellation of internal faces 
#3: sphere (larger sphere, anywhere in space) analytical
#4: wavy surface, domain with constant M vector

#-1: etna topography

benchmark='-1'

###############################################################################
# be careful with the position of the measurement points for 
# benchmark 1. These cannot be above the center of the sphere
# but also not above a diagonal of an element (see ...wtopo)

if benchmark=='1':
   Lx=2
   Ly=2
   Lz=2
   nelx=20
   nely=20
   nelz=20
   Mx0=0     # do not change
   My0=0     # do not change
   Mz0=1     # do not change
   nqdim=4
   sphere_R=1
   sphere_xc=Lx/2
   sphere_yc=Ly/2
   sphere_zc=-Lz/2
   #line meas
   do_line_measurements=True
   xstart=Lx/2+1e-9
   ystart=Ly/2+1e-10
   zstart=0.01      #slightly above surface
   xend=Lx/2+1e-9
   yend=Ly/2+1e-10
   zend=10
   line_nmeas=50 
   #plane meas
   do_plane_measurements=False
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=zend
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=21
   plane_nny=21
   do_spiral_measurements=False
   do_path_measurements=False

if benchmark=='2a':
   Lx=10
   Ly=10
   Lz=10
   nelx=5
   nely=5
   nelz=5
   Mx0=0
   My0=1
   Mz0=0
   nqdim=6
   #plane meas
   do_plane_measurements=True
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=1
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=11
   plane_nny=11
   dz=0.1 #amplitude random
   do_line_measurements=False
   sphere_R=0
   sphere_xc=0
   sphere_yc=0
   sphere_zc=0
   do_spiral_measurements=False
   do_path_measurements=False

if benchmark=='2b':
   Lx=10
   Ly=10
   Lz=10
   nelx=5
   nely=5
   nelz=20
   Mx0=0
   My0=1
   Mz0=0
   nqdim=6
   #plane meas
   do_plane_measurements=True
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=1
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=11
   plane_nny=11
   dz=0.025 #amplitude random
   do_line_measurements=False
   sphere_R=0
   sphere_xc=0
   sphere_yc=0
   sphere_zc=0
   do_spiral_measurements=False
   do_path_measurements=False

if benchmark=='3':
   Lx=20
   Ly=20
   Lz=20
   nelx=60
   nely=nelx
   nelz=nelx
   Mx0=0
   My0=0
   Mz0=7.5
   sphere_R=10
   sphere_xc=Lx/2
   sphere_yc=Ly/2
   sphere_zc=-Lz/2
   nqdim=6
   #plane meas
   do_plane_measurements=False
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=0.4
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=30
   plane_nny=30
   do_line_measurements=False
   do_spiral_measurements=True
   radius_spiral=1.01*sphere_R
   npts_spiral=101 #keep odd
   do_path_measurements=False

if benchmark=='4':
   Lx=100
   Ly=100
   Lz=50
   nelx=40
   nely=40
   nelz=8
   Mx0=0
   My0=4
   Mz0=-6
   nqdim=4
   #topography parameters
   wavelength=25
   A=4
   do_plane_measurements=True
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=0.25
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=50
   plane_nny=50
   do_line_measurements=False
   sphere_R=0
   sphere_xc=0
   sphere_yc=0
   sphere_zc=0
   # to do: code line meas 
   do_spiral_measurements=False
   do_path_measurements=False

   subbench='south'

   if subbench=='east':
      slopex=np.arctan(-6/180*np.pi)
      slopey=np.arctan(0/180*np.pi)
      direction=90/180*np.pi

   if subbench=='north':
      slopex=np.arctan(0/180*np.pi)
      slopey=np.arctan(-6/180*np.pi)
      direction=0/180*np.pi

   if subbench=='west':
      slopex=np.arctan(6/180*np.pi)
      slopey=np.arctan(0/180*np.pi)
      direction=90/180*np.pi

   if subbench=='south':
      slopex=np.arctan(0/180*np.pi)
      slopey=np.arctan(6/180*np.pi)
      direction=0/180*np.pi

   cos_dir=np.cos(direction)
   sin_dir=np.sin(direction)

   do_path_measurements=False

if benchmark=='-1':

   from etna import *

if benchmark=='-1000': #full 5m dem -- too big
   nnx=6370
   nny=6361
   xllcorner=488243.81984712
   yllcorner=4162238.3580075
   nelx=nnx-1
   nely=nny-1
   Lx=5*nelx    # each cell is 5x5 meter!
   Ly=5*nely
   Lz=1000
   nelz=1
   do_plane_measurements=False
   do_line_measurements=False
   do_spiral_measurements=False
   Mx0=0
   My0=4
   Mz0=-6
   nqdim=4

#------------------------------------------------------------------------------

nel=nelx*nely*nelz
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction
NV=nnx*nny*nnz  # number of nodes

###############################################################################

print('========================================')
print('benchmark=',benchmark)
print('Lx,Ly,Lz=',Lx,Ly,Lz)
print('nelx,nely,nelz=',nelx,nely,nelz)
print('nnx,nny,nnz=',nnx,nny,nnz)
print('nel=',nel)
print('NV=',NV)
print('Mx0,My0,Mz0=',Mx0,My0,Mz0)
print('nqdim=',nqdim)
print('do_plane_measurements=',do_plane_measurements)
if do_plane_measurements:
   print('  plane_x0,y0,z0=',plane_x0,plane_y0,plane_z0)
   print('  plane_Lx,plane_Ly=',plane_Lx,plane_Ly)
   print('  plane_nnx,plane_nny=',plane_nnx,plane_nny)
print('do_line_measurements=',do_line_measurements)
if do_line_measurements:
   print('xstart,ystart,zstart=',xstart,ystart,zstart)
   print('xend,yend,zend=',xend,yend,zend)
   print('line_nmeas=',line_nmeas)
print('do_spiral_measurements=',do_spiral_measurements)
if do_spiral_measurements:
   print('npts_spiral',npts_spiral)
   print('radius_spiral',radius_spiral)
print('do_path_measurements=',do_path_measurements)   
if do_path_measurements:                              
   print('site=',site)                                
   print('path=',path)                                
   print('height=',ho,zpath_height)                   
   print('npts path=',npath)                          
   print('zpath_option=',zpath_option)
print('========================================')

###############################################################################
# grid point setup
# if benchmark 2, a small random perturbation is added to the
# z coordinate of the interior nodes.
###############################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    #print( int(i/nnx*100)  ,'% done')
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)-Lz
            if i!=0 and j!=0 and k!=0 and i!=nnx-1 and j!=nny-1 and k!=nnz-1 and (benchmark=='2a' or benchmark=='2b'):
               z[counter]+=random.uniform(-1,+1)*dz
            counter += 1
        #end for
    #end for
#end for
   
print("grid points setup: %.3f s" % (time.time() - start))

###############################################################################
# connectivity
###############################################################################
start = time.time()

icon =np.zeros((8,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    #print(int(i/nelx*100),'% done')
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("grid connectivity setup: %.3f s" % (time.time() - start))

###############################################################################
# adding wavy topography to surface and deform the mesh accordingly

if benchmark=='4':

   for i in range(0,NV):
       if abs(z[i])<1e-6:
          z[i]+=topography(x[i]-Lx/2,y[i]-Ly/2,A,wavelength,cos_dir,sin_dir,slopex,slopey)

   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               LLz=Lz+topography(x[counter]-Lx/2,y[counter]-Ly/2,A,wavelength,cos_dir,sin_dir,slopex,slopey)
               z[counter]=k*LLz/float(nelz)-Lz
               counter += 1
           #end for
       #end for
   #end for

   print('add synthetic topography')

###############################################################################

if benchmark=='-1':

   x[:]+=xllcorner
   y[:]+=yllcorner

   N=nnx*nny

   ztopo=np.empty(N,dtype=np.float64) 

   topo = open(topofile, 'r')
   lines_topo = topo.readlines()
   nlines=np.size(lines_topo)
   print(topofile+' counts ',nlines,' lines',nny)
   counter=0
   for i in range(0,nlines):
       #reading lines backwards bc of how file is built
       line=lines_topo[nlines-1-i].strip()
       columns=line.split()
       for j in range(0,nnx):
           ztopo[counter]=columns[j]
           counter+=1    

   print('topo (min/max):',min(ztopo),max(ztopo))
   print('read topo file ok')

   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               zmax=Lz+ztopo[j*nnx+i]
               zmin= 0+ztopo[j*nnx+i]
               z[counter]=k*(zmax-zmin)/float(nelz)+zmin-Lz
               counter += 1
           #end for
       #end for
   #end for

   print('add etna topography')

   #read in path 

   path = open(pathfile, 'r')
   lines_path = path.readlines()
   nlines=np.size(lines_path)
   print(pathfile+' counts ',nlines,' lines')
   xpath = np.empty(npath,dtype=np.float64)  # x coordinates
   ypath = np.empty(npath,dtype=np.float64)  # y coordinates
   zpath = np.empty(npath,dtype=np.float64)  # z coordinates
   B_inc = np.empty(npath,dtype=np.float64) 
   B_dec = np.empty(npath,dtype=np.float64) 
   B_int = np.empty(npath,dtype=np.float64) 

   for i in range(0,npath):
       #reading lines backwards bc of how file is built
       line=lines_path[npath-1-i].strip()
       columns=line.split()
       xpath[i]=columns[1]
       ypath[i]=columns[2]
       zpath[i]=columns[3]
       B_inc[i]=columns[4]
       B_dec[i]=columns[5]
       B_int[i]=columns[6]

   if zpath_option==2: # based on dem

      start = time.time()

      for i in range(0,npath):
          iel=0
          for ielx in range(0,nelx):
              for iely in range(0,nely):
                  for ielz in range(0,nelz):
                      if ielz==nelz-1 and\
                         xpath[i]>x[icon[0,iel]] and\
                         xpath[i]<x[icon[2,iel]] and\
                         ypath[i]>y[icon[0,iel]] and\
                         ypath[i]<y[icon[2,iel]]:
                         r=((xpath[i]-x[icon[0,iel]])/(x[icon[2,iel]]-x[icon[0,iel]])-0.5)*2
                         s=((ypath[i]-y[icon[0,iel]])/(y[icon[2,iel]]-y[icon[0,iel]])-0.5)*2
                         N1=0.25*(1-r)*(1-s)
                         N2=0.25*(1+r)*(1-s)
                         N3=0.25*(1+r)*(1+s)
                         N4=0.25*(1-r)*(1+s)
                         zpath[i]=z[icon[4,iel]]*N1+\
                                  z[icon[5,iel]]*N2+\
                                  z[icon[6,iel]]*N3+\
                                  z[icon[7,iel]]*N4+\
                                  zpath_height
                      #end if
                      iel+=1
                  #end for
              #end for
          #end for
      #end for

      print("creating path points above DEM (zpath_option=2): %.3f s" % (time.time() - start))

   print('xpath (min/max):',min(xpath),max(xpath))
   print('ypath (min/max):',min(ypath),max(ypath))
   print('zpath (min/max):',min(zpath),max(zpath))

   vtufile=open('path.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npath,npath))
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,npath):
       vtufile.write("%.10e %.10e %.10e \n" %(xpath[i],ypath[i],zpath[i]))
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

###############################################################################
# prescribe M inside each cell
# for benchmarks 1 and 3, M is zero everywhere except inside
# a sphere of radius sphere_R at location (sphere_xc,sphere_yc,sphere_zc)
# we use the center of an element as a representative point.
# For benchmark 2, M is constant in space and equal to (Mx0,My0,Mz0)
###############################################################################
start = time.time()

Mx=np.zeros(nel,dtype=np.float64)
My=np.zeros(nel,dtype=np.float64)
Mz=np.zeros(nel,dtype=np.float64)

if benchmark=='1' or benchmark=='3':
   Mx[:]=0
   My[:]=0
   Mz[:]=0
   for iel in range(0,nel):
       xc=(x[icon[0,iel]]+x[icon[6,iel]])*0.5
       yc=(y[icon[0,iel]]+y[icon[6,iel]])*0.5
       zc=(z[icon[0,iel]]+z[icon[6,iel]])*0.5
       if (xc-sphere_xc)**2+(yc-sphere_yc)**2+(zc-sphere_zc)**2<sphere_R**2:
          Mx[iel]=Mx0
          My[iel]=My0
          Mz[iel]=Mz0

if benchmark=='2a' or benchmark=='2b' or benchmark=='4' or benchmark=='-1':
   Mx[:]=Mx0
   My[:]=My0
   Mz[:]=Mz0

export_mesh_3D(NV,nel,x,y,z,icon,'mesh.vtu',Mx,My,Mz,nnx,nny,nnz)
   
print("prescribe M vector in domain: %.3f s" % (time.time() - start))

###############################################################################
# plane measurements setup
# the plane originates at (plane_x0,plane_y0,plane_z0) and extends 
# in the x,y directions by plane_Lx,plane_Ly
# note that a small perturbation is added to the x,y coordinates
# so as to avoid that a measurement point lies in the plane
# of an element (vertical) face. 
###############################################################################

if do_plane_measurements:

   plane_nmeas=plane_nnx*plane_nny # total number of points in plane

   plane_nelx=plane_nnx-1          # nb of cells in x direction in plane
   plane_nely=plane_nny-1          # nb of cells in y direction in plane
   plane_nel=plane_nelx*plane_nely # total nb of cells in plane

   x_meas=np.empty(plane_nmeas,dtype=np.float64)  # x coordinates of meas points
   y_meas=np.empty(plane_nmeas,dtype=np.float64)  # y coordinates of meas points
   z_meas=np.empty(plane_nmeas,dtype=np.float64)  # y coordinates of meas points

   counter = 0
   for j in range(0,plane_nny):
       for i in range(0,plane_nnx):
           x_meas[counter]=plane_x0+(i+random.uniform(-1,+1)*1e-8)*plane_Lx/float(plane_nnx-1) 
           y_meas[counter]=plane_y0+(j+random.uniform(-1,+1)*1e-8)*plane_Ly/float(plane_nny-1) 
           z_meas[counter]=plane_z0
           if benchmark=='4':
              z_meas[counter]+=topography(x_meas[counter]-Lx/2,y_meas[counter]-Ly/2,A,\
                               wavelength,cos_dir,sin_dir,slopex,slopey)
           counter += 1

   icon_meas =np.zeros((4,plane_nel),dtype=np.int32)
   counter = 0
   for j in range(0,plane_nely):
       for i in range(0,plane_nelx):
           icon_meas[0,counter] = i + j * (plane_nelx + 1)
           icon_meas[1,counter] = i + 1 + j * (plane_nelx + 1)
           icon_meas[2,counter] = i + 1 + (j + 1) * (plane_nelx + 1)
           icon_meas[3,counter] = i + (j + 1) * (plane_nelx + 1)
           counter += 1

   #export_mesh_2D(plane_nmeas,plane_nel,x_meas,y_meas,z_meas,icon_meas,'mesh_plane_measurements.vtu')

   #np.savetxt('mesh_plane_measurements.ascii',np.array([x_meas,y_meas,z_meas]).T)

   print('setup plane measurement points ')


###############################################################################
# measuring B on a plane
# Nomenclature for variables/arrays:
# _vi: volume integral
# _si: surface integral
# _th: analytical value (if applicable)
# The volume integral is parameterised by the number of quadrature 
# points per dimension nqdim.
# Because the integrand is not a polynomial, the volume integral
# remains a numerical solution (which depends on nqdim), while 
# the surface integral is actually analytical (down to machine precision).
###############################################################################
   
if do_plane_measurements:
   print('starting plane measurement ...')

   B_vi=np.zeros((3,plane_nmeas),dtype=np.float64)
   B_si=np.zeros((3,plane_nmeas),dtype=np.float64)
   B_th=np.zeros((3,plane_nmeas),dtype=np.float64)

   for i in range(0,plane_nmeas):
       print('------------------------------')
       print('doing',i,'out of ',plane_nmeas) 
       #print('x,y,z meas',x_meas[i],y_meas[i],z_meas[i])
       B_th[:,i]=compute_analytical_solution(x_meas[i],y_meas[i],z_meas[i],sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)
       #print('analytical ->',B_th[:,i])

       start = time.time()
       for iel in range(0,nel):
           B_vi[:,i]+=compute_B_quadrature(x_meas[i],y_meas[i],z_meas[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
       print("vol int: %.3f s" % (time.time() - start))
       #print('vol int    ->',B_vi[:,i])

       start = time.time()
       for iel in range(0,nel):
           B_si[:,i]+=compute_B_surface_integral_wtopo(x_meas[i],y_meas[i],z_meas[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])
       print("surf int: %.3f s" % (time.time() - start))
       #print('surf int   ->',B_si[:,i])

   export_plane_measurements(plane_nmeas,plane_nel,x_meas,y_meas,z_meas,icon_meas,'plane_measurements.vtu',B_vi,B_si,B_th)

   np.savetxt('plane_measurements.ascii',np.array([x_meas,y_meas,z_meas,\
                                                   B_vi[0,:],B_vi[1,:],B_vi[2,:],\
                                                   B_th[0,:],B_th[1,:],B_th[2,:]]).T)

   exit()

###############################################################################
# measuring B on a line
# the line starts at xstart,ystart,zstart and ends at 
# xend,yend,zend, and is discretised by means of line_nmeas pts
###############################################################################

print('========================================')

if do_line_measurements:

   print('starting line measurement ...')

   x_meas=np.empty(line_nmeas,dtype=np.float64)  # x coordinates
   y_meas=np.empty(line_nmeas,dtype=np.float64)  # y coordinates
   z_meas=np.empty(line_nmeas,dtype=np.float64)  # y coordinates

   linefile=open("measurements_line.ascii","w")
   linefile.write("# 1,2,3, 4    , 5    , 6    , 7    , 8    , 9    , 10   , 11   , 12    \n")
   linefile.write("# x,y,z, Bx_vi, By_vi, Bz_vi, Bx_si, By_si, Bz_si, Bx_th, By_th, Bz_th \n")

   B_vi=np.zeros((3,line_nmeas),dtype=np.float64)
   B_si=np.zeros((3,line_nmeas),dtype=np.float64)
   B_th=np.zeros((3,line_nmeas),dtype=np.float64)

   for i in range(0,line_nmeas):
       print('doing',i,'out of ',line_nmeas) 
       xm=xstart+(xend-xstart)/(line_nmeas-1)*i
       ym=ystart+(yend-ystart)/(line_nmeas-1)*i
       zm=zstart+(zend-zstart)/(line_nmeas-1)*i
       x_meas[i]=xm
       y_meas[i]=ym
       z_meas[i]=zm
       #print(xm,ym,zm)
       for iel in range(0,nel):
           B_vi[:,i]+=compute_B_quadrature      (xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
           B_si[:,i]+=compute_B_surface_integral_wtopo(xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])

       B_th[:,i]=compute_analytical_solution(xm,ym,zm,sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)
       #print(B_th[:,i]) 
       #print(B_vi[:,i]) 
       #print(B_si[:,i]) 
    
       linefile.write("%e %e %e %e %e %e %e %e %e %e %e %e \n" %(xm,ym,zm,\
                                                       B_vi[0,i],B_vi[1,i],B_vi[2,i],\
                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
                                                       B_th[0,i],B_th[1,i],B_th[2,i]))

   export_line_measurements(line_nmeas,x_meas,y_meas,z_meas,'line_measurements.vtu',B_vi,B_si,B_th)

print('========================================')

###############################################################################
# measuring B on a path
###############################################################################

print('========================================')

if do_path_measurements:

   print('starting path measurement ...')

   linefile=open("measurements_path.ascii","w")
   linefile.write("# 1,2,3, 4    , 5    , 6    , 7    , 8    , 9    , 10   , 11   , 12    \n")
   linefile.write("# x,y,z, Bx_vi, By_vi, Bz_vi, Bx_si, By_si, Bz_si, Bx_th, By_th, Bz_th \n")

   B_vi=np.zeros((3,npath),dtype=np.float64)
   B_si=np.zeros((3,npath),dtype=np.float64)
   B_th=np.zeros((3,npath),dtype=np.float64)

   for i in range(0,npath):
       print('doing',i,'out of ',npath) 
       xm=xpath[i]
       ym=ypath[i]
       zm=zpath[i]
       #print(xm,ym,zm)
       for iel in range(0,nel):
           B_vi[:,i]+=compute_B_quadrature      (xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
           B_si[:,i]+=compute_B_surface_integral_wtopo(xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])

       #print(B_vi[:,i]) 
       #print(B_si[:,i]) 

       B_vi[:,i]*=1e-7 #AEH check !
       B_si[:,i]*=1e-7
    
       linefile.write("%e %e %e %e %e %e %e %e %e  \n" %(xm,ym,zm,\
                                                         B_vi[0,i],B_vi[1,i],B_vi[2,i],\
                                                         B_si[0,i],B_si[1,i],B_si[2,i]))

   export_line_measurements(npath,xpath,ypath,zpath,'path_measurements.vtu',B_vi,B_si,B_th)

print('========================================')


###############################################################################

if do_spiral_measurements:

   x_spiral = np.zeros(npts_spiral,dtype=np.float64)  
   y_spiral = np.zeros(npts_spiral,dtype=np.float64)  
   z_spiral = np.zeros(npts_spiral,dtype=np.float64)  
   r_spiral = np.zeros(npts_spiral,dtype=np.float64)  
   theta_spiral = np.zeros(npts_spiral,dtype=np.float64)  
   phi_spiral = np.zeros(npts_spiral,dtype=np.float64)  

   golden_ratio = (1. + np.sqrt(5.))/2.
   golden_angle = 2. * np.pi * (1. - 1./golden_ratio)

   for i in range(0,npts_spiral):
       r_spiral[i] = radius_spiral
       theta_spiral[i] = np.arccos(1. - 2. * i / (npts_spiral - 1.))
       phi_spiral[i] = np.fmod((i*golden_angle), 2.*np.pi)

   x_spiral[:]=r_spiral[:]*np.sin(theta_spiral[:])*np.cos(phi_spiral[:])+sphere_xc
   y_spiral[:]=r_spiral[:]*np.sin(theta_spiral[:])*np.sin(phi_spiral[:])+sphere_yc
   z_spiral[:]=r_spiral[:]*np.cos(theta_spiral[:])                      +sphere_zc

   spiralfile=open("measurements_spiral.ascii","w")
   spiralfile.write("# 1,2,3,4    ,5    ,6    ,7    ,8    ,9    ,10   ,11   ,12    \n")
   spiralfile.write("# x,y,z,Bx_vi,By_vi,Bz_vi,Bx_si,By_si,Bz_si,Bx_th,By_th,Bz_th \n")

   B_vi=np.zeros((3,npts_spiral),dtype=np.float64)
   B_si=np.zeros((3,npts_spiral),dtype=np.float64)
   B_th=np.zeros((3,npts_spiral),dtype=np.float64)

   for i in range(0,npts_spiral):
       print('doing',i,'out of ',npts_spiral) 

       start = time.time()
       for iel in range(0,nel):
           B_vi[:,i]+=compute_B_quadrature (x_spiral[i],y_spiral[i],z_spiral[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
       print("vol int: %.3f s" % (time.time() - start))

       start = time.time()
       for iel in range(0,nel):
           B_si[:,i]+=compute_B_surface_integral_wtopo(x_spiral[i],y_spiral[i],z_spiral[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])
       print("surf int: %.3f s" % (time.time() - start))

       B_th[:,i]=compute_analytical_solution(x_spiral[i],y_spiral[i],z_spiral[i],sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)
    
       spiralfile.write("%e %e %e %e %e %e %e %e %e %e %e %e \n" %(x_spiral[i],y_spiral[i],z_spiral[i],\
                                                       B_vi[0,i],B_vi[1,i],B_vi[2,i],\
                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
                                                       B_th[0,i],B_th[1,i],B_th[2,i]))

   export_spiral_measurements(npts_spiral,x_spiral,y_spiral,z_spiral,'spiral_measurements.vtu',B_vi,B_si,B_th)

print('========================================')

