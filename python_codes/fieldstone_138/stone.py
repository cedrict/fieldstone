import numpy as np
from magnetostatics import *
from tools import *
import random
import time as time
from set_measurement_parameters import *

###################################################################################################
# this function adds reference field to measurements and computes Pmag Int,Inc,Dec

def add_referencefield(B0_name,npath,B0,B_si):
    linefile1=open(f"measurements_path_refField{B0_name}.ascii","w")
    linefile1.write("# 1    , 2      , 3      , 4      , 5      , 6      , 7       \n")
    linefile1.write("# dmeas, Bx_siB0, By_siB0, Bz_siB0, In_siB0, Ic_siB0, Dc_siB0 \n")

    B_siB0=np.zeros((3,npath),dtype=np.float64)
    In_siB0=np.zeros((npath),dtype=np.float64)
    Ic_siB0=np.zeros((npath),dtype=np.float64)
    Dc_siB0=np.zeros((npath),dtype=np.float64)
    for i in range(0,npath):      
        B_siB0[0,i]=B_si[1,i]+B0[0] #adding B0 in pmag coor + B_si in model coor
        B_siB0[1,i]=B_si[0,i]+B0[1]
        B_siB0[2,i]=-B_si[2,i]+B0[2]
        
        In_siB0[i]=np.sqrt(B_siB0[0,i]**2+B_siB0[1,i]**2+B_siB0[2,i]**2)
        Ic_siB0[i]=np.arctan2(B_siB0[2,i],np.sqrt(B_siB0[0,i]**2+B_siB0[1,i]**2))/np.pi*180
        Dc_siB0[i]=np.arctan2(B_siB0[1,i],B_siB0[0,i])/np.pi*180
   
        if benchmark=='4':
           linefile1.write("%e %e %e %e %e %e \n" %(B_siB0[0,i], B_siB0[1,i], B_siB0[2,i],\
                                                    In_siB0[i], Ic_siB0[i], Dc_siB0[i]))    #AEH))
        else:
           linefile1.write("%e %e %e %e %e %e %e \n" %(dmeas[i],\
                                                       B_siB0[0,i], B_siB0[1,i], B_siB0[2,i],\
                                                      In_siB0[i], Ic_siB0[i], Dc_siB0[i]))    #AEH))           
    return B_siB0,In_siB0,Ic_siB0,Dc_siB0

###################################################################################################
# this function returns a topography value at each point x,y passed as argument

def topography(x,y,A,llambda,cos_dir,sin_dir,slopex,slopey):
    pert1=A*np.sin(2*np.pi/llambda*(x*cos_dir+y*sin_dir))
    pert2=slopex*x+slopey*y 
    return pert1+pert2

###################################################################################################
# returns analytical solution (vector B) 

def compute_analytical_solution(x,y,z,R,Mx,My,Mz,xcenter,ycenter,zcenter,benchmark):

   #-----------------------------------------------------------------
   if benchmark=='1': 
      mu0=4*np.pi #*1e-7
      V=4/3*np.pi*R**3
      r=np.sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)
      Bx=0
      By=0
      Bz=2*mu0*V/4/np.pi/r**3*Mz

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

###################################################################################################
#benchmark:
#1: dipole (small sphere, far away), line measurement
#2a: random perturbation internal nodes cubic-> checks cancellation of internal faces 
#2b: random perturbation internal nodes pancake-> checks cancellation of internal faces 
#3: sphere (larger sphere, anywhere in space) analytical
#4: Synthetic shapes, wavy surface, domain with constant M vector, or box #AEH
#-1: etna topography
###################################################################################################

benchmark='4'
add_noise=False
Nf=2  #noise amplitude between -Nf and Nf 
flat_bottom=False #only for benchmark=4 and etna (benchmark=-1)

#TODO AEH: check flat bottom + ETNA!

###################################################################################################
# be careful with the position of the measurement points for 
# benchmark 1. These cannot be above the center of the sphere
# but also not above a diagonal of an element (see ...wtopo)

if benchmark=='1':
   Lx=2
   Ly=2
   Lz=2
   nelx=100
   nely=100
   nelz=100
   Mx0=0     # do not change
   My0=0     # do not change
   #Mz0=1     # do not change
   Mz0=7.5
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
   zend=2
   #zend=0.02
   #zend=100
   line_nmeas=100
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
   nelz=50
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

if benchmark=='3':
   Lx=20
   Ly=20
   Lz=20
   nelx=60
   #nelx=120
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
   radius_spiral=1.025*sphere_R
   #radius_spiral=1.05*sphere_R
   npts_spiral=101 #keep odd
   do_path_measurements=False

if benchmark=='4':   
#   Lx=100
#   Ly=100
#   Lz=50
#   nelx=40
#   nely=40
#   nelz=8
   Lx=250
   Ly=250
   Lz=20
#   Lz=Lx*2.4
   nelx=int(Lx*1.5)
   nely=int(Ly*1.5)
   nelz=10

   Mx0=0
   My0=4.085
   Mz0=-6.29
   nqdim=4
   #topography parameters
   wavelength=25
   A=4
   do_plane_measurements=False
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=1
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=30
   plane_nny=30
   do_line_measurements=True
   xstart=0.23+((Lx-50)/2)
   #xstart=0.23
   ystart=Ly/2-0.221
   zstart=1      #slightly above surface
   xend=49.19+((Lx-50)/2)
   #xend=49.19
   yend=Ly/2-0.221
   zend=1
   line_nmeas=47
   sphere_R=0
   sphere_xc=0
   sphere_yc=0
   sphere_zc=0
   # to do: code line meas 
   do_spiral_measurements=False
   do_path_measurements=False

   subbench='east'

   if subbench=='east':
      slopex=np.arctan(-6/180*np.pi)
      slopey=np.arctan(0/180*np.pi)
      direction=90/180*np.pi
      xstart=Lx/2-0.221
      xend=Lx/2-0.221
      ystart=0.23+((Ly-50)/2)
      yend=49.19+((Ly-50)/2)

   if subbench=='north':
      slopex=np.arctan(0/180*np.pi)
      slopey=np.arctan(-6/180*np.pi)
      direction=0/180*np.pi

   if subbench=='west':
      slopex=np.arctan(6/180*np.pi)
      slopey=np.arctan(0/180*np.pi)
      direction=90/180*np.pi
      #ystart=0.23
      #yend=49.19
      xstart=Lx/2-0.221
      xend=Lx/2-0.221
      ystart=0.23+((Ly-50)/2)
      yend=49.19+((Ly-50)/2)
      
   if subbench=='south':
      slopex=np.arctan(0/180*np.pi)
      slopey=np.arctan(6/180*np.pi)
      direction=0/180*np.pi

   cos_dir=np.cos(direction)
   sin_dir=np.sin(direction)

   do_path_measurements=False

   IGRF_E=1561.2e-9
   IGRF_N=26850.3e-9
   IGRF_D=36305.7e-9  
   IGRFx=IGRF_N
   IGRFy=IGRF_E
   IGRFz=IGRF_D
   IGRFint=np.sqrt(IGRFx**2+IGRFy**2+IGRFz**2)        
   IGRFinc=np.arctan2(IGRFz,np.sqrt(IGRFx**2+IGRFy**2))/np.pi*180
   IGRFdec=np.arctan2(IGRFy,IGRFx)/np.pi*180

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
#print('NV=',NV)
#print('Mx0,My0,Mz0=',Mx0,My0,Mz0)
#print('nqdim=',nqdim)
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
   if benchmark=='4':
       print('subbench, flank=',subbench)
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
   print('resolution DEM=',rDEM)  
   print('size cut DEM=',sDEM)
   print('IGRFx, Brefx=',IGRFx,Brefx)
   print('IGRFy, Brefy=',IGRFy,Brefy)        
   print('IGRFz, Brefz=',IGRFz,Brefz)
   print('magnetization (Mx,My,Mz)',Mx0,My0,Mz0)
   
   if add_noise:
      print(f'noise added to DEM with a noise factor of: {Nf}')
   else:
      print('no noise added')

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
# adding synthetic topography to surface and deform the mesh accordingly

if benchmark=='4':   

   if flat_bottom:

      for i in range(0,NV):
          if abs(z[i])<1e-6: #top layer of nodes!
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

   else:

      counter=0
      for i in range(0,nnx):
          for j in range(0,nny):
              for k in range(0,nnz):
                  z[counter]+=topography(x[counter]-Lx/2,y[counter]-Ly/2,A,wavelength,cos_dir,sin_dir,slopex,slopey)
                  counter += 1
              #end for
          #end for
      #end for


   print('add synthetic topography')

###############################################################################
# adding topography based on DEM, and reading in measurement points from field data
# uses "etna.py" for importing values
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

   if flat_bottom:
      counter=0
      for i in range(0,nnx):
          for j in range(0,nny):
              for k in range(0,nnz):
                  zmax=Lz+ztopo[j*nnx+i]
                  zmin= 0
                  z[counter]=k*(zmax-zmin)/float(nelz)+zmin-Lz
                  counter += 1
              #end for
          #end for
      #end for

   else:

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
   zpathO = np.empty(npath,dtype=np.float64)  # z coordinates original
   Ic_m = np.empty(npath,dtype=np.float64)  # measured inclination
   Dc_m = np.empty(npath,dtype=np.float64)  # measured declination
   In_m = np.empty(npath,dtype=np.float64)  # measured intensity (in microT)
   dmeas_m = np.empty(npath,dtype=np.float64)  # measured intensity (in microT)
   for i in range(0,npath):
       #reading lines backwards bc of how file is built #AEH NO, only DEM backwards, these are the field meas files... 
       line=lines_path[i].strip()   #AEH
       columns=line.split()
       xpath[i]=columns[1]
       ypath[i]=columns[2]
       zpath[i]=columns[3]
       zpathO[i]=zpath[i]
       Ic_m[i]=columns[4]
       Dc_m[i]=columns[5]
       In_m[i]=columns[6]
       dmeas_m[i]=columns[7]

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

   export_path_measurements(npath,xpath,ypath,zpath,'path.vtu')

###############################################################################
# prescribe M inside each cell
# for benchmarks 1 and 3, M is zero everywhere except inside
# a sphere of radius sphere_R at location (sphere_xc,sphere_yc,sphere_zc)
# we use the center of an element as a representative point.
# For benchmark 2a,2b,4 and Etna, M is constant in space and equal to (Mx0,My0,Mz0)
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

   export_mesh_2D(plane_nmeas,plane_nel,x_meas,y_meas,z_meas,icon_meas,'mesh_plane_measurements.vtu')

   np.savetxt('mesh_plane_measurements.ascii',np.array([x_meas,y_meas,z_meas]).T)

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
    #   #print('x,y,z meas',x_meas[i],y_meas[i],z_meas[i])
    #   B_th[:,i]=compute_analytical_solution(x_meas[i],y_meas[i],z_meas[i],sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)
    #   #print('analytical ->',B_th[:,i])

     #  start = time.time()
     #  for iel in range(0,nel):
     #      B_vi[:,i]+=compute_B_quadrature(x_meas[i],y_meas[i],z_meas[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
     #  print("vol int: %.3f s" % (time.time() - start))
     #  #print('vol int    ->',B_vi[:,i])

       start = time.time()
       for iel in range(0,nel):
           B_si[:,i]+=compute_B_surface_integral_wtopo(x_meas[i],y_meas[i],z_meas[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])
       print("surf int: %.3f s" % (time.time() - start))
       #print('surf int   ->',B_si[:,i])

   export_plane_measurements(plane_nmeas,plane_nel,x_meas,y_meas,z_meas,icon_meas,'plane_measurements.vtu',B_si,B_si,B_si)

   np.savetxt('plane_measurements.ascii',np.array([x_meas,y_meas,z_meas,\
                                                   B_si[0,:],B_si[1,:],B_si[2,:],\
                                                   B_si[0,:],B_si[1,:],B_si[2,:]]).T)

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

#   linefile=open("measurements_line.ascii","w")
#   linefile.write("# 1,2,3, 4    , 5    , 6    , 7    , 8    , 9    , 10   , 11   , 12    \n")
#   linefile.write("# x,y,z, Bx_vi, By_vi, Bz_vi, Bx_si, By_si, Bz_si, Bx_th, By_th, Bz_th \n")
   linefile=open("measurements_line.ascii","w")
   linefile.write("# 1,2,3, 4    , 5    , 6    , 7    , 8    , 9     \n")
   linefile.write("# x,y,z, Bx_si, By_si, Bz_si, Bx_th, By_th, Bz_th \n")
   if benchmark=='4':
      linefile1=open("measurements_line_plotfile.ascii","w")
      linefile1.write("# 1 , 2 , 3 , 4      , 5      , 6      , 7      , 8      , 9        \n")
      linefile1.write("# xm, ym, zm, IGRF_In, IGRF_Ic, IGRF_Dc, In_siB0, Ic_siB0, Dc_siB0  \n")
    
 #   B_vi=np.zeros((3,line_nmeas),dtype=np.float64)
   B_si=np.zeros((3,line_nmeas),dtype=np.float64)
   B_th=np.zeros((3,line_nmeas),dtype=np.float64)

   for i in range(0,line_nmeas):
       print('doing',i,'out of ',line_nmeas) 
       xm=xstart+(xend-xstart)/(line_nmeas-1)*i
       ym=ystart+(yend-ystart)/(line_nmeas-1)*i
       zm=zstart+(zend-zstart)/(line_nmeas-1)*i
       if benchmark=='4':
          zm+=topography(xm-Lx/2,ym-Ly/2,A,\
                         wavelength,cos_dir,sin_dir,slopex,slopey)
      
       x_meas[i]=xm
       y_meas[i]=ym
       z_meas[i]=zm
       #print(xm,ym,zm)
       for iel in range(0,nel):
#           B_vi[:,i]+=compute_B_quadrature      (xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
           B_si[:,i]+=compute_B_surface_integral_wtopo(xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])

#       B_th[:,i]=compute_analytical_solution(xm,ym,zm,sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)

       if benchmark=='4':
          B0=np.array([IGRFx,IGRFy,IGRFz])
          B0_name="IGRF"
          (B_siB0,In_siB0,Ic_siB0,Dc_siB0)=add_referencefield(B0_name,line_nmeas,B0,B_si)
          linefile1.write("%e %e %e %e %e %e %e %e %e \n" %(xm, ym, zm,\
                                                            IGRFint,IGRFinc,IGRFdec,\
                                                            In_siB0[i],Ic_siB0[i],Dc_siB0[i]))
       else:
          linefile.write("%e %e %e %e %e %e %e %e %e \n" %(xm,ym,zm,\
                                                           B_si[0,i],B_si[1,i],B_si[2,i],\
                                                           B_th[0,i],B_th[1,i],B_th[2,i]))
#       linefile.write("%e %e %e %e %e %e %e %e %e %e %e %e \n" %(xm,ym,zm,\
#                                                       B_vi[0,i],B_vi[1,i],B_vi[2,i],\
#                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
#                                                       B_th[0,i],B_th[1,i],B_th[2,i]))       

   export_line_measurements(line_nmeas,x_meas,y_meas,z_meas,'line_measurements.vtu',B_th,B_si,B_th)
#   export_line_measurements(line_nmeas,x_meas,y_meas,z_meas,'line_measurements.vtu',B_vi,B_si,B_th)

print('========================================')

###################################################################################################
# measuring B on a path
###################################################################################################

print('========================================')
# Volume integral is always too time consuming here so excluded, B_th does not exist
if do_path_measurements:

   print('starting path measurement ...')

   linefile1=open("measurements_path.ascii","w")
   linefile1.write("# 1, 2, 3, 4   , 5   , 6   , 7   , 8   , 9   , 10  \n")
   linefile1.write("# x, y, z, Bx_si, By_si, Bz_si, In_si, Ic_si, Dc_si, dmeas \n")

   B_si=np.zeros((3,npath),dtype=np.float64)

   In_si=np.zeros((npath),dtype=np.float64)
   Ic_si=np.zeros((npath),dtype=np.float64)
   Dc_si=np.zeros((npath),dtype=np.float64)
   dmeas=np.zeros((npath),dtype=np.float64)

   for i in range(0,npath):
       print('doing',(i+1),'out of ',npath) 
       xm=xpath[i] #xmeas
       ym=ypath[i] #ymeas
       zm=zpath[i] #zmeas
       #print(xm,ym,zm)
       for iel in range(0,nel):
           if add_noise:
              if iel==0: 
                 Noise=random.uniform(-1,+1)*Nf
                 # print("element nr:",iel)
                 # print("Noise:",Noise)
              elif iel%nelz==0: 
                 Noise=random.uniform(-1,+1)*Nf
                 # print("element nr:",iel)
                 # print("Noise:",Noise)

              B_si[:,i]+=compute_B_surface_integral_wtopo_noise(xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],Noise)

           else:    
              B_si[:,i]+=compute_B_surface_integral_wtopo(xm,ym,zm,x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])
 
       ## PROCESSING THE DATA, note converted to Pmag axis (N,E,D as X,Y,Z), pmag xyz /= model xyz, modified equations
       In_si[i]=np.sqrt(B_si[0,i]**2+B_si[1,i]**2+B_si[2,i]**2)
       Ic_si[i]=np.arctan2(-B_si[2,i],np.sqrt(B_si[1,i]**2+B_si[0,i]**2))/np.pi*180
       Dc_si[i]=np.arctan2(B_si[0,i],B_si[1,i])/np.pi*180    
       print('Obs point number=',(i+1), "Bx=",B_si[1,i], "By=",B_si[0,i], "Bz=",-B_si[2,i]) 
       print('Intensity Path:',In_si[i],'Inclination Path:',Ic_si[i],'Declination Path:',Dc_si[i])
           
       if i==0:
          dmeas[i]=0
       else:
          dmeas[i]=np.sqrt((xpath[i]-xpath[i-1])**2+(ypath[i]-ypath[i-1])**2)+dmeas[i-1]
       linefile1.write("%e %e %e %e %e %e %e %e %e %e \n" %(xm,ym,zm,\
                                                            B_si[1,i],B_si[0,i],-B_si[2,i],\
                                                            In_si[i],Ic_si[i],Dc_si[i],\
                                                            dmeas[i] ))    #AEH
       
   export_line_measurements(npath,xpath,ypath,zpath,'path_measurements.vtu',B_si,B_si,B_si)    #did not want to change tool, so useless 3x B_si

print('========================================')

#TO DO: add path hight diff to other option    #AEH

###################################################################################################
# STATISTICS
# and adding reference fields
###################################################################################################

print('========================================')
if do_path_measurements:
   B0=np.array([IGRFx,IGRFy,IGRFz])
   B0_name="IGRF"
#  B0=np.array([Brefx,Brefy,Brefz])
#  B0_name="Bref"
   print('starting processing: adding reference field and statistics ...')
   (B_siB0,In_siB0,Ic_siB0,Dc_siB0)=add_referencefield(B0_name,npath,B0,B_si)
    
   linefile1=open("statistics_IGRF.ascii","w")
   linefile1.write("# 1   , 2   , 3    , 4          , 5          , 6    , 7     \n")
   linefile1.write("# data, Mean, stDEV, min In_siB0, max In_siB0, minus, maxus \n")
# INT
   MeanIn=np.mean(In_siB0) 
   StdevIn=np.std(In_siB0)
   linefile1.write("int %e %e %e %e %e %e \n" %(MeanIn*1e6,StdevIn*1e6,\
                                             min(In_siB0)*1e6, max(In_siB0)*1e6,\
                                             -(MeanIn-min(In_siB0))*1e6,(max(In_siB0)-MeanIn)*1e6 ))                                            
   print('Mean Int IGRF added=',MeanIn)
   print('stDEV Int IGRF added=',StdevIn)
# INC
   MeanIc=np.mean(Ic_siB0)
   StdevIc=np.std(Ic_siB0)
   linefile1.write("inc %e %e %e %e %e %e \n" %(MeanIc,StdevIc,\
                                             min(Ic_siB0), max(Ic_siB0),\
                                             -(MeanIc-min(Ic_siB0)),max(Ic_siB0)-MeanIc))                                                
   print('Mean Inc IGRF added=',MeanIc)
   print('stDEV Inc IGRF added=',StdevIc)
# DEC
   MeanDc=np.mean(Dc_siB0)
   StdevDc=np.std(Dc_siB0)
   linefile1.write("dec %e %e %e %e %e %e \n" %(MeanDc,StdevDc,\
                                             min(Dc_siB0), max(Dc_siB0),\
                                             -(MeanDc-min(Dc_siB0)),max(Dc_siB0)-MeanDc))                                        
   print('Mean Dec IGRF added=',MeanDc)
   print('stDEV Dec IGRF added=',StdevDc)

###################################################################################################
# write one plot file for nice vis

if do_path_measurements:
   if benchmark=='-1' and rDEM==2:
      if site==1 or site==2 or site==3 or site==5:
         poh=8.5 #offset from height of path to height of DEM
      else:
         poh=0
   elif benchmark=='-1' and rDEM==5:
      if site==1 or site==2 or site==5:
         poh=7.5
      elif site==4:
         poh=3
      elif site==6:
         poh=8.5
      else:
         poh=0
    
   linefile1=open("measurements_path_plotfile.ascii","w")
   linefile1.write("# 1      , 2      , 3      , 4    , 5     , 6      , 7      , 8      , 9              , 10  , 11  , 12  , 13    , 14      \n")
   linefile1.write("# IGRF_In, IGRF_Ic, IGRF_Dc, dmeas, height, In_siB0, Ic_siB0, Dc_siB0, height m (-poh), In_m, Ic_m, Dc_m, min hP, max hP  \n")
    
   IGRFint=np.sqrt(IGRFx**2+IGRFy**2+IGRFz**2)        
   IGRFinc=np.arctan2(IGRFz,np.sqrt(IGRFx**2+IGRFy**2))/np.pi*180
   IGRFdec=np.arctan2(IGRFy,IGRFx)/np.pi*180
   for i in range(0,npath):
       linefile1.write("%e %e %e %e %e %e %e %e %e %e %e %e %e %e \n" %(IGRFint,IGRFinc,IGRFdec,\
                                                                        dmeas[i],zpath[i],In_siB0[i],\
                                                                        Ic_siB0[i],Dc_siB0[i],zpathO[i]-poh,\
                                                                        In_m[i],Ic_m[i],Dc_m[i],min(zpath),\
                                                                        max(zpath)))

###################################################################################################

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

 #      start = time.time()
 #      for iel in range(0,nel):
 #          B_vi[:,i]+=compute_B_quadrature (x_spiral[i],y_spiral[i],z_spiral[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel],nqdim)
 #      print("vol int: %.3f s" % (time.time() - start))

       start = time.time()
       for iel in range(0,nel):
           B_si[:,i]+=compute_B_surface_integral_wtopo(x_spiral[i],y_spiral[i],z_spiral[i],x,y,z,icon[:,iel],Mx[iel],My[iel],Mz[iel])
       print("surf int: %.3f s" % (time.time() - start))

       B_th[:,i]=compute_analytical_solution(x_spiral[i],y_spiral[i],z_spiral[i],sphere_R,Mx0,My0,Mz0,sphere_xc,sphere_yc,sphere_zc,benchmark)
    
       spiralfile.write("%e %e %e %e %e %e %e %e %e %e %e %e \n" %(x_spiral[i],y_spiral[i],z_spiral[i],\
                                                       B_vi[0,i],B_vi[1,i],B_vi[2,i],\
                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
                                                       B_th[0,i],B_th[1,i],B_th[2,i]))
#       spiralfile.write("%e %e %e %e %e %e %e %e %e %e %e %e \n" %(x_spiral[i],y_spiral[i],z_spiral[i],\
#                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
#                                                       B_si[0,i],B_si[1,i],B_si[2,i],\
#                                                       B_th[0,i],B_th[1,i],B_th[2,i]))

   export_spiral_measurements(npts_spiral,x_spiral,y_spiral,z_spiral,'spiral_measurements.vtu',B_si,B_si,B_th)
#   export_spiral_measurements(npts_spiral,x_spiral,y_spiral,z_spiral,'spiral_measurements.vtu',B_vi,B_si,B_th)

print('========================================')

###################################################################################################
