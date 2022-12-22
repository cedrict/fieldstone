import numpy as np
from magnetostatics import *
from tools import *
import random
import time as time

#------------------------------------------------------------------------------
# this function returns a topography value at each point x,y passed as argument

def topography(x,y,A,llambda,dx,dy,slopex,slopey):
    pert1=A*np.sin(2*np.pi/llambda*(x*dx+y*dy))
    pert2=slopex*x+slopey*y 
    return pert1+pert2

#------------------------------------------------------------------------------
# returns analytical solution (vector B) 

def compute_analytical_solution(x,y,z,R,Mx,My,Mz,xcenter,ycenter,zcenter,benchmark):

   if benchmark==1: #--------------------------------------
      mu0=4*np.pi #*1e-7
      V=4/3*np.pi*R**3
      r=np.sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)
      Bx=0
      By=0
      Bz=2*mu0*V/4/np.pi/r**3*Mz0

   if benchmark==2 or benchmark==4: #----------------------
      Bx=0
      By=0
      Bz=0

   if benchmark==3: #--------------------------------------
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

###############################################################################
#benchmark:
#1: dipole (small sphere, far away), line measurement
#2: random perturbation internal nodes -> checks cancellation of internal faces 
#3: sphere (larger sphere, anywhere in space) analytical
#4: wavy surface, domain with constant M vector

benchmark=4

###############################################################################
# be careful with the position of the measurement points for 
# benchmark 1. These cannot be above the center of the sphere
# but also not above a diagonal of an element (see ...wtopo)

if benchmark==1:
   Lx=2
   Ly=2
   Lz=2
   nelx=40
   nely=40
   nelz=40
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

if benchmark==2:
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

if benchmark==3:
   Lx=20
   Ly=20
   Lz=20
   nelx=20
   nely=20
   nelz=20
   Mx0=0
   My0=0
   Mz0=7.5
   sphere_R=10
   sphere_xc=Lx/2
   sphere_yc=Ly/2
   sphere_zc=-Lz/2
   nqdim=4
   #plane meas
   do_plane_measurements=True
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=0.4
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=30
   plane_nny=30
   do_line_measurements=False

if benchmark==4:
   Lx=100
   Ly=100
   Lz=100
   nelx=50
   nely=50
   nelz=12
   Mx0=0
   My0=4
   Mz0=-6
   nqdim=4
   #topography parameters
   wavelength=25
   A=1
   angle=20/180*np.pi
   dx=np.cos(angle)
   dy=np.sin(angle)
   slopex=np.arctan(3/180*np.pi)
   slopey=np.arctan(5/180*np.pi)
   do_plane_measurements=True
   plane_x0=-Lx/2
   plane_y0=-Ly/2
   plane_z0=0.25
   plane_Lx=2*Lx
   plane_Ly=2*Ly
   plane_nnx=30
   plane_nny=30
   do_line_measurements=False
   sphere_R=0
   sphere_xc=0
   sphere_yc=0
   sphere_zc=0
   # to do: code line meas 

   #subbench=1,2,3,4
   #if subbench==1:
   #if subbench==2:
   #if subbench==3:
   #if subbench==4:

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

print('========================================')

###############################################################################
# grid point setup
# if benchmark 2, a small random perturbation is added to the
# z coordinate of the interior nodes.
###############################################################################

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            z[counter]=k*Lz/float(nelz)-Lz
            if i!=0 and j!=0 and k!=0 and i!=nnx-1 and j!=nny-1 and k!=nnz-1 and benchmark==2:
               z[counter]+=random.uniform(-1,+1)*dz
            counter += 1
        #end for
    #end for
#end for

###############################################################################
# connectivity
###############################################################################

icon =np.zeros((8,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
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

###############################################################################
# adding wavy topography to surface and deform the mesh accordingly

if benchmark==4:

   for i in range(0,NV):
       if abs(z[i])<1e-6:
          z[i]+=topography(x[i]-Lx/2,y[i]-Ly/2,A,wavelength,dx,dy,slopex,slopey)

   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               LLz=Lz+topography(x[counter]-Lx/2,y[counter]-Ly/2,A,wavelength,dx,dy,slopex,slopey)
               z[counter]=k*LLz/float(nelz)-Lz
               counter += 1
           #end for
       #end for
   #end for

#################################################################
# prescribe M inside each cell
# for benchmarks 1 and 3, M is zero everywhere except inside
# a sphere of radius sphere_R at location (sphere_xc,sphere_yc,sphere_zc)
# For benchmark 2, M is constant in space and equal to (Mx0,My0,Mz0)

Mx=np.zeros(nel,dtype=np.float64)
My=np.zeros(nel,dtype=np.float64)
Mz=np.zeros(nel,dtype=np.float64)

Mx[:]=Mx0
My[:]=My0
Mz[:]=Mz0

if benchmark==1 or benchmark==3:
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

if benchmark==2 or benchmark==4:
   Mx[:]=Mx0
   My[:]=My0
   Mz[:]=Mz0

export_mesh_3D(NV,nel,x,y,z,icon,'mesh.vtu',Mx,My,Mz)

#################################################################
# plane measurements setup
# the plane originates at (plane_x0,plane_y0,plane_z0) and extends 
# in the x,y directions by plane_Lx,plane_Ly
# note that a small perturbation is added to the x,y coordinates
# so as to avoid that a measurement point lies in the plane
# of an element (vertical) face. 
#################################################################

if do_plane_measurements:

   plane_nmeas=plane_nnx*plane_nny

   plane_nelx=plane_nnx-1
   plane_nely=plane_nny-1
   plane_nel=plane_nelx*plane_nely

   x_meas = np.empty(plane_nmeas,dtype=np.float64)  # x coordinates
   y_meas = np.empty(plane_nmeas,dtype=np.float64)  # y coordinates
   z_meas = np.empty(plane_nmeas,dtype=np.float64)  # y coordinates

   counter = 0
   for j in range(0,plane_nny):
       for i in range(0,plane_nnx):
           x_meas[counter]=plane_x0+(i+random.uniform(-1,+1)*1e-8)*plane_Lx/float(plane_nnx-1) 
           y_meas[counter]=plane_y0+(j+random.uniform(-1,+1)*1e-8)*plane_Ly/float(plane_nny-1) 
           z_meas[counter]=plane_z0
           if benchmark==4:
              z_meas[counter]+=topography(x_meas[counter]-Lx/2,y_meas[counter]-Ly/2,A,wavelength,dx,dy,slopex,slopey)
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

   export_mesh_2D(plane_nmeas,plane_nel,x_meas,y_meas,z_meas,icon_meas,'mesh_meas.vtu')

   print('setup plane measurement points ')

#################################################################
# measuring B on a plane
# Nomenclature:
# _vi: volume integral
# _si: surface integral
# _th: analytical value (if applicable)
# The volume integral is parameterised by the number of quadrature 
# points per dimension nqdim.
# Because the integrand is not a polynomial, the volume integral
# remains a numerical solution, while the surface integral 
# is actually analytical (down to machine precision).
#################################################################
   
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

#################################################################
# measuring B on a line
# the line starts at xstart,ystart,zstart and ends at 
# xend,yend,zend, and is discretised by means of line_nmeas pts
#################################################################

print('========================================')

if do_line_measurements:

   print('starting line measurement ...')

   linefile=open("measurements_line.ascii","w")

   B_vi=np.zeros((3,line_nmeas),dtype=np.float64)
   B_si=np.zeros((3,line_nmeas),dtype=np.float64)
   B_th=np.zeros((3,line_nmeas),dtype=np.float64)

   for i in range(0,line_nmeas):
       print('doing',i,'out of ',line_nmeas) 
       xm=xstart+(xend-xstart)/(line_nmeas-1)*i
       ym=ystart+(yend-ystart)/(line_nmeas-1)*i
       zm=zstart+(zend-zstart)/(line_nmeas-1)*i
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

print('========================================')

