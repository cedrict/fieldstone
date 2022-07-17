import numpy as np
import matplotlib.pyplot as plt
import tetrahedron as th
import time 
import numba
from numba import jit

xx=0
yy=1
zz=2

Ggrav=6.6743e-11

###############################################################################
# function which computes the gravity fields at a given point x,y,z
# using the mass points at xC,yC,zC and their volume
# density and G not taken into account

@jit(nopython=True)
def compute_g_at_point(x,y,z,nlat,nlon,xC,yC,zC,volume):
    ggx=0.
    ggy=0.
    ggz=0.
    UUU=0.
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            distx=x-xC[ilat,ilon]
            disty=y-yC[ilat,ilon]
            distz=z-zC[ilat,ilon]
            dist=np.sqrt(distx**2+disty**2+distz**2)
            K=volume[ilat,ilon]/dist**3
            ggx+=K*distx
            ggy+=K*disty
            ggz+=K*distz
            UUU-=volume[ilat,ilon]/dist
        #end for
    #end for
    return ggx,ggy,ggz,UUU

###############################################################################
# resolution:
# 1: 1 degree
# 2: 0.5 degree
# 4: 0.25 degree
# 16: 0.0625 degree
###############################################################################

resolution = 1

if resolution==1:
   nlon=360
   nlat=180
   latmin=89.5
   latmax=-89.5
   lonmin=0.5
   lonmax=359.5
   resol=1

if resolution==2:
   nlon=360*2
   nlat=180*2
   latmin=89.75
   latmax=-89.75
   lonmin=0.25
   lonmax=359.75
   resol=0.5

if resolution==4:
   nlon=360*4
   nlat=180*4
   latmin=89.875
   latmax=-89.875
   lonmin=0.125
   lonmax=359.875
   resol=0.25

if resolution==16:
   nlon=360*16
   nlat=180*16
   latmin=89.968750
   latmax=-89.968750
   lonmin=0.03125
   lonmax=359.968750
   resol=0.0625

Rmars=3389.508e3

rho0=3000

Rsat=Rmars+100e3

nlat2=nlat
nlon2=nlon

method_slow=False
method_jit=False

use_tetrahedra=True

simple_shell=True
shell_thickness=300e3

npts=nlon*nlat
npts2=nlon2*nlat2

###############################################################################

dlon=(lonmax-lonmin)/(nlon-1)
dlat=(latmax-latmin)/(nlat-1)

ddlon=abs(dlon)/2
ddlat=abs(dlat)/2

###############################################################################
start = time.time()

topography =np.zeros((nlat,nlon),dtype=np.float64)

if resolution==1:
   file=open("MOLA_1deg.txt", "r")
   print("using MOLA_1deg.txt")
if resolution==2:
   file=open("MOLA_0.5deg.txt", "r")
   print("using MOLA_0.5deg.txt")
if resolution==4:
   file=open("MOLA_0.25deg.txt", "r")
   print("using MOLA_0.25deg.txt")
if resolution==16:
   file=open("MOLA_0.0625deg.txt", "r")
   print("using MOLA_0.0625deg.txt")

lines = file.readlines()
file.close

print("time read file: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        topography[j,i]=float(values[2])
        counter+=1

plt.imshow(topography) ; plt.colorbar()
plt.savefig('topography.pdf', bbox_inches='tight')
plt.clf()

topomin=-8200 #np.min(topography)

if simple_shell:
   topography[:,:]=0.
   topomin=-shell_thickness

print('   -> topography m/M',np.min(topography),np.max(topography),' m')

print("time topo array: %.3f s" % (time.time() - start))

###############################################################################
# compute cell volume
###############################################################################
start = time.time()

cell_volume =np.zeros((nlat,nlon),dtype=np.float64)
rC =np.zeros((nlat,nlon),dtype=np.float64)
thetaC =np.zeros((nlat,nlon),dtype=np.float64)
phiC =np.zeros((nlat,nlon),dtype=np.float64)

resol*=(np.pi/180)

for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lon=lonmin+ilon*dlon
        lat=latmin+ilat*dlat
        phi=lon/180*np.pi
        theta=(90-lat)/180*np.pi
        theta_min=theta-resol/2
        theta_max=theta+resol/2
        Rmin=Rmars+topomin
        Rmax=Rmars+topography[ilat,ilon]
        rC[ilat,ilon]=0.5*(Rmin+Rmax)
        thetaC[ilat,ilon]=theta
        phiC[ilat,ilon]=phi
        cell_volume[ilat,ilon]=(Rmax**3-Rmin**3)/3*(np.cos(theta_min)-np.cos(theta_max))*resol
    #end for 
#end for 

print('   -> volume m/M',np.min(cell_volume),np.max(cell_volume),'m^3')
print('   -> total volume=',np.sum(cell_volume)) 

if simple_shell:
   volume_analytical=4/3*np.pi*(Rmax**3-Rmin**3)
   print('   -> anal. volume=',volume_analytical)

plt.imshow(cell_volume/1e9) ; plt.colorbar()
plt.savefig('cell_volume.pdf', bbox_inches='tight')
plt.clf()

xC=rC*np.sin(thetaC)*np.cos(phiC)
yC=rC*np.sin(thetaC)*np.sin(phiC)
zC=rC*np.cos(thetaC)

print('   -> xC m/M: %.2f %.2f m' %(np.min(xC),np.max(xC)))
print('   -> yC m/M: %.2f %.2f m' %(np.min(yC),np.max(yC)))
print('   -> zC m/M: %.2f %.2f m' %(np.min(zC),np.max(zC)))

print("time cell volume: %.3f s" % (time.time() - start))

#########################################################################################
# store coordinates of cell vertices
#########################################################################################
start = time.time()

pt1=np.zeros((3,npts),dtype=np.float64)
pt2=np.zeros((3,npts),dtype=np.float64)
pt3=np.zeros((3,npts),dtype=np.float64)
pt4=np.zeros((3,npts),dtype=np.float64)
pt5=np.zeros((3,npts),dtype=np.float64)
pt6=np.zeros((3,npts),dtype=np.float64)
pt7=np.zeros((3,npts),dtype=np.float64)
pt8=np.zeros((3,npts),dtype=np.float64)

counter=0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lon=lonmin+ilon*dlon
        lat=latmin+ilat*dlat

        radius=Rmars+topomin

        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        pt1[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt1[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt1[zz,counter]=radius*np.cos(theta)

        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        pt2[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt2[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt2[zz,counter]=radius*np.cos(theta)

        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        pt3[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt3[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt3[zz,counter]=radius*np.cos(theta)

        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        pt4[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt4[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt4[zz,counter]=radius*np.cos(theta)

        radius=Rmars+topography[ilat,ilon]

        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        pt5[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt5[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt5[zz,counter]=radius*np.cos(theta)

        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        pt6[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt6[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt6[zz,counter]=radius*np.cos(theta)

        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        pt7[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt7[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt7[zz,counter]=radius*np.cos(theta)

        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        pt8[xx,counter]=radius*np.sin(theta)*np.cos(phi)
        pt8[yy,counter]=radius*np.sin(theta)*np.sin(phi)
        pt8[zz,counter]=radius*np.cos(theta)

        counter+=1

    #end for
#end for

print("time vertices coords: %.3f s" % (time.time() - start))

#########################################################################################
# compute coordinates of measurement satellite points
#########################################################################################
start = time.time()

xM =np.zeros((nlat2,nlon2),dtype=np.float64)
yM =np.zeros((nlat2,nlon2),dtype=np.float64)
zM =np.zeros((nlat2,nlon2),dtype=np.float64)

for ilat2 in range(0,nlat2):
    for ilon2 in range(0,nlon2): 
        phi2=360/nlon2*(ilon2+0.5)  * np.pi/180
        theta2=180/nlat2*(ilat2+0.5)* np.pi/180
        xM[ilat2,ilon2]=Rsat*np.sin(theta2)*np.cos(phi2)
        yM[ilat2,ilon2]=Rsat*np.sin(theta2)*np.sin(phi2)
        zM[ilat2,ilon2]=Rsat*np.cos(theta2)
    #end for
#end for

print('   -> xM m/M: %.2f %.2f m' %(np.min(xM),np.max(xM)))
print('   -> yM m/M: %.2f %.2f m' %(np.min(yM),np.max(yM)))
print('   -> zM m/M: %.2f %.2f m' %(np.min(zM),np.max(zM)))

print("time xyzM setup: %.3f s" % (time.time() - start))

#########################################################################################
# computing gravity at satellite orbit 
#########################################################################################
if method_slow:

   start = time.time()

   gx =np.zeros((nlat2,nlon2),dtype=np.float64)
   gy =np.zeros((nlat2,nlon2),dtype=np.float64)
   gz =np.zeros((nlat2,nlon2),dtype=np.float64)
   UU =np.zeros((nlat2,nlon2),dtype=np.float64)

   for ilat2 in range(0,nlat2):
       for ilon2 in range(0,nlon2): # loop over measurement points
           for ilat in range(0,nlat):
               for ilon in range(0,nlon): # loop over cells
                   distx=xM[ilat2,ilon2]-xC[ilat,ilon]
                   disty=yM[ilat2,ilon2]-yC[ilat,ilon]
                   distz=zM[ilat2,ilon2]-zC[ilat,ilon]
                   dist=np.sqrt(distx**2+disty**2+distz**2)
                   K=cell_volume[ilat,ilon]/dist**3
                   gx[ilat2,ilon2]+=K*distx
                   gy[ilat2,ilon2]+=K*disty
                   gz[ilat2,ilon2]+=K*distz
                   UU[ilat2,ilon2]-=cell_volume[ilat,ilon]/dist
               #end for
           #end for
       #end for
   #end for

   gx*=Ggrav*rho0
   gy*=Ggrav*rho0
   gz*=Ggrav*rho0
   UU*=Ggrav*rho0

   gg=np.sqrt(gx**2+gy**2+gz**2)

   print('   -> gx m/M: %.5f %.5f ' %(np.min(gx),np.max(gx)))
   print('   -> gy m/M: %.5f %.5f ' %(np.min(gy),np.max(gy)))
   print('   -> gz m/M: %.5f %.5f ' %(np.min(gz),np.max(gz)))
   print('   -> g  m/M: %.5f %.5f ' %(np.min(gg),np.max(gg)))
   print('   -> UU m/M: %.5f %.5f ' %(np.min(UU),np.max(UU)))

   end = time.time()

   print("time gravity calc.: %.3f s | %d %d | %e" % (end-start,npts,npts2,(end-start)/npts/npts2))

   # slightly different approach - 10% faster

   start = time.time()

   gx =np.zeros((nlat2,nlon2),dtype=np.float64)
   gy =np.zeros((nlat2,nlon2),dtype=np.float64)
   gz =np.zeros((nlat2,nlon2),dtype=np.float64)
   UU =np.zeros((nlat2,nlon2),dtype=np.float64)

   for ilat2 in range(0,nlat2):
       for ilon2 in range(0,nlon2): # loop over measurement points
           ggx=0
           ggy=0
           ggz=0
           UUU=0
           for ilat in range(0,nlat):
               for ilon in range(0,nlon): # loop over cells
                   distx=xM[ilat2,ilon2]-xC[ilat,ilon]
                   disty=yM[ilat2,ilon2]-yC[ilat,ilon]
                   distz=zM[ilat2,ilon2]-zC[ilat,ilon]
                   dist=np.sqrt(distx**2+disty**2+distz**2)
                   K=cell_volume[ilat,ilon]/dist**3
                   ggx+=K*distx
                   ggy+=K*disty
                   ggz+=K*distz
                   UUU-=cell_volume[ilat,ilon]/dist
               #end for
           #end for
           gx[ilat2,ilon2]=ggx
           gy[ilat2,ilon2]=ggy
           gz[ilat2,ilon2]=ggz
           UU[ilat2,ilon2]=UUU
       #end for
   #end for

   gx*=Ggrav*rho0
   gy*=Ggrav*rho0
   gz*=Ggrav*rho0
   UU*=Ggrav*rho0

   gg=np.sqrt(gx**2+gy**2+gz**2)

   print('   -> gx m/M: %.5f %.5f ' %(np.min(gx),np.max(gx)))
   print('   -> gy m/M: %.5f %.5f ' %(np.min(gy),np.max(gy)))
   print('   -> gz m/M: %.5f %.5f ' %(np.min(gz),np.max(gz)))
   print('   -> g  m/M: %.5f %.5f ' %(np.min(gg),np.max(gg)))
   print('   -> UU m/M: %.5f %.5f ' %(np.min(UU),np.max(UU)))

   end = time.time()

   print("time gravity calc.: %.3f s | %d %d | %e" % (end-start,npts,npts2,(end-start)/npts/npts2))

##################################################################
# compute gravity by means of function + jit
##################################################################

if method_jit:

   start = time.time()

   gx =np.zeros((nlat2,nlon2),dtype=np.float64)
   gy =np.zeros((nlat2,nlon2),dtype=np.float64)
   gz =np.zeros((nlat2,nlon2),dtype=np.float64)
   UU =np.zeros((nlat2,nlon2),dtype=np.float64)

   for ilat2 in range(0,nlat2):
       for ilon2 in range(0,nlon2): 
           gx[ilat2,ilon2],gy[ilat2,ilon2],gz[ilat2,ilon2],UU[ilat2,ilon2]=\
             compute_g_at_point(xM[ilat2,ilon2],yM[ilat2,ilon2],zM[ilat2,ilon2],nlat,nlon,xC,yC,zC,cell_volume)
       #end for
   #end for

   gx*=Ggrav*rho0
   gy*=Ggrav*rho0
   gz*=Ggrav*rho0
   UU*=Ggrav*rho0

   gg=np.sqrt(gx**2+gy**2+gz**2)

   print('   -> gx m/M: %.6f %.6f ' %(np.min(gx),np.max(gx)))
   print('   -> gy m/M: %.6f %.6f ' %(np.min(gy),np.max(gy)))
   print('   -> gz m/M: %.6f %.6f ' %(np.min(gz),np.max(gz)))
   print('   -> g  m/M: %.6f %.6f ' %(np.min(gg),np.max(gg)))
   print('   -> UU m/M: %.6f %.6f ' %(np.min(UU),np.max(UU)))

   end = time.time()

   print("time gravity calc.: %.3f s | %d %d | %e" % (end-start,npts,npts2,(end-start)/npts/npts2))

   plt.imshow(gx) ; plt.colorbar() ; plt.savefig('gx.pdf', bbox_inches='tight') ; plt.clf()
   plt.imshow(gy) ; plt.colorbar() ; plt.savefig('gy.pdf', bbox_inches='tight') ; plt.clf()
   plt.imshow(gz) ; plt.colorbar() ; plt.savefig('gz.pdf', bbox_inches='tight') ; plt.clf()
   plt.imshow(gg) ; plt.colorbar() ; plt.savefig('gg.pdf', bbox_inches='tight') ; plt.clf()
   plt.imshow(UU) ; plt.colorbar() ; plt.savefig('UU.pdf', bbox_inches='tight') ; plt.clf()

#########################################################################################
#########################################################################################

start = time.time()

nlat4=1000

xP =np.zeros((nlat4),dtype=np.float64)
yP =np.zeros((nlat4),dtype=np.float64)
zP =np.zeros((nlat4),dtype=np.float64)
thetaP =np.zeros((nlat4),dtype=np.float64)
gx4 =np.zeros((nlat4),dtype=np.float64)
gy4 =np.zeros((nlat4),dtype=np.float64)
gz4 =np.zeros((nlat4),dtype=np.float64)
UU4 =np.zeros((nlat4),dtype=np.float64)
    
phiP=226.2/180*np.pi

for ilat4 in range(0,nlat4): 
    thetaP[ilat4]=180/nlat4*(ilat4+0.5)* np.pi/180
    xP[ilat4]=Rsat*np.sin(thetaP[ilat4])*np.cos(phiP)
    yP[ilat4]=Rsat*np.sin(thetaP[ilat4])*np.sin(phiP)
    zP[ilat4]=Rsat*np.cos(thetaP[ilat4])
    gx4[ilat4],gy4[ilat4],gz4[ilat4],UU4[ilat4]=\
       compute_g_at_point(xP[ilat4],yP[ilat4],zP[ilat4],nlat,nlon,xC,yC,zC,cell_volume)
    #end for
#end for

gx4*=Ggrav*rho0
gy4*=Ggrav*rho0
gz4*=Ggrav*rho0
UU4*=Ggrav*rho0

gg4=np.sqrt(gx4**2+gy4**2+gz4**2)

np.savetxt('gravity_above_olympus_mons.ascii',np.array([xP,yP,zP,gg4,thetaP]).T)

end = time.time()

print("time grav. above olympus: %.3f s" % (end-start))

#########################################################################################
# compute coordinates of measurement line points
#########################################################################################
start = time.time()

if simple_shell:

   npts3=50

   xN=np.zeros(npts3,dtype=np.float64)
   yN=np.zeros(npts3,dtype=np.float64)
   zN=np.zeros(npts3,dtype=np.float64)
   gx3=np.zeros(npts3,dtype=np.float64)
   gy3=np.zeros(npts3,dtype=np.float64)
   gz3=np.zeros(npts3,dtype=np.float64)
   UU3=np.zeros(npts3,dtype=np.float64)
   gga=np.zeros(npts3,dtype=np.float64)

   for i in range(0,npts3):
       phi3=np.pi*0.123456789
       theta3=np.pi*0.123456789
       Rmeas=i*3*Rmars/(npts3-1)
       xN[i]=Rmeas*np.sin(theta3)*np.cos(phi3)
       yN[i]=Rmeas*np.sin(theta3)*np.sin(phi3)
       zN[i]=Rmeas*np.cos(theta3)
       gx3[i],gy3[i],gz3[i],UU3[i]=\
             compute_g_at_point(xN[i],yN[i],zN[i],nlat,nlon,xC,yC,zC,cell_volume)

   #end for

   gx3*=Ggrav*rho0
   gy3*=Ggrav*rho0
   gz3*=Ggrav*rho0
   UU3*=Ggrav*rho0

   gg3=np.sqrt(gx3**2+gy3**2+gz3**2)
   rN=np.sqrt(xN**2+yN**2+zN**2)

   for i in range(0,npts3):
       if rN[i]<=Rmars-shell_thickness:
          gga[i]=0
       elif rN[i]<=Rmars:
          gga[i]=4*np.pi/3*Ggrav*rho0*(rN[i]-(Rmars-shell_thickness)**3/rN[i]**2)
       else:
          gga[i]=Ggrav*rho0*volume_analytical/rN[i]**2

   np.savetxt('gravity_on_line_pm.ascii',np.array([rN,gg3,UU3,gga]).T)

   print("time grav on line with point mass: %.3f s" % (time.time() - start))

#########################################################################################
start = time.time()

if simple_shell and use_tetrahedra:

   gx5=np.zeros(npts3,dtype=np.float64)
   gy5=np.zeros(npts3,dtype=np.float64)
   gz5=np.zeros(npts3,dtype=np.float64)
   UU5=np.zeros(npts3,dtype=np.float64)

   for i in range(0,npts3):

       print(i,npts3)

       pt_meas = np.array([xN[i],yN[i],zN[i]],dtype=np.float64)

       for ic in range(0,npts):
           
           th.export_hexahedron_to_vtu(pt1[:,ic],pt2[:,ic],pt3[:,ic],pt4[:,ic],pt5[:,ic],pt6[:,ic],pt7[:,ic],pt8[:,ic],'hh')

           # first tetrahedron 
           #g5,UUU5=th.compute_gravity_at_point_th(pt2[:,ic],pt3[:,ic],pt6[:,ic],pt1[:,ic],pt_meas,rho0)
           #gx5[i]+=g5[xx]
           #gy5[i]+=g5[yy]
           #gz5[i]+=g5[zz]
           #UU5[i]+=UUU5

           # second tetrahedron 
           th.export_tetrahedron_to_vtu(pt3[:,ic],pt8[:,ic],pt1[:,ic],pt4[:,ic],'th_02.vtu')
           g5,UUU5=th.compute_gravity_at_point_th(pt3[:,ic],pt8[:,ic],pt1[:,ic],pt4[:,ic],pt_meas,rho0)
           gx5[i]+=g5[xx]
           gy5[i]+=g5[yy]
           gz5[i]+=g5[zz]
           UU5[i]+=UUU5

           # third tetrahedron 
           g5,UUU5=th.compute_gravity_at_point_th(pt5[:,ic],pt1[:,ic],pt6[:,ic],pt8[:,ic],pt_meas,rho0)
           gx5[i]+=g5[xx]
           gy5[i]+=g5[yy]
           gz5[i]+=g5[zz]
           UU5[i]+=UUU5

           # fourth tetrahedron 
           #g5,UUU5=th.compute_gravity_at_point_th(pt7[:,ic],pt8[:,ic],pt6[:,ic],pt3[:,ic],pt_meas,rho0)
           #gx5[i]+=g5[xx]
           #gy5[i]+=g5[yy]
           #gz5[i]+=g5[zz]
           #UU5[i]+=UUU5

           # fifth tetrahedron 
           g5,UUU5=th.compute_gravity_at_point_th(pt3[:,ic],pt6[:,ic],pt1[:,ic],pt8[:,ic],pt_meas,rho0)
           gx5[i]+=g5[xx]
           gy5[i]+=g5[yy]
           gz5[i]+=g5[zz]
           UU5[i]+=UUU5

           #end for
       #end for

   #end for

   gg5=np.sqrt(gx5**2+gy5**2+gz5**2)

   np.savetxt('gravity_on_line_th.ascii',np.array([rN,gg5,UU5,gga]).T)

   print("time grav on line with tetrahedra: %.3f s" % (time.time() - start))

#########################################################################################
# export map to vtu 
#########################################################################################
start = time.time()

nel=nlon*nlat

vtufile=open("topo2D.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
counter=0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            vtufile.write("%10f %10f %10f \n" %(lon-ddlon,lat-ddlon,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon+ddlon,lat-ddlon,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon+ddlon,lat+ddlon,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon-ddlon,lat+ddlon,topography[ilat,ilon]/2e3))
            counter+=1
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='topo (m)' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%f\n" % (topography[ilat,ilon]))
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(4*iel,4*iel+1,4*iel+2,4*iel+3))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("time to produce topo2D.vtu: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()

vtufile=open("topo3D.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lon=lonmin+ilon*dlon
        lat=latmin+ilat*dlat
        radius=Rmars+topography[ilat,ilon]

        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        vtufile.write("%.2f %.2f %.2f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        vtufile.write("%.2f %.2f %.2f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        vtufile.write("%.2f %.2f %.2f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        vtufile.write("%.2f %.2f %.2f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='topo (m)' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%.2f\n" % (topography[ilat,ilon]))
    #end for
#end for
vtufile.write("</DataArray>\n")
#
vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%f\n" % (cell_volume[ilat,ilon]))
    #end for
#end for
vtufile.write("</DataArray>\n")
#
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d\n" %(4*iel,4*iel+1,4*iel+2,4*iel+3))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")

print("time to produce topo3D.vtu: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()

vtufile=open("shell3D.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8*nel,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for ic in range(0,npts):
    vtufile.write("%.2f %.2f %.2f \n" %(pt1[0,ic],pt1[1,ic],pt1[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt2[0,ic],pt2[1,ic],pt2[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt3[0,ic],pt3[1,ic],pt3[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt4[0,ic],pt4[1,ic],pt4[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt5[0,ic],pt5[1,ic],pt5[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt6[0,ic],pt6[1,ic],pt6[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt7[0,ic],pt7[1,ic],pt7[2,ic]))
    vtufile.write("%.2f %.2f %.2f \n" %(pt8[0,ic],pt8[1,ic],pt8[2,ic]))
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='topo (m)' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%.1f\n" % (topography[ilat,ilon]))
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
   vtufile.write("%d %d %d %d %d %d %d %d \n" %(8*iel,8*iel+1,8*iel+2,8*iel+3,8*iel+4,8*iel+5,8*iel+6,8*iel+7))
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

print("time to produce shell3D.vtu: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()

if method_jit:

   vtufile=open('gravity_field.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npts2,npts2))
   #--
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for ilat2 in range(0,nlat2):
       for ilon2 in range(0,nlon2): 
           vtufile.write("%10e %10e %10e \n" %(xM[ilat2,ilon2],yM[ilat2,ilon2],zM[ilat2,ilon2]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #--
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for ilat2 in range(0,nlat2):
       for ilon2 in range(0,nlon2): 
           vtufile.write("%10e %10e %10e \n" %(gx[ilat2,ilon2],gy[ilat2,ilon2],gz[ilat2,ilon2]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #--
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,npts2):
       vtufile.write("%d " % i) 
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,npts2):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,npts2):
       vtufile.write("%d " % 1) 
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   print("time to produce gravity_field.vtu: %.3f s" % (time.time() - start))

###############################################################################
start = time.time()

vtufile=open('gravity_above_olympus_mons.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nlat4,nlat4))
#--
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
for ilat4 in range(0,nlat4):
    vtufile.write("%10e %10e %10e \n" %(xP[ilat4],yP[ilat4],zP[ilat4]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#--
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
for ilat4 in range(0,nlat4):
        vtufile.write("%10e %10e %10e \n" %(gx4[ilat4],gy4[ilat4],gz4[ilat4]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#--
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for i in range(0,nlat4):
    vtufile.write("%d " % i) 
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for i in range(0,nlat4):
    vtufile.write("%d " % (i+1))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for i in range(0,nlat4):
    vtufile.write("%d " % 1) 
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("time to produce gravity_above_olympus_mons.vtu: %.3f s" % (time.time() - start))


