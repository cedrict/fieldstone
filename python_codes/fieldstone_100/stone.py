import numpy as np
import matplotlib.pyplot as plt

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

npts=nlon*nlat

Rmars=3389.508e3

rho0=3000

dlon=(lonmax-lonmin)/(nlon-1)
dlat=(latmax-latmin)/(nlat-1)

ddlon=abs(dlon)/2
ddlat=abs(dlat)/2

###############################################################################

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

print('npts=',npts)

###############################################################################

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        topography[j,i]=float(values[2])
        counter+=1

print('topography m/M',np.min(topography),np.max(topography),' m')

#########################################################################################
# compute cell volume

cell_volume =np.zeros((nlat,nlon),dtype=np.float64)

t=np.min(topography)

resol*=(np.pi/180)

counter=0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lon=lonmin+ilon*dlon
        lat=latmin+ilat*dlat
        phi=lon/180*np.pi
        theta=(90-lat)/180*np.pi
        theta_min=theta-resol/2
        theta_max=theta+resol/2
        Rmin=Rmars+t
        Rmax=Rmars+topography[ilat,ilon]
        cell_volume[ilat,ilon]=(Rmax**3-Rmin**3)/3*(np.cos(theta_min)-np.cos(theta_max))*resol
        counter+=1
    #end for 
#end for 

print('volume m/M',np.min(cell_volume),np.max(cell_volume),'m^3')
print('total volume=',np.sum(cell_volume)) #,4/3*np.pi*(Rmax**3-Rmin**3))

plt.imshow(cell_volume/1e9)
plt.colorbar()
plt.savefig('cell_volume.pdf', bbox_inches='tight')
print('produced cell_volume.pdf')

#########################################################################################
# export map to vtu 
#########################################################################################

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

print('produced topo2D.vtu')

###############################################################################

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
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat-ddlat)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon-ddlon)/180*np.pi
        theta=np.pi/2-float(lat+ddlat)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
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
        vtufile.write("%f\n" % (topography[ilat,ilon]))
    #end for
#end for
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%f\n" % (cell_volume[ilat,ilon]))
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

print('produced topo3D.vtu')

