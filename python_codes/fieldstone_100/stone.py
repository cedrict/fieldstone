import numpy as np

###############################################################################

nlon=360
nlat=180
npts=nlon*nlat

latmin=89.5
latmax=-89.5
lonmin=0.5
lonmax=359.5

Rmars=3389.5e6

###############################################################################

topography =np.zeros((nlat,nlon),dtype=np.float64)

file=open("MOLA_1deg.txt", "r")
lines = file.readlines()
file.close

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        topography[j,i]=float(values[2])
        counter+=1

print('moho depth m/M',np.min(topography),np.max(topography))

#########################################################################################
# export map to vtu 
#########################################################################################

dlon=(lonmax-lonmin)/(nlon-1)
dlat=(latmax-latmin)/(nlat-1)

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
            vtufile.write("%10f %10f %10f \n" %(lon-0.5,lat-0.5,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon+0.5,lat-0.5,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon+0.5,lat+0.5,topography[ilat,ilon]/2e3))
            vtufile.write("%10f %10f %10f \n" %(lon-0.5,lat+0.5,topography[ilat,ilon]/2e3))
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

        phi=float(lon-0.5)/180*np.pi
        theta=np.pi/2-float(lat-0.5)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+0.5)/180*np.pi
        theta=np.pi/2-float(lat-0.5)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon+0.5)/180*np.pi
        theta=np.pi/2-float(lat+0.5)/180*np.pi
        vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                            radius*np.sin(theta)*np.sin(phi),\
                                            radius*np.cos(theta)))
        phi=float(lon-0.5)/180*np.pi
        theta=np.pi/2-float(lat+0.5)/180*np.pi
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

