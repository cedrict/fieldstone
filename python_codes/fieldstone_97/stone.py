import numpy as np

#parameters
nlon=360
nlat=180
latmin=89.5
latmax=-89.5
lonmin=0.5
lonmax=359.5

#user chosen
nrad=301  # 301 <-> 300m resolution for -81:9km range

alpha=3e-5
rho0=3300

radmin=6371e3-81e3
radmax=6371e3+9e3

###########################################################

crust_rho =np.zeros((9,nlat,nlon),dtype=np.float64)
crust_bnd =np.zeros((9,nlat,nlon),dtype=np.float64)

###########################################################
# read in density data
###########################################################

file=open("crust1.rho", "r")
lines = file.readlines()
file.close

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        #print(values[:])
        crust_rho[0,j,i]=float(values[0])
        crust_rho[1,j,i]=float(values[1])
        crust_rho[2,j,i]=float(values[2])
        crust_rho[3,j,i]=float(values[3])
        crust_rho[4,j,i]=float(values[4])
        crust_rho[5,j,i]=float(values[5])
        crust_rho[6,j,i]=float(values[6])
        crust_rho[7,j,i]=float(values[7])
        crust_rho[8,j,i]=float(values[8])
        counter+=1

###########################################################
# read in layer boundary data
###########################################################

file=open("crust1.bnds", "r")
lines = file.readlines()
file.close

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        #print(values[:])
        crust_bnd[0,j,i]=float(values[0])
        crust_bnd[1,j,i]=float(values[1])
        crust_bnd[2,j,i]=float(values[2])
        crust_bnd[3,j,i]=float(values[3])
        crust_bnd[4,j,i]=float(values[4])
        crust_bnd[5,j,i]=float(values[5])
        crust_bnd[6,j,i]=float(values[6])
        crust_bnd[7,j,i]=float(values[7])
        crust_bnd[8,j,i]=float(values[8])
        counter+=1

###########################################################
# generate file for ASPECT
# remember that the latitudes go from 89.5 to -89.5 so that 
# these are actually co-latitudes, which is nice
# because they match the spherical coordinate starting at 
# 0 at the north pole
###########################################################

boundaries =np.zeros(9,dtype=np.float64)
rhos =np.zeros(9,dtype=np.float64)

cfile=open('crust1p0_full_for_aspect.ascii',"w")
mfile=open('crust1p0_moho_for_aspect.ascii',"w")

cfile.write("# POINTS: %i %i %i \n" %(nrad,nlon,nlat)) 
mfile.write("# POINTS: %i %i %i \n" %(nrad,nlon,nlat)) 

dlon=(lonmax-lonmin)/(nlon-1)
drad=(radmax-radmin)/(nrad-1)
dlat=(latmax-latmin)/(nlat-1)

print('radial resolution=',drad,'m')

for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        for irad in range(0,nrad):

            rad=radmin+irad*drad
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat

            depth=(rad-6371e3)/1e3

            boundaries[:]=crust_bnd[:,ilat,ilon]
            rhos[:]      =crust_rho[:,ilat,ilon]*1e3

            #######################################

            if depth<boundaries[8]: 
               rho=rhos[8] 
            elif depth<boundaries[7]: 
               rho=rhos[7]
            elif depth<boundaries[6]: 
               rho=rhos[6]
            elif depth<boundaries[5]: 
               rho=rhos[5]
            elif depth<boundaries[4]: 
               rho=rhos[4]
            elif depth<boundaries[3]: 
               rho=rhos[3]
            elif depth<boundaries[2]: 
               rho=rhos[2]
            elif depth<boundaries[1]: 
               rho=rhos[1]
            elif depth<boundaries[0]: 
               rho=rhos[0]
            else:
               #rho=0. #air
               rho=rhos[0] # air replaced by what is just below !!
            #end if

            cfile.write("%e %e %e %e \n" %(rad,lon/180.*np.pi,(90-lat)/180.*np.pi, (1.-rho/rho0)/alpha))
            #cfile.write("%e %e %e %e \n" %(rad,lon,lat,rho))

            #######################################

            if depth<boundaries[8]:
               rho=3300
            else:
               rho=2900

            mfile.write("%e %e %e %e \n" %(rad,lon/180.*np.pi,(90-lat)/180.*np.pi, (1.-rho/rho0)/alpha))

        #end for
    #end for
#end for

cfile.close()
mfile.close()

print('produced crust1p0_full_for_aspect.ascii')
print('produced crust1p0_moho_for_aspect.ascii')

###############################################################################

mohodepth =np.zeros((nlat,nlon),dtype=np.float64)

file=open("depthtomoho.xyz", "r")
lines = file.readlines()
file.close

counter=0
for j in range(0,nlat):
    for i in range(0,nlon):
        values = lines[counter].strip().split()
        #print(values[:])
        mohodepth[j,i]=float(values[2])*1000
        counter+=1

print('moho depth m/M',np.min(mohodepth),np.max(mohodepth))

#########################################################################################
# export map to vtu 
#########################################################################################

nel=nlon*nlat

vtufile=open("moho_map.vtu","w")
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
            vtufile.write("%10f %10f %10f \n" %(lon-0.5,lat-0.5,mohodepth[ilat,ilon]/3500))
            vtufile.write("%10f %10f %10f \n" %(lon+0.5,lat-0.5,mohodepth[ilat,ilon]/3500))
            vtufile.write("%10f %10f %10f \n" %(lon+0.5,lat+0.5,mohodepth[ilat,ilon]/3500))
            vtufile.write("%10f %10f %10f \n" %(lon-0.5,lat+0.5,mohodepth[ilat,ilon]/3500))
            counter+=1
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='depth (m)' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%f\n" % (mohodepth[ilat,ilon]))
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

print('produced moho_map.vtu')

###############################################################################

vtufile=open("moho_shell.vtu","w")
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
        radius=6371e3+mohodepth[ilat,ilon]

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

        counter+=1
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='depth (m)' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        vtufile.write("%f\n" % (mohodepth[ilat,ilon]))
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

print('produced moho_shell.vtu')

#########################################################################################
