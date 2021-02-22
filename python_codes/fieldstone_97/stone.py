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
mfile=open('bench3.ascii',"w")

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
#counter=0
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

        #counter+=1
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
# counting the number of non-zero volume cells for each layer 
#########################################################################################

counter =np.zeros(8,dtype=np.int32)

for il in range(0,8): #layer

    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            boundaries[:]=crust_bnd[:,ilat,ilon]
            if abs(boundaries[il]-boundaries[il+1])>1e-6:
               counter[il]+=1
        #end for
    #end for
#end for

print('water      :',counter[0],'cells out of 64800')
print('ice        :',counter[1],'cells out of 64800')
print('up. seds.  :',counter[2],'cells out of 64800')
print('mid. seds. :',counter[3],'cells out of 64800')
print('low. seds. :',counter[4],'cells out of 64800')
print('up. crust  :',counter[5],'cells out of 64800')
print('mid. crust :',counter[6],'cells out of 64800')
print('low. crust :',counter[7],'cells out of 64800')

#########################################################################################
# produce 8 layer vtu files
#########################################################################################

for il in range(0,8): #layer

    filename = 'layer_{:02d}.vtu'.format(il)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8*counter[il],counter[il]))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")

    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
            if abs(boundaries[il]-boundaries[il+1])>1e-3:
               max_depth=boundaries[il]
               min_depth=boundaries[il+1]
               r_min=6371e3+min_depth
               r_max=6371e3+max_depth

               #node 0
               radius=r_min
               phi=float(lon-0.5)/180*np.pi
               theta=np.pi/2-float(lat-0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 1
               radius=r_min
               phi=float(lon+0.5)/180*np.pi
               theta=np.pi/2-float(lat-0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 2
               radius=r_min
               phi=float(lon+0.5)/180*np.pi
               theta=np.pi/2-float(lat+0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 3
               radius=r_min
               phi=float(lon-0.5)/180*np.pi
               theta=np.pi/2-float(lat+0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 4
               radius=r_max
               phi=float(lon-0.5)/180*np.pi
               theta=np.pi/2-float(lat-0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 5
               radius=r_max
               phi=float(lon+0.5)/180*np.pi
               theta=np.pi/2-float(lat-0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 6
               radius=r_max
               phi=float(lon+0.5)/180*np.pi
               theta=np.pi/2-float(lat+0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
               #node 7
               radius=r_max
               phi=float(lon-0.5)/180*np.pi
               theta=np.pi/2-float(lat+0.5)/180*np.pi
               vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                   radius*np.sin(theta)*np.sin(phi),\
                                                   radius*np.cos(theta)))
            #end if
        #end for
    #end for
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")

    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
            if abs(boundaries[il]-boundaries[il+1])>1:
               vtufile.write("%f\n" % (crust_rho[il,ilat,ilon]*1e3))
        #end for
    #end for
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='thickness' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
            if abs(boundaries[il]-boundaries[il+1])>1:
               vtufile.write("%f\n" % (abs(boundaries[il]-boundaries[il+1])))
        #end for
    #end for
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='longitude' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
            if abs(boundaries[il]-boundaries[il+1])>1:
               vtufile.write("%f\n" % (lon))
        #end for
    #end for
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='latitude' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
            if abs(boundaries[il]-boundaries[il+1])>1:
               vtufile.write("%f\n" % (lat))
        #end for
    #end for
    vtufile.write("</DataArray>\n")






    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,counter[il]):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(8*iel,8*iel+1,8*iel+2,8*iel+3,
                                                   8*iel+4,8*iel+5,8*iel+6,8*iel+7))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,counter[il]):
        vtufile.write("%d \n" %((iel+1)*8))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,counter[il]):
        vtufile.write("%d \n" %12)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")

    print('produced '+filename)

#end do

#########################################################################################
# Same exercise but now with lat-lon bounds
#########################################################################################

min_lon=60 +180
max_lon=100 +180 

min_lat=5
max_lat=40

counter =np.zeros(8,dtype=np.int32)

for il in range(0,8): #layer

    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]
               if abs(boundaries[il]-boundaries[il+1])>1e-6:
                  counter[il]+=1
        #end for
    #end for
#end for

print('min/max lon:',min_lon,max_lon)
print('min/max lat:',min_lat,max_lat)
print('water      :',counter[0],'cells out of 64800')
print('ice        :',counter[1],'cells out of 64800')
print('up. seds.  :',counter[2],'cells out of 64800')
print('mid. seds. :',counter[3],'cells out of 64800')
print('low. seds. :',counter[4],'cells out of 64800')
print('up. crust  :',counter[5],'cells out of 64800')
print('mid. crust :',counter[6],'cells out of 64800')
print('low. crust :',counter[7],'cells out of 64800')




for il in range(0,8): #layer

    filename = 'layer_area_{:02d}.vtu'.format(il)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8*counter[il],counter[il]))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")

    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
               if abs(boundaries[il]-boundaries[il+1])>1e-3:
                  max_depth=boundaries[il]
                  min_depth=boundaries[il+1]
                  r_min=6371e3+min_depth
                  r_max=6371e3+max_depth

                  #node 0
                  radius=r_min
                  phi=float(lon-0.5)/180*np.pi
                  theta=np.pi/2-float(lat-0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 1
                  radius=r_min
                  phi=float(lon+0.5)/180*np.pi
                  theta=np.pi/2-float(lat-0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 2
                  radius=r_min
                  phi=float(lon+0.5)/180*np.pi
                  theta=np.pi/2-float(lat+0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 3
                  radius=r_min
                  phi=float(lon-0.5)/180*np.pi
                  theta=np.pi/2-float(lat+0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 4
                  radius=r_max
                  phi=float(lon-0.5)/180*np.pi
                  theta=np.pi/2-float(lat-0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 5
                  radius=r_max
                  phi=float(lon+0.5)/180*np.pi
                  theta=np.pi/2-float(lat-0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 6
                  radius=r_max
                  phi=float(lon+0.5)/180*np.pi
                  theta=np.pi/2-float(lat+0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
                  #node 7
                  radius=r_max
                  phi=float(lon-0.5)/180*np.pi
                  theta=np.pi/2-float(lat+0.5)/180*np.pi
                  vtufile.write("%10f %10f %10f \n" %(radius*np.sin(theta)*np.cos(phi),\
                                                      radius*np.sin(theta)*np.sin(phi),\
                                                      radius*np.cos(theta)))
               #end if thickness>0
            #end if lat/lon box
        #end for
    #end for
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")

    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
               if abs(boundaries[il]-boundaries[il+1])>1:
                  vtufile.write("%f\n" % (crust_rho[il,ilat,ilon]*1e3))
            #end if lat/lon box
        #end for
    #end for
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='thickness' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
               if abs(boundaries[il]-boundaries[il+1])>1:
                  vtufile.write("%f\n" % (abs(boundaries[il]-boundaries[il+1])))
            #end if lat/lon box
        #end for
    #end for
    vtufile.write("</DataArray>\n")


    vtufile.write("<DataArray type='Float32' Name='longitude' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
               if abs(boundaries[il]-boundaries[il+1])>1:
                  vtufile.write("%f\n" % (lon))
            #end if lat/lon box
        #end for
    #end for
    vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' Name='latitude' Format='ascii'> \n")
    for ilat in range(0,nlat):
        for ilon in range(0,nlon):
            lon=lonmin+ilon*dlon
            lat=latmin+ilat*dlat
            if lon>min_lon and lon<max_lon and lat>min_lat and lat<max_lat: 
               boundaries[:]=crust_bnd[:,ilat,ilon]*1e3
               if abs(boundaries[il]-boundaries[il+1])>1:
                  vtufile.write("%f\n" % (lat))
            #end if lat/lon box
        #end for
    #end for
    vtufile.write("</DataArray>\n")


    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,counter[il]):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(8*iel,8*iel+1,8*iel+2,8*iel+3,
                                                   8*iel+4,8*iel+5,8*iel+6,8*iel+7))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,counter[il]):
        vtufile.write("%d \n" %((iel+1)*8))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,counter[il]):
        vtufile.write("%d \n" %12)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")

    print('produced '+filename)

#end do

#########################################################################################

