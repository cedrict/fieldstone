import numpy as np
import compute_volume_hexahedron
import time as timing

###############################################################################
# part1: producing ascii data file for ASPECT
###############################################################################
# original files counts 259200 lines, i.e. 
# longitudes start at 0.25degree up to 359.75. 
# latitudes start at -89.75 up to 89.75
# (360x2) X (180x2) = 259200

N=259200

lon=np.zeros(N,dtype=np.float64)   
lat=np.zeros(N,dtype=np.float64)   
rho=np.zeros(N,dtype=np.float64)   
colat=np.zeros(N,dtype=np.float64)   
T=np.zeros(N,dtype=np.float64)   

#load data file into arrays

lon[0:N],lat[0:N],rho[0:N]=np.loadtxt('rho_56km_SH_v2.txt',unpack=True,usecols=[0,1,2],skiprows=0)

print('lon (m/M):',np.min(lon),np.max(lon))
print('lat (m/M):',np.min(lat),np.max(lat))
print('rho (m/M):',np.min(rho),np.max(rho))

lon[:]/=180
lat[:]/=180

lon[:]*=np.pi
lat[:]*=np.pi

colat[:]=np.pi/2-lat[:]

print('longitude (m/M): ',np.min(lon),np.max(lon))
print('latitude (m/M): ',np.min(lat),np.max(lat))
print('colatitude (m/M): ',np.min(colat),np.max(colat))

theta=colat
phi=lon

#transforming density into temperature for ASPECT to read in.
#ASPECT then uses this temperature in its material model, and 
#if the same reference density and thermal expansion values are 
#used then the desired density is recovered.
T[:]=(1-rho[:]/3300)/3e-5

print('Temperature (m/M):',np.min(T),np.max(T))

#I write in the file the same temperature value at two different depths/radii, 
#i.e. 1km above and below the actual radii of the layer. 

benchfile=open('bench2.ascii',"w")
benchfile.write("# POINTS: 2 720 360 \n")
for j in range(0,360):
    for i in range(0,720):
        counter=j+i*360
        benchfile.write("%f %f %f %f \n" %(6371e3-81e3,phi[counter],theta[counter],T[counter]))
        benchfile.write("%f %f %f %f \n" %(6371e3-55e3,phi[counter],theta[counter],T[counter]))
        
benchfile.close

print('produced bench2.ascii')

###############################################################################
# part1a: producing input file bench3.txt for aspect, Root et al 2021, case 3 
###############################################################################
# the file Global_Moho_CRUST1.0_version2.xyz also contains 259,200 lines

depths=np.zeros(N,dtype=np.float64)   

lon[0:N],lat[0:N],depths[0:N]=np.loadtxt('Global_Moho_CRUST1.0_version2.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

print('lon (m/M):',np.min(lon),np.max(lon))
print('lat (m/M):',np.min(lat),np.max(lat))
print('depths (m/M):',np.min(depths),np.max(depths))

lon[:]/=180
lat[:]/=180

lon[:]*=np.pi
lat[:]*=np.pi

colat[:]=np.pi/2-lat[:]

depths[:]*=1000

print('longitude (m/M): ',np.min(lon),np.max(lon))
print('latitude (m/M): ',np.min(lat),np.max(lat))
print('colatitude (m/M): ',np.min(colat),np.max(colat))

theta=colat
phi=lon

nrad=180
nlon=720
nlat=360
radmin=6371e3-80e3
radmax=6371e3
drad=(radmax-radmin)/(nrad-1)

benchfile=open('bench3.ascii',"w")
benchfile.write("# POINTS: %i %i %i \n" %(nrad,nlon,nlat))

counter=0
for j in range(0,360):
    for i in range(0,720):
        #counter=j+i*360
        for irad in range(0,nrad):
            rad=radmin+irad*drad
            if rad>6371e3-depths[counter]:
               rho=2900
            else:
               rho=3300
            T=(1-rho/3300)/3e-5
            benchfile.write("%f %f %f %f \n" %(rad,phi[counter],theta[counter],T))
        counter+=1
        
benchfile.close

print('produced bench3.ascii')

exit()

###############################################################################
# part2: producing vtu file of the original lon-lat datas set 
###############################################################################
# there are 259,200 cells
# note that one needs to be careful with thetaa values since these are
# colatitudes: a smaller latitude means a larger colatitude!

inner_radius=6371e3-80e3
outer_radius=6371e3-56e3

deg4=np.pi/180/4

total_volume=4*np.pi/3*(outer_radius**3-inner_radius**3)

vol1=np.zeros(N,dtype=np.float64)   
vol2=np.zeros(N,dtype=np.float64)   
mass1=np.zeros(N,dtype=np.float64)   
mass2=np.zeros(N,dtype=np.float64)   
x=np.zeros(8,dtype=np.float64)   
y=np.zeros(8,dtype=np.float64)   
z=np.zeros(8,dtype=np.float64)   
xc=np.zeros(N,dtype=np.float64)   
yc=np.zeros(N,dtype=np.float64)   
zc=np.zeros(N,dtype=np.float64)   

#uncomment to set density in shell to constant value:
#rho[:]=3300

vtufile=open('layer.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(8*N,N))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")

for i in range(0,N):

    radius=inner_radius
    thetaa=theta[i]+deg4
    phii=phi[i]    -deg4
    x[0]=radius*np.sin(thetaa)*np.cos(phii)
    y[0]=radius*np.sin(thetaa)*np.sin(phii)
    z[0]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[0],y[0],z[0]))

    radius=inner_radius
    thetaa=theta[i]+deg4
    phii=phi[i]    +deg4
    x[1]=radius*np.sin(thetaa)*np.cos(phii)
    y[1]=radius*np.sin(thetaa)*np.sin(phii)
    z[1]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[1],y[1],z[1]))

    radius=inner_radius
    thetaa=theta[i]-deg4
    phii=phi[i]    +deg4
    x[2]=radius*np.sin(thetaa)*np.cos(phii)
    y[2]=radius*np.sin(thetaa)*np.sin(phii)
    z[2]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[2],y[2],z[2]))

    radius=inner_radius
    thetaa=theta[i]-deg4
    phii=phi[i]    -deg4
    x[3]=radius*np.sin(thetaa)*np.cos(phii)
    y[3]=radius*np.sin(thetaa)*np.sin(phii)
    z[3]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[3],y[3],z[3]))

    radius=outer_radius
    thetaa=theta[i]+deg4
    phii=phi[i]    -deg4
    x[4]=radius*np.sin(thetaa)*np.cos(phii)
    y[4]=radius*np.sin(thetaa)*np.sin(phii)
    z[4]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[4],y[4],z[4]))

    radius=outer_radius
    thetaa=theta[i]+deg4
    phii=phi[i]    +deg4
    x[5]=radius*np.sin(thetaa)*np.cos(phii)
    y[5]=radius*np.sin(thetaa)*np.sin(phii)
    z[5]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[5],y[5],z[5]))

    radius=outer_radius
    thetaa=theta[i]-deg4
    phii=phi[i]    +deg4
    x[6]=radius*np.sin(thetaa)*np.cos(phii)
    y[6]=radius*np.sin(thetaa)*np.sin(phii)
    z[6]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[6],y[6],z[6]))

    radius=outer_radius
    thetaa=theta[i]-deg4
    phii=phi[i]    -deg4
    x[7]=radius*np.sin(thetaa)*np.cos(phii)
    y[7]=radius*np.sin(thetaa)*np.sin(phii)
    z[7]=radius*np.cos(thetaa)
    vtufile.write("%10f %10f %10f \n" %(x[7],y[7],z[7]))

    vol1[i]=compute_volume_hexahedron.hexahedron_volume(x,y,z)

    vol2[i]=(outer_radius**3-inner_radius**3)/3*(0.5/180*np.pi)*\
            (np.cos(theta[i]-deg4)-np.cos(theta[i]+deg4))

    mass1[i]=vol1[i]*rho[i]
    mass2[i]=vol2[i]*rho[i]

    #assign center to cell
    radius=(inner_radius+outer_radius)/2
    thetaa=theta[i]
    phii=phi[i]    
    #xc[i]=radius*np.sin(thetaa)*np.cos(phii)
    #yc[i]=radius*np.sin(thetaa)*np.sin(phii)
    #zc[i]=radius*np.cos(thetaa)
    #xc[i]=np.sum(x[:])/8.
    #yc[i]=np.sum(y[:])/8.
    #zc[i]=np.sum(z[:])/8.

    xc[i]=(outer_radius**4-inner_radius**4)/4*\
          (deg4-0.25*(np.sin(2*(thetaa+deg4))-np.sin(2*(thetaa-deg4))))*\
          (np.sin(phii+deg4) - np.sin(phii-deg4))/vol2[i]

    yc[i]=(outer_radius**4-inner_radius**4)/4*\
          (deg4-0.25*(np.sin(2*(thetaa+deg4))-np.sin(2*(thetaa-deg4))))*\
          (-np.cos(phii+deg4) + np.cos(phii-deg4))/vol2[i]

    zc[i]=-0.5*(outer_radius**4-inner_radius**4)/4*(2*deg4)*\
          ((np.cos(thetaa+deg4))**2-(np.cos(thetaa-deg4))**2)/vol2[i]

#end for

vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%e \n" %rho[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='volume (Grandy, 1997)' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%e \n" %vol1[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='volume (Analytical)' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%e \n" %vol2[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='volume (Difference G-A)' Format='ascii'> \n")
for i in range (0,N):
    vtufile.write("%e \n" %(vol2[i]-vol1[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,N):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(8*iel,8*iel+1,8*iel+2,8*iel+3,
                                                8*iel+4,8*iel+5,8*iel+6,8*iel+7))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,N):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,N):
    vtufile.write("%d \n" %12)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()
print('generated layer.vtu')

print('vol1 (m/M):',np.min(vol1),np.max(vol1))
print('vol2 (m/M):',np.min(vol2),np.max(vol2))
print('total vol1:',vol1.sum(),total_volume)
print('total vol2:',vol2.sum(),total_volume)
print('total vol rel. error:', abs(vol1.sum()-total_volume)/total_volume*100, '%')
print('total vol rel. error:', abs(vol2.sum()-total_volume)/total_volume*100, '%')

print('mass1 (m/M/total):',np.min(mass1),np.max(mass1),mass1.sum())
print('mass2 (m/M/total):',np.min(mass2),np.max(mass2),mass2.sum())

#np.savetxt('xyzc.ascii',np.array([xc,yc,zc]).T)
#print('generated xyzc.ascii')

#exit()

###############################################################################
# part3: computing gravity field at 250km heightabove the 6371km surface
###############################################################################

Ggrav=6.67428e-11

radius=6371e3+250e3
nlon=180+1
nlat=90+1

#nlon=41
#nlat=21

gravfile=open('gravity.ascii',"w")

npts=nlon*nlat
nel=(nlon-1)*(nlat-1)

ggx=np.zeros(npts,dtype=np.float64)   
ggy=np.zeros(npts,dtype=np.float64)   
ggz=np.zeros(npts,dtype=np.float64)   
gg=np.zeros(npts,dtype=np.float64)   
ggr=np.zeros(npts,dtype=np.float64)   
UU=np.zeros(npts,dtype=np.float64)   

start = timing.time()

counter=0
for ilat in range(0,nlat):
    thetaa=ilat*np.pi/(nlat-1)
    for ilon in range(0,nlon):
        phii=2*ilon*np.pi/(nlon-1)
        if counter%100==0:
           print(counter/(nlon*nlat)*100,'% done')
        xs=radius*np.sin(thetaa)*np.cos(phii)
        ys=radius*np.sin(thetaa)*np.sin(phii)
        zs=radius*np.cos(thetaa)
        gx=0
        gy=0
        gz=0
        gr=0
        U=0
        for i in range(0,N):
            dx=xs-xc[i]
            dy=ys-yc[i]
            dz=zs-zc[i]
            dx2=dx**2
            dy2=dy**2
            dz2=dz**2
            dist=np.sqrt(dx2+dy2+dz2)
            fact=mass2[i]/dist**3
            gx+= fact*dx 
            gy+= fact*dy
            gz+= fact*dz
            U -= mass2[i]/dist
        #end for
        g=np.sqrt(gx**2+gy**2+gz**2)*Ggrav
        gx*=Ggrav
        gy*=Ggrav
        gz*=Ggrav
        U*=Ggrav
        ggr[counter]=(xs*gx+ys*gy+zs*gz)/radius
        ggx[counter]=gx
        ggy[counter]=gy
        ggz[counter]=gz
        gg[counter]=g        
        UU[counter]=U
        gravfile.write("%e %e %e %e %e %e %e %e %e %e \n" \
                        %(ilon/(nlon-1)*360,ilat/(nlat-1)*180-90,gx,gy,gz,g,U,xs,ys,zs))
        counter+=1
    #end for
#end for

tend=timing.time()

print("compute gravity: %.3f s" % (tend-start))
print("compute gravity per point: %.3f s" % ((tend-start)/npts))

##############################
# export results to vtu file
##############################

x=np.zeros(npts,dtype=np.float64)   
y=np.zeros(npts,dtype=np.float64)   

counter = 0
for j in range(0, nlat):
    for i in range(0, nlon):
        x[counter]=i*360/float(nlon-1)
        y[counter]=j*180/float(nlat-1)
        counter += 1
    #end for
#end for

icon =np.zeros((4,nel),dtype=np.int32)
counter = 0
for j in range(0, nlat-1):
    for i in range(0, nlon-1):
        icon[0, counter] = i + j * (nlon )
        icon[1, counter] = i + 1 + j * (nlon )
        icon[2, counter] = i + 1 + (j + 1) * (nlon )
        icon[3, counter] = i + (j + 1) * (nlon )
        counter += 1
    #end for
#end for

vtufile=open('gravity.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(npts,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,npts):
       vtufile.write("%10e %10e %10e \n" %(x[i],90-y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='g' Format='ascii'> \n")
for i in range(0,npts):
       vtufile.write("%10e \n" %(gg[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='gr' Format='ascii'> \n")
for i in range(0,npts):
       vtufile.write("%10e \n" %(ggr[i]))
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='U' Format='ascii'> \n")
for i in range(0,npts):
       vtufile.write("%10e \n" %(UU[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

gravfile.close()

print('generated gravity.vtu')
print('generated gravity.ascii')
