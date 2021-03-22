import numpy as np

nlon=720
nlat=360
N=259200
latmin=89.75
latmax=-89.75
lonmin=0.25
lonmax=359.75
dlon=(lonmax-lonmin)/(nlon-1)
dlat=(latmax-latmin)/(nlat-1)

lon=np.zeros(N,dtype=np.float64)   
lat=np.zeros(N,dtype=np.float64)
continental=np.zeros(N,dtype=np.float64)
bedrock=np.zeros(N,dtype=np.float64)
ice=np.zeros(N,dtype=np.float64)
moho=np.zeros(N,dtype=np.float64)
rho_c=np.zeros(N,dtype=np.float64)
rho_submoho=np.zeros(N,dtype=np.float64)
rho_20km=np.zeros(N,dtype=np.float64)
rho_36km=np.zeros(N,dtype=np.float64)
rho_56km=np.zeros(N,dtype=np.float64)
rho_80km=np.zeros(N,dtype=np.float64)
rho_110km=np.zeros(N,dtype=np.float64)
rho_150km=np.zeros(N,dtype=np.float64)
rho_200km=np.zeros(N,dtype=np.float64)
rho_260km=np.zeros(N,dtype=np.float64)
rho_330km=np.zeros(N,dtype=np.float64)
rho_400km=np.zeros(N,dtype=np.float64)
colat=np.zeros(N,dtype=np.float64)   
phi=np.zeros(N,dtype=np.float64)   
theta=np.zeros(N,dtype=np.float64)   

###################################################################################################
# there is an error in the lat/lon values of all files except the ETOPO* ones
# so we read lat/lon values from these last.
###################################################################################################

print('fetch data from toobig/f99')
exit()

lon[0:N],lat[0:N],rho_c[0:N]=\
np.loadtxt('data/rho_c_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_submoho[0:N]=\
np.loadtxt('data/rho_submoho_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],moho[0:N]=\
np.loadtxt('data/Global_Moho_CRUST1.0.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_20km[0:N]=\
np.loadtxt('data/rho_20km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_36km[0:N]=\
np.loadtxt('data/rho_36km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_56km[0:N]=\
np.loadtxt('data/rho_56km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_80km[0:N]=\
np.loadtxt('data/rho_80km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_110km[0:N]=\
np.loadtxt('data/rho_110km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_150km[0:N]=\
np.loadtxt('data/rho_150km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_200km[0:N]=\
np.loadtxt('data/rho_200km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_260km[0:N]=\
np.loadtxt('data/rho_260km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_330km[0:N]=\
np.loadtxt('data/rho_330km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],rho_400km[0:N]=\
np.loadtxt('data/rho_400km_out.xyz',unpack=True,usecols=[0,1,2],skiprows=0)


lon[0:N],lat[0:N],continental[0:N]=\
np.loadtxt('data/ETOPO2_km_continental.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],bedrock[0:N]=\
np.loadtxt('data/ETOPO2_km_depth_Bed.xyz',unpack=True,usecols=[0,1,2],skiprows=0)

lon[0:N],lat[0:N],ice[0:N]=\
np.loadtxt('data/ETOPO2_km_depth_Ice.xyz',unpack=True,usecols=[0,1,2],skiprows=0)


continental*=1000
bedrock*=1000
ice*=1000
moho*=1000

continental[:]=6371e3-continental[:]
bedrock[:]=6371e3-bedrock[:]
ice[:]=6371e3-ice[:]
moho[:]=6371e3-moho[:]

###################################################################################################

nel=nlon*nlat
vtufile=open("visualisation_bef.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
counter=0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
            llon=lonmin+ilon*dlon
            llat=latmin+ilat*dlat
            k=ilon+nlon*ilat
            vtufile.write("%10f %10f %10f \n" %(llon-0.5,llat-0.5,0))
            vtufile.write("%10f %10f %10f \n" %(llon+0.5,llat-0.5,0))
            vtufile.write("%10f %10f %10f \n" %(llon+0.5,llat+0.5,0))
            vtufile.write("%10f %10f %10f \n" %(llon-0.5,llat+0.5,0))
            counter+=1
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='continental-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (continental[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='bedrock-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (bedrock[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='ice-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (ice[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='ice thickness' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (ice[k]-bedrock[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")


vtufile.write("<DataArray type='Float32' Name='water thickness' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (continental[k]-bedrock[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")


vtufile.write("<DataArray type='Float32' Name='continental-ice' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (continental[k]-ice[k]))
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

print('produced visualisation_bef.vtu')

###################################################################################################
# looking at the raw data in ParaView, we see two problems:
# there are a few places where there is negative water thickness and negative ice thickness
# we therefore correct the data to remove these problems.
# I somewhat arbitrarily take the bedrock signal as the reference one and correct the other two.
###################################################################################################

# for ice, we make sure that the ice signal is strictly above the bedrock signal
# so that ice-bedrock>0

for k in range(0,N):
    ice[k]=max(ice[k],bedrock[k])

# for water, we make sur that the continental signal is strictly above the bedrock signal
# so that continental-bedrock>0

for k in range(0,N):
    continental[k]=max(continental[k],bedrock[k])


vtufile=open("visualisation_aft.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
counter=0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
            llon=lonmin+ilon*dlon
            llat=latmin+ilat*dlat
            k=ilon+nlon*ilat
            vtufile.write("%10f %10f %10f \n" %(llon-0.5,llat-0.5,bedrock[k]/1e3-6371))
            vtufile.write("%10f %10f %10f \n" %(llon+0.5,llat-0.5,bedrock[k]/1e3-6371))
            vtufile.write("%10f %10f %10f \n" %(llon+0.5,llat+0.5,bedrock[k]/1e3-6371))
            vtufile.write("%10f %10f %10f \n" %(llon-0.5,llat+0.5,bedrock[k]/1e3-6371))
            counter+=1
    #end for
#end for
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='continental-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (continental[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='bedrock-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (bedrock[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='ice-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (ice[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='ice thickness' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (ice[k]-bedrock[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='water thickness' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (continental[k]-bedrock[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_c' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_c[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='moho-6371e3' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (moho[k]-6371e3))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_submoho' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_submoho[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_20km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_20km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_36km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_36km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_56km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_56km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_80km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_80km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_110km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_110km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_150km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_150km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_200km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_200km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_260km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_260km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_330km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_330km[k]))
    #end for
#end for
vtufile.write("</DataArray>\n")
####
vtufile.write("<DataArray type='Float32' Name='rho_440km' Format='ascii'> \n")
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        k=ilon+nlon*ilat
        vtufile.write("%f\n" % (rho_400km[k]))
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


print('produced visualisation_aft.vtu')

###################################################################################################
# the data set goes from -400km to approx +7km topography.
###################################################################################################

#nrad=409
#nrad=103
nrad=150
nlon=720
nlat=360

radmin=6371e3-400e3
radmax=6371e3+  8e3
drad=(radmax-radmin)/(nrad-1)

print('lon (m/M):',np.min(lon),np.max(lon))
print('lat (m/M):',np.min(lat),np.max(lat))

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


benchfile=open('bench4.ascii',"w")
benchfile.write("# POINTS: %i %i %i \n" %(nrad,nlon,nlat))

counter=0
for ilat in range(0,360):
    for ilon in range(0,720):
        k=ilon+nlon*ilat
        for irad in range(0,nrad):
            rad=radmin+irad*drad
            rho=0

            #we first deal with the concentric layers
            if rad<6371e3-330e3: # between 400 and 330
               rho= (rad-(6371e3-400e3))/70e3*(rho_330km[k]-rho_400km[k])+rho_400km[k]
            elif rad<6371e3-260e3: # between 330 and 260
               rho= (rad-(6371e3-330e3))/70e3*(rho_260km[k]-rho_330km[k])+rho_330km[k]
            elif rad<6371e3-200e3: # between 260 and 200
               rho= (rad-(6371e3-260e3))/60e3*(rho_200km[k]-rho_260km[k])+rho_260km[k]
            elif rad<6371e3-150e3: # between 200 and 150
               rho= (rad-(6371e3-200e3))/50e3*(rho_150km[k]-rho_200km[k])+rho_200km[k]
            elif rad<6371e3-110e3: # between 150 and 110
               rho= (rad-(6371e3-150e3))/40e3*(rho_110km[k]-rho_150km[k])+rho_150km[k]
            elif rad<6371e3-80e3: # between 110 and 80
               rho= (rad-(6371e3-110e3))/30e3*(rho_80km[k]-rho_110km[k])+rho_110km[k]
            elif rad<6371e3-56e3: # between 80 and 56
               rho= (rad-(6371e3-80e3))/24e3*(rho_56km[k]-rho_80km[k])+rho_80km[k]
            elif rad<6371e3-36e3: # between 56 and 36
               rho= (rad-(6371e3-56e3))/20e3*(rho_36km[k]-rho_56km[k])+rho_56km[k]
            elif rad<6371e3-20e3: # between 36 and 20
               rho= (rad-(6371e3-36e3))/16e3*(rho_20km[k]-rho_36km[k])+rho_36km[k]
            elif rad<moho[k]: # between moho and 20
               rho= (rad-(6371e3-20e3))/(moho[k]-(6371e3-20e3))*(rho_submoho[k]-rho_20km[k])+rho_20km[k]

            #taking care of the are between moho and bedrock
            rtop=bedrock[k]
            rbot=moho[k]
            if rad<rtop and rad>rbot:
               rho=rho_c[k]  

            #taking care of water 
            rtop=continental[k]
            rbot=bedrock[k]
            if rad<rtop and rad>rbot:
               rho=1030  

            #taking care of ice 
            rtop=ice[k]
            rbot=bedrock[k]
            if rad<rtop and rad>rbot:
               rho=910  

            T=(1-rho/3300)/3e-5
            benchfile.write("%f %f %f %f \n" %(rad,phi[k],theta[k],T))
        counter+=1
        
benchfile.close

print('produced bench4.ascii')


