import numpy as np

nlat=179
nlon=360

###################################################################################################

S20RTS_dv  = np.zeros(nlat*nlon,dtype=np.float64)  
stone_dv  = np.zeros(nlat*nlon,dtype=np.float64)  

f = open('S20RTS_deep.xyz','r')
lines = f.readlines()
f.close
counter=0
for ilon in range(0,360):
    for ilat in range(0,179):
        vals=lines[counter].strip().split()
        S20RTS_dv[nlon*ilat+ilon]=vals[2]  
        counter+=1

f = open('seismic_velocity_anomaly.ascii','r')
lines = f.readlines()
f.close
counter=0
for ilat in range(0,179):
    for ilon in range(0,360):
        vals=lines[counter].strip().split()
        stone_dv[counter]=vals[2]  
        counter+=1

###################################################################################################
# generate lats and lons for plotting
###################################################################################################

lons = np.empty(nlat*nlon, dtype=np.float64)
lats = np.empty(nlat*nlon, dtype=np.float64)

counter = 0
for ilat in range(0,nlat):
    for ilon in range(0,nlon):
        lons[counter]=    ilon*360/float(nlon)  #starts at 0 then goes to 360
        lats[counter]=-89+ilat*179/float(nlat)   #starts at -90 then goes to +90
        counter += 1
    #end for
#end for

np.savetxt('both.ascii',np.array([lons,lats,stone_dv,S20RTS_dv,stone_dv-S20RTS_dv]).T,  fmt='%1.5e')

###################################################################################################
