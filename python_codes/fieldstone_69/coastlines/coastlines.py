import numpy as np
from vtu_tools import *  

#------------------------------------------------------------------------------

np_europe=338058
np_africa=311712
np_asia=924434
np_samer=390832
np_namer=819761

radius=6371e3

#------------------------------------------------
filename='europe_coastlines.txt'

lon=np.zeros(np_europe,dtype=np.float64)   
lat=np.zeros(np_europe,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lat[counter]=columns[0]
    lon[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('europe_coastlines_map1.vtu',np_europe,lon,lat)
export_to_vtu_sphere('europe_coastlines_sphere1.vtu',np_europe,lon,lat,radius)

for i in range(0,np_europe):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('europe_coastlines_map2.vtu',np_europe,lon,lat)

for i in range(0,np_europe):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('europe_coastlines_map3.vtu',np_europe,lon,lat)
export_to_vtu_sphere('europe_coastlines_sphere3.vtu',np_europe,lon,lat,radius)

#------------------------------------------------
filename='asia_coastlines.txt'

lon=np.zeros(np_asia,dtype=np.float64)   
lat=np.zeros(np_asia,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lat[counter]=columns[0]
    lon[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('asia_coastlines_map1.vtu',np_asia,lon,lat)
export_to_vtu_sphere('asia_coastlines_sphere1.vtu',np_asia,lon,lat,radius)

for i in range(0,np_asia):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('asia_coastlines_map2.vtu',np_asia,lon,lat)

for i in range(0,np_asia):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('asia_coastlines_map3.vtu',np_asia,lon,lat)
export_to_vtu_sphere('asia_coastlines_sphere3.vtu',np_asia,lon,lat,radius)


#------------------------------------------------
filename='africa_coastlines.txt'

lon=np.zeros(np_africa,dtype=np.float64)   
lat=np.zeros(np_africa,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lat[counter]=columns[0]
    lon[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('africa_coastlines_map1.vtu',np_africa,lon,lat)
export_to_vtu_sphere('africa_coastlines_sphere1.vtu',np_africa,lon,lat,radius)

for i in range(0,np_africa):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('africa_coastlines_map2.vtu',np_africa,lon,lat)

for i in range(0,np_africa):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('africa_coastlines_map3.vtu',np_africa,lon,lat)
export_to_vtu_sphere('africa_coastlines_sphere3.vtu',np_africa,lon,lat,radius)


#------------------------------------------------
filename='namer_coastlines.txt'

lon=np.zeros(np_namer,dtype=np.float64)   
lat=np.zeros(np_namer,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lat[counter]=columns[0]
    lon[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('namer_coastlines_map1.vtu',np_namer,lon,lat)
export_to_vtu_sphere('namer_coastlines_sphere1.vtu',np_namer,lon,lat,radius)

for i in range(0,np_namer):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('namer_coastlines_map2.vtu',np_namer,lon,lat)

for i in range(0,np_namer):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('namer_coastlines_map3.vtu',np_namer,lon,lat)
export_to_vtu_sphere('namer_coastlines_sphere3.vtu',np_namer,lon,lat,radius)


#------------------------------------------------
filename='samer_coastlines.txt'

lon=np.zeros(np_samer,dtype=np.float64)   
lat=np.zeros(np_samer,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lat[counter]=columns[0]
    lon[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('samer_coastlines_map1.vtu',np_samer,lon,lat)
export_to_vtu_sphere('samer_coastlines_sphere1.vtu',np_samer,lon,lat,radius)

for i in range(0,np_samer):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('samer_coastlines_map2.vtu',np_samer,lon,lat)

for i in range(0,np_samer):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('samer_coastlines_map3.vtu',np_samer,lon,lat)
export_to_vtu_sphere('samer_coastlines_sphere3.vtu',np_samer,lon,lat,radius)

