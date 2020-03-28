import numpy as np
from vtu_tools import *  

#------------------------------------------------

np_europe=196266
np_africa=597474
np_asia=911477
np_samer=460583
np_namer=332854

radius=6371e3

#------------------------------------------------
filename='europe_rivers.txt'

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

#export_to_vtu_flat('europe_rivers.vtu',np_europe,lon,lat)
export_to_vtu_sphere('europe_rivers.vtu',np_europe,lon,lat,radius)

#------------------------------------------------
filename='asia_rivers.txt'

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

#export_to_vtu_flat('asia_rivers.vtu',np_asia,lon,lat)
export_to_vtu_sphere('asia_rivers.vtu',np_asia,lon,lat,radius)

#------------------------------------------------
filename='africa_rivers.txt'

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

#export_to_vtu_flat('africa_rivers.vtu',np_africa,lon,lat)
export_to_vtu_sphere('africa_rivers.vtu',np_africa,lon,lat,radius)

#------------------------------------------------
filename='samer_rivers.txt'

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

#export_to_vtu_flat('samer_rivers.vtu',np_samer,lon,lat)
export_to_vtu_sphere('samer_rivers.vtu',np_samer,lon,lat,radius)

#------------------------------------------------
filename='namer_rivers.txt'

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

#export_to_vtu_flat('namer_rivers.vtu',np_namer,lon,lat)
export_to_vtu_sphere('namer_rivers.vtu',np_namer,lon,lat,radius)
























































































































































