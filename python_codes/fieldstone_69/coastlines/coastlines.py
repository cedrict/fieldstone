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

#export_to_vtu_flat('europe_coastlines.vtu',np_europe,lon,lat)
export_to_vtu_sphere('europe_coastlines.vtu',np_europe,lon,lat,radius)

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

#export_to_vtu_flat('asia_coastlines.vtu',np_asia,lon,lat)
export_to_vtu_sphere('asia_coastlines.vtu',np_asia,lon,lat,radius)

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

#export_to_vtu_flat('africa_coastlines.vtu',np_africa,lon,lat)
export_to_vtu_sphere('africa_coastlines.vtu',np_africa,lon,lat,radius)

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

#export_to_vtu_flat('namer_coastlines.vtu',np_namer,lon,lat)
export_to_vtu_sphere('namer_coastlines.vtu',np_namer,lon,lat,radius)


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

#export_to_vtu_flat('samer_coastlines.vtu',np_samer,lon,lat)
export_to_vtu_sphere('samer_coastlines.vtu',np_samer,lon,lat,radius)

