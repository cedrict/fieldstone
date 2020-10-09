import numpy as np
from vtu_tools import *  

#------------------------------------------------------------------------------

np_world=22878

radius=6371e3

#------------------------------------------------
filename='coastlines_los_alamos.ascii'

lon=np.zeros(np_world,dtype=np.float64)   
lat=np.zeros(np_world,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lon[counter]=columns[0]
    lat[counter]=columns[1]
    counter+=1
#end for

export_to_vtu_flat('world_coastlines_map1.vtu',np_world,lon,lat)
export_to_vtu_sphere('world_coastlines_sphere1.vtu',np_world,lon,lat,radius)

for i in range(0,np_world):
    if lon[i]<0:
       lon[i]+=360
export_to_vtu_flat('world_coastlines_map2.vtu',np_world,lon,lat)

for i in range(0,np_world):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('world_coastlines_map3.vtu',np_world,lon,lat)
export_to_vtu_sphere('world_coastlines_sphere3.vtu',np_world,lon,lat,radius)

