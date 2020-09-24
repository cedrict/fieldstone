import numpy as np
from vtu_tools import * 

#------------------------------------------------

#np_pb=6048
np_pb=5974

radius=6371e3

#------------------------------------------------
filename='bird2003/pb2002_boundaries_copy.dig'

lon=np.zeros(np_pb,dtype=np.float64)   
lat=np.zeros(np_pb,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lon[counter]=float(columns[0])
    lat[counter]=float(columns[1])
    counter+=1
#end for

export_to_vtu_flat('plate_boundaries_map.vtu',np_pb,lon,lat)
export_to_vtu_sphere('plate_boundaries.vtu',np_pb,lon,lat,radius)


for i in range(0,np_pb):
    #lon[i]+=180
    if lon[i]<0:
       lon[i]+=360
 
export_to_vtu_flat('plate_boundaries_map2.vtu',np_pb,lon,lat)

for i in range(0,np_pb):
    lon[i]+=180
    if lon[i]>360:
       lon[i]-=360
export_to_vtu_flat('plate_boundaries_map3.vtu',np_pb,lon,lat)
