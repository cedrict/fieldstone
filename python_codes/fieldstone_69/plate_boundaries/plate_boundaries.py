import numpy as np
from vtu_tools import * 

#------------------------------------------------

np_pb=6048

radius=6371e3

#------------------------------------------------
filename='plate_boundaries.txt'


lon=np.zeros(np_pb,dtype=np.float64)   
lat=np.zeros(np_pb,dtype=np.float64)    

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    lon[counter]=columns[0]
    lat[counter]=columns[1]
    counter+=1
#end for

#export_to_vtu_flat('plate_boundaries.vtu',np_samer,lon,lat)
export_to_vtu_sphere('plate_boundaries.vtu',np_pb,lon,lat,radius)



 
