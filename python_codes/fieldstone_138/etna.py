from set_measurement_parameters import *

do_line_measurements=False
do_plane_measurements=False
do_spiral_measurements=False
do_path_measurements=True

Mx0=0
My0=4.085
Mz0=-6.290

nqdim=2

zpath_option=2 #based on dem + zpath_height

# 6 sites with ~3 paths & ~2 heights 
# rDEM: 2 or 5m (resolution)
# sDEM: 1 (largest ~2km), 2 (small ~300m), 3 (very small ~200m)
# site: 1,2,3,4,5,6
# path: 1,2,3 (except site 6, only 1)
# ho  : 1,2 (height option), site5 has 4 options.

rDEM=5
sDEM=5
site=6
path=1
ho=1

Lx,Ly,Lz,nelx,nely,nelz,xllcorner,yllcorner,npath,zpath_height,pathfile,topofile,error,IGRFx,IGRFy,IGRFz,Brefx,Brefy,Brefz=\
      set_measurement_parameters(rDEM,sDEM,site,path,ho)

if error:
   exit('combination does not exist -> terminate')
   
###############################################################################
# setup for case study into Mt. Etna field measurements, reproducing sites
# site measurement locations were ~1m apart, three paths per site
# 6 sites, nomenclature: 
# rDEM: resolution of DEM (either 2m or 5m, but 5m does not exist for site 3) 
# sDEM: size of DEM cuts (cuts were made with path kept ~ in the middle) see below for 
# path: the number allocated to each path on the sites (for 1-5: 3 paths)
#
#
#
#
#site  rDEM  sDEM  rough_size
# 1    2m    1     2000
# 1    2m    2     1400
# 1    2m    3     1000
# 1    2m    4     1400
# 1    2m    5     1000
# 1    2m    6     1400
# 1    2m    7     1000
# 1    2m    2     1400

# 1    5m    1     2000
# 1    5m    1     300
# 1    5m    1     200
#######################
# 2    2m    -     2000

# 2    5m    1     2000
# 2    5m    2     300
# 2    5m    3     200
#######################
# 3
#######################
# 4
#######################
# 5
#######################
# 6
###############################################################################

