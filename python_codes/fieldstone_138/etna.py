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
sDEM=1
site=1
path=1
ho=1

Lx,Ly,Lz,nelx,nely,nelz,xllcorner,yllcorner,npath,zpath_height,pathfile,topofile,error=\
      set_measurement_parameters(rDEM,sDEM,site,path,ho)

if error:
   exit('combination does not exist -> terminate')
