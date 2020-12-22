import numpy as np


rad=0.123456789
size=1./8.

#experiment=1 # stokes sphere in the middle
#experiment=2 # stokes sphere under surface
#experiment=3 # square block
experiment=4  # vaks97 RT experiment

n_p=64 # must be even , default is 32

nstep=500
CFL=0.25

# the following parameterisation is 
# somewhat arbitrary

np_top=n_p
np_bottom=n_p
np_left=2
np_right=2

if experiment==1:
   gx=0
   gy=-1
   Lx=1
   Ly=1
   end_time=0
   np_surf=3*n_p
   xobject=0.5
   yobject=0.5
   bc='noslip'
   np_object=5*n_p

if experiment==2:
   gx=0
   gy=-1
   Lx=1
   Ly=1
   end_time=30
   np_surf=5*n_p
   xobject=0.5
   yobject=0.6
   bc='freeslip'
   np_object=5*n_p

if experiment==3:
   gx=0
   gy=-1
   Lx=1
   Ly=1
   end_time=0
   np_surf=3*n_p
   xobject=0.5
   yobject=0.5
   bc='noslip'
   np_object=4*n_p

if experiment==4:
   gx=0
   gy=-10
   Lx=0.9142
   Ly=1
   end_time=200
   np_surf=8*n_p
   xobject=0.
   yobject=0.
   bc='freeslip'
   np_object=0


