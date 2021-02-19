import numpy as np


rad=0.123456789
size=1./8.

#experiment=1 # stokes sphere in the middle
#experiment=2 # stokes sphere under surface
#experiment=3 # square block
#experiment=4 # vaks97 RT experiment
#experiment=5 # slab under surface
experiment=6 # scbe08

n_p=32 # must be even , default is 32

nstep=100
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

if experiment==5:
   gx=0
   gy=-1
   Lx=1
   Ly=1
   end_time=1000
   np_surf=5*n_p
   bc='freeslip'
   np_object=9*n_p
   y1=0.55
   delta=0.05
   xB=0.5
   yB=y1
   xA=xB-0.345
   yA=y1
   xC=xB+0.123
   yC=y1-0.123 
   xD=xB
   yD=y1-0.123
   xobject=xB
   yobject=yB


if experiment==6:
   gx=0.
   gy=-10
   Lx=3000e3
   Ly=700e3
   end_time=3e15
   np_surf=10*n_p
   bc='freeslip'
   np_object=5*n_p
   xobject=1500e3
   yobject=250e3

