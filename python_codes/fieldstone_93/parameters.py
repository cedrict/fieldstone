import numpy as np

gx=0
gy=-1

Lx=1
Ly=1

rad=0.123456789

n_p=32 # must be even

nstep=100
CFL=0.2
end_time=200

# the following parameterisation is 
# somewhat arbitrary

np_top=n_p
np_bottom=n_p
np_left=n_p
np_right=n_p
np_sphere=5*n_p

np_surf=5*n_p
