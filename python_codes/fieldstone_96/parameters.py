import numpy as np

R_outer=3.397e6
R_inner=R_outer-1600e3

np_inner=80   # max 150
np_outer=120  # max 450

np_blob=150
R_blob=300e3
z_blob=R_outer-1000e3
rho_blob=-200
eta_blob=6e20

eta_core=1e25

eta_max=1e25

np_wall=60 # max 200+ ?
nnr=np_wall
nnt=np_outer




rho_surf=3000 # dyn topo

#-------------------------------------
# elasto-viscous stuff
#-------------------------------------
mu=1e11
year=365.25*3600*24
dt=50*year
use_ev=False

#-------------------------------------
# viscosity model
#-------------------------------------

viscosity_model = 3

eta_crust=1e24
eta_lith=1e21
eta_mantle=6e20

eta0=6e20

#-------------------------------------
#boundary conditions at planet surface
#-------------------------------------
#0: no-slip
#1: free-slip
#2: free top surface

surface_bc=1

#-------------------------------------
# gravity acceleration
#-------------------------------------

use_isog=True
g0=3.72

#-------------------------------------
#do not change
np_grav=0
nel_phi=20

#rho_ref=3389

R_moho=R_outer-49.5e3
np_moho=0

R_trans=R_outer-1111.5e3
np_trans=0
