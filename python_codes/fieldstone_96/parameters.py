import numpy as np

R_inner=1.422e6
R_outer=3.389e6

np_inner=80   # max 150
np_outer=120  # max 450

np_blob=150
R_blob=200e3
z_blob=2800e3
rho_blob=-4000
eta_blob=1e22

eta_core=1e25

eta_max=1e23

np_wall=60 # max 200+ ?
nnr=np_wall
nnt=np_outer


#-------------------------------------
# elasto-viscous stuff
#-------------------------------------
mu=1e11
year=365.25*3600*24
dt=50*year
use_ev=True


#-------------------------------------
#forcing constant viscosity in domain
#-------------------------------------

isoviscous=False
eta0=1e22

#-------------------------------------
#boundary conditions at planet surface
#-------------------------------------
#0: no-slip
#1: free-slip

surface_bc=1





#-------------------------------------
#do not change
np_grav=0
nel_phi=20

rho_ref=3389

R_moho=R_outer-49.5e3
np_moho=0

R_trans=R_outer-1111.5e3
np_trans=0
