import numpy as np

R_outer=3.397e6
R_inner=R_outer-1600e3

eta_max=1e25

# main parameter which controls resolution
# shound be 1,2,3,4, or 5
res=1

nnr=res*16+1 
nnt=res*80 

#-------------------------------------
# elasto-viscous rheology
#-------------------------------------

use_ev=False
mu=1e11
year=365.25*3600*24
dt=50*year

#-------------------------------------
# viscosity model
#-------------------------------------
# 1: isoviscous
# 2: steinberger data
# 3: three layer model

viscosity_model = 3

rho_crust=0#3300
eta_crust=1e25

rho_lith=0#3500
eta_lith=1e21

rho_mantle=0#3700
eta_mantle=6e20

eta0=6e20 # isoviscous case

eta_core=1e25
rho_core=0 #7200

#rho_crust+=3700
#rho_lith+=3700
#rho_mantle+=3700

#-------------------------------------
# blob setup 
#-------------------------------------

np_blob=res*30
R_blob=300e3
z_blob=R_outer-1000e3
rho_blob=rho_mantle-200
eta_blob=6e20

#-------------------------------------
#boundary conditions at planet surface
#-------------------------------------
#0: no-slip
#1: free-slip
#2: free (only top surface)

surface_bc=1

cmb_bc=1

#-------------------------------------
# gravity acceleration
#-------------------------------------

use_isog=True
g0=3.72

#-------------------------------------
#do not change
np_grav=0
nel_phi=20

