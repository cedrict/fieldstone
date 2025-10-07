import numpy as np
import sys as sys
import numba
from scipy.sparse import lil_matrix
import time as clock
import random
from scipy import special
import scipy.sparse as sps

###############################################################################

@numba.njit
def interpolate_vel_on_pt(xm,ym,hx,hy,xV,yV,iconV,nelx,u,v):
    ielx=int(xm/hx)
    iely=int(ym/hy)
    iel=nelx*(iely)+ielx
    xmin=xV[iconV[0,iel]] 
    ymin=yV[iconV[0,iel]] 
    rm=((xm-xmin)/hx-0.5)*2
    sm=((ym-ymin)/hy-0.5)*2
    N_V=basis_functions_V(rm,sm,order)
    um=np.dot(N_V,u[icon_V[:,iel]])
    vm=np.dot(N_V,v[icon_V[:,iel]])
    return um,vm,rm,sm,iel 

###############################################################################
# viscosity function
###############################################################################

def eta(T,e,y):
    TT=min(T,1)
    TT=max(T,0)
    if viscosity_model==0: # constant
       val=1
       rh=0

    if viscosity_model==1: # bugg08
       if y <= 0.77:
          val=50*np.exp(-6.9*TT) # lower mantle
          rh=1
       elif y <= 0.9:
          val=0.8*np.exp(-6.9*TT) # upper mantle
          rh=1
       else:
          sigma_y=1.5e5  
          eta_v=10*np.exp(-6.9*TT)
          eta_p=sigma_y/2/e
          if eta_v<eta_p:
             val=eta_v
             rh=1
          else:
             val=eta_p
             rh=2

    if viscosity_model==2: # brhv08
       bcoeff=np.log(1000)
       rh=0
       if y>0.9551:
          val=1000
       elif y>0.7187:
          val=np.exp(-bcoeff*TT)
       else:
          val=30*np.exp(-bcoeff*TT)

    if viscosity_model==3: # budt14
       rh=0
       if y>0.9585:
          val=np.exp(9.2103*(0.5-TT))
       elif y>0.7682:
          val=1./30.*np.exp(9.2103*(0.5-TT))
       else:
          val=(-6.24837*y + 6.8)*np.exp(9.2103*(0.5-TT))

    if viscosity_model==4: #mayw11
       rh=0
       etaref=0.01
       z=Ly-y
       if z<=0.23:
          G=1.
       elif z<=0.42:
          G=1+0.2*(z-0.23)/0.19
       elif z<=0.55:
          G=1.2-0.1*(z-0.42)/0.13
       else:
          G=1.1+0.7*(z-0.55)/0.45
       val=etaref*np.exp(12.66*(G/(0.15+1.7*TT)-1))
       val=min(1e3,val)
       val=max(1e-4,val)

       #print (y,z,TT,G,val)

    return val,rh

###############################################################################
# velocity shape functions
###############################################################################
# 
#  Q2          Q1
#  6---7---8   2-------3
#  |       |   |       |
#  3   4   5   |       | 
#  |       |   |       |
#  0---1---2   0-------1
#
###############################################################################

@numba.njit
def basis_functions_V(r,s,order):
    if order==1:
       N0=0.25*(1.-r)*(1.-s)
       N1=0.25*(1.+r)*(1.-s)
       N2=0.25*(1.-r)*(1.+s)
       N3=0.25*(1.+r)*(1.+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)
    if order==2:
       N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N1=    (1.-r**2) * 0.5*s*(s-1.)
       N2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N3= 0.5*r*(r-1.) *    (1.-s**2)
       N4=    (1.-r**2) *    (1.-s**2)
       N5= 0.5*r*(r+1.) *    (1.-s**2)
       N6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N7=    (1.-r**2) * 0.5*s*(s+1.)
       N8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)
    if order==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       N00= N1r*N1t 
       N01= N2r*N1t 
       N02= N3r*N1t 
       N03= N4r*N1t 
       N04= N1r*N2t 
       N05= N2r*N2t 
       N06= N3r*N2t 
       N07= N4r*N2t 
       N08= N1r*N3t 
       N09= N2r*N3t 
       N10= N3r*N3t 
       N11= N4r*N3t 
       N12= N1r*N4t 
       N13= N2r*N4t 
       N14= N3r*N4t 
       N15= N4r*N4t 
       return np.array([N00,N01,N02,N03,N04,N05,N06,N07,\
                        N08,N09,N10,N11,N12,N13,N14,N15],dtype=np.float64)
    if order==4:
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4) 
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4) 
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       N00=N1r*N1s
       N01=N2r*N1s
       N02=N3r*N1s
       N03=N4r*N1s
       N04=N5r*N1s
       N05=N1r*N2s
       N06=N2r*N2s
       N07=N3r*N2s
       N08=N4r*N2s
       N09=N5r*N2s
       N10=N1r*N3s
       N11=N2r*N3s
       N12=N3r*N3s
       N13=N4r*N3s
       N14=N5r*N3s
       N15=N1r*N4s
       N16=N2r*N4s
       N17=N3r*N4s
       N18=N4r*N4s
       N19=N5r*N4s
       N20=N1r*N5s
       N21=N2r*N5s
       N22=N3r*N5s
       N23=N4r*N5s
       N24=N5r*N5s
       return np.array([N00,N01,N02,N03,N04,\
                        N05,N06,N07,N08,N09,\
                        N10,N11,N12,N13,N14,\
                        N15,N16,N17,N18,N19,\
                        N20,N21,N22,N23,N24],dtype=np.float64)

###############################################################################
# velocity shape functions derivatives
###############################################################################

def basis_functions_V_dr(r,s,order):
    if order==1:
       dNdr0=-0.25*(1.-s) 
       dNdr1=+0.25*(1.-s) 
       dNdr2=-0.25*(1.+s) 
       dNdr3=+0.25*(1.+s) 
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)
    if order==2:
       dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr1=       (-2.*r) * 0.5*s*(s-1)
       dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr3= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr4=       (-2.*r) *   (1.-s**2)
       dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr7=       (-2.*r) * 0.5*s*(s+1)
       dNdr8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr_4,\
                        dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)
    if order==3:
       dN1rdr=( +1 +18*r -27*r**2)/16
       dN2rdr=(-27 -18*r +81*r**2)/16
       dN3rdr=(+27 -18*r -81*r**2)/16
       dN4rdr=( -1 +18*r +27*r**2)/16
       N1s=(-1    +s +9*s**2 - 9*s**3)/16
       N2s=(+9 -27*s -9*s**2 +27*s**3)/16
       N3s=(+9 +27*s -9*s**2 -27*s**3)/16
       N4s=(-1    -s +9*s**2 + 9*s**3)/16
       dNdr_00= dN1rdr* N1s 
       dNdr_01= dN2rdr* N1s 
       dNdr_02= dN3rdr* N1s 
       dNdr_03= dN4rdr* N1s 
       dNdr_04= dN1rdr* N2s 
       dNdr_05= dN2rdr* N2s 
       dNdr_06= dN3rdr* N2s 
       dNdr_07= dN4rdr* N2s 
       dNdr_08= dN1rdr* N3s 
       dNdr_09= dN2rdr* N3s 
       dNdr_10= dN3rdr* N3s 
       dNdr_11= dN4rdr* N3s 
       dNdr_12= dN1rdr* N4s 
       dNdr_13= dN2rdr* N4s 
       dNdr_14= dN3rdr* N4s 
       dNdr_15= dN4rdr* N4s 
       return np.array([dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,dNdr_05,dNdr_06,dNdr_07,\
                        dNdr_08,dNdr_09,dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,dNdr_15],dtype=np.float64)
    if order==4:
       dN1dr=(    1 - 2*r -12*r**2 +16*r**3)/6
       dN2dr=(   -8 +32*r +24*r**2 -64*r**3)/6
       dN3dr=(      -10*r          +16*r**3) 
       dN4dr=(  8   +32*r -24*r**2 -64*r**3)/6
       dN5dr=(   -1 - 2*r +12*r**2 +16*r**3)/6
       N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
       N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
       N3s=(1     - 5*s**2         + 4*s**4) 
       N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
       N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
       dNdr_00= dN1dr*N1s
       dNdr_01= dN2dr*N1s
       dNdr_02= dN3dr*N1s
       dNdr_03= dN4dr*N1s
       dNdr_04= dN5dr*N1s
       dNdr_05= dN1dr*N2s
       dNdr_06= dN2dr*N2s
       dNdr_07= dN3dr*N2s
       dNdr_08= dN4dr*N2s
       dNdr_09= dN5dr*N2s
       dNdr_10= dN1dr*N3s
       dNdr_11= dN2dr*N3s
       dNdr_12= dN3dr*N3s
       dNdr_13= dN4dr*N3s
       dNdr_14= dN5dr*N3s
       dNdr_15= dN1dr*N4s
       dNdr_16= dN2dr*N4s
       dNdr_17= dN3dr*N4s
       dNdr_18= dN4dr*N4s
       dNdr_19= dN5dr*N4s
       dNdr_20= dN1dr*N5s
       dNdr_21= dN2dr*N5s
       dNdr_22= dN3dr*N5s
       dNdr_23= dN4dr*N5s
       dNdr_24= dN5dr*N5s
       return np.array([dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,\
                        dNdr_05,dNdr_06,dNdr_07,dNdr_08,dNdr_09,\
                        dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,\
                        dNdr_15,dNdr_16,dNdr_17,dNdr_18,dNdr_19,\
                        dNdr_20,dNdr_21,dNdr_22,dNdr_23,dNdr_24],dtype=np.float64)

def basis_functions_V_ds(r,s,order):
    if order==1:
       dNds0=-0.25*(1.-r)
       dNds1=-0.25*(1.+r)
       dNds2=+0.25*(1.-r)
       dNds3=+0.25*(1.+r)
       return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)
    if order==2:
       dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds1=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds3= 0.5*r*(r-1.) *       (-2.*s)
       dNds4=    (1.-r**2) *       (-2.*s)
       dNds5= 0.5*r*(r+1.) *       (-2.*s)
       dNds6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds7=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,\
                        dNds5,dNds6,dNds7,dNds8],dtype=np.float64)
    if order==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       dNds_00= N1r*dN1sds 
       dNds_01= N2r*dN1sds 
       dNds_02= N3r*dN1sds 
       dNds_03= N4r*dN1sds 
       dNds_04= N1r*dN2sds 
       dNds_05= N2r*dN2sds 
       dNds_06= N3r*dN2sds 
       dNds_07= N4r*dN2sds 
       dNds_08= N1r*dN3sds 
       dNds_09= N2r*dN3sds 
       dNds_10= N3r*dN3sds 
       dNds_11= N4r*dN3sds 
       dNds_12= N1r*dN4sds 
       dNds_13= N2r*dN4sds 
       dNds_14= N3r*dN4sds 
       dNds_15= N4r*dN4sds
       return np.array([dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,dNds_05,dNds_06,dNds_07,\
                        dNds_08,dNds_09,dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,dNds_15],dtype=np.float64)
    if order==4:
       N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
       N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
       N3r=(1     - 5*r**2         + 4*r**4) 
       N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
       N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
       dN1ds=(    1 - 2*s -12*s**2 +16*s**3)/6
       dN2ds=( -8*1 +32*s +24*s**2 -64*s**3)/6
       dN3ds=(      -10*s          +16*s**3) 
       dN4ds=(  8   +32*s -24*s**2 -64*s**3)/6
       dN5ds=(   -1 - 2*s +12*s**2 +16*s**3)/6
       dNds_00= N1r*dN1ds
       dNds_01= N2r*dN1ds
       dNds_02= N3r*dN1ds
       dNds_03= N4r*dN1ds
       dNds_04= N5r*dN1ds
       dNds_05= N1r*dN2ds
       dNds_06= N2r*dN2ds
       dNds_07= N3r*dN2ds
       dNds_08= N4r*dN2ds
       dNds_09= N5r*dN2ds
       dNds_10= N1r*dN3ds
       dNds_11= N2r*dN3ds
       dNds_12= N3r*dN3ds
       dNds_13= N4r*dN3ds
       dNds_14= N5r*dN3ds
       dNds_15= N1r*dN4ds
       dNds_16= N2r*dN4ds
       dNds_17= N3r*dN4ds
       dNds_18= N4r*dN4ds
       dNds_19= N5r*dN4ds
       dNds_20= N1r*dN5ds
       dNds_21= N2r*dN5ds
       dNds_22= N3r*dN5ds
       dNds_23= N4r*dN5ds
       dNds_24= N5r*dN5ds
       return np.array([dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,\
                        dNds_05,dNds_06,dNds_07,dNds_08,dNds_09,\
                        dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,\
                        dNds_15,dNds_16,dNds_17,dNds_18,dNds_19,\
                        dNds_20,dNds_21,dNds_22,dNds_23,dNds_24],dtype=np.float64)

###############################################################################
# pressure shape functions 
###############################################################################

def basis_functions_P(r,s,order):
    if order==1:
       return np.array([1],dtype=np.float64)
    if order==2:
       N0=0.25*(1-r)*(1-s)
       N1=0.25*(1+r)*(1-s)
       N2=0.25*(1-r)*(1+s)
       N3=0.25*(1+r)*(1+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)
    if order==3:
       N0= 0.5*r*(r-1) * 0.5*s*(s-1)
       N1=    (1-r**2) * 0.5*s*(s-1)
       N2= 0.5*r*(r+1) * 0.5*s*(s-1)
       N3= 0.5*r*(r-1) *    (1-s**2)
       N4=    (1-r**2) *    (1-s**2)
       N5= 0.5*r*(r+1) *    (1-s**2)
       N6= 0.5*r*(r-1) * 0.5*s*(s+1)
       N7=    (1-r**2) * 0.5*s*(s+1)
       N8= 0.5*r*(r+1) * 0.5*s*(s+1)
       return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)
    if order==4:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       N00=N1r*N1t ; N01=N2r*N1t ; N02=N3r*N1t ; N03=N4r*N1t 
       N04=N1r*N2t ; N05=N2r*N2t ; N06=N3r*N2t ; N07=N4r*N2t 
       N08=N1r*N3t ; N09=N2r*N3t ; N10=N3r*N3t ; N11=N4r*N3t 
       N12=N1r*N4t ; N13=N2r*N4t ; N14=N3r*N4t ; N15=N4r*N4t 
       return np.array([N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
                        N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15],dtype=np.float64)

###############################################################################
# constants

eps=1e-10
sqrt2=np.sqrt(2)

###############################################################################

print("*******************************")
print("********** stone 88 ***********")
print("*******************************")

ndim=2    # number of dimensions
ndof_V=2  # number of velocity degrees of freedom per node

Lx=4.
Ly=1.

if int(len(sys.argv) == 7):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   order = int(sys.argv[4])
   Ra    = float(sys.argv[5])
   nstep = int(sys.argv[6])
else:
   nelx = 96   # number of elements in x direction
   nely = 32   # number of elements in y direction
   visu = 1    # trigger for vtu output
   order= 1    # polynomial order for velocity
   Ra   = 1e4  # Rayleigh number
   nstep= 1000 # number of time steps

nmarker_per_dim=4 
random_markers=True

###################
# viscosity model:
# 0: constant
# 1: bugg08
# 2: brhv08
# 3: budt14
# 4: mayv11
viscosity_model=1

debug=True

tfinal=1e2

CFL_nb=0.9

apply_filter=False # Lenardic & Kaula filter

# streamline upwind stabilisation
# 0: none
# 1: standard
# 2: using sqrt15
supg_type=1

every_vtu=1 # how often vtu files are generated
every_profile=1 # how often vtu files are generated

top_bc_noslip=False
bot_bc_noslip=False

nel=nelx*nely
nnx=order*nelx+1  # number of V nodes, x direction
nny=order*nely+1  # number of V nodes, y direction
nn_V=nnx*nny

if order==1:
   nn_P=nelx*nely
   m_V=4 
   m_P=1 
   r_V=[-1,+1,-1,+1]
   s_V=[-1,-1,+1,+1]
   r_P=[0]
   s_P=[0]

if order==2:
   nn_P=(nelx+1)*(nely+1)
   m_V=9 
   m_P=4 
   r_V=[-1,0,+1,-1,0,+1,-1,0,+1]
   s_V=[-1,-1,-1,0,0,0,+1,+1,+1]
   r_P=[-1,+1,-1,+1]
   s_P=[-1,-1,+1,+1]

if order==3:
   nn_P=(2*nelx+1)*(2*nely+1)
   m_V=16 
   m_P=9  
   r_V=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   s_V=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]
   r_P=[-1,0,+1,-1,0,+1,-1,0,+1]
   s_P=[-1,-1,-1,0,0,0,+1,+1,+1]

if order==4:
   nn_P=(3*nelx+1)*(3*nely+1)
   m_V=25 
   m_P=16 
   r_V=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]
   s_V=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
   r_P=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   s_P=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs
Nfem_T=nn_V        # nb of temperature dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

###############################################################################
# Gauss quadrature setup
###############################################################################

nq_per_dim=order+1

if nq_per_dim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nq_per_dim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

###############################################################################
# open output files
###############################################################################

vrms_file=open('vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
conv_file=open('conv.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")
ustats_file=open('stats_u.ascii',"w")
vstats_file=open('stats_v.ascii',"w")

###############################################################################

print ('Ra       =',Ra)
print ('order    =',order)
print ('nnx      =',nnx)
print ('nny      =',nny)
print ('nn_V     =',nn_V)
print ('nn_P     =',nn_P)
print ('nel      =',nel)
print ('Nfem_V   =',Nfem_V)
print ('Nfem_P   =',Nfem_P)
print ('Nfem     =',Nfem)
print ('nq_per_dim =',nq_per_dim)
print("-----------------------------")

###############################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i],order))

###############################################################################
# checking that all pressure shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i],order))

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/order
        y_V[counter]=j*hy/order
        counter+=1
    #end for
#end for

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
top=np.zeros(nel,dtype=bool)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                icon_V[counter2,counter]=i*order+l+j*order*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        top[counter]=(j==nely-1)
        counter += 1
    #end for
#end for

# creating a dedicated connectivity array to plot the solution on Q1 space
# different icon array but same velocity nodes.

nel2=(nnx-1)*(nny-1)
iconQ1=np.zeros((4,nel2),dtype=np.int32)
counter=0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid
# if Q1P0 elements are used then the pressure node is in the center of elt 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64) # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64) # y coordinates

if order==1:
   for iel in range(0,nel):
       x_P[iel]=sum(x_V[icon_V[0:m_V,iel]])/m_V
       y_P[iel]=sum(y_V[icon_V[0:m_V,iel]])/m_V
    #end for
else:      
   counter=0    
   for j in range(0,(order-1)*nely+1):
       for i in range(0,(order-1)*nelx+1):
           x_P[counter]=i*hx/(order-1)
           y_P[counter]=j*hy/(order-1)
           counter+=1
       #end for
    #end for
#end if

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

if order==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_P[0,counter]=counter
           counter += 1
       #end for
   #end for
else:
   om1=order-1
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   icon_P[counter2,counter]=i*om1+l+j*om1*(om1*nelx+1)+(om1*nelx+1)*k 
                   counter2+=1
               #end for
           #end for
           counter += 1
       #end for
   #end for

print("build icon_P: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]= 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]= 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1]= 0.
       if bot_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]= 0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1]= 0.
       if top_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]= 0.

print("velocity b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_V):
    if y_V[i]<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=1.
    if y_V[i]>(Ly-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=0.
#end for

print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)
T_prev=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    #T[i]=1.-y_V[i]  # conductive profile
    T[i]=0.5
    kappa=1
    T[i]-=0.5*special.erfc((Ly-y_V[i])/2/np.sqrt(kappa*0.001))
    T[i]+=0.5*special.erfc(y_V[i]/2/np.sqrt(kappa*0.001))
    T[i]-=0.03*(np.cos(2.132*np.pi*x_V[i]/Lx)+\
                np.cos(3.333*np.pi*x_V[i]/Lx)+\
                np.cos(7.123*np.pi*x_V[i]/Lx)) *np.sin(np.pi*y_V[i]/Ly)

T_prev[:]=T[:]

if debug: np.savetxt('temperature_init.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))


###############################################################################
# marker setup
###############################################################################
start=clock.time()

nmarker_per_element=nmarker_per_dim**2
nmarker=nel*nmarker_per_element

swarm_x=np.zeros(nmarker,dtype=np.float64)  
swarm_y=np.zeros(nmarker,dtype=np.float64)  
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64)  
swarm_active=np.zeros(nmarker,dtype=bool) 

if random_markers:
   counter=0
   for iel in range(0,nel):
       x1=x_V[icon_V[0,iel]] ; y1=y_V[icon_V[0,iel]]
       x2=x1+hx              ; y2=y1
       x3=x1+hx              ; y3=y1+hy
       x4=x1                 ; y4=y1+hy
       for im in range(0,nmarker_per_element):
           # generate random numbers r,s between 0 and 1
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N1=0.25*(1-r)*(1-s)
           N2=0.25*(1+r)*(1-s)
           N3=0.25*(1+r)*(1+s)
           N4=0.25*(1-r)*(1+s)
           swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
           swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
           counter+=1
       #end for
   #end for
else:
   counter=0
   for iel in range(0,nel):
       x1=x_V[icon_V[0,iel]] ; y1=y_V[icon_V[0,iel]]
       x2=x1+hx              ; y2=y1
       x3=x1+hx              ; y3=y1+hy
       x4=x1                 ; y4=y1+hy
       for j in range(0,nmarker_per_dim):
           for i in range(0,nmarker_per_dim):
               r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
               s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               counter+=1
           #end for
       #end for
   #end for

swarm_active[:]=True

print("     -> nmarker %d " % nmarker)
print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("markers setup: %.3f s" % (clock.time() - start))

###############################################################################
# marker paint
###############################################################################
start=clock.time()

swarm_mat=np.zeros(nmarker,dtype=np.int32)  

for i in [0,2,4,6,8,10,12,14]:
    dx=Lx/16
    for im in range (0,nmarker):
        if swarm_x[im]>i*dx and swarm_x[im]<(i+1)*dx:
           swarm_mat[im]+=1

for i in [0,2,4,6,8]:
    dy=Ly/4
    for im in range (0,nmarker):
        if swarm_y[im]>i*dy and swarm_y[im]<(i+1)*dy:
           swarm_mat[im]+=1

swarm_y0=np.zeros(nmarker,dtype=np.float64)  
swarm_y0[:]=swarm_y[:]

print("markers paint: %.3f s" % (clock.time()-start))


###################################################################################################
###################################################################################################
# time stepping loop
###################################################################################################
###################################################################################################

u_prev=np.zeros(nn_V,dtype=np.float64)
v_prev=np.zeros(nn_V,dtype=np.float64)
exx_n =np.zeros(nn_V,dtype=np.float64)  
eyy_n =np.zeros(nn_V,dtype=np.float64)  
exy_n =np.zeros(nn_V,dtype=np.float64)  
Tvect =np.zeros(m_V,dtype=np.float64)   
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

time=0


for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start = clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    b_fem=np.zeros(Nfem,dtype=np.float64) 
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
    N_mat=np.zeros((3,m_P),dtype=np.float64) 
    jcbi=np.zeros((ndim,ndim),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy
    jcob=hx*hy/4

    for iel in range(0,nel):

        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_V=basis_functions_V(rq,sq,order)
                N_P=basis_functions_P(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)

                # jcob,jcbi precomputed
                JxWq=jcob*weightq

                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])

                exxq=np.dot(N_V,exx_n[icon_V[:,iel]])
                eyyq=np.dot(N_V,eyy_n[icon_V[:,iel]])
                exyq=np.dot(N_V,exy_n[icon_V[:,iel]])
                eq=np.sqrt(0.5*(exxq**2+eyyq**2)+exyq**2)

                etaq,dum=eta(Tq,eq,yq)

                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                K_el+=B.T.dot(C.dot(B))*etaq*JxWq

                # compute elemental rhs vector
                for i in range(0,m_V):
                    f_el[ndof_V*i+1]+=N_V[i]*JxWq*Ra*Tq
                #end for

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=0.
                #end for

                G_el-=B.T.dot(N_mat)*JxWq

            # end for jq
        # end for iq

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0
                #end if
            #end for
        #end for

        # assemble matrix and right hand side
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1+i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2+i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_fem[m1,m2] += K_el[ikk,jkk]
                        #end if
                    #end for
                #end for
                for k2 in range(0,m_P):
                    m2 =icon_P[k2,iel]
                    A_fem[m1,Nfem_V+m2]+=G_el[ikk,k2]
                    A_fem[Nfem_V+m2,m1]+=G_el[ikk,k2]
                #end for
                b_fem[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            b_fem[Nfem_V+m2]+=h_el[k2]
        #end for

    #end for iel

    print("build FE matrix Stokes: %.3fs" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    ustats_file.write("%6e %6e %6e\n" % (time,np.min(u),np.max(u)))
    vstats_file.write("%6e %6e %6e\n" % (time,np.min(v),np.max(v)))
    ustats_file.flush()
    vstats_file.flush()

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time() - start))

    ###########################################################################
    # normalise pressure
    ###########################################################################

    if order==1: 
       avrgP=0
       for iel in range(0,nel):
           if top[iel]:
              avrgP+=p[iel]*hx

    elif order==2:
       avrgP=0
       for iel in range(0,nel):
           if top[iel]:
              for iq in range(0,nq_per_dim):
                  rq=qcoords[iq]
                  sq=1
                  N_P=basis_functions_P(rq,sq,order)
                  pq=np.dot(N_P,p[icon_P[:,iel]])
                  avrgP+=pq*qweights[iq]*hx/2

    avrgP/=Lx
    #print('********************',avrgP)

    p[:]-=avrgP

    ###########################################################################
    # compute timestep value
    ###########################################################################

    dt1=CFL_nb*hx/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*hx**2 #/(hcond/hcapa/rho0)=1 

    dt=np.min([dt1,dt2])

    print('     -> dt1 = %.6f' %dt1)
    print('     -> dt2 = %.6f' %dt2)
    print('     -> dt  = %.6f' %dt)

    print('     -> time= %.6f; tfinal= %.6f' %(time,tfinal))

    dt_file.write("%e %e %e %e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start = clock.time()

    A_mat=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    rhs=np.zeros(Nfem_T,dtype=np.float64)           # FE rhs 
    B=np.zeros((2,m_V),dtype=np.float64)  # gradient matrix B 
    N_mat=np.zeros((m_V,1),dtype=np.float64)        # shape functions
    N_mat_supg=np.zeros((m_V,1),dtype=np.float64)   # shape functions
    tau_supg=np.zeros(nel*nq_per_dim**ndim,dtype=np.float64)

    counterq=0   
    for iel in range (0,nel):

        b_el=np.zeros(m_V,dtype=np.float64)
        A_el=np.zeros((m_V,m_V),dtype=np.float64)
        Ka=np.zeros((m_V,m_V),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_V,m_V),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_V,m_V),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m_V):
            Tvect[k]=T[icon_V[k,iel]]
        #end for

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[:,0]=basis_functions_V(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)

                # jacobian matrix already precomputed

                vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
                vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])

                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                B[0,:]=dNdx_V[:]
                B[1,:]=dNdy_V[:]

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2)/2/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
    
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B))

                # compute mass matrix
                MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob

                # compute diffusion matrix
                Kd+=B.T.dot(B)*weightq*jcob

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B))*weightq*jcob

                counterq+=1

            #end for
        #end for

        A_el=MM+0.5*(Ka+Kd)*dt
        b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_V):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            for k2 in range(0,m_V):
                m2=icon_V[k2,iel]
                A_mat[m1,m2]+=A_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel

    print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix temperature: %.3f s" % (clock.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = clock.time()

    Traw = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(Traw),np.max(Traw)))
    Tstats_file.flush()

    print("solve T time: %.3f s" % (clock.time() - start))

    #################################################################
    # apply Lenardic & Kaula filter
    #################################################################
    start = clock.time()

    if apply_filter:

       # step 1: compute the initial sum 'sum0' of all values of T

       sum0=np.sum(Traw)

       # step 2: find the minimum value Tmin of T
  
       minT=np.min(Traw)
  
       # step 3: find the maximum value Tmax of T
  
       maxT=np.max(Traw)

       # step 4: set T=0 if T<=|Tmin|

       for i in range(0,NV):
           if Traw[i]<=abs(minT):
              Traw[i]=0

       # step 5: set T=1 if T>=2-Tmax

       for i in range(0,NV):
           if Traw[i]>=2-maxT:
              Traw[i]=1

       # step 6: compute the sum sum1 of all values of T

       sum1=np.sum(Traw)

       # step 7: compute the number num of 0<T<1

       num=0
       for i in range(0,NV):
           if Traw[i]>0 and Traw[i]<1:
              num+=1

       # step 8: add (sum1-sum0)/num to all 0<T<1
       
       for i in range(0,NV):
           if Traw[i]>0 and Traw[i]<1:
              Traw[i]+=(sum1-sum0)/num 

       print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    #end if
       
    T[:]=Traw[:]

    print("apply L&K filter: %.3f s" % (clock.time() - start))

    #################################################################
    # compute vrms & <T>
    #################################################################
    start = clock.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                JxWq=jcob*weightq
                N_V=basis_functions_V(rq,sq,order)
                N_P=basis_functions_P(rq,sq,order)
                dNdr_V=basis_functions_V_dr(rq,sq,order)
                dNds_V=basis_functions_V_ds(rq,sq,order)
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                vrms+=(uq**2+vq**2)*JxWq
                Tavrg+=Tq*JxWq
            #end for jq
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly))
    Tavrg/=(Lx*Ly)

    Tavrg_file.write("%e %e\n" % (time,Tavrg))
    Tavrg_file.flush()

    vrms_file.write("%e %.10f\n" % (time,vrms))
    vrms_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (clock.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = clock.time()
    
    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  
    sr_n=np.zeros(nn_V,dtype=np.float64)  
    eta_n=np.zeros(nn_V,dtype=np.float64)  
    rh_n=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)
    c=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,m_V):
            rq=r_V[i]
            sq=s_V[i]
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            #jacobian pre computed
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            e_xx=np.dot(dNdx_V,u[icon_V[:,iel]])
            e_yy=np.dot(dNdy_V,v[icon_V[:,iel]])
            e_xy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

            inode=icon_V[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            q[inode]+=np.dot(N_P,p[icon_P[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    q/=count

    sr_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

    for i in range(0,nn_V):
        eta_n[i],rh_n[i]=eta(T[i],sr_n[i],y_V[i])

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> sr_n (m,M) %.6e %.6e " %(np.min(sr_n),np.max(sr_n)))
    print("     -> eta_n (m,M) %.6e %.6e " %(np.min(eta_n),np.max(eta_n)))

    if debug:
       np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')
       np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (clock.time() - start))

    #####################################################################
    # compute temperature & visc profile
    #####################################################################
    start = clock.time()

    if istep%every_profile==0:
       eta_profile=np.zeros(nny,dtype=np.float64)  
       eta_profile_min=np.zeros(nny,dtype=np.float64) ; eta_profile_min[:]=1e20   
       eta_profile_max=np.zeros(nny,dtype=np.float64)
       T_profile=np.zeros(nny,dtype=np.float64)  
       T_profile_min=np.ones(nny,dtype=np.float64)  
       T_profile_max=np.zeros(nny,dtype=np.float64)  
       y_profile=np.zeros(nny,dtype=np.float64)  

       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               eta_profile[j]+=eta_n[counter]/nnx
               T_profile[j]+=T[counter]/nnx
               T_profile_min[j]=min(T[counter],T_profile_min[j])
               T_profile_max[j]=max(T[counter],T_profile_max[j])
               eta_profile_min[j]=min(eta_n[counter],eta_profile_min[j])
               eta_profile_max[j]=max(eta_n[counter],eta_profile_max[j])
               y_profile[j]=y_V[counter]
               counter+=1
           #end for
       #end for

       np.savetxt('profile_T_{:06d}.ascii'.format(istep),\
                   np.array([y_profile,T_profile,T_profile_min,T_profile_max]).T)
       np.savetxt('profile_eta_{:06d}.ascii'.format(istep),\
                   np.array([y_profile,eta_profile,eta_profile_min,eta_profile_max]).T)

    print("compute profiles: %.3f s" % (clock.time() - start))

    #####################################################################
    # advect markers
    #####################################################################
    start = clock.time()

    RKorder=1 # cheap!

    if RKorder==1: 

       for im in range(0,nmarker):
           if swarm_active[im]:
              swarm_u[im],swarm_v[im],rm,sm,iel=interpolate_vel_on_pt(swarm_x[im],swarm_y[im],\
                                                hx,hy,x_V,y_V,icon_V,nelx,u,v)
              swarm_x[im]+=swarm_u[im]*dt
              swarm_y[im]+=swarm_v[im]*dt
              if swarm_x[im]<0 or swarm_x[im]>Lx or swarm_y[im]<0 or swarm_y[im]>Ly:
                 swarm_active[im]=False
                 swarm_x[im]=-0.0123
                 swarm_y[im]=-0.0123
           # end if active
       # end for im

    else:

       exit('no higher order RK yet')

    print("advect markers: %.3f s" % (clock.time() - start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start = clock.time()

    if istep%every_vtu==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='press' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='delta T' Format='ascii'> \n")
       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               vtufile.write("%e \n" %(T[counter]-T_profile[j]))
               counter+=1
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %eta_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rheology' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %rh_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %sr_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev stress' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" %(2*eta_n[i]*sr_n[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       ########################################################################
       # export markers to vtu file
       ########################################################################

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%e %e %e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%e %e %e \n" %(swarm_u[im],swarm_v[im],0.))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='paint' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%e \n" % swarm_mat[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='y0' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%e \n" % swarm_y0[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu file: %.3f s" % (clock.time() - start))

    ###################################

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]

    time+=dt

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
