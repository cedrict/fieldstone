import numpy as np
import sys as sys
import scipy
import math as math
#import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing
import random
from scipy import special
import scipy.sparse as sps

###############################################################################

def interpolate_vel_on_pt(xm,ym):
    ielx=int(xm/Lx*nelx)
    iely=int(ym/Ly*nely)
    #if ielx<0:
    #   exit('ielx<0')
    #if iely<0:
    #   exit('iely<0')
    #if ielx>=nelx:
    #   exit('ielx>nelx')
    #if iely>=nely:
    #   exit('iely>nely')
    iel=nelx*(iely)+ielx
    xmin=xV[iconV[0,iel]] ; xmax=xmin+hx #xV[iconV[8,iel]]
    ymin=yV[iconV[0,iel]] ; ymax=ymin+hy #yV[iconV[8,iel]]
    rm=((xm-xmin)/(xmax-xmin)-0.5)*2
    sm=((ym-ymin)/(ymax-ymin)-0.5)*2
    NNNV[0:mV]=NNV(rm,sm,order)
    um=0.
    vm=0.
    for k in range(0,mV):
        um+=NNNV[k]*u[iconV[k,iel]]
        vm+=NNNV[k]*v[iconV[k,iel]]
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

def NNV(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)
    if order==2:
       N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N_1=    (1.-r**2) * 0.5*s*(s-1.)
       N_2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N_3= 0.5*r*(r-1.) *    (1.-s**2)
       N_4=    (1.-r**2) *    (1.-s**2)
       N_5= 0.5*r*(r+1.) *    (1.-s**2)
       N_6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N_7=    (1.-r**2) * 0.5*s*(s+1.)
       N_8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)
    if order==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       N_00= N1r*N1t 
       N_01= N2r*N1t 
       N_02= N3r*N1t 
       N_03= N4r*N1t 
       N_04= N1r*N2t 
       N_05= N2r*N2t 
       N_06= N3r*N2t 
       N_07= N4r*N2t 
       N_08= N1r*N3t 
       N_09= N2r*N3t 
       N_10= N3r*N3t 
       N_11= N4r*N3t 
       N_12= N1r*N4t 
       N_13= N2r*N4t 
       N_14= N3r*N4t 
       N_15= N4r*N4t 
       return np.array([N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
                        N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15],dtype=np.float64)
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
       N_00= N1r*N1s
       N_01= N2r*N1s
       N_02= N3r*N1s
       N_03= N4r*N1s
       N_04= N5r*N1s
       N_05= N1r*N2s
       N_06= N2r*N2s
       N_07= N3r*N2s
       N_08= N4r*N2s
       N_09= N5r*N2s
       N_10= N1r*N3s
       N_11= N2r*N3s
       N_12= N3r*N3s
       N_13= N4r*N3s
       N_14= N5r*N3s
       N_15= N1r*N4s
       N_16= N2r*N4s
       N_17= N3r*N4s
       N_18= N4r*N4s
       N_19= N5r*N4s
       N_20= N1r*N5s
       N_21= N2r*N5s
       N_22= N3r*N5s
       N_23= N4r*N5s
       N_24= N5r*N5s
       return np.array([N_00,N_01,N_02,N_03,N_04,\
                        N_05,N_06,N_07,N_08,N_09,\
                        N_10,N_11,N_12,N_13,N_14,\
                        N_15,N_16,N_17,N_18,N_19,\
                        N_20,N_21,N_22,N_23,N_24],dtype=np.float64)

#------------------------------------------------------------------------------
# velocity shape functions derivatives
#------------------------------------------------------------------------------

def dNNVdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s) 
       dNdr_1=+0.25*(1.-s) 
       dNdr_2=-0.25*(1.+s) 
       dNdr_3=+0.25*(1.+s) 
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)
    if order==2:
       dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr_1=       (-2.*r) * 0.5*s*(s-1)
       dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr_3= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr_4=       (-2.*r) *   (1.-s**2)
       dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr_6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr_7=       (-2.*r) * 0.5*s*(s+1)
       dNdr_8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,\
                        dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)
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

def dNNVds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)
    if order==2:
       dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds_1=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds_3= 0.5*r*(r-1.) *       (-2.*s)
       dNds_4=    (1.-r**2) *       (-2.*s)
       dNds_5= 0.5*r*(r+1.) *       (-2.*s)
       dNds_6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds_7=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds_8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)
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

#------------------------------------------------------------------------------
# pressure shape functions 
#------------------------------------------------------------------------------

def NNP(r,s,order):
    if order==1:
       N_1=1.
       return np.array([N_1],dtype=np.float64)
    if order==2:
       N_0=0.25*(1-r)*(1-s)
       N_1=0.25*(1+r)*(1-s)
       N_2=0.25*(1-r)*(1+s)
       N_3=0.25*(1+r)*(1+s)
       return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)
    if order==3:
       N_0= 0.5*r*(r-1) * 0.5*s*(s-1)
       N_1=    (1-r**2) * 0.5*s*(s-1)
       N_2= 0.5*r*(r+1) * 0.5*s*(s-1)
       N_3= 0.5*r*(r-1) *    (1-s**2)
       N_4=    (1-r**2) *    (1-s**2)
       N_5= 0.5*r*(r+1) *    (1-s**2)
       N_6= 0.5*r*(r-1) * 0.5*s*(s+1)
       N_7=    (1-r**2) * 0.5*s*(s+1)
       N_8= 0.5*r*(r+1) * 0.5*s*(s+1)
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
       N_00= N1r*N1t 
       N_01= N2r*N1t 
       N_02= N3r*N1t 
       N_03= N4r*N1t 
       N_04= N1r*N2t 
       N_05= N2r*N2t 
       N_06= N3r*N2t 
       N_07= N4r*N2t 
       N_08= N1r*N3t 
       N_09= N2r*N3t 
       N_10= N3r*N3t 
       N_11= N4r*N3t 
       N_12= N1r*N4t 
       N_13= N2r*N4t 
       N_14= N3r*N4t 
       N_15= N4r*N4t 
       return np.array([N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
                        N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15],dtype=np.float64)

###############################################################################
# constants

eps=1e-9
sqrt2=np.sqrt(2)

###############################################################################

print("-----------------------------")
print("--------- stone 88 ----------")
print("-----------------------------")

ndim=2   # number of dimensions
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom per node
ndofT=1  # number of temperature degrees of freedom per node

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


tfinal=1e2

CFL_nb=0.9

apply_filter=False # Lenardic & Kaula filter

# streamline upwind stabilisation
# 0: none
# 1: standard
# 2: using sqrt15
supg_type=1

every=1 # how often vtu files are generated

top_bc_noslip=False
bot_bc_noslip=False

nel=nelx*nely
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
NV=nnx*nny

if order==1:
   NP=nelx*nely
   mV=4     # number of velocity nodes making up an element
   mP=1     # number of pressure nodes making up an element
   rVnodes=[-1,+1,-1,+1]
   sVnodes=[-1,-1,+1,+1]
   rPnodes=[0]
   sPnodes=[0]

if order==2:
   NP=(nelx+1)*(nely+1)
   mV=9     # number of velocity nodes making up an element
   mP=4     # number of pressure nodes making up an element
   rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
   rPnodes=[-1,+1,-1,+1]
   sPnodes=[-1,-1,+1,+1]

if order==3:
   NP=(2*nelx+1)*(2*nely+1)
   mV=16    # number of velocity nodes making up an element
   mP=9     # number of pressure nodes making up an element
   rVnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sVnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]
   rPnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sPnodes=[-1,-1,-1,0,0,0,+1,+1,+1]

if order==4:
   NP=(3*nelx+1)*(3*nely+1)
   mV=25    # number of velocity nodes making up an element
   mP=16     # number of pressure nodes making up an element
   rVnodes=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]
   sVnodes=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
   rPnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sPnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs
NfemT=NV*ndofT       # nb of temperature dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

###############################################################################
# Gauss quadrature setup
###############################################################################

nqperdim=order+1

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nqperdim==5:
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
print ('NV       =',NV)
print ('NP       =',NP)
print ('nel      =',nel)
print ('NfemV    =',NfemV)
print ('NfemP    =',NfemP)
print ('Nfem     =',Nfem)
print ('nqperdim =',nqperdim)
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
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/order
        yV[counter]=j*hy/order
        counter+=1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("build V grid: %.3f s" % (timing.time() - start))

###############################################################################
# connectivity
###############################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
top=np.zeros(nel,dtype=bool)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                iconV[counter2,counter]=i*order+l+j*order*nnx+nnx*k
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
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

print("build iconV: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure grid
# if Q1P0 elements are used then the pressure node is in the center of elt 
###############################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64) # x coordinates
yP=np.empty(NP,dtype=np.float64) # y coordinates

if order==1:
   for iel in range(0,nel):
       xP[iel]=sum(xV[iconV[0:mV,iel]])*0.25
       yP[iel]=sum(yV[iconV[0:mV,iel]])*0.25
    #end for
#end if 
      
if order>1:
   counter=0    
   for j in range(0,(order-1)*nely+1):
       for i in range(0,(order-1)*nelx+1):
           xP[counter]=i*hx/(order-1)
           yP[counter]=j*hy/(order-1)
           counter+=1
       #end for
    #end for
#end if

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure connectivity array 
# if Q1P0 elements are used then the pressure node is in the center of elt 
###############################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)

if order==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           iconP[0,counter]=counter
           counter += 1
       #end for
   #end for

if order>1:
   om1=order-1
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   iconP[counter2,counter]=i*om1+l+j*om1*(om1*nelx+1)+(om1*nelx+1)*k 
                   counter2+=1
               #end for
           #end for
           counter += 1
       #end for
   #end for

print("build iconP: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if bot_bc_noslip:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if top_bc_noslip:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.

print("velocity b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]<eps:
       bc_fixT[i]=True ; bc_valT[i]=1.
    if yV[i]>(Ly-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
#end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# initial temperature
###############################################################################

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    #T[i]=1.-yV[i]  # conductive profile
    T[i]=0.5
    kappa=1
    T[i]-=0.5*special.erfc((Ly-yV[i])/2/np.sqrt(kappa*0.001))
    T[i]+=0.5*special.erfc(yV[i]/2/np.sqrt(kappa*0.001))
    T[i]-=0.03*(np.cos(2.132*np.pi*xV[i]/Lx)+\
                np.cos(3.333*np.pi*xV[i]/Lx)+\
                np.cos(7.123*np.pi*xV[i]/Lx)) *np.sin(np.pi*yV[i]/Ly)
#end for

T_prev[:]=T[:]

np.savetxt('temperature_init.ascii',np.array([xV,yV,T]).T,header='# x,y,T')

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# marker setup
###############################################################################
start = timing.time()

nmarker_per_element=nmarker_per_dim**2
nmarker=nel*nmarker_per_element

swarm_x=np.empty(nmarker,dtype=np.float64)  
swarm_y=np.empty(nmarker,dtype=np.float64)  
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64)  
swarm_active=np.zeros(nmarker,dtype=bool) 

if random_markers:
   counter=0
   for iel in range(0,nel):
       x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
       x2=x1+hx            ; y2=y1
       x3=x1+hx            ; y3=y1+hy
       x4=x1               ; y4=y1+hy
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
       x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
       x2=x1+hx            ; y2=y1
       x3=x1+hx            ; y3=y1+hy
       x4=x1               ; y4=y1+hy
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

print("markers setup: %.3f s" % (timing.time() - start))

###############################################################################
# marker paint
###############################################################################
start = timing.time()

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

print("markers paint: %.3f s" % (timing.time() - start))

###################################################################################################
###################################################################################################
# time stepping loop
###################################################################################################
###################################################################################################

u       = np.zeros(NV,dtype=np.float64) # x-component velocity
v       = np.zeros(NV,dtype=np.float64) # y-component velocity
u_prev  = np.zeros(NV,dtype=np.float64) # x-component velocity
v_prev  = np.zeros(NV,dtype=np.float64) # y-component velocity
NNNV    = np.zeros(mV,dtype=np.float64) # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64) # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64) # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64) # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64) # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64) # shape functions derivatives
Tvect   = np.zeros(mV,dtype=np.float64)   
exx_n   = np.zeros(NV,dtype=np.float64)  
eyy_n   = np.zeros(NV,dtype=np.float64)  
exy_n   = np.zeros(NV,dtype=np.float64)  
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

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
    start = timing.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    f_rhs=np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs=np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat=np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat=np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    jcbi=np.zeros((ndim,ndim),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy
    jcob=hx*hy/4

    for iel in range(0,nel):

        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:mV]=NNV(rq,sq,order)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
                dNNNVds[0:mV]=dNNVds(rq,sq,order)
                NNNP[0:mP]=NNP(rq,sq,order)

                # calculate jacobian matrix
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,mV):
                #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #jcbi = np.linalg.inv(jcb)
                #jcob = np.linalg.det(jcb)

                # compute dNdx & dNdy
                xq=0.
                yq=0.
                Tq=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                    exxq+=NNNV[k]*exx_n[iconV[k,iel]]
                    eyyq+=NNNV[k]*eyy_n[iconV[k,iel]]
                    exyq+=NNNV[k]*exy_n[iconV[k,iel]]
                    #dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    #dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                    dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]
                #end for

                e=np.sqrt(0.5*(exxq**2+eyyq**2)+exyq**2)

                # compute elemental a_mat matrix
                etaq,dum=eta(Tq,e,yq)
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*Ra*Tq
                #end for

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.
                #end for

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            # end for jq
        # end for iq

        # impose b.c. 
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                if bc_fix[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,mV*ndofV):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0
                #end if
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1+i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2+i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                        #end if
                    #end for
                #end for
                for k2 in range(0,mP):
                    m2 =iconP[k2,iel]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,k2]
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,k2]
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
        #end for

    #end for iel

    print("build FE matrix Stokes: %.3fs" % (timing.time()-start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (timing.time() - start))

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    sparse_matrix=A_sparse.tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    ustats_file.write("%6e %6e %6e\n" % (time,np.min(u),np.max(u)))
    vstats_file.write("%6e %6e %6e\n" % (time,np.min(v),np.max(v)))
    ustats_file.flush()
    vstats_file.flush()

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

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
              for iq in range(0,nqperdim):
                  sq=1
                  jcob=hx/2
                  NNNP[0:mP]=NNP(rq,sq,order)
                  pq=0
                  for k in range(0,mP):
                      pq+=NNNP[k]*p[iconP[k,iel]]
                  avrgP+=pq*qweights[iq]*jcob

    avrgP/=Lx
    #print('********************',avrgP)

    p[:]-=avrgP

    ###########################################################################
    # compute timestep value
    ###########################################################################

    dt1=CFL_nb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*(Lx/nelx)**2 #/(hcond/hcapa/rho0)=1 

    dt=np.min([dt1,dt2])

    print('     -> dt1 = %.6f' %dt1)
    print('     -> dt2 = %.6f' %dt2)
    print('     -> dt  = %.6f' %dt)

    time+=dt

    print('     -> time= %.6f; tfinal= %.6f' %(time,tfinal))

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start = timing.time()

    A_mat=np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs=np.zeros(NfemT,dtype=np.float64)           # FE rhs 
    B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)  # gradient matrix B 
    N_mat=np.zeros((mV,1),dtype=np.float64)        # shape functions
    N_mat_supg=np.zeros((mV,1),dtype=np.float64)   # shape functions
    tau_supg=np.zeros(nel*nqperdim**ndim,dtype=np.float64)

    counterq=0   
    for iel in range (0,nel):

        b_el=np.zeros(mV*ndofT,dtype=np.float64)
        a_el=np.zeros((mV*ndofT,mV*ndofT),dtype=np.float64)
        Ka=np.zeros((mV,mV),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mV,mV),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mV,mV),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,mV):
            Tvect[k]=T[iconV[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:mV,0]=NNV(rq,sq,order)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
                dNNNVds[0:mV]=dNNVds(rq,sq,order)

                # calculate jacobian matrix
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,mV):
                #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for
                #jcob = np.linalg.det(jcb)
                #jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,mV):
                    vel[0,0]+=N_mat[k,0]*u[iconV[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    B_mat[0,k]=dNNNVdx[k]
                    B_mat[1,k]=dNNNVdy[k]
                #end for

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2)/2/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
    
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B_mat))

                # compute mass matrix
                MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*weightq*jcob

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B_mat))*weightq*jcob

                counterq+=1

            #end for
        #end for

        a_el=MM+0.5*(Ka+Kd)*dt

        b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

        # apply boundary conditions
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mV):
                   m2=iconV[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end for
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            for k2 in range(0,mV):
                m2=iconV[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel

    print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix temperature: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    Traw = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(Traw),np.max(Traw)))
    Tstats_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # apply Lenardic & Kaula filter
    #################################################################
    start = timing.time()

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

    print("apply L&K filter: %.3f s" % (timing.time() - start))

    #################################################################
    # compute vrms & <T>
    #################################################################
    start = timing.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:mV]=NNV(rq,sq,order)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
                dNNNVds[0:mV]=dNNVds(rq,sq,order)
                #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                #for k in range(0,mV):
                #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #jcob = np.linalg.det(jcb)
                uq=0.
                vq=0.
                Tq=0.
                for k in range(0,mV):
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                #end for
                vrms+=(uq**2+vq**2)*weightq*jcob
                Tavrg+=Tq*weightq*jcob
            #end for jq
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly))
    Tavrg/=(Lx*Ly)

    Tavrg_file.write("%10e %10e\n" % (time,Tavrg))
    Tavrg_file.flush()

    vrms_file.write("%10e %.10f\n" % (time,vrms))
    vrms_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = timing.time()
    
    exx_n=np.zeros(NV,dtype=np.float64)  
    eyy_n=np.zeros(NV,dtype=np.float64)  
    exy_n=np.zeros(NV,dtype=np.float64)  
    sr_n=np.zeros(NV,dtype=np.float64)  
    eta_n=np.zeros(NV,dtype=np.float64)  
    rh_n=np.zeros(NV,dtype=np.float64)  
    count=np.zeros(NV,dtype=np.int32)  
    q=np.zeros(NV,dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]
            NNNV[0:mV]=NNV(rq,sq,order)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
            dNNNVds[0:mV]=dNNVds(rq,sq,order)
            NNNP[0:mP]=NNP(rq,sq,order)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            e_xx=0.
            e_yy=0.
            e_xy=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    q/=count

    sr_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

    for i in range(0,NV):
        eta_n[i],rh_n[i]=eta(T[i],sr_n[i],yV[i])

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> sr_n (m,M) %.6e %.6e " %(np.min(sr_n),np.max(sr_n)))
    print("     -> eta_n (m,M) %.6e %.6e " %(np.min(eta_n),np.max(eta_n)))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute temperature & visc profile
    #####################################################################
    start = timing.time()

    if istep%every==0:
       eta_profile = np.zeros(nny,dtype=np.float64)  
       eta_profile_min = np.zeros(nny,dtype=np.float64) ; eta_profile_min[:]=1e20   
       eta_profile_max = np.zeros(nny,dtype=np.float64)
       T_profile = np.zeros(nny,dtype=np.float64)  
       T_profile_min = np.ones(nny,dtype=np.float64)  
       T_profile_max = np.zeros(nny,dtype=np.float64)  
       y_profile = np.zeros(nny,dtype=np.float64)  

       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               eta_profile[j]+=eta_n[counter]/nnx
               T_profile[j]+=T[counter]/nnx
               T_profile_min[j]=min(T[counter],T_profile_min[j])
               T_profile_max[j]=max(T[counter],T_profile_max[j])
               eta_profile_min[j]=min(eta_n[counter],eta_profile_min[j])
               eta_profile_max[j]=max(eta_n[counter],eta_profile_max[j])
               y_profile[j]=yV[counter]
               counter+=1
           #end for
       #end for

       np.savetxt('profile_T_{:06d}.ascii'.format(istep),\
                   np.array([y_profile,T_profile,T_profile_min,T_profile_max]).T)
       np.savetxt('profile_eta_{:06d}.ascii'.format(istep),\
                   np.array([y_profile,eta_profile,eta_profile_min,eta_profile_max]).T)

    print("compute profiles: %.3f s" % (timing.time() - start))

    #####################################################################
    # advect markers
    #####################################################################
    start = timing.time()

    RKorder=1 # cheap!

    if RKorder==1: 

       for im in range(0,nmarker):
           if swarm_active[im]:
              swarm_u[im],swarm_v[im],rm,sm,iel =interpolate_vel_on_pt(swarm_x[im],swarm_y[im])
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

    print("advect markers: %.3f s" % (timing.time() - start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start = timing.time()

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='press' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='delta T' Format='ascii'> \n")
       counter=0    
       for j in range(0,nny):
           for i in range(0,nnx):
               vtufile.write("%10e \n" %(T[counter]-T_profile[j]))
               counter+=1
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eta_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rheology' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %rh_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %sr_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev stress' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %(2*eta_n[i]*sr_n[i]))
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
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e %10e %10e \n" %(swarm_u[im],swarm_v[im],0.))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='paint' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_mat[im])
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

       print("export to vtu file: %.3f s" % (timing.time() - start))

    ###################################

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
