import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing

#------------------------------------------------------------------------------

def rho(rho0,alphaT,T,T0):
    val=rho0*(1.-alphaT*(T-T0))-rho0
    return val

def eta(T):
    val=1.
    return val

#------------------------------------------------------------------------------

def NNV(r,s,order):
    if order==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.-r)*(1.+s)
       N_3=0.25*(1.+r)*(1.+s)
       return N_0,N_1,N_2,N_3
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
       return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8
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
       return N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
              N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15
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
       return N_00,N_01,N_02,N_03,N_04,\
              N_05,N_06,N_07,N_08,N_09,\
              N_10,N_11,N_12,N_13,N_14,\
              N_15,N_16,N_17,N_18,N_19,\
              N_20,N_21,N_22,N_23,N_24

def dNNVdr(r,s,order):
    if order==1:
       dNdr_0=-0.25*(1.-s) 
       dNdr_1=+0.25*(1.-s) 
       dNdr_2=-0.25*(1.+s) 
       dNdr_3=+0.25*(1.+s) 
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3
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
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8
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
       return dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,dNdr_05,dNdr_06,dNdr_07,\
              dNdr_08,dNdr_09,dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,dNdr_15
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
       return dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,\
              dNdr_05,dNdr_06,dNdr_07,dNdr_08,dNdr_09,\
              dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,\
              dNdr_15,dNdr_16,dNdr_17,dNdr_18,dNdr_19,\
              dNdr_20,dNdr_21,dNdr_22,dNdr_23,dNdr_24

def dNNVds(r,s,order):
    if order==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.-r)
       dNds_3=+0.25*(1.+r)
       return dNds_0,dNds_1,dNds_2,dNds_3
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
       return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8
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
       return dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,dNds_05,dNds_06,dNds_07,\
              dNds_08,dNds_09,dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,dNds_15
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
       return dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,\
              dNds_05,dNds_06,dNds_07,dNds_08,dNds_09,\
              dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,\
              dNds_15,dNds_16,dNds_17,dNds_18,dNds_19,\
              dNds_20,dNds_21,dNds_22,dNds_23,dNds_24


def NNP(r,s,order):
    if order==1:
       N_1=1.
       return N_1
    if order==2:
       N_0=0.25*(1-r)*(1-s)
       N_1=0.25*(1+r)*(1-s)
       N_2=0.25*(1-r)*(1+s)
       N_3=0.25*(1+r)*(1+s)
       return N_0,N_1,N_2,N_3
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
       return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8
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
       return N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
              N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1  # number of pressure degrees of freedom 

Lx=1.
Ly=1.

if int(len(sys.argv) == 7):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   order = int(sys.argv[4])
   Ra    = float(sys.argv[5])
   nstep = int(sys.argv[6])
else:
   nelx = 32
   nely = 32
   visu = 1
   order= 2
   Ra = 1e6
   nstep=100

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

ndofV=2
ndofP=1

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs
NfemT=NV*ndofT       # nb of temperature dofs

eps=1e-9

hx=Lx/nelx
hy=Ly/nely

sparse=False

#################################################################

alphaT=1e-4   # thermal expansion coefficient
hcond=1.      # thermal conductivity
hcapa=1.      # heat capacity
rho0=1        # reference density
T0=0          # reference temperature
relax=0.95    # relaxation coefficient (0,1)
gx=0.         # gravity vector component
gy=-Ra/alphaT # vertical component of gravity vector
tol_ss=1e-6   # tolerance for steady state 

#################################################################

if order==1:
   nqperdim=2
if order==2:
   nqperdim=3
if order==3:
   nqperdim=4
if order==4:
   nqperdim=5

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

if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

alpha=0.5

Nu_vrms_file=open('Nu_vrms.ascii',"w")
Nu_vrms_file.write("#time,Nusselt,vrms\n")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
conv_file=open('conv.ascii',"w")

#################################################################

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

#################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i],order))

#################################################################
# checking that all pressure shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i],order))

#################################################################
# build velocity nodes coordinates 
#################################################################
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

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

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
        counter += 1
    #end for
#end for

#print("-------iconV--------")
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range (0,mV):
#        print ("node ",i,':',iconV[i,iel],"at pos.",xV[iconV[i,iel]], yV[iconV[i,iel]])

# creating a dedicated connectivity array to plot the solution on Q1 space

nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int16)
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

#################################################################
# build pressure grid 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

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

#################################################################
# build pressure connectivity array 
#################################################################
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

#print("-------iconP--------")
#for iel in range (0,nel):
#    print ("iel=",iel)
#    for i in range(0,mP):
#        print ("node ",i,':',iconP[i,iel],"at pos.",xP[iconP[i,iel]], yP[iconP[i,iel]])

print("build iconP: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
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

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]<eps:
       bc_fixT[i]=True ; bc_valT[i]=1.
    if yV[i]>(Ly-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
#end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=1.-yV[i]-0.01*np.cos(np.pi*xV[i]/Lx)*np.sin(np.pi*yV[i]/Ly)
#end for

T_prev[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

#################################################################
# compute area of elements
#################################################################
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

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
u_prev  = np.zeros(NV,dtype=np.float64)           # x-component velocity
v_prev  = np.zeros(NV,dtype=np.float64)           # y-component velocity
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
Tvect   = np.zeros(mV,dtype=np.float64)   
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

Nusselt_prev=1.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    if sparse:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    else:   
       K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
       G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

    f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

    for iel in range(0,nel):

        # set arrays to 0 every loop
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
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.
                yq=0.
                Tq=0.
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]
                #end for

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(Tq)*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rho(rho0,alphaT,Tq,T0)*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho(rho0,alphaT,Tq,T0)*gy
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
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        if sparse:
                           A_sparse[m1,m2] += K_el[ikk,jkk]
                        else:
                           K_mat[m1,m2]+=K_el[ikk,jkk]
                        #end if
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    if sparse:
                       A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                       A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                    else:
                       G_mat[m1,m2]+=G_el[ikk,jkk]
                    #end if
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
        #end for

    #end for iel

    if not sparse:
       print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
       print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

    print("build FE matrix: %.3fs" % (timing.time()-start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    if not sparse:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

    print("assemble blocks: %.3f s" % (timing.time() - start))

    ######################################################################
    # assign extra pressure b.c. to remove null space
    ######################################################################

    if sparse:
       A_sparse[Nfem-1,:]=0
       A_sparse[:,Nfem-1]=0
       A_sparse[Nfem-1,Nfem-1]=1
       rhs[Nfem-1]=0
    else:
       a_mat[Nfem-1,:]=0
       a_mat[:,Nfem-1]=0
       a_mat[Nfem-1,Nfem-1]=1
       rhs[Nfem-1]=0
    #end if

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    if sparse:
       sparse_matrix=A_sparse.tocsr()
    else:
       sparse_matrix=sps.csr_matrix(a_mat)

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    #####################################################################
    # relaxation step
    #####################################################################

    u=relax*u+(1-relax)*u_prev
    v=relax*v+(1-relax)*v_prev

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions

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
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

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

                # compute mass matrix
                MM+=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

            #end for
        #end for

        a_el=Ka+Kd

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

    print("build FE matrix : %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # relax
    #################################################################

    T=relax*T+(1-relax)*T_prev

    #################################################################
    # compute vrms 
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
                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
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

    Tavrg_file.write("%10e %10e\n" % (istep,Tavrg))
    Tavrg_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = timing.time()
    
    exx_n = np.zeros(NV,dtype=np.float64)  
    eyy_n = np.zeros(NV,dtype=np.float64)  
    exy_n = np.zeros(NV,dtype=np.float64)  
    count = np.zeros(NV,dtype=np.int16)  
    qx_n = np.zeros(NV,dtype=np.float64)  
    qy_n = np.zeros(NV,dtype=np.float64)  
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
            q_x=0.
            q_y=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
                q_x -= hcond*dNNNVdx[k]*T[iconV[k,iel]]
                q_y -= hcond*dNNNVdy[k]*T[iconV[k,iel]]
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            qx_n[inode]+=q_x
            qy_n[inode]+=q_y
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    qx_n/=count
    qy_n/=count
    q/=count

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (timing.time() - start))

    #################################################################
    # compute Nusselt number at top
    #################################################################
    start = timing.time()

    Nusselt=0
    for iel in range(0,nel):
        if yV[iconV[mV-1,iel]]>1-eps: 
           for iq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=+1.
               weightq=qweights[iq]
               NNNV[0:mV]=NNV(rq,sq,order)
               xq=0.
               q_y=0.
               for k in range(0,mV):
                   xq += NNNV[k]*xV[iconV[k,iel]]
                   q_y -= NNNV[k]*qy_n[iconV[k,iel]]
               #end for
               jcob=hx/2.
               Nusselt+=q_y*jcob*weightq
               #print (xq,q_y)
           #end for
        #end if
    #end for

    Nusselt=np.abs(Nusselt)/Lx

    Nu_vrms_file.write("%10e %.10f %.10f\n" % (istep,Nusselt,vrms))
    Nu_vrms_file.flush()

    print("     istep= %d ; Nusselt= %e ; Ra= %e " %(istep,Nusselt,Ra))

    print("compute Nu: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute temperature profile
    #####################################################################
    start = timing.time()

    T_profile = np.zeros(nny,dtype=np.float64)  
    y_profile = np.zeros(nny,dtype=np.float64)  

    counter=0    
    for j in range(0,nny):
        for i in range(0,nnx):
            T_profile[j]+=T[counter]/nnx
            y_profile[j]=yV[counter]
            counter+=1
        #end for
    #end for

    np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='# x,T')

    print("compute T profile: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    if visu==1:
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
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(qx_n[i],qy_n[i],0.))
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

       print("export to vtu file: %.3f s" % (timing.time() - start))

    #############################

    T_diff=np.sum(abs(T-T_prev))/NV
    Nu_diff=np.abs(Nusselt-Nusselt_prev)/Nusselt

    print("T conv, T_diff, Nu_diff: " , T_diff<tol_ss,T_diff,Nu_diff)
    #print("u conv, u_diff, <u>: " , u_diff<tol_ss*vrms,u_diff,vrms)
    #print("v conv, v_diff, <v>: " , v_diff<tol_ss*vrms,v_diff,vrms)

    conv_file.write("%10e %10e %10e %10e\n" % (istep,T_diff,Nu_diff,tol_ss))
    conv_file.flush()

    if T_diff<tol_ss and Nu_diff<tol_ss:
       print("convergence reached")
       break

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]
    
    Nusselt_prev=Nusselt

#end istep
    
print("     script ; Nusselt= %e ; Ra= %e ; order= %d" %(Nusselt,Ra,order))

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
