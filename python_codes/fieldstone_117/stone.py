import numpy as np
import sys as sys
import scipy
import math as math
import solvi as solvi
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing

#------------------------------------------------------------------------------

def NNV(r,s,element):
    if element==1:
       N_0=0.25*(1.-r)*(1.-s)
       N_1=0.25*(1.+r)*(1.-s)
       N_2=0.25*(1.+r)*(1.+s)
       N_3=0.25*(1.-r)*(1.+s)
       return N_0,N_1,N_2,N_3
    if element==2 or element==5 or element==6:
       N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N_1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N_2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       N_3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N_4=    (1.-r**2) * 0.5*s*(s-1.)
       N_5= 0.5*r*(r+1.) *    (1.-s**2)
       N_6=    (1.-r**2) * 0.5*s*(s+1.)
       N_7= 0.5*r*(r-1.) *    (1.-s**2)
       N_8=    (1.-r**2) *    (1.-s**2)
       return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8
    if element==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       N_00= N1r*N1t ; N_01= N2r*N1t ; N_02= N3r*N1t ; N_03= N4r*N1t 
       N_04= N1r*N2t ; N_05= N2r*N2t ; N_06= N3r*N2t ; N_07= N4r*N2t 
       N_08= N1r*N3t ; N_09= N2r*N3t ; N_10= N3r*N3t ; N_11= N4r*N3t 
       N_12= N1r*N4t ; N_13= N2r*N4t ; N_14= N3r*N4t ; N_15= N4r*N4t 
       return N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
              N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15
    if element==4:
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
       N_00= N1r*N1s ; N_01= N2r*N1s ; N_02= N3r*N1s ; N_03= N4r*N1s ; N_04= N5r*N1s
       N_05= N1r*N2s ; N_06= N2r*N2s ; N_07= N3r*N2s ; N_08= N4r*N2s ; N_09= N5r*N2s
       N_10= N1r*N3s ; N_11= N2r*N3s ; N_12= N3r*N3s ; N_13= N4r*N3s ; N_14= N5r*N3s
       N_15= N1r*N4s ; N_16= N2r*N4s ; N_17= N3r*N4s ; N_18= N4r*N4s ; N_19= N5r*N4s
       N_20= N1r*N5s ; N_21= N2r*N5s ; N_22= N3r*N5s ; N_23= N4r*N5s ; N_24= N5r*N5s
       return N_00,N_01,N_02,N_03,N_04,\
              N_05,N_06,N_07,N_08,N_09,\
              N_10,N_11,N_12,N_13,N_14,\
              N_15,N_16,N_17,N_18,N_19,\
              N_20,N_21,N_22,N_23,N_24

def dNNVdr(r,s,element):
    if element==1:
       dNdr_0=-0.25*(1.-s) 
       dNdr_1=+0.25*(1.-s) 
       dNdr_2=+0.25*(1.+s) 
       dNdr_3=-0.25*(1.+s) 
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3
    if element==2 or element==5 or element==6:
       dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr_1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       dNdr_3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr_4=       (-2.*r) * 0.5*s*(s-1)
       dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr_6=       (-2.*r) * 0.5*s*(s+1)
       dNdr_7= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr_8=       (-2.*r) *   (1.-s**2)
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8
    if element==3:
       dN1rdr=( +1 +18*r -27*r**2)/16
       dN2rdr=(-27 -18*r +81*r**2)/16
       dN3rdr=(+27 -18*r -81*r**2)/16
       dN4rdr=( -1 +18*r +27*r**2)/16
       N1s=(-1    +s +9*s**2 - 9*s**3)/16
       N2s=(+9 -27*s -9*s**2 +27*s**3)/16
       N3s=(+9 +27*s -9*s**2 -27*s**3)/16
       N4s=(-1    -s +9*s**2 + 9*s**3)/16
       dNdr_00=dN1rdr*N1s ; dNdr_01=dN2rdr*N1s ; dNdr_02=dN3rdr*N1s ; dNdr_03=dN4rdr*N1s 
       dNdr_04=dN1rdr*N2s ; dNdr_05=dN2rdr*N2s ; dNdr_06=dN3rdr*N2s ; dNdr_07=dN4rdr*N2s 
       dNdr_08=dN1rdr*N3s ; dNdr_09=dN2rdr*N3s ; dNdr_10=dN3rdr*N3s ; dNdr_11=dN4rdr*N3s 
       dNdr_12=dN1rdr*N4s ; dNdr_13=dN2rdr*N4s ; dNdr_14=dN3rdr*N4s ; dNdr_15=dN4rdr*N4s 
       return dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,dNdr_05,dNdr_06,dNdr_07,\
              dNdr_08,dNdr_09,dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,dNdr_15
    if element==4:
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
       dNdr_00=dN1dr*N1s ; dNdr_01=dN2dr*N1s ; dNdr_02=dN3dr*N1s ; dNdr_03=dN4dr*N1s ; dNdr_04=dN5dr*N1s
       dNdr_05=dN1dr*N2s ; dNdr_06=dN2dr*N2s ; dNdr_07=dN3dr*N2s ; dNdr_08=dN4dr*N2s ; dNdr_09=dN5dr*N2s
       dNdr_10=dN1dr*N3s ; dNdr_11=dN2dr*N3s ; dNdr_12=dN3dr*N3s ; dNdr_13=dN4dr*N3s ; dNdr_14=dN5dr*N3s
       dNdr_15=dN1dr*N4s ; dNdr_16=dN2dr*N4s ; dNdr_17=dN3dr*N4s ; dNdr_18=dN4dr*N4s ; dNdr_19=dN5dr*N4s
       dNdr_20=dN1dr*N5s ; dNdr_21=dN2dr*N5s ; dNdr_22=dN3dr*N5s ; dNdr_23=dN4dr*N5s ; dNdr_24=dN5dr*N5s
       return dNdr_00,dNdr_01,dNdr_02,dNdr_03,dNdr_04,\
              dNdr_05,dNdr_06,dNdr_07,dNdr_08,dNdr_09,\
              dNdr_10,dNdr_11,dNdr_12,dNdr_13,dNdr_14,\
              dNdr_15,dNdr_16,dNdr_17,dNdr_18,dNdr_19,\
              dNdr_20,dNdr_21,dNdr_22,dNdr_23,dNdr_24

def dNNVds(r,s,element):
    if element==1:
       dNds_0=-0.25*(1.-r)
       dNds_1=-0.25*(1.+r)
       dNds_2=+0.25*(1.+r)
       dNds_3=+0.25*(1.-r)
       return dNds_0,dNds_1,dNds_2,dNds_3
    if element==2 or element==5 or element==6:
       dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds_1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       dNds_3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds_4=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds_5= 0.5*r*(r+1.) *       (-2.*s)
       dNds_6=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds_7= 0.5*r*(r-1.) *       (-2.*s)
       dNds_8=    (1.-r**2) *       (-2.*s)
       return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8
    if element==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       dNds_00=N1r*dN1sds ; dNds_01=N2r*dN1sds ; dNds_02=N3r*dN1sds ; dNds_03=N4r*dN1sds 
       dNds_04=N1r*dN2sds ; dNds_05=N2r*dN2sds ; dNds_06=N3r*dN2sds ; dNds_07=N4r*dN2sds 
       dNds_08=N1r*dN3sds ; dNds_09=N2r*dN3sds ; dNds_10=N3r*dN3sds ; dNds_11=N4r*dN3sds 
       dNds_12=N1r*dN4sds ; dNds_13=N2r*dN4sds ; dNds_14=N3r*dN4sds ; dNds_15=N4r*dN4sds
       return dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,dNds_05,dNds_06,dNds_07,\
              dNds_08,dNds_09,dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,dNds_15
    if element==4:
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
       dNds_00=N1r*dN1ds ; dNds_01=N2r*dN1ds ; dNds_02=N3r*dN1ds ; dNds_03=N4r*dN1ds ; dNds_04=N5r*dN1ds
       dNds_05=N1r*dN2ds ; dNds_06=N2r*dN2ds ; dNds_07=N3r*dN2ds ; dNds_08=N4r*dN2ds ; dNds_09=N5r*dN2ds
       dNds_10=N1r*dN3ds ; dNds_11=N2r*dN3ds ; dNds_12=N3r*dN3ds ; dNds_13=N4r*dN3ds ; dNds_14=N5r*dN3ds
       dNds_15=N1r*dN4ds ; dNds_16=N2r*dN4ds ; dNds_17=N3r*dN4ds ; dNds_18=N4r*dN4ds ; dNds_19=N5r*dN4ds
       dNds_20=N1r*dN5ds ; dNds_21=N2r*dN5ds ; dNds_22=N3r*dN5ds ; dNds_23=N4r*dN5ds ; dNds_24=N5r*dN5ds
       return dNds_00,dNds_01,dNds_02,dNds_03,dNds_04,\
              dNds_05,dNds_06,dNds_07,dNds_08,dNds_09,\
              dNds_10,dNds_11,dNds_12,dNds_13,dNds_14,\
              dNds_15,dNds_16,dNds_17,dNds_18,dNds_19,\
              dNds_20,dNds_21,dNds_22,dNds_23,dNds_24

#------------------------------------------------------------------------------

def NNP(r,s,element):
    if element==1:
       N_1=1.
       return N_1
    if element==2:
       N_0=0.25*(1-r)*(1-s)
       N_1=0.25*(1+r)*(1-s)
       N_2=0.25*(1-r)*(1+s)
       N_3=0.25*(1+r)*(1+s)
       return N_0,N_1,N_2,N_3
    if element==3:
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
    if element==4:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       N1t=(-1    +s +9*s**2 - 9*s**3)/16
       N2t=(+9 -27*s -9*s**2 +27*s**3)/16
       N3t=(+9 +27*s -9*s**2 -27*s**3)/16
       N4t=(-1    -s +9*s**2 + 9*s**3)/16
       N_00= N1r*N1t ; N_01= N2r*N1t ; N_02= N3r*N1t ; N_03= N4r*N1t 
       N_04= N1r*N2t ; N_05= N2r*N2t ; N_06= N3r*N2t ; N_07= N4r*N2t 
       N_08= N1r*N3t ; N_09= N2r*N3t ; N_10= N3r*N3t ; N_11= N4r*N3t 
       N_12= N1r*N4t ; N_13= N2r*N4t ; N_14= N3r*N4t ; N_15= N4r*N4t 
       return N_00,N_01,N_02,N_03,N_04,N_05,N_06,N_07,\
              N_08,N_09,N_10,N_11,N_12,N_13,N_14,N_15
    if element==5:
       N_0=1-2*r-2*s
       N_1=2*r
       N_2=2*s
       return N_0,N_1,N_2
    if element==6:
       return 0,0,0

#------------------------------------------------------------------------------

def bx(x,y):
    if experiment==0:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==1:
       val=0
    return val

def by(x,y):
    if experiment==0:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==1:
       val=0
    return val

#------------------------------------------------------------------------------

def eta(x,y):
    if experiment==0:
       val=1
    if experiment==1:
       if (np.sqrt(x*x+y*y) < 0.2):
          val=1e3
       else:
          val=1.
    return val

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if experiment==0:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if experiment==1:
       val,vi,pi=solvi.SolViSolution(x,y) 
    return val

def velocity_y(x,y):
    if experiment==0:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==1:
       ui,val,pi=solvi.SolViSolution(x,y) 
    return val

def pressure(x,y):
    if experiment==0:
       val=x*(1.-x)#-1./6.
    if experiment==1:
       ui,vi,val=solvi.SolViSolution(x,y) 
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.
Ly=1.

experiment=1

element= 2

correct_mesh=True

if element==1:
   mV=4
   mP=1
   rVnodes=[-1,+1,+1,-1]
   sVnodes=[-1,-1,+1,+1]
if element==2:
   mV=9
   mP=4
   rVnodes=[-1,+1,+1,-1, 0,+1, 0,-1,0]
   sVnodes=[-1,-1,+1,+1,-1, 0,+1, 0,0]
if element==3:
   mV=16
   mP=9
   rVnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sVnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]
if element==4:
   mV=25
   mP=16
   rVnodes=[-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1,-1,-0.5,0,0.5,1]
   sVnodes=[-1,-1,-1,-1,-1,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1]
if element==5 or element==6:
   mV=9
   mP=3
   rVnodes=[-1,1,1,-1,0,1,0,-1,0]
   sVnodes=[-1,-1,1,1,-1,0,1,0,0]

eps=1e-9

# allowing for argument parsing through command line
if int(len(sys.argv) == 2):
   folder=sys.argv[1]
else:
   folder='08'

print (folder)

###############################################################################

if element==1: nqperdim=2
if element==2: nqperdim=3
if element==3: nqperdim=4
if element==4: nqperdim=5
if element==5: nqperdim=3
if element==6: nqperdim=3

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

##############################################################################

print ('element  =',element)
print ('nqperdim =',nqperdim)
print("-----------------------------")

##############################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i],order))

###############################################################################
# read in node coordinates and connectivity
###############################################################################
start = timing.time()

if element==1: # Q1Q0
   orderV='Q1'
   orderP='Q0'

if element==2: # Q2Q1
   orderV='Q2'
   orderP='Q1'

if element==3: # Q3Q2
   orderV='Q3'
   orderP='Q2'

if element==4: # Q4Q3
   orderV='Q4'
   orderP='Q3'

if element==5: # Q2P1
   orderV='Q2'
   orderP='P1'

if element==6: # Q2P1
   orderV='Q2'
   orderP='P1'


f_vel = open('./meshes/'+folder+'/coordinates_'+orderV+'.dat', 'r') 
lines_vel = f_vel.readlines()
line=lines_vel[0].strip()
columns=line.split()
NV=int(columns[0])

g_vel = open('./meshes/'+folder+'/connectivity_'+orderV+'.dat', 'r') 
lines_iconV = g_vel.readlines()
line=lines_iconV[0].strip()
columns=line.split()
nel=int(columns[0])

if orderP[0]=='Q':
   f_press = open('./meshes/'+folder+'/coordinates_'+orderP+'.dat', 'r') 
   lines_press = f_press.readlines()
   line=lines_press[0].strip()
   columns=line.split()
   NP=int(columns[0])

   g_press = open('./meshes/'+folder+'/connectivity_'+orderP+'.dat', 'r') 
   lines_iconP = g_press.readlines()
   line=lines_iconP[0].strip()
   columns=line.split()
   nel=int(columns[0])

if orderP=='P1':
   NP=3*nel

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs

print ('NV       =',NV)
print ('NP       =',NP)
print ('nel      =',nel)
print ('NfemV    =',NfemV)
print ('NfemP    =',NfemP)
print ('Nfem     =',Nfem)

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

for i in range(0,NV):
    line=lines_vel[i+1].strip()
    columns=line.split()
    xV[i]=float(columns[0])
    yV[i]=float(columns[1])
#end for
np.savetxt('meshV.ascii',np.array([xV,yV]).T)

iconV=np.zeros((mV,nel),dtype=np.int32)

for i in range(1,nel+1):
    line=lines_iconV[i].strip()
    columns=line.split()
    for k in range(0,mV):
        iconV[k,i-1]=int(columns[k])-1
    #print(iconV[:,i-1])
#end for

xP=np.empty(NfemP,dtype=np.float64)  
yP=np.empty(NfemP,dtype=np.float64) 
iconP=np.zeros((mP,nel),dtype=np.int32)

if orderP[0]=='Q':
   counter=0
   for i in range(1,NP+1):
       line=lines_press[i].strip()
       columns=line.split()
       xP[counter]=float(columns[0])
       yP[counter]=float(columns[1])
       counter+=1
       #end for

   counter=0
   for i in range(1,nel+1):
       line=lines_iconP[i].strip()
       columns=line.split()
       for k in range(0,mP):
           iconP[k,counter]=int(columns[k])-1
       counter+=1
   #end for

if orderP=='P1':
   NNNV = np.zeros(mV,dtype=np.float64) 
   for iel in range(nel):
       iconP[0,iel]=3*iel
       iconP[1,iel]=3*iel+1
       iconP[2,iel]=3*iel+2
   counter=0
   for iel in range(nel):
       rq=0
       sq=0
       NNNV[0:mV]=NNV(rq,sq,element)
       xP[counter]=NNNV.dot(xV[iconV[0:mV,iel]])
       yP[counter]=NNNV.dot(yV[iconV[0:mV,iel]])
       counter+=1
       rq=0.5
       sq=0
       NNNV[0:mV]=NNV(rq,sq,element)
       xP[counter]=NNNV.dot(xV[iconV[0:mV,iel]])
       yP[counter]=NNNV.dot(yV[iconV[0:mV,iel]])
       counter+=1
       rq=0
       sq=0.5
       NNNV[0:mV]=NNV(rq,sq,element)
       xP[counter]=NNNV.dot(xV[iconV[0:mV,iel]])
       yP[counter]=NNNV.dot(yV[iconV[0:mV,iel]])
       counter+=1
   #end for
      

np.savetxt('meshP.ascii',np.array([xP,yP]).T)


print("read in meshes : %.3f s" % (timing.time() - start))

###############################################################################
#straighten things out
###############################################################################

if element==2:
   for iel in range(0,nel):
       xV[iconV[4,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
       yV[iconV[4,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[1,iel]])
       xV[iconV[5,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
       yV[iconV[5,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]])
       xV[iconV[6,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[3,iel]])
       yV[iconV[6,iel]]=0.5*(yV[iconV[2,iel]]+yV[iconV[3,iel]])
       xV[iconV[7,iel]]=0.5*(xV[iconV[3,iel]]+xV[iconV[0,iel]])
       yV[iconV[7,iel]]=0.5*(yV[iconV[3,iel]]+yV[iconV[0,iel]])
       xV[iconV[8,iel]]=0.25*(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]])
       yV[iconV[8,iel]]=0.25*(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]])

#np.savetxt('meshV2.ascii',np.array([xV,yV]).T)

#for iel in range (0,1):
#     print ("iel=",iel)
#     print ("node 1",iconV[0][iel],"at pos.",xV[iconV[0][iel]], yV[iconV[0][iel]])
#     print ("node 2",iconV[1][iel],"at pos.",xV[iconV[1][iel]], yV[iconV[1][iel]])
#     print ("node 3",iconV[2][iel],"at pos.",xV[iconV[2][iel]], yV[iconV[2][iel]])
#     print ("node 4",iconV[3][iel],"at pos.",xV[iconV[3][iel]], yV[iconV[3][iel]])

###############################################################################
# flag nodes on viscosity contrast interface
###############################################################################
interface=np.zeros(NV,dtype=np.int16)  
r=np.zeros(NV,dtype=np.float64)  
theta=np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    r[i]=np.sqrt(xV[i]**2+yV[i]**2)
    theta[i]=np.arctan2(yV[i],xV[i])
    if abs(r[i]-0.2)<1e-6:
       interface[i]=1
       r[i]=0.2

if correct_mesh:

   for iel in range(0,nel):
       if element==2:
          if interface[iconV[0,iel]]==1 and interface[iconV[1,iel]]==1: 
             mid=iconV[4,iel] 
             r[mid]=0.2
             xV[mid]=r[mid]*np.cos(theta[mid])
             yV[mid]=r[mid]*np.sin(theta[mid])
             interface[mid]=1
          if interface[iconV[1,iel]]==1 and interface[iconV[2,iel]]==1: 
             mid=iconV[5,iel] 
             r[mid]=0.2
             xV[mid]=r[mid]*np.cos(theta[mid])
             yV[mid]=r[mid]*np.sin(theta[mid])
             interface[mid]=1
          if interface[iconV[2,iel]]==1 and interface[iconV[3,iel]]==1: 
             mid=iconV[6,iel] 
             r[mid]=0.2
             xV[mid]=r[mid]*np.cos(theta[mid])
             yV[mid]=r[mid]*np.sin(theta[mid])
             interface[mid]=1
          if interface[iconV[3,iel]]==1 and interface[iconV[0,iel]]==1: 
             mid=iconV[7,iel] 
             r[mid]=0.2
             xV[mid]=r[mid]*np.cos(theta[mid])
             yV[mid]=r[mid]*np.sin(theta[mid])
             interface[mid]=1

   np.savetxt('meshV3.ascii',np.array([xV,yV,interface,r,theta]).T)

#exit()

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    ui=velocity_x(xV[i],yV[i])
    vi=velocity_y(xV[i],yV[i])
    if xV[i]<eps:
          bc_fix[i*ndofV+0]   = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1]   = True ; bc_val[i*ndofV+1] = vi
    if xV[i]>(Lx-eps):
          bc_fix[i*ndofV+0]   = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1]   = True ; bc_val[i*ndofV+1] = vi
    if yV[i]<eps:
          bc_fix[i*ndofV+0]   = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1]   = True ; bc_val[i*ndofV+1] = vi
    if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+0]   = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1]   = True ; bc_val[i*ndofV+1] = vi

print("boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
area_inc= np.zeros(nel,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
            dNNNVds[0:mV]=dNNVds(rq,sq,element)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            if abs(jcob)<1e-10:
               print(jcb)
               print(iel)
               print(xV[iconV[:,iel]])
               print(yV[iconV[:,iel]])
               exit('opla')
            area[iel]+=jcob*weightq
            if r[iconV[8,iel]]<=0.2:
               area_inc[iel]+=jcob*weightq
         

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))
print("     -> area inclusion %.10f %.10f %d" %(area_inc.sum(),0.25*np.pi*0.2**2,nel))

print("compute elements areas: %.3f s" % (timing.time() - start))

#for iel in range (585,586):
#     print ("iel=",iel)
#     print ("node 1",iconV[0][iel],"at pos.",xV[iconV[0][iel]], yV[iconV[0][iel]])
#     print ("node 2",iconV[1][iel],"at pos.",xV[iconV[1][iel]], yV[iconV[1][iel]])
#     print ("node 3",iconV[2][iel],"at pos.",xV[iconV[2][iel]], yV[iconV[2][iel]])
#     print ("node 4",iconV[3][iel],"at pos.",xV[iconV[3][iel]], yV[iconV[3][iel]])
#     print ("node 5",iconV[4][iel],"at pos.",xV[iconV[4][iel]], yV[iconV[4][iel]])
#     print ("node 6",iconV[5][iel],"at pos.",xV[iconV[5][iel]], yV[iconV[5][iel]])
#     print ("node 7",iconV[6][iel],"at pos.",xV[iconV[6][iel]], yV[iconV[6][iel]])
#     print ("node 8",iconV[7][iel],"at pos.",xV[iconV[7][iel]], yV[iconV[7][iel]])
#     print ("node 8",iconV[8][iel],"at pos.",xV[iconV[8][iel]], yV[iconV[8][iel]])
#exit()

###############################################################################

p_analytical = np.zeros(NP,dtype=np.float64)
for i in range(0,NP):
    p_analytical[i]=pressure(xP[i],yP[i])

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat    = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat    = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u        = np.zeros(NV,dtype=np.float64)           # x-component velocity
v        = np.zeros(NV,dtype=np.float64)           # y-component velocity
c_mat    = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    if element==6:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

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

            NNNV[0:mV]=NNV(rq,sq,element)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
            dNNNVds[0:mV]=dNNVds(rq,sq,element)
            NNNP[0:mP]=NNP(rq,sq,element)

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
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            if element==6:
               NNNP[0]=(m11+m21*xq+m31*yq)
               NNNP[1]=(m12+m22*xq+m32*yq)
               NNNP[2]=(m13+m23*xq+m33*yq)

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

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
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

######################################################################
# assemble rhs
######################################################################
start = timing.time()
   
rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

######################################################################
# assign extra pressure b.c. to remove null space
######################################################################

for i in range(0,Nfem):
    A_sparse[Nfem-1,i]=0
    A_sparse[i,Nfem-1]=0
A_sparse[Nfem-1,Nfem-1]=1
rhs[Nfem-1]=0

######################################################################
# solve system
######################################################################
start = timing.time()

sparse_matrix=A_sparse.tocsr()
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

#np.savetxt('vel.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

#####################################################################
# normalise pressure
#####################################################################
start = timing.time()

pavrg=0.
for iel in range (0,nel):

    if element==6:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq,element)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
            dNNNVds[0:mV]=dNNVds(rq,sq,element)
            NNNP[0:mP]=NNP(rq,sq,element)
            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)

            if element==6:
               xq=NNNV[:].dot(xV[iconV[:,iel]])
               yq=NNNV[:].dot(yV[iconV[:,iel]])
               NNNP[0]=(m11+m21*xq+m31*yq)
               NNNP[1]=(m12+m22*xq+m32*yq)
               NNNP[2]=(m13+m23*xq+m33*yq)

            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            # end for k
            pavrg+=pq*weightq*jcob
        # end for jq
    # end for iq
# end for iel

p[:]-=pavrg/Lx/Ly

print("     -> pavrg=",pavrg)

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (timing.time() - start))

#####################################################################
# compute strainrate at element center
#####################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.5
    sq = 0.5
    weightq = 2 
    NNNV[0:mV]=NNV(rq,sq,element)
    dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
    dNNNVds[0:mV]=dNNVds(rq,sq,element)
    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    for k in range(0,mV):
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (timing.time() - start))

#####################################################################
# project pressure onto velocity grid
#####################################################################
start = timing.time()

q=np.zeros(NV,dtype=np.float64)
c=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        NNNP[0:mP]=NNP(rVnodes[i],sVnodes[i],element)
        q[iconV[i,iel]]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        c[iconV[i,iel]]+=1.

q=q/c

np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (timing.time() - start))

#####################################################################
# compute error fields for plotting
#####################################################################
start = timing.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_p = np.empty(NP,dtype=np.float64)
error_q = np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i])
    error_v[i]=v[i]-velocity_y(xV[i],yV[i])
    error_q[i]=q[i]-pressure(xV[i],yV[i])

for i in range(0,NP): 
    error_p[i]=p[i]-pressure(xP[i],yP[i])

print("     -> error_u (m,M) %.4e %.4e " %(np.min(error_u),np.max(error_u)))
print("     -> error_v (m,M) %.4e %.4e " %(np.min(error_v),np.max(error_v)))
print("     -> error_p (m,M) %.4e %.4e " %(np.min(error_p),np.max(error_p)))
print("     -> error_q (m,M) %.4e %.4e " %(np.min(error_q),np.max(error_q)))

print("compute error fields: %.3f s" % (timing.time() - start))

#####################################################################
# compute L2 errors
#####################################################################
start = timing.time()

errv=0.
errp=0.
errq=0.
for iel in range (0,nel):

    if element==6:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq,element)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
            dNNNVds[0:mV]=dNNVds(rq,sq,element)
            NNNP[0:mP]=NNP(rq,sq,element)

            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)

            xq=0.
            yq=0.
            uq=0.
            vq=0.
            qq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                qq+=NNNV[k]*q[iconV[k,iel]]
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errq+=(qq-pressure(xq,yq))**2*weightq*jcob

            if element==6:
               NNNP[0]=(m11+m21*xq+m31*yq)
               NNNP[1]=(m12+m22*xq+m32*yq)
               NNNP[2]=(m13+m23*xq+m33*yq)
            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            errp+=(pq-pressure(xq,yq))**2*weightq*jcob

        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)

hmin=np.sqrt(min(area))
hmax=np.sqrt(max(area))

print("     -> nel= %6d ; errv= %.8e ; errp= %.8e ; errq= %.8e ; hmin/max= %.6e %.6e" %(nel,errv,errp,errq,hmin,hmax))

print("compute errors: %.3f s" % (timing.time() - start))

#####################################################################
# extract pressure profile at bottom
#####################################################################

r=np.sqrt(xP**2+yP**2)

np.savetxt('p.ascii',np.array([xP,yP,p,r,p_analytical]).T,header='# x,y,p,r')

#####################################################################
# plot of solution
#####################################################################

if True==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta(xc[iel],yc[iel])))
    vtufile.write("</DataArray>\n")
    #--
    if element==1 or element==5 or element==6:
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%12e \n" %p[iconP[0,iel]])
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eyy[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exy[iel]))
    vtufile.write("</DataArray>\n")

    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='interface' Format='ascii'> \n")
    for i in range(0,NV):
           vtufile.write("%2e \n" %interface[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
    for i in range(0,NV):
           vtufile.write("%2e \n" %theta[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error u' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_u[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error v' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_v[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='error q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%.5e \n" %error_q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    if element==1:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
    if element==2 or element==5 or element==6:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
    if element==3:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[3,iel],iconV[15,iel],iconV[12,iel]))
    if element==4:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[4,iel],iconV[24,iel],iconV[20,iel]))

    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
