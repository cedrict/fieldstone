import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import time as clock

debug=False

###############################################################################

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
       return np.array([N00,N01,N02,N03,N04,N05,N06,N07,
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
       N00= N1r*N1s
       N01= N2r*N1s
       N02= N3r*N1s
       N03= N4r*N1s
       N04= N5r*N1s
       N05= N1r*N2s
       N06= N2r*N2s
       N07= N3r*N2s
       N08= N4r*N2s
       N09= N5r*N2s
       N10= N1r*N3s
       N11= N2r*N3s
       N12= N3r*N3s
       N13= N4r*N3s
       N14= N5r*N3s
       N15= N1r*N4s
       N16= N2r*N4s
       N17= N3r*N4s
       N18= N4r*N4s
       N19= N5r*N4s
       N20= N1r*N5s
       N21= N2r*N5s
       N22= N3r*N5s
       N23= N4r*N5s
       N24= N5r*N5s
       return np.array([N00,N01,N02,N03,N04,\
                        N05,N06,N07,N08,N09,\
                        N10,N11,N12,N13,N14,\
                        N15,N16,N17,N18,N19,\
                        N20,N21,N22,N23,N24],dtype=np.float64)

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
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)
    if order==3:
       dN1rdr=( +1 +18*r -27*r**2)/16
       dN2rdr=(-27 -18*r +81*r**2)/16
       dN3rdr=(+27 -18*r -81*r**2)/16
       dN4rdr=( -1 +18*r +27*r**2)/16
       N1s=(-1    +s +9*s**2 - 9*s**3)/16
       N2s=(+9 -27*s -9*s**2 +27*s**3)/16
       N3s=(+9 +27*s -9*s**2 -27*s**3)/16
       N4s=(-1    -s +9*s**2 + 9*s**3)/16
       dNdr00= dN1rdr* N1s 
       dNdr01= dN2rdr* N1s 
       dNdr02= dN3rdr* N1s 
       dNdr03= dN4rdr* N1s 
       dNdr04= dN1rdr* N2s 
       dNdr05= dN2rdr* N2s 
       dNdr06= dN3rdr* N2s 
       dNdr07= dN4rdr* N2s 
       dNdr08= dN1rdr* N3s 
       dNdr09= dN2rdr* N3s 
       dNdr10= dN3rdr* N3s 
       dNdr11= dN4rdr* N3s 
       dNdr12= dN1rdr* N4s 
       dNdr13= dN2rdr* N4s 
       dNdr14= dN3rdr* N4s 
       dNdr15= dN4rdr* N4s 
       return np.array([dNdr00,dNdr01,dNdr02,dNdr03,dNdr04,dNdr05,dNdr06,dNdr07,
                        dNdr08,dNdr09,dNdr10,dNdr11,dNdr12,dNdr13,dNdr14,dNdr15],dtype=np.float64)

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
       dNdr00= dN1dr*N1s
       dNdr01= dN2dr*N1s
       dNdr02= dN3dr*N1s
       dNdr03= dN4dr*N1s
       dNdr04= dN5dr*N1s
       dNdr05= dN1dr*N2s
       dNdr06= dN2dr*N2s
       dNdr07= dN3dr*N2s
       dNdr08= dN4dr*N2s
       dNdr09= dN5dr*N2s
       dNdr10= dN1dr*N3s
       dNdr11= dN2dr*N3s
       dNdr12= dN3dr*N3s
       dNdr13= dN4dr*N3s
       dNdr14= dN5dr*N3s
       dNdr15= dN1dr*N4s
       dNdr16= dN2dr*N4s
       dNdr17= dN3dr*N4s
       dNdr18= dN4dr*N4s
       dNdr19= dN5dr*N4s
       dNdr20= dN1dr*N5s
       dNdr21= dN2dr*N5s
       dNdr22= dN3dr*N5s
       dNdr23= dN4dr*N5s
       dNdr24= dN5dr*N5s
       return np.array([dNdr00,dNdr01,dNdr02,dNdr03,dNdr04,
                        dNdr05,dNdr06,dNdr07,dNdr08,dNdr09,
                        dNdr10,dNdr11,dNdr12,dNdr13,dNdr14,
                        dNdr15,dNdr16,dNdr17,dNdr18,dNdr19,
                        dNdr20,dNdr21,dNdr22,dNdr23,dNdr24],dtype=np.float64)

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
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)
    if order==3:
       N1r=(-1    +r +9*r**2 - 9*r**3)/16
       N2r=(+9 -27*r -9*r**2 +27*r**3)/16
       N3r=(+9 +27*r -9*r**2 -27*r**3)/16
       N4r=(-1    -r +9*r**2 + 9*r**3)/16
       dN1sds=( +1 +18*s -27*s**2)/16
       dN2sds=(-27 -18*s +81*s**2)/16
       dN3sds=(+27 -18*s -81*s**2)/16
       dN4sds=( -1 +18*s +27*s**2)/16
       dNds00= N1r*dN1sds 
       dNds01= N2r*dN1sds 
       dNds02= N3r*dN1sds 
       dNds03= N4r*dN1sds 
       dNds04= N1r*dN2sds 
       dNds05= N2r*dN2sds 
       dNds06= N3r*dN2sds 
       dNds07= N4r*dN2sds 
       dNds08= N1r*dN3sds 
       dNds09= N2r*dN3sds 
       dNds10= N3r*dN3sds 
       dNds11= N4r*dN3sds 
       dNds12= N1r*dN4sds 
       dNds13= N2r*dN4sds 
       dNds14= N3r*dN4sds 
       dNds15= N4r*dN4sds
       return np.array([dNds00,dNds01,dNds02,dNds03,dNds04,dNds05,dNds06,dNds07,
                        dNds08,dNds09,dNds10,dNds11,dNds12,dNds13,dNds14,dNds15],dtype=np.float64)
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
       dNds00= N1r*dN1ds
       dNds01= N2r*dN1ds
       dNds02= N3r*dN1ds
       dNds03= N4r*dN1ds
       dNds04= N5r*dN1ds
       dNds05= N1r*dN2ds
       dNds06= N2r*dN2ds
       dNds07= N3r*dN2ds
       dNds08= N4r*dN2ds
       dNds09= N5r*dN2ds
       dNds10= N1r*dN3ds
       dNds11= N2r*dN3ds
       dNds12= N3r*dN3ds
       dNds13= N4r*dN3ds
       dNds14= N5r*dN3ds
       dNds15= N1r*dN4ds
       dNds16= N2r*dN4ds
       dNds17= N3r*dN4ds
       dNds18= N4r*dN4ds
       dNds19= N5r*dN4ds
       dNds20= N1r*dN5ds
       dNds21= N2r*dN5ds
       dNds22= N3r*dN5ds
       dNds23= N4r*dN5ds
       dNds24= N5r*dN5ds
       return np.array([dNds00,dNds01,dNds02,dNds03,dNds04,
                        dNds05,dNds06,dNds07,dNds08,dNds09,
                        dNds10,dNds11,dNds12,dNds13,dNds14,
                        dNds15,dNds16,dNds17,dNds18,dNds19,
                        dNds20,dNds21,dNds22,dNds23,dNds24],dtype=np.float64)


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
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)
    if order==4:
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
       return np.array([N00,N01,N02,N03,N04,N05,N06,N07,
                        N08,N09,N10,N11,N12,N13,N14,N15],dtype=np.float64)

###############################################################################

eps=1e-6

print("*******************************")
print("********** stone 042 **********")
print("*******************************")

ndim=2
ndof_V=2  # number of velocity degrees of freedom per node

Lx=8.*2
Ly=6.

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   geom = int(sys.argv[3])
   order = int(sys.argv[4])
else:
   nelx = 16
   nely = 8
   geom = 2
   order= 2

gy=-1

slope=0.6

nel=nelx*nely
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
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

rho=1.
eta=1.

hx=Lx/nelx
hy=Ly/nely

###############################################################################

if order==1: nq_per_dim=2
if order==2: nq_per_dim=3
if order==3: nq_per_dim=4
if order==4: nq_per_dim=5

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

if nq_per_dim==6:
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

###############################################################################

print ('order      =',order)
print ('nnx        =',nnx)
print ('nny        =',nny)
print ('nn_V       =',nn_V)
print ('nn_P       =',nn_P)
print ('nel        =',nel)
print ('Nfem_V     =',Nfem_V)
print ('Nfem_P     =',Nfem_P)
print ('Nfem       =',Nfem)
print ('nq_per_dim =',nq_per_dim)
print("-----------------------------")

###############################################################################
# check that all vel basis functions are 1 on their node and zero elsewhere

if debug:
   for i in range(0,m_V):
       print ('node',i,':',basis_functions_V(r_V[i],s_V[i],order))

###############################################################################
# check that all press basis functions are 1 on their node and zero elsewhere

if debug:
   for i in range(0,m_P):
       print ('node',i,':',basis_functions_S(r_P[i],s_P[i],order))

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

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                icon_V[counter2,counter]=i*order+l+j*order*nnx+nnx*k
                counter2+=1
        counter+=1

if debug:
   print("-------icon_V--------")
   for iel in range (0,nel):
       print ("iel=",iel)
       for i in range (0,m_V):
           print ("node ",i,':',icon_V[i,iel],"at pos.",x_V[icon_V[i,iel]],y_V[icon_V[i,iel]])

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)

if order==1:
   for iel in range(0,nel):
       x_P[iel]=sum(x_V[icon_V[0:m_V,iel]])*0.25
       y_P[iel]=sum(y_V[icon_V[0:m_V,iel]])*0.25
      
if order>1:
   counter=0    
   for j in range(0,(order-1)*nely+1):
       for i in range(0,(order-1)*nelx+1):
           x_P[counter]=i*hx/(order-1)
           y_P[counter]=j*hy/(order-1)
           counter+=1

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

if order>1:
   om1=order-1
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   icon_P[counter2,counter]=i*om1+l+j*om1*(om1*nelx+1)+(om1*nelx+1)*k 
                   counter2+=1
           counter += 1

if debug:
   print("-------icon_P--------")
   for iel in range (0,nel):
       print ("iel=",iel)
       for i in range(0,m_P):
           print ("node ",i,':',icon_P[i,iel],"at pos.",x_P[icon_P[i,iel]],y_P[icon_P[i,iel]])

print("build icon_P: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# deform mesh
###############################################################################

if geom==1:
   for i in range(0,nn_V):
       if x_V[i]>Lx/2:
          y_V[i]-=slope*(x_V[i]-Lx/2)
   for i in range(0,nn_P):
       if x_P[i]>Lx/2:
          y_P[i]-=slope*(x_P[i]-Lx/2)

if geom==2:
   ymax=Ly
   counter=0    
   for j in range(0,nny):
       for i in range(0,nnx):
           if x_V[counter]>Lx/2:
              ymin=-slope*(x_V[i]-Lx/2)
              y_V[counter]=ymin+(ymax-ymin)/(order*nely)*j
           counter+=1

   if order==1:
      for iel in range(0,nel):
          x_P[iel]=sum(x_V[icon_V[0:m_V,iel]])*0.25
          y_P[iel]=sum(y_V[icon_V[0:m_V,iel]])*0.25
   else:
      counter=0    
      for j in range(0,(order-1)*nely+1):
          for i in range(0,(order-1)*nelx+1):
              if x_P[counter]>Lx/2:
                 ymin=-slope*(x_P[i]-Lx/2)
                 y_P[counter]=ymin+(ymax-ymin)/((order-1)*nely)*j
              counter+=1

if debug:
   np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')
   np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

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
            N_V=basis_functions_V(rq,sq,order)
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
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
rhs=np.zeros(Nfem,dtype=np.float64) 
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
N_mat=np.zeros((3,m_P),dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
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
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[i]*JxWq*rho*gy

            N_mat[0,:]=N_P[:]
            N_mat[1,:]=N_P[:]

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
            rhs[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        rhs[Nfem_V+m2]+=h_el[k2]

print("build FE matrix: %.3fs - %d elts" % (clock.time()-start,nel))

###############################################################################
# assign extra pressure b.c. to remove null space
###############################################################################

A_sparse[Nfem-1,:]=0
A_sparse[:,Nfem-1]=0
A_sparse[Nfem-1,Nfem-1]=1
rhs[Nfem-1]=0

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_sparse.tocsr(),rhs)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# project pressure onto velocity grid
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)
c=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,m_V):
        N_P=basis_functions_P(r_V[i],s_V[i],order)
        q[icon_V[i,iel]]+=np.dot(N_P,p[icon_P[:,iel]])
        c[icon_V[i,iel]]+=1.

q/=c

if debug: np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute L2 errors
###############################################################################
start=clock.time()

vrms=0.
prms=0
qrms=0
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq,order)
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            qq=np.dot(N_V,q[icon_V[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])
            vrms+=(uq**2+vq**2)*JxWq
            qrms+=qq**2*JxWq
            prms+=pq**2*JxWq
        # end for jq
    # end for iq
# end for iel

vrms=np.sqrt(vrms/area.sum())
prms=np.sqrt(prms/area.sum())
qrms=np.sqrt(qrms/area.sum())

print("     -> nel= %6d ; vrms= %.8e ; prms= %.8e ; qrms= %.8e" %(nel,vrms,prms,qrms))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################

if True:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
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
    if order==1:
       vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
       for i in range(0,nel):
           vtufile.write("%10e \n" %p[i])
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    if order==1:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[3,iel],icon_V[2,iel]))
    if order==2:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[2,iel],icon_V[8,iel],icon_V[6,iel]))
    if order==3:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[3,iel],icon_V[15,iel],icon_V[12,iel]))
    if order==4:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[4,iel],icon_V[24,iel],icon_V[20,iel]))

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

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
