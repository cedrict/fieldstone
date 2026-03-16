import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import time as clock

###############################################################################
# density & viscosity function
###############################################################################

def rho(rho0,alphaT,T,T0):
    val=rho0*(1.-alphaT*(T-T0)) 
    return val

def eta(T,eta0):
    return eta0

###############################################################################
# velocity basis functions
###############################################################################

def basis_functions_V(r,s,order):
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

##############################################################################
# velocity basis functions derivatives
##############################################################################

def basis_functions_V_dr(r,s,order):
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

def basis_functions_V_ds(r,s,order):
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

##############################################################################
# pressure basis functions 
##############################################################################

def basis_functions_P(r,s,order):
    if order==2:
       N0=0.25*(1-r)*(1-s)
       N1=0.25*(1+r)*(1-s)
       N2=0.25*(1-r)*(1+s)
       N3=0.25*(1+r)*(1+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)
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

def basis_functions_P_dr(r,s,order):
    if order==2:
       dNdr_0=-0.25*(1-s)
       dNdr_1=+0.25*(1-s)
       dNdr_2=-0.25*(1+s)
       dNdr_3=+0.25*(1+s)
       return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def basis_functions_P_ds(r,s,order):
    if order==2:
       dNds_0=-0.25*(1-r)
       dNds_1=-0.25*(1+r)
       dNds_2=+0.25*(1-r)
       dNds_3=+0.25*(1+r)
       return dNds_0,dNds_1,dNds_2,dNds_3

###############################################################################

eps=1e-9

print("*******************************")
print("********** stone 110 **********")
print("*******************************")

ndim=2   # number of dimensions
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.
Ly=1.

if int(len(sys.argv) == 7):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   order = int(sys.argv[4])
   Ra_nb = float(sys.argv[5])
   nstep = int(sys.argv[6])
else:
   nelx = 48
   nely = nelx
   visu = 1
   order= 2
   Ra_nb= 1e4
   nstep= 100

tol_ss=1e-7   # tolerance for steady state 

top_bc_noslip=False
bot_bc_noslip=False

nel=nelx*nely
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
nn_V=nnx*nny

if order==2:
   nn_P=(nelx+1)*(nely+1)
   m_V=9     # number of velocity nodes making up an element
   m_P=4     # number of pressure nodes making up an element
   r_V=[-1,0,+1,-1,0,+1,-1,0,+1]
   s_V=[-1,-1,-1,0,0,0,+1,+1,+1]
   r_P=[-1,+1,-1,+1]
   s_P=[-1,-1,+1,+1]
if order==3:
   nn_P=(2*nelx+1)*(2*nely+1)
   m_V=16    # number of velocity nodes making up an element
   m_P=9     # number of pressure nodes making up an element
   r_V=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   s_V=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]
   r_P=[-1,0,+1,-1,0,+1,-1,0,+1]
   s_P=[-1,-1,-1,0,0,0,+1,+1,+1]
if order==4:
   nn_P=(3*nelx+1)*(3*nely+1)
   m_V=25    # number of velocity nodes making up an element
   m_P=16     # number of pressure nodes making up an element
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

sparse=True # storage of FEM matrix 

EBA=True

###############################################################################
# definition: Ra_nb=alphaT*abs(gy)*Ly**3*rho0**2*hcapa/hcond/eta

alphaT=2.5e-3 # thermal expansion coefficient
hcond=1.      # thermal conductivity
hcapa=1e-2    # heat capacity
rho0=20       # reference density
T0=0          # reference temperature
relax=0.25    # relaxation coefficient (0,1)
gy=-1 #Ra/alphaT # vertical component of gravity vector

eta0 = alphaT*abs(gy)*Ly**3*rho0**2*hcapa/hcond/Ra_nb

Di_nb=alphaT*abs(gy)*Ly/hcapa

debug=False

###############################################################################

L_ref=Ly
T_ref=1
eta_ref=eta0
kappa_ref=hcond/hcapa/rho0
vel_ref=kappa_ref/L_ref
t_ref=L_ref**2/kappa_ref
q_ref=eta_ref*L_ref/t_ref**2

print('L_ref    =',L_ref)
print('T_ref    =',T_ref)
print('eta_ref  =',eta_ref)
print('kappa_ref=',kappa_ref)
print('vel_ref  =',vel_ref)
print('t_ref    =',t_ref)
print('q_ref    =',q_ref)
print("-----------------------------")

###############################################################################

nq_per_dim=order+1 # dubious/not necessary

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

Nu_vrms_file=open('Nu_vrms.ascii',"w")
Nu_vrms_file.write("#istep,Nusselt,vrms\n")
Tavrg_file=open('T_avrg.ascii',"w")
conv_file=open('conv.ascii',"w")

###############################################################################

print ('Ra       =',Ra_nb)
print ('Di       =',Di_nb)
print ('eta0     =',eta0)
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
# checking that all velocity basis fcts are 1 on their node and zero elsewhere

if debug:
   for i in range(0,m_V):
       print ('node',i,':',basis_functions_V(r_V[i],s_V[i],order))

###############################################################################
# checking that all pressure basis fcts are 1 on their node and zero elsewhere

if debug:
   for i in range(0,m_P):
       print ('node',i,':',basis_functions_P(r_P[i],s_P[i],order))

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/order
        y_V[counter]=j*hy/order
        counter+=1
    #end for
#end for

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
            #end for
        #end for
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

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)

counter=0    
for j in range(0,(order-1)*nely+1):
    for i in range(0,(order-1)*nelx+1):
        x_P[counter]=i*hx/(order-1)
        y_P[counter]=j*hy/(order-1)
        counter+=1
    #end for
#end for

if debug: np.savetxt('gridP.ascii',np.array([xi_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start=clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

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
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if bot_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if top_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.

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

print("temperature b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)
T_prev=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    T[i]=1.-y_V[i]-0.01*np.cos(np.pi*x_V[i]/Lx)*np.sin(np.pi*y_V[i]/Ly)

T_prev[:]=T[:]

if debug: np.savetxt('temperature_init.ascii',np.array([x_V,y_V,T]).T,header='# x,y,T')

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements / sanity check
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
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
u_prev=np.zeros(nn_V,dtype=np.float64)
v_prev=np.zeros(nn_V,dtype=np.float64)
Tvect=np.zeros(m_V,dtype=np.float64)   
C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

Nusselt_prev=1.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    if sparse:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    else:   
       K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
       G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

    b_fem=np.zeros(Nfem,dtype=np.float64)   
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
    N_mat=np.zeros((3,m_P),dtype=np.float64) 

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
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                K_el+=B.T.dot(C.dot(B))*eta(Tq,eta0)*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i+1]+=N_V[i]*rho(rho0,alphaT,Tq,T0)*gy*JxWq
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

        # assemble matrix and right hand side vector 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        if sparse:
                           A_sparse[m1,m2] += K_el[ikk,jkk]
                        else:
                           K_mat[m1,m2]+=K_el[ikk,jkk]
                        #end if
                    #end for
                #end for
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    if sparse:
                       A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                       A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                    else:
                       G_mat[m1,m2]+=G_el[ikk,jkk]
                    #end if
                b_fem[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            b_fem[Nfem_V+m2]+=h_el[k2]
        #end for

    #end for iel

    print("build FE matrix: %.3fs" % (clock.time()-start))

    ###########################################################################
    # assemble K, G, GT, f, h into A and rhs
    ###########################################################################
    start=clock.time()

    if not sparse:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
       a_mat[0:Nfem_V,0:Nfem_V]=K_mat
       a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
       a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

    print("assemble blocks: %.3f s" % (clock.time()-start))

    ###########################################################################
    # assign extra pressure b.c. to remove null space
    ###########################################################################

    if sparse:
       A_sparse[Nfem-1,:]=0
       A_sparse[:,Nfem-1]=0
       A_sparse[Nfem-1,Nfem-1]=1
       b_fem[Nfem-1]=0
    else:
       a_mat[Nfem-1,:]=0
       a_mat[:,Nfem-1]=0
       a_mat[Nfem-1,Nfem-1]=1
       b_fem[Nfem-1]=0
    #end if

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    if sparse:
       sparse_matrix=A_sparse.tocsr()
    else:
       sparse_matrix=sps.csr_matrix(a_mat)

    sol=sps.linalg.spsolve(sparse_matrix,b_fem)

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

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    ###########################################################################
    # relaxation step
    ###########################################################################

    u=relax*u+(1-relax)*u_prev
    v=relax*v+(1-relax)*v_prev

    ###########################################################################
    # compute nodal strainrate and heat flux 
    ###########################################################################
    start=clock.time()
    
    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)
    c=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,m_V):
            rq=r_V[i]
            sq=s_V[i]

            N_V=basis_functions_V(rq,sq,order)
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
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

    print("     -> exx_n (m,M) %.4e %.4e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.4e %.4e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.4e %.4e " %(np.min(exy_n),np.max(exy_n)))

    if debug:
       np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')
       np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute nodal press & sr: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64)
    b_fem=np.zeros(Nfem_T,dtype=np.float64)     
    B_mat=np.zeros((ndim,m_V),dtype=np.float64) 
    N_mat=np.zeros((m_V,1),dtype=np.float64)   

    for iel in range (0,nel):

        A_el=np.zeros((m_V,m_V),dtype=np.float64)
        Ka=np.zeros((m_V,m_V),dtype=np.float64)  
        Kd=np.zeros((m_V,m_V),dtype=np.float64) 
        MM=np.zeros((m_V,m_V),dtype=np.float64)
        vel=np.zeros((1,ndim),dtype=np.float64)
        b_el=np.zeros(m_V,dtype=np.float64)

        Tvect[:]=T[icon_V[:,iel]]

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
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
                vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
                vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])
                B_mat[0,:]=dNdx_V[:]
                B_mat[1,:]=dNdy_V[:]

                # is it ok to use dqdx instead of dpdx ?
                dpdxq=np.dot(dNdx_V,q[icon_V[:,iel]])
                dpdyq=np.dot(dNdy_V,q[icon_V[:,iel]])
                #dNNNPdr[0:mP]=dNNPdr(rq,sq,order)
                #dNNNPds[0:mP]=dNNPds(rq,sq,order)
                #dNNNPdx[0:mP]=dNNNPdr[0:mP]*2/hx
                #dNNNPdy[0:mP]=dNNNPds[0:mP]*2/hy
                #for k in range(0,mP):
                #    dpdxq += dNNNPdx[k]*p[icon_P[k,iel]]
                #    dpdyq += dNNNPdy[k]*p[icon_P[k,iel]]

                # compute mass matrix
                #MM+=N_mat.dot(N_mat.T)*rho0*hcapa*JxWq

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*JxWq

                # compute advection matrix
                N_mat[:,0]=N_V
                Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*JxWq

                if EBA:
                   #b_el[:]+=N_mat[:,0]*weightq*jcob* 2*eta(Tq,eta0)*(exxq**2+eyyq**2+2*exyq**2) # viscous dissipation
                   #b_el[:]+=N_mat[:,0]*weightq*jcob* alphaT*Tq*(vel[0,0]*dpdxq+vel[0,1]*dpdyq) 
                   # viscous dissipation
                   b_el[:]+=N_mat[:,0]*JxWq* 2*eta(Tq,eta0)*(2./3*exxq**2+2./3*eyyq**2-2./3*exxq*eyyq+2*exyq**2) 
                   # adiabatic heating
                   MM-=N_mat.dot(N_mat.T)*alphaT*(vel[0,0]*dpdxq+vel[0,1]*dpdyq)*JxWq # adiabatic heating

            #end for
        #end for

        A_el=Ka+Kd+MM

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

        # assemble matrix and right hand side vector 
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            for k2 in range(0,m_V):
                m2=icon_V[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    #end for iel

    print("build FE matrix : %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # relax
    ###########################################################################
    start=clock.time()

    T=relax*T+(1-relax)*T_prev

    print("relax temperature field: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start=clock.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
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
                JxWq=np.linalg.det(jcb)*weightq
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                vrms+=(uq**2+vq**2)*JxWq
                Tavrg+=Tq*JxWq
            #end for jq
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly)) / vel_ref
    Tavrg/=(Lx*Ly)             / T_ref

    Tavrg_file.write("%d %.10e\n" % (istep,Tavrg))
    Tavrg_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal pressure gradient and heat flux 
    ###########################################################################
    start=clock.time()
    
    count = np.zeros(nn_V,dtype=np.int32)  
    qx_n = np.zeros(nn_V,dtype=np.float64)  
    qy_n = np.zeros(nn_V,dtype=np.float64)  
    dpdx_n = np.zeros(nn_V,dtype=np.float64)  
    dpdy_n = np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            rq=r_V[i]
            sq=s_V[i]
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            inode=icon_V[i,iel]
            qx_n[inode]-=hcond*np.dot(dNdx_V,T[icon_V[:,iel]])
            qy_n[inode]-=hcond*np.dot(dNdy_V,T[icon_V[:,iel]])
            dpdx_n[inode]+=np.dot(dNdx_V,q[icon_V[:,iel]])
            dpdy_n[inode]+=np.dot(dNdy_V,q[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count
    dpdx_n/=count
    dpdy_n/=count

    print("     -> qx_n (m,M) %.4e %.4e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.4e %.4e " %(np.min(qy_n),np.max(qy_n)))

    np.savetxt('heatflux_bot.ascii',np.array([x_V[0:nnx],qx_n[0:nnx],qy_n[0:nnx]]).T)
    np.savetxt('heatflux_top.ascii',np.array([x_V[nn_V-nnx:nn_V],qx_n[nn_V-nnx:nn_V],qy_n[nn_V-nnx:nn_V]]).T)
    if debug: np.savetxt('heatflux.ascii',np.array([x_V,y_V,qx_n,qy_n]).T)

    print("compute nodal heat flux: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute heat flux and Nusselt number
    # qy_top1,qy_bot1 obtained by integration of the nodal heat flux on top/bottom
    # qy_top2,qy_bot2 obtained using only mid-edge value 
    ###########################################################################
    start=clock.time()

    qy_top1=0
    qy_top2=0
    qy_bot1=0
    qy_bot2=0
    Nusselt=0
    for iel in range(0,nel):
        #--------------
        # top boundary 
        #--------------
        if y_V[icon_V[m_V-1,iel]]>1-eps: 
           for iq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=+1
               weightq=qweights[iq]
               N_V=basis_functions_V(rq,sq,order)
               xq=np.dot(N_V,x_V[icon_V[:,iel]])
               q_y=np.dot(N_V,qy_n[icon_V[:,iel]])
               jcob=hx/2.
               Nusselt+=q_y*jcob*weightq
               qy_top1+=q_y*jcob*weightq
           #end for
           qy_top2+=qy_n[icon_V[7,iel]]*hx
        #end if
        #--------------
        # bottom boundary 
        #--------------
        if y_V[icon_V[0,iel]]<eps: 
           for iq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=-1
               weightq=qweights[iq]
               N_V=basis_functions_V(rq,sq,order)
               xq=np.dot(N_V,x_V[icon_V[:,iel]])
               q_y=np.dot(N_V,qy_n[icon_V[:,iel]])
               jcob=hx/2.
               qy_bot1+=q_y*jcob*weightq *-1
           #end for
           qy_bot2+=qy_n[icon_V[1,iel]]*hx *-1
        #end if
    #end for

    Nusselt=np.abs(Nusselt)/Lx

    Nu_vrms_file.write("%d %e %e %.8e %.8e %.8e %.8e\n" % (istep,Nusselt,vrms,qy_bot1,qy_top1,qy_bot1,qy_top2))
    Nu_vrms_file.flush()

    print("     istep= %d ; Nusselt= %e ; Ra= %e " %(istep,Nusselt,Ra_nb))

    print("compute Nu: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute temperature and vel profile 
    # this approach is actually too naive and yield results which 
    # are quite off with respect to aspect
    ###########################################################################
    #start = timing.time()
    #T_profile = np.zeros(nny,dtype=np.float64)  
    #y_profile = np.zeros(nny,dtype=np.float64)  
    #v_profile = np.zeros(nny,dtype=np.float64)  
    #counter=0    
    #for j in range(0,nny):
    #    for i in range(0,nnx):
    #        T_profile[j]+=T[counter]/nnx
    #        y_profile[j]=yV[counter]
    #        v_profile[j]+=np.sqrt(u[counter]**2+v[counter]**2)/nnx
    #        counter+=1
    #    #end for
    #end for
    #np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='#y,T')
    #np.savetxt('vel_profile.ascii',np.array([y_profile,v_profile]).T,header='#y,vel')
    #print("compute T & vel profile: %.3f s" % (timing.time() - start))

    ###########################################################################
    # compute temperature and vel profile 
    # looping over each element. Considering the bottom 3 nodes, three nodes 
    # and top three nodes (only for elements at the surface). Integrate 
    # velocity and temperature on each face and add to the profile.
    ###########################################################################
    start=clock.time()

    T_profile=np.zeros(nny,dtype=np.float64)  
    y_profile=np.zeros(nny,dtype=np.float64)  
    vel_profile=np.zeros(nny,dtype=np.float64)  

    iel=0
    for j in range(0,nely):
        for i in range(0,nelx):
            for iq in range(0,nq_per_dim):
                rq=qcoords[iq]
                weightq=qweights[iq]
                jcob=hx/2.
                #---------------------
                #-- bottom row s=-1 --
                #---------------------
                sq=-1
                N_V=basis_functions_V(rq,sq,order)
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])


                jnode=2*j     
                y_profile[jnode]=yq
                vel_profile[jnode]+=np.sqrt(uq**2+vq**2)*jcob*weightq
                T_profile[jnode]+=Tq*jcob*weightq

                #--------------------
                #-- middle row s=0 --
                #--------------------
                sq=0
                N_V=basis_functions_V(rq,sq,order)
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                jnode=2*j+1     
                y_profile[jnode]=yq
                vel_profile[jnode]+=np.sqrt(uq**2+vq**2)*jcob*weightq
                T_profile[jnode]+=Tq*jcob*weightq

                #-----------------
                #-- top row s=1 --
                #-----------------
                if j==nely-1:
                   sq=1
                   N_V=basis_functions_V(rq,sq,order)
                   yq=np.dot(N_V,y_V[icon_V[:,iel]])
                   uq=np.dot(N_V,u[icon_V[:,iel]])
                   vq=np.dot(N_V,v[icon_V[:,iel]])
                   Tq=np.dot(N_V,T[icon_V[:,iel]])
                   jnode=2*j+2
                   y_profile[jnode]=yq
                   vel_profile[jnode]+=np.sqrt(uq**2+vq**2)*jcob*weightq
                   T_profile[jnode]+=Tq*jcob*weightq
                #end if

            #end for
            iel+=1

        #end for
    #end for

    np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='#y,T')
    np.savetxt('vel_profile.ascii',np.array([y_profile,vel_profile]).T,header='#y,vel')

    print("compute T & vel profile fancy: %.3f s" % (clock.time()-start))

    ###########################################################################
    # assess convergence of iterations
    ###########################################################################
    start=clock.time()

    T_diff=np.sum(abs(T-T_prev))/nn_V
    Nu_diff=np.abs(Nusselt-Nusselt_prev)/Nusselt

    print("T conv, T_diff, Nu_diff: " , T_diff<tol_ss,T_diff,Nu_diff)

    conv_file.write("%10e %10e %10e %10e\n" % (istep,T_diff,Nu_diff,tol_ss))
    conv_file.flush()

    if T_diff<tol_ss and Nu_diff<tol_ss:
       print("convergence reached")
       converged=True
    else:
       converged=False

    print("assess convergence: %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if visu==1 or converged or istep==nstep-1:
       filename = 'solution_{:04d}.vtu'.format(istep)
       if converged or istep==nstep-1:
          filename = 'solution.vtu'.format(istep)

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
       q.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       T.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (rho(rho0,alphaT,T[i],T0)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       exx_n.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       eyy_n.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       exy_n.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eff. strain rate' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (np.sqrt(0.5*(exx_n[i]**2+eyy_n[i]**2)+exy_n[i]**2)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='shear heating (2*eta*e)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (2*eta(T[i],eta0)*np.sqrt(exx_n[i]**2+eyy_n[i]**2+2*exy_n[i]**2)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='adiab heating (linearised)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (alphaT*T[i]*rho0*v[i]*gy))
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Float32' Name='adiab heating (true)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i]))) 
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Float32' Name='adiab heating (diff)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i])-\
                                       alphaT*T[i]*rho0*v[i]*gy))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(qx_n[i],qy_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='pressure gradient' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(dpdx_n[i],dpdy_n[i],0.))
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

       print("export to vtu file: %.3f s" % (clock.time()-start))

    T_prev[:]=T[:]
    u_prev[:]=u[:]
    v_prev[:]=v[:]
    Nusselt_prev=Nusselt

    if converged:
       break

#end for istep
    
print("     script ; Nusselt= %e ; Ra= %e ; order= %d" %(Nusselt,Ra_nb,order))

print("*******************************")
print("********** the end ************")
print("*******************************")

##############################################################################
