import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing
import random

#------------------------------------------------------------------------------
# velocity shape functions
#------------------------------------------------------------------------------
# Q2          Q1
# 6---7---8   2-------3
# |       |   |       |
# 3   4   5   |       |    etc ...
# |       |   |       |
# 0---1---2   0-------1

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

#------------------------------------------------------------------------------
# velocity shape functions derivatives
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# constants

cm=1e-2
km=1e3
eps=1e-9
year=365.25*24*3600
sqrt2=np.sqrt(2)
ndim=2   # number of dimensions
ndofT=1  # number of temperature degrees of freedom per node

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")


#####################################################################
# TODO:
# - make function for velocity profile
# - stretch mesh in vertical direction - careful with jacobian!!
# - matrix does not change in time, precompute!
# - compute heat flux at surface
# - benchmark advection and diffusion
#####################################################################
Lx=100*km          # horizontal dimension of domain
Ly=25*km           # vertical dimension of domain
Tsurf=0            # temperature at the top
Tbase=1480         # temperature at the bottom
Tintrusion=1250    # temperature prescribed at intrusion
Hmagma=2500        # thickness of magma channel
Umagma=20*cm/year  # maximum velocity
Ymagma=10.5*km     # channel middle depth 
hcapa_rock=800     # heat capacity
hcond_rock=1.5     # heat conductivity
rho_rock=2900      # density
hcapa_magma=1100   # heat capacity
hcond_magma=1.5    # heat conductivity
rho_magma=2700     # density
tfinal=10e6*year   # duration of simulation
nelx = 80          # number of elements in x direction
nely = 100         # number of elements in y direction
order= 1           # polynomial basis for basis functions
nstep= 2000        # maximum number of time steps
CFL_nb = 1         # CFL number
supg_type=0        # toggle switch for SUPG advection stabilisation
#####################################################################

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
if order==2:
   NP=(nelx+1)*(nely+1)
   mV=9     # number of velocity nodes making up an element
   mP=4     # number of pressure nodes making up an element
   rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
if order==3:
   NP=(2*nelx+1)*(2*nely+1)
   mV=16    # number of velocity nodes making up an element
   mP=9     # number of pressure nodes making up an element
   rVnodes=[-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1,-1,-1./3.,+1./3.,+1]
   sVnodes=[-1,-1,-1,-1,-1./3.,-1./3.,-1./3.,-1./3.,+1./3.,+1./3.,+1./3.,+1./3.,+1,+1,+1,+1]

NfemT=NV*ndofT       # nb of temperature dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

#################################################################
# quadrature setup
#################################################################

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

#################################################################
# open output files

dt_file=open('dt.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")

#################################################################

kappa_rock  =hcond_rock /rho_rock /hcapa_rock
kappa_magma =hcond_magma/rho_magma/hcapa_magma

print ('order       =',order)
print ('hx          =',hx)
print ('hy          =',hy)
print ('nnx         =',nnx)
print ('nny         =',nny)
print ('nel         =',nel)
print ('NfemT       =',NfemT)
print ('nqperdim    =',nqperdim)
print ('tfinal      =',tfinal/year,' year')
print ('kappa_rock  =',kappa_rock)
print ('kappa_magma =',kappa_magma)
print ('Umagma      =',Umagma/cm*year,'cm/year')
print ('Hmagma      =',Hmagma,'m')
print ('-----------------------------')

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

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]/Ly<eps:
       bc_fixT[i]=True ; bc_valT[i]=Tbase
    if yV[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=Tsurf
    if xV[i]/Lx<eps and abs(yV[i]-Ymagma)<Hmagma/2:
       bc_fixT[i]=True ; bc_valT[i]=Tintrusion

print("temperature b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)
T_init = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=(Tsurf-Tbase)/Ly*yV[i]+Tbase

T_init[:]=T[:]

print("initial temperature: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
# not strictly necessary, but good test of basis functions
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

#################################################################
# define velocity field 
#################################################################
start = timing.time()

u = np.zeros(NV,dtype=np.float64) # x-component velocity
v = np.zeros(NV,dtype=np.float64) # y-component velocity

for i in range(0,NV):
    if abs(yV[i]-Ymagma)<Hmagma/2:
       Y=yV[i]-Ymagma+Hmagma/2
       u[i]=-(Y**2-Y*Hmagma)/Hmagma**2*4 * Umagma

print("define velocity field: %.3f s" % (timing.time() - start))

#################################################################
# define elemental parameters 
#################################################################
start = timing.time()
    
hcond = np.zeros(nel,dtype=np.float64)
hcapa = np.zeros(nel,dtype=np.float64)
rho = np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc=np.sum(xV[iconV[:,iel]])/mV
    yc=np.sum(yV[iconV[:,iel]])/mV
    if abs(yc-Ymagma)<Hmagma/2:
       hcapa[iel]=hcapa_magma
       hcond[iel]=hcond_magma
       rho[iel]=rho_magma
    else:
       hcapa[iel]=hcapa_rock
       hcond[iel]=hcond_rock
       rho[iel]=rho_rock

print("elemental params: %.3f s" % (timing.time() - start))

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
Tvect   = np.zeros(mV,dtype=np.float64)   

time=0

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # compute timestep value
    #################################################################

    dt1=CFL_nb*min(hx,hy) /np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*min(hx,hy)**2 / max(kappa_rock,kappa_magma)
    dt=np.min([dt1,dt2])
    time+=dt

    print('     -> dt  = %.6f yr' %(dt/year))
    print('     -> time= %.6f yr' %(time/year))

    dt_file.write("%10e %10e %10e %10e\n" % (time/year,dt1/year,dt2/year,dt/year))
    dt_file.flush()

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions
    N_mat_supg = np.zeros((mV,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*nqperdim**ndim,dtype=np.float64)

    #precompute jacobian
    jcob=hx*hy/4
    jcbi=np.zeros((ndim,ndim),dtype=np.float64)
    jcbi[0,0]=2/hx
    jcbi[1,1]=2/hy

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
                xq=0
                yq=0
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,mV):
                    vel[0,0]+=N_mat[k,0]*u[iconV[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[iconV[k,iel]]
                    xq+=N_mat[k,0]*xV[iconV[k,iel]]
                    yq+=N_mat[k,0]*yV[iconV[k,iel]]
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
                MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob*rho[iel]*hcapa[iel]

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*weightq*jcob*hcond[iel]

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B_mat))*weightq*jcob*rho[iel]*hcapa[iel]

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

    #print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix : %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     T (m,M): %.4f %.4f " %(np.min(T),np.max(T)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(T),np.max(T)))
    Tstats_file.flush()

    print("solve linear system: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute heat flux
    #####################################################################
    start = timing.time()

    qx = np.zeros(nel,dtype=np.float64)
    qy = np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):
        rq = 0.0 
        sq = 0.0 
        dNNNVdr[0:mV]=dNNVdr(rq,sq,order)
        dNNNVds[0:mV]=dNNVds(rq,sq,order)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            qx[iel]-=hcond[iel]*dNNNVdx[k]*T[iconV[k,iel]]
            qy[iel]-=hcond[iel]*dNNNVdy[k]*T[iconV[k,iel]]
        #end for
    #end for

    print("     qx (m,M): %.4f %.4f " %(np.min(qx),np.max(qx)))
    print("     qy (m,M): %.4f %.4f " %(np.min(qy),np.max(qy)))

    print("compute heat flux: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    every=1
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
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T-T_init' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %(T[i]-T_init[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='qx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % qx[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='qy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % qy[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % rho[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='hcapa' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % hcapa[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='hcond' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % hcond[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
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

       print("export to vtu: %.3f s" % (timing.time() - start))

    ###################################

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
