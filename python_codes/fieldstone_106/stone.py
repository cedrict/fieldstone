import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing
import random
from scipy import special
from scipy.interpolate import interp1d

#------------------------------------------------------------------------------
# denisty and viscosity functions
#------------------------------------------------------------------------------

def eta(T):
    Tused=min(T,Tpatch)
    Tused=max(Tused,Tsurf)
    val=eta0*np.exp(E*(1./(Tused+T0)-1./(1+T0)))    
    val=min(val,eta_max)
    return val

#------------------------------------------------------------------------------
# velocity shape functions
#------------------------------------------------------------------------------
# Q2          Q1
# 6---7---8   2-------3
# |       |   |       |
# 3   4   5   |       |
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

#------------------------------------------------------------------------------
# pressure shape functions 
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# constants

eps=1e-9
R=8.3145

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2   # number of dimensions
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom per node
ndofT=1  # number of temperature degrees of freedom per node

order = 2

if int(len(sys.argv) == 6):
   nelr  = int(sys.argv[1])
   visu  = int(sys.argv[2])
   nstep = int(sys.argv[3])
   exp   = int(sys.argv[4])
   every = int(sys.argv[5])
else:
   nelr = 24
   visu = 1
   nstep= 1000
   exp  = 1
   every= 1

axisymmetric=True

#DeltaT=3000
#Tsurf=293
#Tpatch=Tsurf+DeltaT
#hcond=3
#hcapa=1250
#rho0=3300
#g0=9.81
#alpha=3e-5
#eta0=rho0**2*hcapa*g0*alpha*DeltaT*(Router-Rinner)**3/Ra/hcond

Rinner=1.2222 
Router=2.2222 
Ra=1e7
Tsurf=0
Tpatch=1
eta0=1
kappa=1
eta_max=1e3*eta0
T0=0.1

if exp==1:
   E=0
if exp==2:
   E=0.25328
if exp==3:
   E=3

viscosity_model=1 # do not change for now

tfinal=1

CFL_nb=1.
apply_filter=False
supg_type=0


nelt=int(0.75*nelr)
nel=nelt*nelr
nnt=order*nelt+1  # number of elements, x direction
nnr=order*nelr+1  # number of elements, y direction
NV=nnt*nnr

if order==1:
   NP=nelt*nelr
   mV=4     # number of velocity nodes making up an element
   mP=1     # number of pressure nodes making up an element
   rVnodes=[-1,+1,-1,+1]
   sVnodes=[-1,-1,+1,+1]
   rPnodes=[0]
   sPnodes=[0]
if order==2:
   NP=(nelt+1)*(nelr+1)
   mV=9     # number of velocity nodes making up an element
   mP=4     # number of pressure nodes making up an element
   rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
   sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
   rPnodes=[-1,+1,-1,+1]
   sPnodes=[-1,-1,+1,+1]

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs
NfemT=NV*ndofT       # nb of temperature dofs

ht=np.pi/8/nelt         # element size in tangential direction (radians)
hr=(Router-Rinner)/nelr # element size in radial direction (meters)

sparse=True # storage of FEM matrix 

eta_ref=eta0

use_fs_on_sides=True


#################################################################

sqrt2=np.sqrt(2)

nqperdim=order+1

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

#################################################################
# open output files

vrms_file=open('vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")
vrstats_file=open('stats_vr.ascii',"w")
vtstats_file=open('stats_vt.ascii',"w")

#################################################################

#print ('Ra          =',rho0*alpha*g0*DeltaT*(Router-Rinner)**3/eta0/(hcond/hcapa/rho0) )
print ('Ra          =',Ra)
print ('Rinner/Router=',Rinner/Router)
print ('nelr        =',nelr)
print ('nelt        =',nelt)
print ('NV          =',NV)
print ('NP          =',NP)
print ('nel         =',nel)
print ('NfemV       =',NfemV)
print ('NfemP       =',NfemP)
print ('Nfem        =',Nfem)
print ('nqperdim    =',nqperdim)
print ('eta0        =',eta0)
print ('E           =',E)
print ("-----------------------------")

#################################################################
# reading in steinberger profile 
#################################################################

#profile_r=np.empty(2821,dtype=np.float64)
#profile_eta=np.empty(2821,dtype=np.float64)
##profile_r[1:11511],profile_rho[1:11511]=np.loadtxt('data/rho_prem.ascii',unpack=True,usecols=[0,1])
#profile_r,profile_eta=np.loadtxt('../../images/viscosity_profile/steinberger2/visc_sc06.d',unpack=True,usecols=[1,0])
#profile_r=(6371-profile_r)*1000
#profile_r=np.flip(profile_r)
#profile_eta=np.flip(profile_eta)
#print(np.min(profile_r),np.max(profile_r))
#print(np.min(profile_eta),np.max(profile_eta))
#f_cubic   = interp1d(profile_r, profile_eta, kind='cubic')

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
rV=np.zeros(NV,dtype=np.float64)  # x coordinates
tV=np.zeros(NV,dtype=np.float64)  # x coordinates

counter=0    
for j in range(0,nnr):
    for i in range(0,nnt):
        #tV[counter]=5*np.pi/8-i*ht/order
        tV[counter]=4*np.pi/8-i*ht/order
        rV[counter]=Rinner+j*hr/order
        xV[counter]=rV[counter]*np.cos(tV[counter])
        yV[counter]=rV[counter]*np.sin(tV[counter])
        counter+=1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV,tV,rV]).T,header='# x,y')
#print(np.min(rV),np.max(rV))

print("build V grid: %.3f s" % (timing.time() - start))

#################################################################
#################################################################

flag_el_1=np.zeros(nel,dtype=np.bool)  
flag_el_2=np.zeros(nel,dtype=np.bool)  
flag_el_3=np.zeros(nel,dtype=np.bool)  
flag_el_4=np.zeros(nel,dtype=np.bool)  

counter=0
for j in range(0,nelr):
    for i in range(0,nelt):
        if i==0:
           flag_el_1[counter]=True
        if i==nelt-1:
           flag_el_2[counter]=True
        if j==0:
           flag_el_3[counter]=True
        if j==nelr-1:
           flag_el_4[counter]=True
        counter += 1
    #end for
#end for

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0
for j in range(0,nelr):
    for i in range(0,nelt):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                iconV[counter2,counter]=i*order+l+j*order*nnt+nnt*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

# creating a dedicated connectivity array to plot the solution on Q1 space
# different icon array but same velocity nodes.

nel2=(nnt-1)*(nnr-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
counter = 0
for j in range(0,nnr-1):
    for i in range(0,nnt-1):
        iconQ1[0,counter]=i+j*nnt
        iconQ1[1,counter]=i+1+j*nnt
        iconQ1[2,counter]=i+1+(j+1)*nnt
        iconQ1[3,counter]=i+(j+1)*nnt
        counter += 1
    #end for
#end for

# creating a dedicated connectivity array to plot the solution on P1 space
# different icon array but same velocity nodes.

nel_P1=4*nel2
iconP1 =np.zeros((3,nel_P1),dtype=np.int32)
counter=0
for iel in range(0,nel2):
    iconP1[0,counter]=iconQ1[0,iel]
    iconP1[1,counter]=iconQ1[1,iel]
    iconP1[2,counter]=NV+iel
    counter += 1
    iconP1[0,counter]=iconQ1[1,iel]
    iconP1[1,counter]=iconQ1[2,iel]
    iconP1[2,counter]=NV+iel
    counter += 1
    iconP1[0,counter]=iconQ1[2,iel]
    iconP1[1,counter]=iconQ1[3,iel]
    iconP1[2,counter]=NV+iel
    counter += 1
    iconP1[0,counter]=iconQ1[3,iel]
    iconP1[1,counter]=iconQ1[0,iel]
    iconP1[2,counter]=NV+iel
    counter += 1
#end for

NV_P1=NV+nel2
xV_P1=np.zeros(NV_P1,dtype=np.float64)  # x coordinates
yV_P1=np.zeros(NV_P1,dtype=np.float64)  # y coordinates

xV_P1[0:NV]=xV[0:NV]
yV_P1[0:NV]=yV[0:NV]

for iel in range(0,nel2):
    xc=0.25*np.sum(xV[iconQ1[0:4,iel]])
    yc=0.25*np.sum(yV[iconQ1[0:4,iel]])
    xV_P1[NV+iel]=xc
    yV_P1[NV+iel]=yc

#np.savetxt('gridV_P1.ascii',np.array([xV_P1,yV_P1]).T,header='# x,y')

print("build iconV: %.3f s" % (timing.time() - start))

#################################################################
# build pressure grid 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
tP=np.empty(NP,dtype=np.float64)     # x coordinates
rP=np.empty(NP,dtype=np.float64)     # y coordinates

if order==1:
   for iel in range(0,nel):
       xP[iel]=sum(xV[iconV[0:mV,iel]])*0.25
       yP[iel]=sum(yV[iconV[0:mV,iel]])*0.25
    #end for
#end if 
      
if order>1:
   counter=0    
   for j in range(0,(order-1)*nelr+1):
       for i in range(0,(order-1)*nelt+1):
           tP[counter]=5*np.pi/8-i*ht/(order-1)
           rP[counter]=Rinner+j*hr/(order-1)
           xP[counter]=rP[counter]*np.cos(tP[counter])
           yP[counter]=rP[counter]*np.sin(tP[counter])
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
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconP[0,counter]=counter
           counter += 1
       #end for
   #end for

if order>1:
   om1=order-1
   counter=0
   for j in range(0,nelr):
       for i in range(0,nelt):
           counter2=0
           for k in range(0,order):
               for l in range(0,order):
                   iconP[counter2,counter]=i*om1+l+j*om1*(om1*nelt+1)+(om1*nelt+1)*k 
                   counter2+=1
               #end for
           #end for
           counter += 1
       #end for
   #end for

print("build iconP: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

flag_1=np.zeros(NV,dtype=np.bool)  
flag_2=np.zeros(NV,dtype=np.bool)  
flag_3=np.zeros(NV,dtype=np.bool)  
flag_4=np.zeros(NV,dtype=np.bool)  

for i in range(0,NV):
    if abs(tV[i]-(4*np.pi/8))<eps: 
       flag_1[i]=True
    if abs(tV[i]-(3*np.pi/8))<eps: 
       flag_2[i]=True
    if abs(rV[i]-Rinner)<eps: 
       flag_3[i]=True
    if abs(rV[i]-Router)<eps: 
       flag_4[i]=True
    #if flag_1[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_2[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_3[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    #if flag_4[i]:
    #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
    if not use_fs_on_sides: # must remove nullspace
       if abs(xV[i])<0.001 and flag_3[i]:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0.
        

#end for

print("velocity b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if flag_4[i]:
       bc_fixT[i]=True ; bc_valT[i]=Tsurf
    if flag_3[i] and abs(tV[i]-np.pi/2)<np.pi/16:
       bc_fixT[i]=True ; bc_valT[i]=Tpatch
#end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=0.25 #*DeltaT+Tsurf
#end for

T_prev[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([xV,yV,T]).T,header='# x,y,T')

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

area=np.pi/8*(Router**2-Rinner**2)
print("     -> total area %.6f " %( area  ))

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
exx_n   = np.zeros(NV,dtype=np.float64)  
eyy_n   = np.zeros(NV,dtype=np.float64)  
exy_n   = np.zeros(NV,dtype=np.float64)  




time=0

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

    if axisymmetric:
       c_mat   = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
       b_mat   = np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
       N_mat   = np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
    else:
       c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
       N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

    if sparse:
       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    else:   
       K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
       G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

    f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 

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

                etaq=eta(Tq)

                #Cartesian components of -\vec{e}_r
                erxq=-xq/np.sqrt(xq**2+yq**2)  
                eryq=-yq/np.sqrt(xq**2+yq**2) 

                if axisymmetric:

                   # construct 3x8 b_mat matrix
                   for i in range(0,mV):
                       b_mat[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                                [NNNV[i]/xq,0.       ],
                                                [0.        ,dNNNVdy[i]],
                                                [dNNNVdy[i],dNNNVdx[i]]]
                   #end for

                   K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob * 2*np.pi*xq

                   for i in range(0,mV):
                       f_el[ndofV*i  ]-=NNNV[i]*jcob*weightq*Tq*erxq*Ra *2*np.pi*xq
                       f_el[ndofV*i+1]-=NNNV[i]*jcob*weightq*Tq*eryq*Ra *2*np.pi*xq
                   #end for

                   for i in range(0,mP):
                       N_mat[0,i]=NNNP[i]
                       N_mat[1,i]=NNNP[i]
                       N_mat[2,i]=NNNP[i]
                       N_mat[3,i]=0.
                   #end for

                   G_el-=b_mat.T.dot(N_mat)*weightq*jcob *2*np.pi*xq

                else:

                   # construct 3x8 b_mat matrix
                   for i in range(0,mV):
                       b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                                [0.        ,dNNNVdy[i]],
                                                [dNNNVdy[i],dNNNVdx[i]]]
                   #end for

                   K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                   for i in range(0,mV):
                       f_el[ndofV*i  ]-=NNNV[i]*jcob*weightq*Tq*erxq*Ra
                       f_el[ndofV*i+1]-=NNNV[i]*jcob*weightq*Tq*eryq*Ra
                   #end for

                   for i in range(0,mP):
                       N_mat[0,i]=NNNP[i]
                       N_mat[1,i]=NNNP[i]
                       N_mat[2,i]=0.
                   #end for

                   G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                #end if

            # end for jq
        # end for iq

        #impose dirichlet b.c. 
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

        #free slip at bottom and top
        if flag_el_3[iel] or flag_el_4[iel]:
           for k in range(0,mV):
               inode=iconV[k,iel]
               if flag_3[inode] or flag_4[inode]:
                  RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                  for i in range(0,mV*ndofV):
                      RotMat[i,i]=1.
                  angle=tV[inode]
                  RotMat[2*k  ,2*k]= np.cos(angle) ; RotMat[2*k  ,2*k+1]=np.sin(angle)
                  RotMat[2*k+1,2*k]=-np.sin(angle) ; RotMat[2*k+1,2*k+1]=np.cos(angle)
                  # apply counter rotation 
                  K_el=RotMat.dot(K_el.dot(RotMat.T))
                  f_el=RotMat.dot(f_el)
                  G_el=RotMat.dot(G_el)
                  # apply boundary conditions
                  # x-component set to 0
                  ikk=ndofV*k     
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,mV*ndofV):
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=0#K_ref*bc_val[m1]
                  #h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
                  # rotate back 
                  K_el=RotMat.T.dot(K_el.dot(RotMat))
                  f_el=RotMat.T.dot(f_el)
                  G_el=RotMat.T.dot(G_el)
               #end if
           #end for
        #end if

        if (flag_el_1[iel] or flag_el_2[iel]) and use_fs_on_sides:
           for k in range(0,mV):
               inode=iconV[k,iel]
               if flag_1[inode] or flag_2[inode]:
                  RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                  for i in range(0,mV*ndofV):
                      RotMat[i,i]=1.
                  angle=tV[inode]
                  RotMat[2*k  ,2*k]= np.cos(angle) ; RotMat[2*k  ,2*k+1]=np.sin(angle)
                  RotMat[2*k+1,2*k]=-np.sin(angle) ; RotMat[2*k+1,2*k+1]=np.cos(angle)
                  # apply counter rotation 
                  K_el=RotMat.dot(K_el.dot(RotMat.T))
                  f_el=RotMat.dot(f_el)
                  G_el=RotMat.dot(G_el)
                  # apply boundary conditions
                  # x-component set to 0
                  ikk=ndofV*k   +1  
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,mV*ndofV):
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=0#K_ref*bc_val[m1]
                  #h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
                  # rotate back 
                  K_el=RotMat.T.dot(K_el.dot(RotMat))
                  f_el=RotMat.T.dot(f_el)
                  G_el=RotMat.T.dot(G_el)
               #end if
           #end for
        #end if

        G_el*=eta_ref/Rinner
        h_el*=eta_ref/Rinner

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
    p=sol[NfemV:Nfem]*eta_ref/Rinner

    print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    #################################################################
    # Compute velocity in polar coordinates
    #################################################################

    vr = np.zeros(NV,dtype=np.float64)   
    vt = np.zeros(NV,dtype=np.float64)  

    vr= u*np.cos(tV)+v*np.sin(tV)
    vt=-u*np.sin(tV)+v*np.cos(tV)

    vrstats_file.write("%6e %6e %6e\n" % (time,np.min(vr),np.max(vr)))
    vtstats_file.write("%6e %6e %6e\n" % (time,np.min(vt),np.max(vt)))
    vrstats_file.flush()
    vtstats_file.flush()

    print("     -> vr (m,M) %.4e %.4e " %(np.min(vr),np.max(vr)))
    print("     -> vt (m,M) %.4e %.4e " %(np.min(vt),np.max(vt)))

    #################################################################
    # compute timestep value
    #################################################################

    dt1=CFL_nb*hr/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*hr**2 / kappa #(hcond/hcapa/rho0)

    dt_candidate=np.min([dt1,dt2])

    if istep==0:
       dt=1e-6
    else:
       dt=min(dt_candidate,2*dt)

    print('     -> dt1 = %.8f ' %(dt1))
    print('     -> dt2 = %.8f ' %(dt2))
    print('     -> dtc = %.8f ' %(dt_candidate))
    print('     -> dt  = %.8f ' %(dt))

    time+=dt

    print('     -> time= %.6f; tfinal= %.6f' %(time,tfinal))

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
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
                MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob #*hcapa*rho0

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*weightq*jcob*kappa #hcond

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B_mat))*weightq*jcob #*hcapa*rho0

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

    print("build FE matrix : %.3f s" % (timing.time() - start))

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

    vrms=np.sqrt(vrms/area)
    Tavrg/=area

    Tavrg_file.write("%10e %10e\n" % (time,Tavrg))
    Tavrg_file.flush()

    vrms_file.write("%10e %.10e\n" % (time,vrms))
    vrms_file.flush()

    print("     istep= %.6d ; vrms  = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = timing.time()
    
    exx_n = np.zeros(NV,dtype=np.float64)  
    eyy_n = np.zeros(NV,dtype=np.float64)  
    exy_n = np.zeros(NV,dtype=np.float64)  
    sr_n  = np.zeros(NV,dtype=np.float64)  
    eta_n = np.zeros(NV,dtype=np.float64)  
    rh_n  = np.zeros(NV,dtype=np.float64)  
    count = np.zeros(NV,dtype=np.int32)  
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
        if viscosity_model==1:
           eta_n[i]=eta(T[i])
        else:
           if rV[i]> 6371e3-200e3:
              eta_n[i]=1e24
           else:
              eta_n[i]=10**(f_cubic(rV[i]))


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
       filename = 'profile.ascii'.format(istep)
       vprofile=open(filename,"w")
       for i in range(0,NV):
           if abs(xV[i])<0.001:
              vprofile.write("%10e %10e %10e\n" % (yV[i],T[i],eta_n[i]))
       vprofile.close()

    print("compute profiles: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
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
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vr' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %(vr[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vt' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %(vt[i]))
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
       #vtufile.write("<DataArray type='Float32' Name='eta (S&C,2006)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%10e \n" %(f_cubic(rV[i])))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eta_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_1' Format='ascii'> \n")
       for i in range(0,NV):
           if flag_1[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_2' Format='ascii'> \n")
       for i in range(0,NV):
           if flag_2[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_3' Format='ascii'> \n")
       for i in range(0,NV):
           if flag_3[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_4' Format='ascii'> \n")
       for i in range(0,NV):
           if flag_4[i]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
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
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_1' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_1[iel]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_2' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_2[iel]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_3' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_3[iel]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='flag_4' Format='ascii'> \n")
       for iel in range(0,nel):
           if flag_el_4[iel]:
              vtufile.write("%10e \n" %1.)
           else:
              vtufile.write("%10e \n" %0.)
       vtufile.write("</DataArray>\n")
       #-
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[2,iel],iconV[8,iel],iconV[6,iel],\
                                                       iconV[1,iel],iconV[5,iel],iconV[7,iel],iconV[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       #filename = 'solution_HR_{:04d}.vtu'.format(istep)
       #vtufile=open(filename,"w")
       #vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       #vtufile.write("<UnstructuredGrid> \n")
       #vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #vtufile.write("<Points> \n")
       #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       #vtufile.write("</DataArray>\n")
       #vtufile.write("</Points> \n")
       #vtufile.write("<PointData Scalars='scalars'>\n")
       #vtufile.write("<DataArray type='Float32' Name='vr' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%10e \n" % vr[i])
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='vt' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%10e \n" % vt[i])
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%10e \n" %T[i])
       #vtufile.write("</DataArray>\n")
       #vtufile.write("</PointData>\n")
       #vtufile.write("<Cells>\n")
       #vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       #for iel in range (0,nel2):
       #    vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       #for iel in range (0,nel2):
       #    vtufile.write("%d \n" %((iel+1)*4))
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       #for iel in range (0,nel2):
       #    vtufile.write("%d \n" %9)
       #vtufile.write("</DataArray>\n")
       #vtufile.write("</Cells>\n")
       #vtufile.write("</Piece>\n")
       #vtufile.write("</UnstructuredGrid>\n")
       #vtufile.write("</VTKFile>\n")
       #vtufile.close()


       # create fields for P1 mesh output
       u_P1=np.zeros(NV_P1,dtype=np.float64) 
       v_P1=np.zeros(NV_P1,dtype=np.float64) 
       T_P1=np.zeros(NV_P1,dtype=np.float64) 
       q_P1=np.zeros(NV_P1,dtype=np.float64) 
       vr_P1=np.zeros(NV_P1,dtype=np.float64)
       vt_P1=np.zeros(NV_P1,dtype=np.float64)
       T_P1[0:NV]=T[0:NV]
       u_P1[0:NV]=u[0:NV]
       v_P1[0:NV]=v[0:NV]
       q_P1[0:NV]=q[0:NV]
       vr_P1[0:NV]=vr[0:NV]
       vt_P1[0:NV]=vt[0:NV]
       for iel in range(0,nel2):
           T_P1[NV+iel]=0.25*np.sum(T[iconQ1[0:4,iel]])
           q_P1[NV+iel]=0.25*np.sum(q[iconQ1[0:4,iel]])
           vr_P1[NV+iel]=0.25*np.sum(vr[iconQ1[0:4,iel]])
           vt_P1[NV+iel]=0.25*np.sum(vt[iconQ1[0:4,iel]])
           u_P1[NV+iel]=0.25*np.sum(u[iconQ1[0:4,iel]])
           v_P1[NV+iel]=0.25*np.sum(v[iconQ1[0:4,iel]])

       filename = 'solution_P1_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV_P1,nel_P1))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10e %10e %10e \n" %(xV_P1[i],yV_P1[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10f \n" %T_P1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10e %10e %10e \n" %(u_P1[i],v_P1[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vr' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10f \n" %vr_P1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='vt' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10f \n" %vt_P1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10f \n" %eta(T_P1[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,NV_P1):
           vtufile.write("%10e \n" %q_P1[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel_P1):
           vtufile.write("%d %d %d \n" %(iconP1[0,iel],iconP1[1,iel],iconP1[2,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel_P1):
           vtufile.write("%d \n" %((iel+1)*3))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel_P1):
           vtufile.write("%d \n" %5)
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
