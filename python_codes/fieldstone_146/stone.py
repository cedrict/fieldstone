import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as timing
from scipy.sparse import lil_matrix
from numpy import linalg as LA
import numba
import matplotlib.pyplot as plt
from scipy import special
import os
import matplotlib.tri as tri
import scipy.optimize as optimize

###############################################################################

TKelvin=273.15
year=3600*24*365.
eps=1.e-10
MPa=1e6

###############################################################################
# Glen flow law function
###############################################################################

@numba.njit
def compute_eta_glen(sr,T):
    n=3
    if T<263:
          A=3.61e5
          Q=60e3
    else:
          A=1.73e21
          Q=139e3
    #end if
    return 0.5e6*A**(-1./n)*sr**(1./n-1)*np.exp(Q/n/8.314/T)

###############################################################################
# Dislocation creep functions
###############################################################################

@numba.njit
def coeffs_disl(T):
    n=4
    if T<262:
       A=5e5 *MPa**(-n)
       Q=64.e3
    else:
       A=6.96e23 *MPa**(-n)
       Q=155.e3
    return n,A,Q

@numba.njit
def compute_eta_disl(sr,T):
    n,A,Q=coeffs_disl(T)
    return 0.5*A**(-1./n)*sr**(1./n-1)*np.exp(Q/n/8.314/T)

@numba.njit
def compute_sr_disl(T,sig):
    n,A,Q=coeffs_disl(T)
    return A*np.exp(-Q/8.314/T)*sig**n  

@numba.njit
def compute_sigma_disl(sr,T):
    n,A,Q=coeffs_disl(T)
    return (sr/A)**(1/n) * np.exp(Q/n/8.314/T) 

###############################################################################
# GBS functions
###############################################################################

@numba.njit
def coeffs_gbs(T):
    n=1.8
    p=1.4
    if T<262:
       A=1.1e2 *MPa**(-n)
       Q=70.e3
    else:
       A=8.5e37*MPa**(-n)
       Q=250.e3
    return n,A,Q,p

@numba.njit
def compute_eta_gbs(sr,T,d):    
    n,A,Q,p=coeffs_gbs(T)
    return 0.5*A**(-1./n)*d**(p/n) *sr**(1./n-1)*np.exp(Q/n/8.314/T)

@numba.njit
def compute_sr_gbs(T,sig,d):
    n,A,Q,p=coeffs_gbs(T)
    return A*d**(-p)*np.exp(-Q/8.314/T)*sig**n

@numba.njit
def compute_sigma_gbs(sr,T,d):
    n,A,Q,p=coeffs_gbs(T)
    return (sr/A)**(1/n)*d**(p/n)* np.exp(Q/n/8.314/T) 

###############################################################################
# definition of newton raphson function
# f(x)=sr - (sr_dis+sr_gbs) with x= shear stress
###############################################################################

def fff(x,sr,gs,T):
    sr_disl=compute_sr_disl(T,x)
    sr_gbs=compute_sr_gbs(T,x,gs)
    return sr -sr_disl-sr_gbs

###############################################################################

def viscosity(exx,eyy,exy,iter,T,d,rheology):
    if iter==0:
       e2=1e-10
    else:
       e2=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)

    #--------------------------------------------
    if rheology==0: #linear viscous 
       val=3.5e14

    #--------------------------------------------
    elif rheology==1: # glen's flow law
       val=compute_eta_glen(e2,T)

    #--------------------------------------------
    elif rheology==2: # dislocation (indep of d)
       val=compute_eta_disl(e2,T)

    #--------------------------------------------
    elif rheology==3: #GBS-limited creep
       val=compute_eta_gbs(e2,T,d)

    #--------------------------------------------
    elif rheology==4: # cheap composite
       val=1/(1/compute_eta_gbs(e2,T,d)+1/compute_eta_disl(e2,T))

    #--------------------------------------------
    else:

       # Assume all strainrate is produced by each mechanism and calculate stress
       # and select minimum of stresses as best guess for the optimization procedure 
       sig=min(compute_sigma_disl(e2,T),compute_sigma_gbs(e2,T,d))

       tau_NR = optimize.newton(fff,sig,args=(e2,d,T),tol=1e-3,maxiter=10,fprime=None,fprime2=None)
        
       sr_disl=compute_sr_disl(T,tau_NR)
       sr_gbs=compute_sr_gbs(T,tau_NR,d)

       val=1/(1/compute_eta_gbs(sr_gbs,T,d)+1/compute_eta_disl(sr_disl,T))

    #viscosity cutoffs
    val=min(5.e17,val)
    val=max(1.e11,val)

    return val

###############################################################################

@numba.njit
def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,\
                     NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

@numba.njit
def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,\
                     dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

@numba.njit
def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,\
                     dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

@numba.njit
def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################
# This function is used to prescribe a velocity profile on the 
# left boundary of the domain (ie at NEEM core)

def NEEM_velocity(y,H,n,ubc):
    vx=-ubc*(1-(1-y/H)**(n+1))
    vy=0
    return vx,vy

###############################################################################



# deformation map




###############################################################################

print("------------------------------")
print("---------- stone 146 ---------")
print("------------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

aquarium=False
Neumann=True

if aquarium:
   nelx=25
   nely=25
   Lx=3000 
   Ly=3000
else:
   nelx=500 
   nely=25
   Lx=382600 
   Ly=3000

nnx=2*nelx+1                  # number of elements, x direction
nny=2*nely+1                  # number of elements, y direction
NV=nnx*nny                    # number of nodes
nel=nelx*nely                 # number of elements, total
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
hx=Lx/nelx
hy=Ly/nely

#--------------------------------------
#rheology=0: lin viscous
#rheology=1: glen
#rheology=2: disl
#rheology=3: gbs
#rheology=4: disl+gbs (cheap)
#rheology=5: disl+gbs (right thing :)

rheology=5

rho=917
gy=-9.8

u_surf = 5.5/year # m/yr Based on velocity map from C3S

tol_nl=1e-5

eta_ref=1.e13      # scaling of G blocks
scaling_coeff=eta_ref/Ly

niter_min=1
niter=100

###############################################################################

nqel=9
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("Nfem=",Nfem)
print("hx",hx)
print("hy",hy)
print("----------------------------------------")

vel_stats_file=open('velocity_statistics.ascii',"w")
press_stats_file=open('pressure_statistics.ascii',"w")

###############################################################################
# grid point setup
###############################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

bottom=np.zeros(NV,dtype=np.bool) 
left=np.zeros(NV,dtype=np.bool)   
right=np.zeros(NV,dtype=np.bool)  
top=np.zeros(NV,dtype=np.bool)    

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        bottom[counter]=(j==0)
        left[counter]=(i==0)
        right[counter]=(i==nnx-1)
        top[counter]=(j==nny-1)
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

###############################################################################
# define coordinates of A,B,C,D,E,F,G,...
# Load in bed and surface data saved in same directory as script
###############################################################################

if not aquarium:

   __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
   f = open(os.path.join(__location__, "data/Interpolation_BedMachine_Bed+Surface.txt"))
   bed_and_surface = np.genfromtxt(f)

   # Some values taken from same script that produces BedMachine interpolation
   x_CC = 138300.0
   i_CC = 2766

   # the origin is at sea level, below point F !!
   # adapted to fit data

   xA=bed_and_surface[0,0] # 0
   yA=bed_and_surface[0,2] # 1014.03

   xB=x_CC
   yB=bed_and_surface[i_CC,2] # 1878.63

   xC=bed_and_surface[-1,0] # 382600
   yC=bed_and_surface[-1,2] # 2450.3

   xD=xC 
   yD=bed_and_surface[-1,1] # -68.6301

   xE=xB
   yE=bed_and_surface[i_CC,1] # 527.665

   xF=xA 
   yF=bed_and_surface[0,1] # 625.373


###############################################################################
# create new mesh
# the current assumption is that the length of the initial 
# domain is the same as the new domain.
# we use a bilinear mapping to map the blocks left and right of CC
###############################################################################
start = timing.time()

if not aquarium:

   xVnew=np.zeros(NV,dtype=np.float64)  # x coordinates
   yVnew=np.zeros(NV,dtype=np.float64)  # y coordinates

   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           #inside FEBA block
           if xV[counter]<=xB: 
              r=2*(xV[counter]-xF)/abs(xF-xE)-1
              s=2*(yV[counter])/Ly-1
              xVnew[counter]=0.25*(1-r)*(1-s)*xF+\
                             0.25*(1+r)*(1-s)*xE+\
                             0.25*(1+r)*(1+s)*xB+\
                             0.25*(1-r)*(1+s)*xA
              yVnew[counter]=0.25*(1-r)*(1-s)*yF+\
                             0.25*(1+r)*(1-s)*yE+\
                             0.25*(1+r)*(1+s)*yB+\
                             0.25*(1-r)*(1+s)*yA
           else: #inside EDCB
              r=2*(xV[counter]-xE)/abs(xD-xE)-1
              s=2*(yV[counter])/Ly-1
              xVnew[counter]=0.25*(1-r)*(1-s)*xE+\
                             0.25*(1+r)*(1-s)*xD+\
                             0.25*(1+r)*(1+s)*xC+\
                             0.25*(1-r)*(1+s)*xB
              yVnew[counter]=0.25*(1-r)*(1-s)*yE+\
                             0.25*(1+r)*(1-s)*yD+\
                             0.25*(1+r)*(1+s)*yC+\
                             0.25*(1-r)*(1+s)*yB
           #end if
           counter+=1
       #end for
   #end for

   #np.savetxt('gridnew.ascii',np.array([xVnew,yVnew]).T,header='# x,y')

   xV[:]=xVnew[:]
   yV[:]=yVnew[:]

print("setup: map mesh blocks: %.3f s" % (timing.time() - start))

#################################################################
# prescribe topography of surface and bed rock
#################################################################
start = timing.time()

if not aquarium:

   xGrid = xV[:nnx] # Array of x coordinates, as these are not shifting

   # Interpolate bed and surface from already interpolated data, OK because original data had horizontal resolution of approx 1 km
   bed_interp = np.interp(xGrid,bed_and_surface[:,0],bed_and_surface[:,1])
   surface_interp = np.interp(xGrid,bed_and_surface[:,0],bed_and_surface[:,2])

   # Set bottom and top to values from interpolator
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           if bottom[counter]:
               yV[counter] = bed_interp[i]
           if top[counter]:
               yV[counter] = surface_interp[i]
           counter += 1

print("setup: surface and bedrock topo: %.3f s" % (timing.time() - start))

#################################################################
# further internally deform the mesh according to layers
#################################################################

if not aquarium:

   # Make vertical spacing linear between the new bottom and top values
   for i in range(0,nnx):
           ytop = yV[top][i]
           ybottom =  yV[bottom][i]
           
           # Index with step nnx to grab a new y value from same column
           yV[i::nnx] = np.linspace(ybottom,ytop,nny)

print("setup: surface and bedrock topo + internal deformation: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (timing.time() - start))

###############################################################################
# recenter mid-edge nodes
###############################################################################

for iel in range(0,nel):
    xV[iconV[4,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
    yV[iconV[4,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[1,iel]])
    xV[iconV[5,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
    yV[iconV[5,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]])
    xV[iconV[6,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[3,iel]])
    yV[iconV[6,iel]]=0.5*(yV[iconV[2,iel]]+yV[iconV[3,iel]])
    xV[iconV[7,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[3,iel]])
    yV[iconV[7,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[3,iel]])
    xV[iconV[8,iel]]=0.5*(xV[iconV[4,iel]]+xV[iconV[6,iel]])
    yV[iconV[8,iel]]=0.5*(yV[iconV[4,iel]]+yV[iconV[6,iel]])

###############################################################################
# compute depth and thickness
###############################################################################
start = timing.time()

depth=np.zeros(NV,dtype=np.float64)  
thickness = np.zeros(nnx,dtype=np.float64)  

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        ytop=yV[top][i]
        depth[counter]=ytop-yV[counter]
        counter += 1

for i in range(0,nnx):
    ytop=yV[top][i]
    ybottom=yV[bottom][i]
    thickness[i] = ytop - ybottom

#np.savetxt('thickness.ascii',np.array([xV[:nnx],thickness]).T,header='# x,y')

print("setup: compute depth & thickness: %.3f s" % (timing.time() - start))

###############################################################################
# compute lithostatic pressure 
###############################################################################

p_lithostatic = np.zeros(NV,dtype=np.float64)  

p_lithostatic[:] = depth[:] * rho * abs(gy)

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)    # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value

if not aquarium:

   thickness_at_NEEM = thickness[-1]
   n_Glen = 1

   for i in range(0,NV):

       if left[i]: # free slip
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 # u=0

       if right[i]: # prescribed flow
          ui,vi=NEEM_velocity(yV[i],thickness_at_NEEM,n_Glen,u_surf)
          #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ui # u
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi # v

       if bottom[i]: #no slip
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 # u=0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # v=0

else:

   for i in range(0,NV):
       if left[i]: # free slip
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 # u=0
       if right[i] and not Neumann: # free slip
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 # u=0
       if bottom[i]: #free slip
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # v=0

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# prescribe temperature in the ice sheet 
###############################################################################
start = timing.time()
   
T = np.zeros(NV,dtype=np.float64)

if not aquarium:

   # Constants (source: Yardim (2022))
   K = 45      # m^2/yr, thermal diffusivity
   kappa = 2.7 # W/m*K, thermal conductivity

   #  Measurements at CC (source: Yardim (2022))
   G_CC = 50e-3 # W/m^2, geothermal flux
   M_CC = 24e-2 # m/yr, accumulation rate
   T_e_CC = 248.5 # K, surface temperature

   # Measurements at NEEM (source: Yardim (2022))
   G_NEEM = 88e-3 # W/m^2, geothermal flux
   M_NEEM = 26.5e-2 # m/yr, accumulation rate
   T_e_NEEM = 243.8 # K, surface temperature

   # Robin model for temperature (1977), only works on divide
   def calc_T(z,T_e,G,M,H):
       z_flip = H - z
       q = (M/(2*K*H))**0.5
       factor = G*np.sqrt(np.pi)/(2*kappa*q)
       errorfunctionterms = special.erf(z_flip*q) - special.erf(H*q)
       return T_e - factor*errorfunctionterms

   # Offset to place origin under F
   offset = x_CC

   # Linear inter- and extrapolation function from the only two datapoints we have on G and M
   def linearGfunction(x):
       return (x-offset)*(G_NEEM-G_CC)/(xC-xB)+G_CC

   def linearMfunction(x):
       return (x-offset)*(M_NEEM-M_CC)/(xC-xB)+M_CC

   # Surface temperature values extracted manually from figure 4 of Yardim (2022)
   # x in km from CC, y in K
   # Opened correctly when file is in same directory as script
   f = open(os.path.join(__location__, "data/Surf_temp_extracted.txt"))
   surface_temp_rounded = np.genfromtxt(f)

   # Rescaling to meter, adding offset
   surface_temp_rounded[:,0] = surface_temp_rounded[:,0]*1000 + offset

   # 1 additional data point from Thule airbase
   x_Thule = -65e3 # position of Thule, negative because outside of our grid
   T_Thule = 261.8 # K

   surface_temperature = np.zeros((len(surface_temp_rounded)+1,2))
   surface_temperature[0,:] = [x_Thule,T_Thule]
   surface_temperature[1:,:] = surface_temp_rounded

   # using linear interpolation for all surface temperatures
   surface_temp_interp = np.interp(xGrid,surface_temperature[:,0],surface_temperature[:,1])

   # extra plot for insight into inputs of Robin profile
   #fig, axs = plt.subplots(2)
   #axs[0].plot(xGrid,linearGfunction(xGrid))
   #axs[0].plot(xGrid,linearMfunction(xGrid))
   #axs[1].plot(xGrid,surface_temp_interp)

   # Calculate T for every point

   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           T[counter]=calc_T(depth[counter],\
                             surface_temp_interp[i],\
                             linearGfunction(xGrid[i]),\
                             linearMfunction(xGrid[i]),\
                             thickness[i])
           counter += 1

print("setup: initial temperature: %.3f s" % (timing.time() - start))

###############################################################################
# prescribe grain size in the ice sheet 
# Create boundary layer wrt bottom. 
# At CC delta=0 and at NEEM delta = bound_height_NEEM. 
# Thickness of layer increases linearly.
###############################################################################
start = timing.time()
   
d=np.zeros(NV,dtype=np.float64)  

if not aquarium:

   large_grain_boundary = np.zeros(nnx,dtype=np.float64)

   bound_height_NEEM = 300

   for i in range(0,nnx):
       delta = (xGrid[i] - x_CC)*bound_height_NEEM/(xGrid[-1] - x_CC)
       large_grain_boundary[i] = thickness[i] - delta

   # Define all grainsizes with delta and layer as thick as ice at CC, parallel to surface

   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           if depth[counter] < yB - yE: # Thickness at CC
               d[counter] = 0.005
           elif depth[counter] > large_grain_boundary[i]:
               d[counter] = 0.03
           else:
               d[counter] = 0.002
           counter += 1

print("setup: grain size: %.3f s" % (timing.time() - start))

###############################################################################
# Exploratory plots
###############################################################################

if False:

   # Set up triangulation for tricontourf
   triang = tri.Triangulation(xV, yV)

   # Extract triangle location for applying mask, because of concave shape
   xtri = xV[triang.triangles].mean(axis=1)
   ytri = yV[triang.triangles].mean(axis=1)

   # Use interpolation of bottom to see if triangles are outside domain
   mask = ytri < np.interp(xtri, xV[bottom], yV[bottom])
   triang.set_mask(mask)

   # Plot different values
   fig, ax = plt.subplots()
   ax.scatter(xV,yV)

   fig, ax = plt.subplots()
   im = ax.tricontourf(triang,T,levels=100)
   plt.colorbar(im)

   # Note that the thin layer between the two bottom layer is just a plotting artifact
   fig, ax = plt.subplots()
   im = ax.tricontourf(triang,d,levels=[0.001,0.003,0.02,0.04])
   plt.colorbar(im)

   fig, ax = plt.subplots()
   im = ax.tricontourf(triang,depth)
   plt.colorbar(im)
   #fig.show()  fix??

   #raise Exception('my personal exit message')

###############################################################################
###############################################################################
# non-linear iterations
###############################################################################
###############################################################################

method=1
if method==1:
   c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
elif method==2:
   c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
p       = np.zeros(NfemP,dtype=np.float64)        # pressure field 
Res     = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol     = np.zeros(Nfem,dtype=np.float64)         # solution vector 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
rhs     = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
conv_inf= np.zeros(niter,dtype=np.float64)        
conv_two= np.zeros(niter,dtype=np.float64)        
conv_inf_Ru = np.zeros(niter,dtype=np.float64)        
conv_inf_Rv = np.zeros(niter,dtype=np.float64)        
conv_inf_Rp = np.zeros(niter,dtype=np.float64)        

for iter in range(0,niter): #nonlinear iteration loop

   print("------------------------------------")
   print("iter=", iter)
   print("------------------------------------")

   #################################################################
   # build FE matrix
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################

   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
   f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
   h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
   xq    = np.zeros(nqel*nel,dtype=np.float64)      # x coords of q points 
   yq    = np.zeros(nqel*nel,dtype=np.float64)      # y coords of q points 
   etaq  = np.zeros(nqel*nel,dtype=np.float64)      

   counterq=0
   for iel in range(0,nel):

       # set arrays to 0 every loop
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)

       for jq in [0,1,2]:
           for iq in [0,1,2]:

               # position & weight of quad. point
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               NNNV[0:9]=NNV(rq,sq)
               dNNNVdr[0:9]=dNNVdr(rq,sq)
               dNNNVds[0:9]=dNNVds(rq,sq)
               NNNP[0:4]=NNP(rq,sq)

               # calculate jacobian matrix
               jcb=np.zeros((ndim,ndim),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb)

               # compute dNdx & dNdy & strainrate
               exxq=0.0
               eyyq=0.0
               exyq=0.0
               Tq=0.0
               for k in range(0,mV):
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                   exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                   eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                   exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
                   Tq+=NNNV[k]*T[iconV[k,iel]]
                
               dq=0.0
               for k in range(0,mP):
                   dq+=NNNP[k]*d[iconV[k,iel]]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute effective plastic viscosity
               etaq[counterq]=viscosity(exxq,eyyq,exyq,iter,Tq,dq,rheology)

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob

               counterq+=1
           # end for iq 
       # end for jq 

       #Neumann bc on right side
       if Neumann and right[iconV[5,iel]]:
          p_right=-p_lithostatic[iconV[5,iel]] 
          hhy=yV[iconV[2,iel]]-yV[iconV[1,iel]]
          f_el[ 2]+=p_right*hhy/6.
          f_el[ 4]+=p_right*hhy/6.
          f_el[10]+=p_right*hhy/6.*4.

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
                  #end for jkk
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
               # end if 
           # end for i1 
       #end for k1 

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
                       # end if
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*scaling_coeff
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
                   #end if
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

   # end for iel 

   print("     -> etaq (m,M) %.4e %.4e " %(np.min(etaq),np.max(etaq)))

   print("build FE matrix: %.3f s" % (timing.time() - start))

   ######################################################################
   # assemble K, G, GT, f, h into A and rhs
   ######################################################################
   start = timing.time()

   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   sparse_matrix=A_sparse.tocsr()
   Res=sparse_matrix.dot(sol)-rhs
   sol=sps.linalg.spsolve(sparse_matrix,rhs)

   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e (m/s)" %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e (m/s)" %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e (Pa)" %(np.min(p),np.max(p)))
   print("     -> u (m,M) %.4e %.4e (m/year)" %(np.min(u*year),np.max(u*year)))
   print("     -> v (m,M) %.4e %.4e (m/year)" %(np.min(v*year),np.max(v*year)))

   vel_stats_file.write("%d %10e %10e %10e %10e \n" % (iter,np.min(u*year),np.max(u*year),\
                                                            np.min(v*year),np.max(v*year)))
   vel_stats_file.flush()
   press_stats_file.write("%d %10e %10e \n" % (iter,np.min(p),np.max(p)))
   press_stats_file.flush()

   print("solve system: %.3f s - Nfem %d" % (timing.time() - start, Nfem))

   ############################################################################
   # compute non-linear residual
   ############################################################################
   start = timing.time()

   if iter==0:
      Res0_inf=LA.norm(Res,np.inf)
      Res0_two=LA.norm(Res,2)

   Res_inf=LA.norm(Res,np.inf)
   Res_two=LA.norm(Res,2)

   print("Nonlinear residual (inf. norm) %.7e" % (Res_inf/Res0_inf))
   print("Nonlinear residual (two  norm) %.7e" % (Res_two/Res0_two))

   conv_inf[iter]=Res_inf/Res0_inf
   conv_two[iter]=Res_two/Res0_two

   np.savetxt('nonlinear_conv_inf.ascii',np.array(conv_inf[0:niter]).T)
   np.savetxt('nonlinear_conv_two.ascii',np.array(conv_two[0:niter]).T)

   Res_u,Res_v=np.reshape(Res[0:NfemV],(NV,2)).T
   Res_p=Res[NfemV:Nfem]
   
   conv_inf_Ru[iter]=LA.norm(Res_u,np.inf)
   conv_inf_Rv[iter]=LA.norm(Res_v,np.inf)
   conv_inf_Rp[iter]=LA.norm(Res_p,np.inf)

   np.savetxt('nonlinear_conv_inf_Ru.ascii',np.array(conv_inf_Ru[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rv.ascii',np.array(conv_inf_Rv[0:niter]).T)
   np.savetxt('nonlinear_conv_inf_Rp.ascii',np.array(conv_inf_Rp[0:niter]).T)

   #np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
   #np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x,y,u,v]).T,header='# x,y,u,v')

   if Res_two/Res0_two<tol_nl and iter>niter_min:
      break

   print("computing res norms: %.3f s" % (timing.time() - start))

   ############################################################################
   # interpolate pressure onto velocity grid points
   ############################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)
   Res_q=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):
       q[iconV[0,iel]]=p[iconP[0,iel]]
       q[iconV[1,iel]]=p[iconP[1,iel]]
       q[iconV[2,iel]]=p[iconP[2,iel]]
       q[iconV[3,iel]]=p[iconP[3,iel]]
       q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
       q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
       q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
       q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
       q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+\
                        p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

   for iel in range(0,nel):
       Res_q[iconV[0,iel]]=Res_p[iconP[0,iel]]
       Res_q[iconV[1,iel]]=Res_p[iconP[1,iel]]
       Res_q[iconV[2,iel]]=Res_p[iconP[2,iel]]
       Res_q[iconV[3,iel]]=Res_p[iconP[3,iel]]
       Res_q[iconV[4,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]])*0.5
       Res_q[iconV[5,iel]]=(Res_p[iconP[1,iel]]+Res_p[iconP[2,iel]])*0.5
       Res_q[iconV[6,iel]]=(Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.5
       Res_q[iconV[7,iel]]=(Res_p[iconP[3,iel]]+Res_p[iconP[0,iel]])*0.5
       Res_q[iconV[8,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]]+\
                            Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.25

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (timing.time() - start))

   ############################################################################
   # generate vtu output at every nonlinear iteration
   ############################################################################
   start = timing.time()

   filename = 'solution_nl_{:04d}.vtu'.format(iter)
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
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='eta (middle)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (etaq[iel*nqel+4]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_u[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_v[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %Res_q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                   iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
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

   print("export to vtu: %.3f s" % (timing.time() - start))

###############################################################################
###############################################################################
# end of non-linear iterations
###############################################################################
###############################################################################
   
print("------------------------------------")
print("end of nonlinear iterations  ")
print("------------------------------------")

############################################################################
# compute strainrate 
############################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
Tc = np.zeros(nel,dtype=np.float64)  
dc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
sr  = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

       rq = 0.0
       sq = 0.0
       weightq = 2.0 * 2.0

       NNNV[0:9]=NNV(rq,sq)
       dNNNVdr[0:9]=dNNVdr(rq,sq)
       dNNNVds[0:9]=dNNVds(rq,sq)

       jcb=np.zeros((ndim,ndim),dtype=np.float64)
       for k in range(0,mV):
           jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
           jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
           jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
           jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
       jcbi=np.linalg.inv(jcb)

       for k in range(0,mV):
           dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
           dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

       for k in range(0,mV):
           xc[iel] += NNNV[k]*xV[iconV[k,iel]]
           yc[iel] += NNNV[k]*yV[iconV[k,iel]]
           Tc[iel] += NNNV[k]*T[iconV[k,iel]]
           dc[iel] += NNNV[k]*d[iconV[k,iel]]
           exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
           eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
           exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

       sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
print("     -> sr  (m,M) %.5e %.5e " %(np.min(sr),np.max(sr)))
print("     -> Tc  (m,M) %.5e %.5e " %(np.min(Tc),np.max(Tc)))
print("     -> dc  (m,M) %.5e %.5e " %(np.min(dc),np.max(dc)))

print("compute press & sr: %.3f s" % (timing.time() - start))

############################################################################
# project strainrate onto velocity grid
############################################################################
start = timing.time()

exxn=np.zeros(NV,dtype=np.float64)
eyyn=np.zeros(NV,dtype=np.float64)
exyn=np.zeros(NV,dtype=np.float64)
srn=np.zeros(NV,dtype=np.float64)
c=np.zeros(NV,dtype=np.float64)

rVnodes=[-1,+1,1,-1, 0,1,0,-1,0]
sVnodes=[-1,-1,1,+1,-1,0,1, 0,0]

for iel in range(0,nel):
    for i in range(0,mV):
        dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i])
        dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i])
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
        e_xx=0.
        e_yy=0.
        e_xy=0.
        for k in range(0,mV):
            e_xx += dNNNVdx[k]*u[iconV[k,iel]]
            e_yy += dNNNVdy[k]*v[iconV[k,iel]]
            e_xy += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNVdx[k]*v[iconV[k,iel]]
        exxn[iconV[i,iel]]+=e_xx
        eyyn[iconV[i,iel]]+=e_yy
        exyn[iconV[i,iel]]+=e_xy
        c[iconV[i,iel]]+=1.
    # end for i
# end for iel

exxn/=c
eyyn/=c
exyn/=c

srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

print("     -> exx (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
print("     -> exy (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
print("     -> sr  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

print("compute nod strain rate: %.3f s" % (timing.time() - start))

############################################################################
start = timing.time()

etan=np.zeros(NV,dtype=np.float64)

for i in range (0,NV):
    etan[i]=viscosity(exxn[i],eyyn[i],exyn[i],iter,T[i],d[i],rheology)

print("compute nodal viscosity: %.3f s" % (timing.time() - start))

#####################################################################

np.savetxt('surface_velocity.ascii',np.array([xV[top],u[top],v[top],np.sqrt(u[top]**2+v[top]**2)]).T)

#####################################################################

if False:
   core1file=open('core1.ascii',"w")
   core2file=open('core2.ascii',"w")
   for i in range(0,NV):
       if abs(xV[i]-Lx/2)/Lx<eps:
          core1file.write("%5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e\n"\
                       %(xV[i],yV[i],u[i],v[i],q[i],T[i],d[i],exxn[i],eyyn[i],exyn[i],srn[i],etan[i]))
       if abs(xV[i]-Lx/4)/Lx<eps:
          core2file.write("%5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e %5e\n"\
                       %(xV[i],yV[i],u[i],v[i],q[i],T[i],d[i],exxn[i],eyyn[i],exyn[i],srn[i],etan[i]))

   core1file.close()
   core2file.close()

#####################################################################
# plot of solution
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 
#####################################################################
start = timing.time()

filename = 'solution.vtu'
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
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xx (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xy (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate (middle)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (sr[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='grain size' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (dc[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='aspect ratio (hx/hy)' Format='ascii'> \n")
for iel in range (0,nel):
    hhx=xV[iconV[1,iel]]-xV[iconV[0,iel]]
    hhy=min(  yV[iconV[2,iel]]-yV[iconV[1,iel]] , yV[iconV[3,iel]]-yV[iconV[0,iel]] ) 
    vtufile.write("%10e\n" % (hhx/hhy))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='hx' Format='ascii'> \n")
for iel in range (0,nel):
    hhx=xV[iconV[1,iel]]-xV[iconV[0,iel]]
    vtufile.write("%10e\n" % hhx )
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='hy' Format='ascii'> \n")
for iel in range (0,nel):
    hhy=min(  yV[iconV[2,iel]]-yV[iconV[1,iel]] , yV[iconV[3,iel]]-yV[iconV[0,iel]] ) 
    vtufile.write("%10e\n" % hhy )
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],iter,Tc[iel],dc[iel],rheology)
    vtufile.write("%10e\n" %eta) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity (log)' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],iter,Tc[iel],dc[iel],rheology)
    vtufile.write("%10e\n" %(np.log10(eta))) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],iter,Tc[iel],dc[iel],rheology)
    vtufile.write("%10e\n" %(2.*eta*exx[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],iter,Tc[iel],dc[iel],rheology)
    vtufile.write("%10e\n" %(2.*eta*eyy[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
for iel in range (0,nel):
    eta= viscosity(exx[iel],eyy[iel],exy[iel],iter,Tc[iel],dc[iel],rheology)
    vtufile.write("%10e\n" %(2.*eta*exy[iel])) 
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='pressure (lithostatic)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %p_lithostatic[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='pressure (dynamic)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % (q[i]-p_lithostatic[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='depth' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %depth[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %etan[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_u[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_v[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exxn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exyn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %srn[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T (K)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(T[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='T (C)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(T[i]-TKelvin))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='d' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(d[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
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
   
print("export to vtu: %.3f s" % (timing.time() - start))

print("---------------------------------------")
print("-----------------the end---------------")
print("---------------------------------------")
