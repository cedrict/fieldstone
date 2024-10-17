import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as timing
from numpy import linalg as LA

#------------------------------------------------------------------------------

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
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

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
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

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
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------

def strpt(x,L,stretch_alpha,stretch_beta1,stretch_beta2):
    if x<L/2: 
       val = (2*x/L)**stretch_alpha * L/2
    else:
       val = L- (2*(L-x)/L)**stretch_alpha * L/2
    return val

#------------------------------------------------------------------------------

def viscosity_density(exx,eyy,exy,iiter,x,y,nnn):

    #varepsilon_reg=np.sqrt(exx**2+eyy**2+2*exy**2 + reg**2)
    varepsilon_reg=np.sqrt(0.5*(exx**2+eyy**2+2*exy**2) + reg**2)

    #-----[cavity]-----
    if experiment==1:
       beta=1
       alpha=1./nnn-1
       rho=0

    #-----[brick]-----
    if experiment==2:
       alpha=1./nnn-1
       B=40.e6/2./1e-15/1.e-15**alpha
       beta=B*2**(-alpha/2)
       rho=0

    #-----[slab detach (case 1a)]-----
    if experiment==3:
       if y>580e3 or  (y>Ly-(80e3+250e3) and abs(x-Lx/2)<40e3) : 
          #n=4 
          alpha=0.25-1
          beta=4.75e11
          rho=3300-3150
       else:
          #n=1 
          alpha=0
          beta=1.e21
          rho=3150-3150

    #-----[slab detach (case 1b)]-----
    if experiment==4:
       if y>580e3 or  (y>Ly-(80e3+250e3) and abs(x-Lx/2)<40e3) : 
          #n=4 
          alpha=0.25-1
          beta=4.75e11
          rho=3300-3150
       else:
          #n=3 
          alpha=1./3.-1
          beta=4.54e10
          rho=3150-3150

    # porphyroclast
    if experiment==5:
       if (x-Lx/2)**2+(y-Ly/2)**2<0.01:
          alpha=1./nnn-1
          beta=5
          rho=0
       else:
          #n=1
          alpha=0
          beta=1
          rho=0

    # Stokes sphere
    if experiment==6:
       alphaT=3e-5
       Q=540e3
       Rgas=8.314
       A=2.417e-16
       nnn=3.5
       if (x-Lx/2)**2+(y-Ly/2)**2<100e3**2:
          T=550+273
       else:
          T=(-1330+550)*(y)/Ly +1330 + 273
       alpha=1./nnn-1
       beta=0.5 * A**(-1./nnn) * np.exp ( Q / nnn / Rgas / T )
       rho=3300*(1-alphaT*(T-273))

    # Barr & houseman
    if experiment==7:
       alpha=1./nnn-1
       eps0=0.01/year/Ly
       beta=1e20*eps0**-alpha
       if x>Lx/2 and abs(y-Ly/2)<hy_min:
          beta=1e16
          alpha=0
       rho=0

    # nonlinear poiseuille flow
    if experiment==8:
       alpha=1/nnn-1
       C1=2**(nnn-1)
       beta=C1**(-1./nnn)*2**alpha
       #print (beta)
       rho=0

    # punch
    if experiment==9:
       alpha=1/nnn-1
       beta=1
       rho=0

    # spmw16
    if experiment==10:
       xC=58.1e3
       xD=61.9e3
       yC=9.4e3
       yD=9.4e3
       sr_lim=1.e-15
       tau_lim=100.e6
       eta_lim=tau_lim/2.e0/sr_lim
       if y<7.5e3 or \
          abs(x-60e3)<2.5e3 and y<9.4e3 or \
          abs(x-60e3)<1.9e3 and y<10.e3 or \
          (x-xC)**2+(y-yC)**2<0.6e3**2  or \
          (x-xD)**2+(y-yD)**2<0.6e3**2  : 
          #n=1 
          alpha = 0  
          beta=1.e21
       else:
          alpha=1/nnn-1
          beta=eta_lim/sr_lim**alpha
       rho=0

    # flow around cylinder
    if experiment==11:
       if x**2+y**2<0.5**2:
          alpha=0
          beta=1000
       else:
          alpha=1/nnn-1
          beta=1
       rho=0


    # compute effective power law viscosity
    eta=beta*varepsilon_reg**alpha

    #print(beta,eta,rho)

    return eta,rho,alpha

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cm=0.01
year=3600*24*365.
eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

experiment=1

# allowing for argument parsing through command line
#theta_method=1: theta=1
#theta_method=2: theta=1-R^k/R^Npicard
#theta_method=3: theta=1-sqrt(R^k/R^Npicard)

if int(len(sys.argv) == 5):
   nnn          = int(sys.argv[1])
   Npicard      = int(sys.argv[2])
   niter        = int(sys.argv[3])
   theta_method = int(sys.argv[4])
else:
   nnn=5
   Npicard=4
   niter=51
   theta_method=3

#...........................................
if experiment==1: # cavity
   Lx=1.
   Ly=1.
   nelx=32
   eta_ref=1. 
   reg=1e-6  
   gx=0
   gy=0
   grid_stretch=False

#...........................................
if experiment==2: # brick
   Lx=40e3
   Ly=10e3
   nelx=64
   eta_ref=1e22 
   reg=1e-20  
   gx=0
   gy=0
   grid_stretch=False

#...........................................
if experiment==3 or experiment==4: # slab detachment
   Lx=1000e3
   Ly=660e3
   nelx=100
   eta_ref=1e22 
   reg=1e-20  
   gx=0
   gy=-10
   grid_stretch=False

#...........................................
if experiment==5:
   Lx=1
   Ly=1
   nelx=80
   eta_ref=100
   reg=1e-5
   gx=0
   gy=0
   grid_stretch=False

#...........................................
if experiment==6: # Stokes sphere
   Lx=600e3
   Ly=600e3
   nelx=176
   eta_ref=1e21
   reg=1e-18
   gx=0
   gy=-9.81
   grid_stretch=False

#...........................................
if experiment==7: # def around fault
   R0=10e3
   Lx=2*R0
   Ly=R0
   nelx=102
   eta_ref=1e19
   reg=1e-16
   gx=0
   gy=0
   grid_stretch=True

#...........................................
if experiment==8: # poiseuille  
   Lx=2
   Ly=1
   nelx=80
   gx=0
   gy=0
   eta_ref=10
   reg=1e-8
   grid_stretch=False
   p_bc_left=1
   p_bc_right=-1

#...........................................
if experiment==9: # punch  
   Lx=1
   Ly=0.5
   nelx=80
   gx=0
   gy=0
   eta_ref=10
   reg=1e-6
   grid_stretch=False

#...........................................
if experiment==10: # spmw16
   Lx=120e3
   Ly=30e3
   nelx=100
   gx=0
   gy=0
   eta_ref=1e22
   reg=1e-17
   grid_stretch=False

#...........................................
if experiment==11: # flow around cylinder
   Lx=4
   Ly=8
   nelx=80
   gx=0
   gy=0
   eta_ref=1
   reg=1e-4
   grid_stretch=False

#...........................................

use_russians=False

tol_nl=1e-8 # nonlinear tolerance
every=1
produce_nl_vtu=False
use_srn=False

#################################################################

nely = int(nelx*Ly/Lx)        # number of elements y direction
nnx=2*nelx+1                  # number of nodes, x direction
nny=2*nely+1                  # number of nodes, y direction
NV=nnx*nny                    # total number of nodes
nel=nelx*nely                 # total number of elements
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
hx=Lx/nelx                    # mesh size in x direction
hy=Ly/nely                    # mesh size in y direction

#################################################################
# quadrature parameters

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nq=9*nel

#################################################################

scaling_coeff=eta_ref/Ly

#################################################################

ustats_file=open("stats_u.ascii","w")
vstats_file=open("stats_v.ascii","w")
pstats_file=open("stats_p.ascii","w")
dustats_file=open("stats_du.ascii","w")
dvstats_file=open("stats_dv.ascii","w")
dpstats_file=open("stats_dp.ascii","w")
theta_file=open("stats_theta.ascii","w")
srqstats_file=open("stats_srq.ascii","w")
etaqstats_file=open("stats_etaq.ascii","w")
convfile=open('conv.ascii',"w")
vrmsfile=open('vrms.ascii',"w")

#################################################################
#################################################################

print("experiment=",experiment)
print("nnn",nnn)
print("niter",niter)
print("Npicard",Npicard)
print("theta_method=",theta_method)
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
print("niter",niter)
print("------------------------------")

###################################################################################################
# grid point setup
###################################################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (timing.time() - start))

###################################################################################################
# build connectivity arrays for velocity and pressure
###################################################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###################################################################################################
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
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (timing.time() - start))

###################################################################################################
# stretching of mesh
###################################################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1

if grid_stretch:
   stretch_alpha_y=0.65
   stretch_beta1_y=0.25
   stretch_beta2_y=0.375
   for i in range(0,NV):
       xV[i]=strpt (xV[i],Lx,stretch_alpha_y,stretch_beta1_y,stretch_beta2_y)
       yV[i]=strpt (yV[i],Ly,stretch_alpha_y,stretch_beta1_y,stretch_beta2_y)

   # now there is a pb since nodes 4,5,6,7,8 are no more in the middle 
   # of the edges/element. We need to remedy this:

   for iel in range(0,nel):
       xV[iconV[4,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]])*0.5
       xV[iconV[5,iel]]=(xV[iconV[1,iel]]+xV[iconV[2,iel]])*0.5
       xV[iconV[6,iel]]=(xV[iconV[2,iel]]+xV[iconV[3,iel]])*0.5
       xV[iconV[7,iel]]=(xV[iconV[3,iel]]+xV[iconV[0,iel]])*0.5
       xV[iconV[8,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+\
                         xV[iconV[2,iel]]+xV[iconV[3,iel]])*0.25
       yV[iconV[4,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]])*0.5
       yV[iconV[5,iel]]=(yV[iconV[1,iel]]+yV[iconV[2,iel]])*0.5
       yV[iconV[6,iel]]=(yV[iconV[2,iel]]+yV[iconV[3,iel]])*0.5
       yV[iconV[7,iel]]=(yV[iconV[3,iel]]+yV[iconV[0,iel]])*0.5
       yV[iconV[8,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+\
                         yV[iconV[2,iel]]+yV[iconV[3,iel]])*0.25
       
   #np.savetxt('grid.ascii',np.array([xV,yV]).T)

hy_min=Ly
hy_max=0
for iel in range(0,nel):
    hy_min=min(hy_min,yV[iconV[2,iel]]-yV[iconV[0,iel]])
    hy_max=max(hy_max,yV[iconV[2,iel]]-yV[iconV[0,iel]])
          
print('     -> h_min=',hy_min)
print('     -> h_max=',hy_max)

###################################################################################################
# define boundary conditions
###################################################################################################
start = timing.time()

u     =np.zeros(NV,dtype=np.float64)    # x-component velocity
v     =np.zeros(NV,dtype=np.float64)    # y-component velocity
bc_fix=np.zeros(NfemV,dtype=bool)    # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value

if experiment==1: # cavity 
   for i in range(0,NV):
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = xV[i]*(Lx-xV[i]) 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==2: # brick
   velbc=1e-15*(Lx/2)
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -velbc
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +velbc
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly<eps:
          if xV[i]<Lx/2:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -velbc
          if xV[i]>Lx/2:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +velbc
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==3 or experiment==4: # slab detachment
   for i in range(0,NV):
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==5:
   for i in range(0,NV):
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +0.5
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -0.5
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==6:
   for i in range(0,NV):
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
    #end for
#end if

if experiment==7:
   for i in range(0,NV):
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -5e-3/year
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if yV[i]/Ly>1-eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 5e-3/year
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       #if abs(yV[i]-Ly/2)/Ly<eps:
       #   bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0

       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 5e-3/year*(yV[i]-Ly/2)/(Ly/2)
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          if yV[i]>Ly/2+0.9*hy_min:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 5e-3/year
          if yV[i]<Ly/2-0.9*hy_min:
             bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -5e-3/year
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==8:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly<eps: 
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end for
#end if

if experiment==9:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly<eps: 
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly>(1-eps) and abs(xV[i]-Lx/2)<0.0987654:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -1
    #end for
#end if

if experiment==10:
   velbc=0.0025/year
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = velbc
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = -velbc
       if yV[i]/Ly<eps: 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

if experiment==11:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0 
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0
       if yV[i]/Ly<eps: 
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          if xV[i]<0.5: 
             bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -1


print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###################################################################################################
# put bc velocity is solution vector
###################################################################################################

solution=np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b

solution[0:NfemV]=bc_val[0:NfemV]

###################################################################################################
# Neumann bc
###################################################################################################
Neumann_bc  = np.zeros(NfemV,dtype=np.float64)      

if experiment==8:

   for iel in range(0,nel):
       if abs(xV[iconV[7,iel]]/Lx)<eps: #element on left boundary
          if yV[iconV[0,iel]]>0:
             Neumann_bc[2*iconV[0,iel]]+=p_bc_left*hy/3/2
          if yV[iconV[3,iel]]<Ly:
             Neumann_bc[2*iconV[3,iel]]+=p_bc_left*hy/3/2
          Neumann_bc[2*iconV[7,iel]]+=p_bc_left*hy/3*4/2

       if abs(xV[iconV[5,iel]]/Lx-1)<eps: #element on right boundary
          if yV[iconV[1,iel]]>0:
             Neumann_bc[2*iconV[1,iel]]-=p_bc_right*hy/3/2
          if yV[iconV[2,iel]]<Ly:
             Neumann_bc[2*iconV[2,iel]]-=p_bc_right*hy/3/2
          Neumann_bc[2*iconV[5,iel]]-=p_bc_right*hy/3*4/2
   #end for

###################################################################################################
# non-linear iterations
###################################################################################################
J_mat = np.array([[1,0,0],[0,1,0],[0,0,0.5]],dtype=np.float64) 

dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
p       = np.zeros(NfemP,dtype=np.float64)        # pressure field 
Res     = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
solP    = np.zeros(NfemP,dtype=np.float64)        # P solution vector 
solV    = np.zeros(NfemV,dtype=np.float64)        # V solution vector
exxn    = np.zeros(NV,dtype=np.float64)           # nodal exx
eyyn    = np.zeros(NV,dtype=np.float64)           # nodal eyy
exyn    = np.zeros(NV,dtype=np.float64)           # nodal exy

Rnorm=0
Rnorm0=1

for iiter in range(0,niter):

   print("--------------------------")
   print("iter=", iiter)
   print("--------------------------")

   #################################################################
   # compute theta
   #################################################################

   if iiter<Npicard: 
      # use Picard iteration
      theta=0.
   else: 
      # use Newton iteration
      if theta_method==1:
         theta = 1
      if theta_method==2:
         theta = max(1-(Rnorm/RnormN),0)
      if theta_method==3:
         theta = max(1-np.sqrt(Rnorm/RnormN),0)

   print('     -> theta=',theta)

   theta_file.write("%3d %10e \n" %(iiter,theta)) 
   theta_file.flush()

   #################################################################
   # build FE matrix A and rhs 
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################

   A_sparse= lil_matrix((Nfem,Nfem),dtype=np.float64) # FEM stokes matrix 
   rhs     = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
   N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # N matrix  
   f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
   h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
   xq      = np.zeros(9*nel,dtype=np.float64)         # x coords of q points 
   yq      = np.zeros(9*nel,dtype=np.float64)         # y coords of q points 
   etaq    = np.zeros(9*nel,dtype=np.float64)         # viscosity of q points 
   rhoq    = np.zeros(9*nel,dtype=np.float64)         # density of q points 
   pq      = np.zeros(9*nel,dtype=np.float64)         # pressure of q points 
   srq     = np.zeros(9*nel,dtype=np.float64)         # total strain rate of q points 

   f_rhs+=Neumann_bc

   counter=0
   for iel in range(0,nel):

       # set arrays to 0 for each element 
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       K_el0 =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       K_el1 =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)

       V_el=np.zeros((mV*ndofV),dtype=np.float64)
       for i in range(0,mV):
           V_el[2*i+0]=solution[2*iconV[i,iel]+0]
           V_el[2*i+1]=solution[2*iconV[i,iel]+1]

       P_el=np.zeros((mP*ndofP),dtype=np.float64)
       for i in range(0,mP):
           P_el[i]=p[iconP[i,iel]]

       # integrate viscous term at 9 quadrature points
       for jq in [0,1,2]:
           for iq in [0,1,2]:

               # position & weight of quad. point
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               NNNP[0:mP]=NNP(rq,sq)

               # calculate jacobian matrix
               jcb=np.zeros((ndim,ndim),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb)

               #only valid for rectangular elements!
               #jcbi=np.zeros((ndim,ndim),dtype=np.float64)
               #jcob=hx*hy/4
               #jcbi[0,0] = 2/hx 
               #jcbi[1,1] = 2/hy

               # compute dNdx & dNdy & strainrate
               exxq=0.0
               eyyq=0.0
               exyq=0.0
               for k in range(0,mV):
                   xq[counter]+=NNNV[k]*xV[iconV[k,iel]]
                   yq[counter]+=NNNV[k]*yV[iconV[k,iel]]
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                   #dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                   #dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                   exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                   eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                   exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

               if use_srn:
                  exxq=0.0
                  eyyq=0.0
                  exyq=0.0
                  for k in range(0,mV):
                      exxq+=NNNV[k]*exxn[iconV[k,iel]]
                      eyyq+=NNNV[k]*eyyn[iconV[k,iel]]
                      exyq+=NNNV[k]*exyn[iconV[k,iel]]

               # effective strain rate at qpoint                
               ee=0.5*(exxq**2+eyyq**2+2*exyq**2) + reg**2
               srq[counter]=np.sqrt(ee)

               # compute pressure at qpoint - not needed yet
               #for k in range(0,mP):
               #    pq[counter]+=NNNP[k]*p[iconP[k,iel]]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute effective viscosity and density
               etaq[counter],rhoq[counter],alpha=viscosity_density(exxq,eyyq,exyq,iiter,\
                                                                   xq[counter],yq[counter],nnn)

               D_mat = np.array([[exxq*exxq,exxq*eyyq,exxq*exyq],\
                                 [eyyq*exxq,eyyq*eyyq,eyyq*exyq],\
                                 [exyq*exxq,exyq*eyyq,exyq*exyq]],dtype=np.float64) 

               coef=2*etaq[counter]*alpha/2/ee 
               # compute elemental a_mat matrix
               K_el0+=b_mat.T.dot(J_mat.dot(b_mat))*2*etaq[counter]*weightq*jcob
               K_el1+=b_mat.T.dot(D_mat.dot(b_mat))*coef*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*gx*rhoq[counter]
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq[counter]

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob

               counter+=1
           # end for iq 
       # end for jq 

       # finish building rhs which is -residual
       f_el-=K_el0.dot(V_el)+G_el.dot(P_el) # b - K.V - G.P 
       h_el-=G_el.T.dot(V_el)               # h - G^T.V 

       K_el=K_el0+theta*K_el1

       # impose b.c. 
       for k1 in range(0,mV):
           for i1 in range(0,ndofV):
               ikk=ndofV*k1          +i1
               m1 =ndofV*iconV[k1,iel]+i1
               if bc_fix[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,mV*ndofV):
                      f_el[jkk]-=K_el[jkk,ikk]*(bc_val[m1]-solution[m1])
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  #end for jkk
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*(bc_val[m1]-solution[m1])
                  h_el[:]-=G_el[ikk,:]*(bc_val[m1]-solution[m1])
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
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*scaling_coeff
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,mP):
           m1=iconP[k1,iel]
           h_rhs[m1]+=h_el[k1]*scaling_coeff
       #end for k1

   # end for iel 

   srqstats_file.write("%3d %e %e \n" %(iiter,np.min(srq),np.max(srq)))
   srqstats_file.flush()

   etaqstats_file.write("%3d %e %e \n" %(iiter,np.min(etaq),np.max(etaq)))
   etaqstats_file.flush()

   #np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
   #np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')

   print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
   print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

   print("     -> etaq (m,M) %.5e %.5e " %(np.min(etaq),np.max(etaq)))
   print("     -> rhoq (m,M) %.5e %.5e " %(np.min(rhoq),np.max(rhoq)))

   print("build FE matrix: %.3f s" % (timing.time() - start))

   ######################################################################
   # pressure nullspace removal
   # requires last pressure dof to be at p=0
   ######################################################################

   if experiment==1 or experiment==3 or \
      experiment==4 or experiment==5 : 
      for i in range(0,Nfem):
          A_sparse[Nfem-1,i]=0
          A_sparse[i,Nfem-1]=0
          A_sparse[Nfem-1,Nfem-1]=1
          h_rhs[NfemP-1]=0

   ######################################################################
   # build rhs ( = - residual) 
   ######################################################################
   start = timing.time()

   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs

   Rnorm=LA.norm(rhs,2) # residual 2-norm	
   RVnorm=LA.norm(rhs[0:NfemV],2) # residual 2-norm	
   RPnorm=LA.norm(rhs[NfemV:Nfem],2) # residual 2-norm	
   print('     -> Residual 2-norm=',Rnorm,RVnorm,RPnorm)

   if iiter==0:
      Rnorm0=Rnorm
      RVnorm0=RVnorm
      RPnorm0=RPnorm
   if iiter==Npicard-1:
      RnormN=Rnorm

   ######################################################################
   # solving system
   ######################################################################

   sparse_matrix=A_sparse.tocsr()
   sol=sps.linalg.spsolve(sparse_matrix,rhs)

   du,dv=np.reshape(sol[0:NfemV],(NV,2)).T
   dp=sol[NfemV:Nfem]*scaling_coeff

   dustats_file.write("%d %8e %8e \n" %(iiter,np.min(du),np.max(du)))
   dvstats_file.write("%d %8e %8e \n" %(iiter,np.min(dv),np.max(dv)))
   dpstats_file.write("%d %8e %8e \n" %(iiter,np.min(dp),np.max(dp)))
   dustats_file.flush()
   dvstats_file.flush()
   dpstats_file.flush()


   print("     -> du (m,M) %.4e %.4e " %(np.min(du),np.max(du)))
   print("     -> dv (m,M) %.4e %.4e " %(np.min(dv),np.max(dv)))
   print("     -> dp (m,M) %.4e %.4e " %(np.min(dp),np.max(dp)))

   ######################################################################
   # update solution
   ######################################################################

   solution[:]+=sol[:] 

   u,v=np.reshape(solution[0:NfemV],(NV,2)).T
   p=solution[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))

   ustats_file.write("%d %8e %8e \n" %(iiter,np.min(u),np.max(u)))
   vstats_file.write("%d %8e %8e \n" %(iiter,np.min(v),np.max(v)))
   ustats_file.flush()
   vstats_file.flush()

   ######################################################################
   # assess convergence
   ######################################################################

   dvel=sol[0:NfemV]
   vel=solution[0:NfemV]

   index_res=Rnorm/Rnorm0
   index_vel=LA.norm(dvel,2)/LA.norm(vel,2)
   index_p=LA.norm(dp,2)/LA.norm(p,2)

   convfile.write("%3d %10e %10e %10e %10e\n" %(iiter,index_res,index_vel,index_p,tol_nl)) 
   convfile.flush()

   converged=(index_res<tol_nl and index_vel<tol_nl and index_p<tol_nl)

   if index_res>2:
      exit()

   #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

   print("solve system: %.3f s - Nfem %d" % (timing.time() - start, Nfem))

   #################################################################
   ###insert russians
   #################################################################

   if use_russians and iiter>Npicard: 

      rhs     = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
      f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
      h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

      counter=0
      for iel in range(0,nel):

          # set arrays to 0 for each element 
          K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
          K_el0 =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
          K_el1 =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
          G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
          f_el =np.zeros((mV*ndofV),dtype=np.float64)
          h_el=np.zeros((mP*ndofP),dtype=np.float64)

          V_el=np.zeros((mV*ndofV),dtype=np.float64)
          for i in range(0,mV):
              V_el[2*i+0]=solution[2*iconV[i,iel]+0]
              V_el[2*i+1]=solution[2*iconV[i,iel]+1]

          P_el=np.zeros((mP*ndofP),dtype=np.float64)
          for i in range(0,mP):
              P_el[i]=p[iconP[i,iel]]

          # integrate viscous term at 9 quadrature points
          for jq in [0,1,2]:
              for iq in [0,1,2]:
   
                  # position & weight of quad. point
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]

                  NNNV[0:mV]=NNV(rq,sq)
                  dNNNVdr[0:mV]=dNNVdr(rq,sq)
                  dNNNVds[0:mV]=dNNVds(rq,sq)
                  NNNP[0:mP]=NNP(rq,sq)

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
                  for k in range(0,mV):
                      dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                      dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                      exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                      eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                      exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

                  if use_srn:
                     exxq=0.0
                     eyyq=0.0
                     exyq=0.0
                     for k in range(0,mV):
                         exxq+=NNNV[k]*exxn[iconV[k,iel]]
                         eyyq+=NNNV[k]*eyyn[iconV[k,iel]]
                         exyq+=NNNV[k]*exyn[iconV[k,iel]]

                  # effective strain rate at qpoint                
                  ee=0.5*(exxq**2+eyyq**2+2*exyq**2) + reg**2
                  srq[counter]=np.sqrt(ee)

                  # compute pressure at qpoint - not needed yet
                  #for k in range(0,mP):
                  #    pq[counter]+=NNNP[k]*p[iconP[k,iel]]

                  # construct 3x8 b_mat matrix
                  for i in range(0,mV):
                      b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                               [0.        ,dNNNVdy[i]],
                                               [dNNNVdy[i],dNNNVdx[i]]]

                  # compute effective viscosity and density
                  etaq[counter],rhoq[counter],alpha=viscosity_density(exxq,eyyq,exyq,iiter,\
                                                                      xq[counter],yq[counter],nnn)

                  D_mat = np.array([[exxq*exxq,exxq*eyyq,exxq*exyq],\
                                    [eyyq*exxq,eyyq*eyyq,eyyq*exyq],\
                                    [exyq*exxq,exyq*eyyq,exyq*exyq]],dtype=np.float64) 

                  coef=2*etaq[counter]*alpha/2/ee 
                  # compute elemental a_mat matrix
                  K_el0+=b_mat.T.dot(J_mat.dot(b_mat))*2*etaq[counter]*weightq*jcob
                  K_el1+=b_mat.T.dot(D_mat.dot(b_mat))*coef*weightq*jcob

                  # compute elemental rhs vector
                  for i in range(0,mV):
                      f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*gx*rhoq[counter]
                      f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rhoq[counter]

                  for i in range(0,mP):
                      N_mat[0,i]=NNNP[i]
                      N_mat[1,i]=NNNP[i]
                      N_mat[2,i]=0.

                  G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                  counter+=1
              # end for iq 
          # end for jq 

          # finish building rhs which is -residual
          f_el-=K_el0.dot(V_el)+G_el.dot(P_el) # b - K.V - G.P 
          h_el-=G_el.T.dot(V_el)               # h - G^T.V 

          K_el=K_el0+theta*K_el1

          # impose b.c. 
          for k1 in range(0,mV):
              for i1 in range(0,ndofV):
                  ikk=ndofV*k1          +i1
                  m1 =ndofV*iconV[k1,iel]+i1
                  if bc_fix[m1]:
                     K_ref=K_el[ikk,ikk] 
                     for jkk in range(0,mV*ndofV):
                         f_el[jkk]-=K_el[jkk,ikk]*(bc_val[m1]-solution[m1])
                         K_el[ikk,jkk]=0
                         K_el[jkk,ikk]=0
                     #end for jkk
                     K_el[ikk,ikk]=K_ref
                     f_el[ikk]=K_ref*(bc_val[m1]-solution[m1])
                     h_el[:]-=G_el[ikk,:]*(bc_val[m1]-solution[m1])
                     G_el[ikk,:]=0
                  # end if 
              # end for i1 
          #end for k1 

          # assemble right hand side rhs
          for k1 in range(0,mV):
              for i1 in range(0,ndofV):
                  ikk=ndofV*k1          +i1
                  m1 =ndofV*iconV[k1,iel]+i1
                  f_rhs[m1]+=f_el[ikk]
                  #end for k2
              #end for i1
          #end for k1 
          for k1 in range(0,mP):
              m1=iconP[k1,iel]
              h_rhs[m1]+=h_el[k1]*scaling_coeff
          #end for k1

      # end for iel 

      rhs[0:NfemV]=f_rhs
      rhs[NfemV:Nfem]=h_rhs

      print('f_rhs',f_rhs)
      print('h_rhs',f_rhs)
      #print(rhs)

      RnormR=LA.norm(rhs,2) # residual 2-norm	

      print('     -> ResidualR 2-norm=',RnormR)

      tau=max(Rnorm**2/(Rnorm**2+RnormR**2),0.5)

      print('     -> tau=',tau)

      solution[:]+=(tau-1)*sol[:]

      tau_file.write("%3d %10e \n" %(iiter,tau)) 

   #################################################################
   #normalise pressure (for experiments with p nullspace)
   #################################################################
   start = timing.time()

   if experiment==1 or experiment==3 or \
      experiment==4 or experiment==5 or experiment==7:

      int_p=0
      for iel in range(0,nel):
          for jq in [0,1,2]:
              for iq in [0,1,2]:
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]
                  NNNP[0:mP]=NNP(rq,sq)
                  jcob=hx*hy/4
                  p_q=NNNP[0:mP].dot(p[iconP[0:mP,iel]])
                  int_p+=p_q*weightq*jcob
              #end for
          #end for
      #end for

      avrg_p=int_p/Lx/Ly

      print("     -> int_p %e " %(int_p))
      print("     -> avrg_p %e " %(avrg_p))

      p[:]-=avrg_p

   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   pstats_file.write("%d %8e %8e \n" %(iiter,np.min(p),np.max(p)))
   pstats_file.flush()

   print("normalise pressure: %.3f s" % (timing.time() - start))

   #####################################################################
   # interpolate pressure onto velocity grid points (for plotting)
   #####################################################################
   # velocity    pressure
   # 3---6---2   3-------2
   # |       |   |       |
   # 7   8   5   |       |
   # |       |   |       |
   # 0---4---1   0-------1
   #################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)
   dq=np.zeros(NV,dtype=np.float64)

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
       dq[iconV[0,iel]]=dp[iconP[0,iel]]
       dq[iconV[1,iel]]=dp[iconP[1,iel]]
       dq[iconV[2,iel]]=dp[iconP[2,iel]]
       dq[iconV[3,iel]]=dp[iconP[3,iel]]
       dq[iconV[4,iel]]=(dp[iconP[0,iel]]+dp[iconP[1,iel]])*0.5
       dq[iconV[5,iel]]=(dp[iconP[1,iel]]+dp[iconP[2,iel]])*0.5
       dq[iconV[6,iel]]=(dp[iconP[2,iel]]+dp[iconP[3,iel]])*0.5
       dq[iconV[7,iel]]=(dp[iconP[3,iel]]+dp[iconP[0,iel]])*0.5
       dq[iconV[8,iel]]=(dp[iconP[0,iel]]+dp[iconP[1,iel]]+\
                         dp[iconP[2,iel]]+dp[iconP[3,iel]])*0.25

   #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute strainrate at center of element 
   ######################################################################
   start = timing.time()

   xc = np.zeros(nel,dtype=np.float64)  
   yc = np.zeros(nel,dtype=np.float64)  
   pc = np.zeros(nel,dtype=np.float64)  
   exxc = np.zeros(nel,dtype=np.float64)  
   eyyc = np.zeros(nel,dtype=np.float64)  
   exyc = np.zeros(nel,dtype=np.float64)  
   src  = np.zeros(nel,dtype=np.float64)  

   for iel in range(0,nel):

       rq = 0.
       sq = 0.

       NNNV[0:mV]=NNV(rq,sq)
       dNNNVdr[0:mV]=dNNVdr(rq,sq)
       dNNNVds[0:mV]=dNNVds(rq,sq)
       NNNP[0:mP]=NNP(rq,sq)

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
           exxc[iel] += dNNNVdx[k]*u[iconV[k,iel]]
           eyyc[iel] += dNNNVdy[k]*v[iconV[k,iel]]
           exyc[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

       src[iel]=np.sqrt(0.5*(exxc[iel]*exxc[iel]+eyyc[iel]*eyyc[iel])+exyc[iel]*exyc[iel])

       for k in range(0,mP):
           pc[iel] += NNNP[k]*p[iconP[k,iel]]

   #end if

   print("     -> exxc (m,M) %.5e %.5e " %(np.min(exxc),np.max(exxc)))
   print("     -> eyyc (m,M) %.5e %.5e " %(np.min(eyyc),np.max(eyyc)))
   print("     -> exyc (m,M) %.5e %.5e " %(np.min(exyc),np.max(exyc)))
   print("     -> src  (m,M) %.5e %.5e " %(np.min(src),np.max(src)))
   print("     -> pc   (m,M) %.5e %.5e " %(np.min(pc),np.max(pc)))

   print("compute press & sr: %.3f s" % (timing.time() - start))

   #####################################################################
   # compute strainrate on velocity grid
   #####################################################################
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
           NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i])
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

   print("     -> exxn (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
   print("     -> eyyn (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
   print("     -> exyn (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
   print("     -> srn  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

   print("compute nod strain rate: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute nodal viscosity
   ######################################################################
   start = timing.time()

   etan=np.zeros(NV,dtype=np.float64)
   rhon=np.zeros(NV,dtype=np.float64)

   for i in range(0,NV):
       etan[i],rhon[i],alpha=viscosity_density(exxn[i],eyyn[i],exyn[i],iiter,xV[i],yV[i],nnn)

   print("     -> etan (m,M) %.6e %.6e " %(np.min(etan),np.max(etan)))

   #np.savetxt('etan_{:04d}.ascii'.format(iter),np.array([xV,yV,etan]).T,header='# x,y,eta')

   print("compute nodal viscosity: %.3f s" % (timing.time() - start))


   ######################################################################
   # compute vrms
   ######################################################################
   start = timing.time()

   vrms=0.
   counterq=0
   for iel in range(0,nel):
       for iq in [0,1,2]:
           for jq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               jcob=hx*hy/4 #only for rect elements!
               uq=0.
               vq=0.
               for k in range(0,mV):
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
               #end for
               vrms+=(uq**2+vq**2)*weightq*jcob
               counterq+=1
           #end for
       #end for
   #end for

   vrms=np.sqrt(vrms/(Lx*Ly))

   vrmsfile.write("%3d %10e\n" %(iiter,vrms))
   vrmsfile.flush()

   print("     -> vrms= %.7e " %vrms)

   print("compute vrms: %.3f s" % (timing.time() - start))

   ######################################################################
   # generate vtu output at every nonlinear iteration
   ######################################################################
   start = timing.time()

   if iiter%every==0 and produce_nl_vtu:

      filename = 'solution_q_nl_{:04d}'.format(iiter)+'.vtu'
      vtufile=open(filename,"w")
      vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
      vtufile.write("<UnstructuredGrid> \n")
      vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
      #####
      vtufile.write("<Points> \n")
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e %10e %10e \n" %(xq[iq],yq[iq],0.))
      vtufile.write("</DataArray>\n")
      vtufile.write("</Points> \n")
      #####
      vtufile.write("<PointData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % etaq[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % rhoq[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain_rate (T)' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % srq[iq])
      vtufile.write("</DataArray>\n")
      #vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
      #for iq in range(0,nq):
      #    vtufile.write("%10e \n" % pq[iq])
      #vtufile.write("</DataArray>\n")
      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iq in range (0,nq):
          vtufile.write("%d\n" % iq ) 
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iq in range (0,nq):
          vtufile.write("%d \n" % (iq+1) )
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
      for iq in range (0,nq):
          vtufile.write("%d \n" % 1) 
      vtufile.write("</DataArray>\n")
      vtufile.write("</Cells>\n")
      #####
      vtufile.write("</Piece>\n")
      vtufile.write("</UnstructuredGrid>\n")
      vtufile.write("</VTKFile>\n")
      vtufile.close()

      filename = 'solution_g_nl_{:04d}.vtu'.format(iiter)
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

      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel. corr.' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e %10e %10e \n" %(du[i],dv[i],0.))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %q[i])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='dq' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %dq[i])
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%10e\n" % (etan[i]))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%10e\n" % (rhon[i]))
      vtufile.write("</DataArray>\n")

      vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%e\n" % (srn[i]))
      vtufile.write("</DataArray>\n")

      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                      iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*8))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %23)
      vtufile.write("</DataArray>\n")
      vtufile.write("</Cells>\n")
      #####
      vtufile.write("</Piece>\n")
      vtufile.write("</UnstructuredGrid>\n")
      vtufile.write("</VTKFile>\n")
      vtufile.close()

   print("write nl iter vtu file: %.3f s" % (timing.time() - start))


   if converged:
      print('##### converged #####')
      break

#------------------------------------------------------------------------------
# end of non-linear iterations
#------------------------------------------------------------------------------
   
######################################################################
# export measurements for slab detachment
######################################################################

if experiment==3 or experiment==4:
   vert_file=open('vertical_profile.ascii',"w")
   for i in range(0,NV):
       if abs(xV[i]-Lx/2)/Lx<eps:
          vert_file.write("%e %e %e %e %e %e\n" %(yV[i],etan[i],srn[i],u[i],v[i],q[i]))
   vert_file.close()
   hor_file=open('horizontal_profile.ascii',"w")
   for i in range(0,NV):
       if abs(yV[i]-550e3)/Lx<eps:
          hor_file.write("%e %e %e %e %e %e\n" %(xV[i],etan[i],srn[i],u[i],v[i],q[i]))
   hor_file.close()

######################################################################
# export measurements for inclusion 
######################################################################

if experiment==5:
   diag_file=open('diagonal_profile.ascii',"w")
   for i in range(0,NV):
       if abs(xV[i]-yV[i])<1e-6 and xV[i]>0.4999999:
          r_i=np.sqrt((xV[i]-0.5)**2+(yV[i]-0.5)**2)
          diag_file.write("%e %e %e %e %e %e\n" %(r_i,etan[i],srn[i],u[i],v[i],q[i]))
   diag_file.close()

######################################################################
# export measurements for Stokes sphere
######################################################################

if experiment==6:
   vert_file=open('vertical_profile.ascii',"w")
   for i in range(0,NV):
       if abs(xV[i]-Lx/2)/Lx<eps:
          vert_file.write("%e %e %e %e %e %e\n" %(yV[i],etan[i],srn[i],u[i],v[i],q[i]))
   vert_file.close()
   hor_file=open('horizontal_profile.ascii',"w")
   for i in range(0,NV):
       if abs(yV[i]-Ly/2)/Ly<eps:
          hor_file.write("%e %e %e %e %e %e\n" %(xV[i],etan[i],srn[i],u[i],v[i],q[i]))
   hor_file.close()

######################################################################
# export measurements for slab detachment
######################################################################

if experiment==8:
   vert_file=open('vertical_profile.ascii',"w")
   for i in range(0,NV):
       if abs(xV[i]-Lx/2)/Lx<eps:
          vert_file.write("%e %e %e %e\n" %(yV[i],etan[i],srn[i],u[i]))
   vert_file.close()

######################################################################
# export measurements for punch
######################################################################

if experiment==9:
   surf_file=open('surface_profile.ascii',"w")
   for i in range(0,NV):
       if abs(yV[i]-Ly)/Ly<eps:
          surf_file.write("%e %e %e %e %e %e\n" %(xV[i],etan[i],srn[i],u[i],v[i],q[i]))
   surf_file.close()

######################################################################
# export measurements for flow around cylinder
######################################################################

if experiment==11:
   bot_file=open('bottom_profile.ascii',"w")
   for i in range(0,NV):
       if abs(yV[i])<eps:
          bot_file.write("%e %e %e %e %e %e\n" %(xV[i],etan[i],srn[i],u[i],v[i],q[i]))
   bot_file.close()

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

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
#vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
#for iel in range (0,nel):
#    vtufile.write("%10e\n" % (exxc[iel]+eyyc[iel]))
#vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='(effective) viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    eta,dum,dum=viscosity_density(exxc[iel],eyyc[iel],exyc[iel],iiter,xc[iel],yc[iel],nnn)
    vtufile.write("%10e\n" %eta) 
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
#for i in range(0,NV):
#    vtufile.write("%10e \n" %Res_u[i])
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
#for i in range(0,NV):
#    vtufile.write("%10e \n" %Res_v[i])
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
#for i in range(0,NV):
#    vtufile.write("%10e \n" %Res_q[i])
#vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exxn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exyn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='(effective) strain rate' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %srn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='(effective) viscosity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %etan[i])
vtufile.write("</DataArray>\n")

vtufile.write("<DataArray type='Float32' Name='(effective) dev. stress' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %(2*etan[i]*srn[i]))
vtufile.write("</DataArray>\n")




vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %23)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
