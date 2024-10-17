import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import random
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

###############################################################################

def powerlaw_viscosity(tau,nnn,A,eta0):
    if tau==0:
        powerlawvisc=1e50
    else:
        powerlawvisc=0.5*A**(-1)*tau**(1-nnn)
    combinedvisc=1/(1/powerlawvisc+1/eta0)
    return combinedvisc

###############################################################################

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

###############################################################################

xx=0
yy=1
xy=2
sqrt3=np.sqrt(3.)
sqrt2=np.sqrt(2.)
eps=1.e-10 
cm=0.01
year=365.25*24.*3600.

print("-----------------------------")
print("---------- stone 129 --------")
print("-----------------------------")

mV=9   # nb of nodes per element
ndof=2 # nb of degrees of freedom per node
ndim=2 # nb of dimensions

experiment=5

if experiment==1: #elasto-viscous folding
   Lx=6
   Ly=1
   nelx=80
   nely=40
   #phase1: top layer
   #phase2: bottom layer
   #phase3: thin middle layer
   viscosity = np.array([1e18,1e18,1e20],dtype=np.float64)
   density = np.array([2700,2700,2700],dtype=np.float64)
   poisson_ratio=np.array([0.3,0.3,0.3],dtype=np.float64)
   young_modulus=np.array([1e11,1e11,1e11],dtype=np.float64)
   shear_modulus=young_modulus/(2*(1+poisson_ratio)) #shear modulus
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   gx=0
   gy=9.8
   nstep=800
   dt=0.8*year
   viscous_rheology='linear'

if experiment==2 or experiment==7: #pure(2)/simple(7) shear
   Lx=50e3
   Ly=50e3
   nelx=10
   nely=10
   viscosity = np.array([1e21],dtype=np.float64)
   density = np.array([0],dtype=np.float64)
   shear_modulus=np.array([1e10],dtype=np.float64)
   poisson_ratio=np.array([0.45],dtype=np.float64)
   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   nstep=20
   dt=100*year
   gx=0
   gy=0
   viscous_rheology='linear'

if experiment==3: #bending slab a la Gerya
   Lx=1000e3
   Ly=1000e3
   nelx=50
   nely=50
   viscosity = np.array([1e27,1e21],dtype=np.float64)
   density = np.array([4000,1],dtype=np.float64)
   shear_modulus=np.array([1e10,1e15],dtype=np.float64) #15 not 20!
   poisson_ratio=np.array([0.27,0.27],dtype=np.float64)
   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   dt=200*year
   nstep=20
   gx=0
   gy=10
   viscous_rheology='linear'

if experiment==4: #flexure elastic plate a la Choi et al 2013
   Lx=50e3
   Ly=17.5e3
   nelx=75
   nely=25
   density = np.array([2700,1890,2700],dtype=np.float64)
   poisson_ratio=np.array([0.25,0.25,0.25],dtype=np.float64)
   #shear_modulus=np.array([30e9,30e9,1e30],dtype=np.float64)
   viscosity = np.array([1e30,1e30,1e17],dtype=np.float64)
   shear_modulus=np.array([30e9,30e9,30e14],dtype=np.float64)

   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   dt=5*year
   nstep=6
   gx=0
   gy=9.8
   viscous_rheology='linear'
   if viscous_rheology=='powerlaw':
      nnn=np.array([1,1,3.5],dtype=np.float64)
      AAA=np.array([1e30,1e30,1e-32],dtype=np.float64)

if experiment==5: # parallel-plate viscosimeter
   Lx=20
   Ly=10
   nelx=40
   nely=20
   density = np.array([0],dtype=np.float64)
   viscosity = np.array([1e9],dtype=np.float64)
   shear_modulus=np.array([500e6],dtype=np.float64)
   poisson_ratio=np.array([0.35],dtype=np.float64)
   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   dt=1
   nstep=500
   gx=0
   gy=0
   viscous_rheology='linear'

if experiment==6: #analytical benchmark
   Lx=50e3
   Ly=50e3
   nelx=16
   nely=16
   viscosity = np.array([1e21],dtype=np.float64)
   density = np.array([3300],dtype=np.float64)
   shear_modulus=np.array([1e10],dtype=np.float64)
   poisson_ratio=np.array([0.49],dtype=np.float64)
   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   nstep=100
   dt=5000*year
   gx=0
   gy=9.81
   viscous_rheology='linear'

if experiment==8: #Rayleigh-Taylor instability
   Lx=500e3
   Ly=500e3
   nelx=50
   nely=50
   A0=2e3
   R=10
   viscosity = np.array([1e21*R,1e21],dtype=np.float64)
   density = np.array([3300+100,3300],dtype=np.float64)
   shear_modulus=np.array([1e10,1e10],dtype=np.float64)
   poisson_ratio=np.array([0.499999,0.499999],dtype=np.float64)
   young_modulus=2*shear_modulus*(1+poisson_ratio)
   bulk_modulus=young_modulus/(3*(1-2*poisson_ratio)) 
   nstep=1
   dt=5000*year
   gx=0
   gy=9.81
   viscous_rheology='linear'

every=1

nnx=2*nelx+1  # number of nodes, x direction
nny=2*nely+1  # number of nodes, y direction
NV=nnx*nny    # total number of nodes
nel=nelx*nely # total number of elements
Nfem=NV*ndof  # Total number of velocity dofs

stats_vel_file=open('stats_vel.ascii',"w")
stats_exx_file=open('stats_exx.ascii',"w")
stats_eyy_file=open('stats_eyy.ascii',"w")
stats_exy_file=open('stats_exy.ascii',"w")
stats_sxx_file=open('stats_sxx.ascii',"w")
stats_syy_file=open('stats_syy.ascii',"w")
stats_sxy_file=open('stats_sxy.ascii',"w")
stats_txx_file=open('stats_txx.ascii',"w")
stats__xy_file=open('stats__xy.ascii',"w")

#####################################################################
# Gauss numerical quadrature points and weights

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

nq=nel*nqperdim**ndim # total number of quadrature points

#####################################################################

print('experiment        =',experiment)
print('nelx              =',nelx)
print('nely              =',nely)
print('nnx               =',nnx)
print('nny               =',nny)
print('NV                =',NV)
print('nel               =',nel)
print('dt(yr)            =',dt/year)
print('nstep             =',nstep)
print('viscosity eta     =',viscosity)
print('Young modulus E   =',young_modulus)
print('shear modulus mu  =',shear_modulus)
print('bulk modulus K    =',bulk_modulus)
print('bulk modulus K*dt =',bulk_modulus*dt)
print('poisson ratio nu  =',poisson_ratio)
print('eta_eff           =',viscosity*dt/(dt+viscosity/shear_modulus))
print('Maxwel times      =',viscosity[:]/shear_modulus[:])
print("-----------------------------")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

hx=Lx/float(nelx)
hy=Ly/float(nely)

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2
        yV[counter]=j*hy/2
        counter += 1
    #end for
#end for

xV0 = np.empty(NV,dtype=np.float64)  # x coordinates
yV0 = np.empty(NV,dtype=np.float64)  # y coordinates
xV0[:]=xV[:]
yV0[:]=yV[:]

print("mesh: %.3fs" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

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

print("connectivity: %.3fs" % (timing.time() - start))

#####################################################################
# assigning material to elements 
#####################################################################
start = timing.time()

phase   = np.zeros(nel,dtype=np.int32)
xc      = np.zeros(nel,dtype=np.float64) 
yc      = np.zeros(nel,dtype=np.float64) 
eta     = np.zeros(nel,dtype=np.float64)
mu      = np.zeros(nel,dtype=np.float64)
K       = np.zeros(nel,dtype=np.float64)
E       = np.zeros(nel,dtype=np.float64)
rho     = np.zeros(nel,dtype=np.float64)
eta_eff = np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc[iel]=0.5*(xV[iconV[0,iel]]+xV[iconV[2,iel]])
    yc[iel]=0.5*(yV[iconV[0,iel]]+yV[iconV[2,iel]])

if experiment==1:
   for iel in range(0,nel):
       if yc[iel]>0.55:
          phase[iel]=1
       elif yc[iel]>0.45:
          phase[iel]=3
       else:
          phase[iel]=2

if experiment==2 or experiment==6 or experiment==7:
   for iel in range(0,nel):
       phase[iel]=1

if experiment==3:
   for iel in range(0,nel):
       if xc[iel]<800e3 and yc[iel]<800e3 and yc[iel]>200e3:
          phase[iel]=1
       else:
          phase[iel]=2

if experiment==4:
   for iel in range(0,nel):
       if yc[iel]>12.5e3:
          phase[iel]=1
       else:
          phase[iel]=3
          if xc[iel]>45e3 and yc[iel]>7.5e3: 
             phase[iel]=2

if experiment==5:
   for iel in range(0,nel):
       phase[iel]=1

if experiment==8:
   for iel in range(0,nel):
       if yc[iel]>400e3:
          phase[iel]=1
       else:
          phase[iel]=2

for iel in range(0,nel):
    eta[iel]=viscosity[phase[iel]-1]
    mu[iel]=shear_modulus[phase[iel]-1]
    K[iel]=bulk_modulus[phase[iel]-1]
    E[iel]=young_modulus[phase[iel]-1]
    rho[iel]=density[phase[iel]-1]
    eta_eff[iel]=eta[iel]*dt/(dt+eta[iel]/mu[iel])

print("material layout: %.3f s" % (timing.time() - start))

#################################################################
# add random perturbation to central layer (exp=1)
#################################################################
start = timing.time()

if experiment==1:
   for i in range(0,NV):
       if abs(yV[i]-0.45)/Ly<eps:
          yV[i]+=hy*0.05*random.uniform(-1,1)
       if abs(yV[i]-0.55)/Ly<eps:
          yV[i]+=hy*0.05*random.uniform(-1,1)

   print("add perturbation to layer: %.3f s" % (timing.time() - start))

#################################################################
# add sin perturbation to nodes (exp=8)
#################################################################
start = timing.time()

if experiment==8:
   for i in range(0,NV):
       if abs(yV[i]-400e3)/Ly<eps:
          yV[i]+=A0*np.sin(np.pi*xV[i]/Lx)

   for iel in range(0,nel):
       yV[iconV[7,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[3,iel]])
       yV[iconV[5,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]])
       yV[iconV[8,iel]]=0.5*(yV[iconV[4,iel]]+yV[iconV[6,iel]])
       

   print("add sinusoidal perturbation: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 # vx=0 
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = -5e-3/year # vx  
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 # vy

if experiment==2:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0  
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 1*cm/year  
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if yV[i]/Ly>1-eps: #top boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1*cm/year

if experiment==3:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if yV[i]/Ly>1-eps: #top boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 

if experiment==4:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0  
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 

if experiment==5:
   for i in range(0, NV):
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +1e-4 
       if yV[i]/Ly>1-eps: #top boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1e-4

if experiment==6:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 # vx=0 
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 0 
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 # vy

if experiment==7:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = -1*cm/year
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if yV[i]/Ly>1-eps: #top boundary  
          bc_fix[i*ndof  ] = True ; bc_val[i*ndof  ] = 1*cm/year
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 

if experiment==8:
   for i in range(0, NV):
       if xV[i]/Lx<eps: #Left boundary  
          bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0 
       if xV[i]/Lx>1-eps: #right boundary  
          bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0 
       if yV[i]/Ly<eps: #bottom boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 
       if yV[i]/Ly>1-eps: #top boundary  
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0 

print("define boundary conditions: %.3f s" % (timing.time() - start))

#==============================================================================
# time stepping loop
#==============================================================================

model_time=0.
    
u                 = np.zeros(NV,dtype=np.float64) 
v                 = np.zeros(NV,dtype=np.float64) 
xq                = np.zeros(nq,dtype=np.float64) 
yq                = np.zeros(nq,dtype=np.float64) 
stress0_vector    = np.zeros((3,nq),dtype=np.float64) # stress vector memory
stress_vector     = np.zeros((3,nq),dtype=np.float64) # stress vector 
strainrate_vector = np.zeros((3,nq),dtype=np.float64) # strain rate vector
devstress_vector  = np.zeros((3,nq),dtype=np.float64) # deviatoric stress vector 
devstress         = np.zeros(nq,dtype=np.float64)     # effective deviatoric stress vector 
etaq              = np.zeros(nq,dtype=np.float64)     # effective viscosity at quad point

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep,'/',nstep-1)
    print("-----------------------------")

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64) # FE stiffness matrix 'KM'
    rhs      = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
    NNNV     = np.zeros(mV,dtype=np.float64)            # shape functions V
    dNNNVdr  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdx  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    b_mat    = np.zeros((3,ndof*mV),dtype=np.float64)   # gradient matrix B 

    counterq=0
    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndof),dtype=np.float64)
        K_el =np.zeros((mV*ndof,mV*ndof),dtype=np.float64)

        # integrate viscous term at quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq[counterq]=0
                yq[counterq]=0
                exxq=0
                eyyq=0
                exyq=0
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    xq[counterq]+=NNNV[k]*xV[iconV[k,iel]]
                    yq[counterq]+=NNNV[k]*yV[iconV[k,iel]]
                    exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                    eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                    exyq+=dNNNVdy[k]*u[iconV[k,iel]]*0.5+\
                          dNNNVdx[k]*v[iconV[k,iel]]*0.5

                strainrate_vector[0,counterq]=exxq
                strainrate_vector[1,counterq]=eyyq
                strainrate_vector[2,counterq]=exyq

                if viscous_rheology=='linear':
                   etaq[counterq]=eta[iel]
                elif viscous_rheology=='powerlaw':
                   etaq[counterq]=powerlaw_viscosity(devstress[counterq],nnn[phase[iel]-1],AAA[phase[iel]-1],eta[iel])
                else:
                   exit('viscous_rheology not specified!')

                #viscoelastic material matrix tilde(D)
                di=dt*(3*etaq[counterq]*K[iel]+3*dt*mu[iel]*K[iel]+4*mu[iel]*etaq[counterq])
                od=dt*(-2*mu[iel]*etaq[counterq]+3*etaq[counterq]*K[iel]+3*dt*mu[iel]*K[iel])
                d=3*(etaq[counterq]+dt*mu[iel])
                ed=etaq[counterq]*dt*mu[iel]/(etaq[counterq]+dt*mu[iel])
                Dee = np.array([[di/d, od/d,  0],\
                                [od/d, di/d,  0],\
                                [0,       0, ed]],dtype=np.float64)


                #stress matrix Ds for rhs
                di=3*etaq[counterq]+dt*mu[iel]
                od=dt*mu[iel] 
                ed=etaq[counterq]/(etaq[counterq]+dt*mu[iel]) 
                Dees = np.array([[di/d, od/d,  0],\
                                 [od/d, di/d,  0],\
                                 [0,       0, ed]],dtype=np.float64)

                #print(phase[iel],di/d,od/d,ed)

                stress_vector[:,counterq]=Dee.dot(strainrate_vector[:,counterq])+\
                                          Dees.dot(stress0_vector[:,counterq])

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(Dee.dot(b_mat))*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndof*i  ]-=NNNV[i]*jcob*weightq*gx*rho[iel]
                    f_el[ndof*i+1]-=NNNV[i]*jcob*weightq*gy*rho[iel]

                #f_el-=b_mat.T.dot(stress_vector[:,counterq])*weightq*jcob
                f_el-=b_mat.T.dot(Dees.dot(stress0_vector[:,counterq]))*weightq*jcob

                counterq+=1
            #end for
        #end for

        # apply boundary conditions
        for k1 in range(0,mV):
            for i1 in range(0,ndof):
                ikk=ndof*k1          +i1
                m1 =ndof*iconV[k1,iel]+i1
                if bc_fix[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,mV*ndof):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for 
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                #end if 
            #end for 
        #end for 

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mV):
             for i1 in range(0,ndof):
                 ikk=ndof*k1          +i1
                 m1 =ndof*iconV[k1,iel]+i1
                 for k2 in range(0,mV):
                     for i2 in range(0,ndof):
                         jkk=ndof*k2          +i2
                         m2 =ndof*iconV[k2,iel]+i2
                         A_sparse[m1,m2]+=K_el[ikk,jkk]
                     #end for 
                 #end for 
                 rhs[m1]+=f_el[ikk]
             #end for 
         #end for 

    #end for iel

    print("     -> viscosity (m,M) %e %e (Pa s)" %(np.min(etaq),np.max(etaq)))

    print("building FEM matrix and rhs: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    sol = sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

    u,v=np.reshape(sol,(NV,2)).T

    print("     -> u (m,M) %e %e (cm/year)" %(np.min(u/cm*year),np.max(u/cm*year)))
    print("     -> v (m,M) %e %e (cm/year)" %(np.min(v/cm*year),np.max(v/cm*year)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("solve FE system: %.3f s" % (timing.time() - start))

    #################################################################
    # compute time step
    #################################################################

    CFL=1
    dt1=CFL*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    print(dt1/year,dt/year)

    #################################################################
    start = timing.time()

    devstress_vector[xx,:]=0.5*(stress_vector[xx,:]-stress_vector[yy,:])
    devstress_vector[yy,:]=0.5*(stress_vector[yy,:]-stress_vector[xx,:])
    devstress_vector[xy,:]=stress_vector[xy,:]

    devstress[:]=np.sqrt(0.5*(devstress_vector[xx,:]**2+devstress_vector[yy,:]**2)+devstress_vector[xy,:]**2)

    print("     -> deviatoric stress (m,M) %e %e (Pa)" %(np.min(devstress),np.max(devstress)))

    stats_vel_file.write("%e %e %e %e %e\n" % (model_time,np.min(u),np.max(u),np.min(v),np.max(v)))
    stats_vel_file.flush()

    if istep>0:
       stats_exx_file.write("%e %e %e \n" % (model_time,np.min(strainrate_vector[xx,:]),np.max(strainrate_vector[xx,:])))
       stats_exx_file.flush()
       stats_eyy_file.write("%e %e %e \n" % (model_time,np.min(strainrate_vector[yy,:]),np.max(strainrate_vector[yy,:])))
       stats_eyy_file.flush()
       stats_exy_file.write("%e %e %e \n" % (model_time,np.min(strainrate_vector[xy,:]),np.max(strainrate_vector[xy,:])))
       stats_exy_file.flush()
       stats_sxx_file.write("%e %e %e \n" % (model_time,np.min(stress_vector[xx,:]),np.max(stress_vector[xx,:])))
       stats_sxx_file.flush()
       stats_syy_file.write("%e %e %e \n" % (model_time,np.min(stress_vector[yy,:]),np.max(stress_vector[yy,:])))
       stats_syy_file.flush()
       stats_sxy_file.write("%e %e %e \n" % (model_time,np.min(stress_vector[xy,:]),np.max(stress_vector[xy,:])))
       stats_sxy_file.flush()
       stats_txx_file.write("%e %e %e \n" % (model_time,np.min(devstress_vector[xx,:]),np.max(devstress_vector[xx,:])))
       stats_txx_file.flush()

    print("export stats in files: %.3f s" % (timing.time() - start))

    #################################################################
    # moving mesh nodes
    #################################################################
    start = timing.time()

    xV[:]+=u[:]*dt
    yV[:]+=v[:]*dt
    
    print("     -> xV (m,M) %e %e (m)" %(np.min(xV),np.max(xV)))
    print("     -> yV (m,M) %e %e (m)" %(np.min(yV),np.max(yV)))

    stats__xy_file.write("%e %e %e %e %e\n" % (model_time,np.min(xV),np.max(xV),np.min(yV),np.max(yV)))
    stats__xy_file.flush()

    print("evolve mesh: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation
    # stress components are exported in the solution file by 
    # simply using the value for the quadrature point in the middle
    # of the element. 
    #################################################################
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
           vtufile.write("%e %e %e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='phase' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % phase[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % eta[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % eta_eff[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='bulk modulus K' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % K[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='shear modulus mu' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % mu[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='young modulus E' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % E[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='density rho' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % rho[iel])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % (stress_vector[0,iel*9+4]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % (stress_vector[1,iel*9+4]))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % (stress_vector[2,iel*9+4]))
       vtufile.write("</DataArray>\n")
  
       vtufile.write("<DataArray type='Float32' Name='devstress' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % (devstress[iel*9+4]))
       vtufile.write("</DataArray>\n")
  


       vtufile.write("<DataArray type='Float32' Name='sigma_m' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e \n" % (0.5*stress_vector[0,iel*9+4]+0.5*stress_vector[1,iel*9+4]))
       vtufile.write("</DataArray>\n")

       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(u[i]/cm*year,v[i]/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e %e %e \n" %(xV[i]-xV0[i],yV[i]-yV0[i],0))
       vtufile.write("</DataArray>\n")



       #--
       #vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
       #for i in range(0,NV):
       #    if bc_fix[i*2]:
       #       val=1
       #    else:
       #       val=0
       #    vtufile.write("%10e \n" %val)
       #vtufile.write("</DataArray>\n")
       #-- 
       #vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
       #for i in range(0,NV):
       #    if bc_fix[i*2+1]:
       #       val=1
       #    else:
       #       val=0
       #    vtufile.write("%10e \n" %val)
       #vtufile.write("</DataArray>\n")
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

       filename = 'qpts_{:04d}.vtu'.format(istep) 
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
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % stress_vector[xx,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % stress_vector[yy,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % stress_vector[xy,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sigma_m' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % (0.5*(stress_vector[0,iq]+stress_vector[1,iq])))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % strainrate_vector[xx,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % strainrate_vector[yy,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % strainrate_vector[xy,iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % (devstress_vector[xx,iq]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % (devstress[iq]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % (etaq[iq]))
       vtufile.write("</DataArray>\n")


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
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to files: %.3f s" % (timing.time() - start))

    #end if

    model_time+=dt
    print ("model_time=",model_time/year,'yr')

    stress0_vector[:,:]=stress_vector[:,:]

    if experiment==1:
       shortening=(1-max(xV)/Lx)*100
       print('max(x)=',max(xV),', Lx(t=0)=',Lx)
       print('shortening=',shortening,'%')
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
