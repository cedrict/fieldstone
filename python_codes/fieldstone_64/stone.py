import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import time as clock 
import random

###############################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################
# bi-quadratic Bernstein polynomial
# https://en.wikipedia.org/wiki/Bernstein_polynomial

def BernsteinPolynomial(r,s): 
    B0= 0.25*(1-r)**2 * 0.25*(1-s)**2
    B1= 0.25*(1+r)**2 * 0.25*(1-s)**2
    B2= 0.25*(1+r)**2 * 0.25*(1+s)**2
    B3= 0.25*(1-r)**2 * 0.25*(1+s)**2
    B4=  0.5*(1-r**2) * 0.25*(1-s)**2
    B5= 0.25*(1+r)**2 *  0.5*(1-s**2)
    B6=  0.5*(1-r**2) * 0.25*(1+s)**2
    B7= 0.25*(1-r)**2 *  0.5*(1-s**2)
    B8=  0.5*(1-r**2) *  0.5*(1-s**2)
    return np.array([B0,B1,B2,B3,B4,B5,B6,B7,B8],dtype=np.float64)

###############################################################################

def gy(time):
    if benchmark==11 or benchmark==12: val=0 

    if benchmark==2:
       if time<20e3*year :
          val=-10
       else:
          val=0

    if benchmark==3:
       if time<50e3*year :
          val=-10
       else:
          val=0

    if benchmark==4: val=-9.81

    if benchmark==5: val=-g0

    return val

def compute_rs(xM,yM,iel):
    x=x_V[icon_V[0:m_V,iel]]
    y=y_V[icon_V[0:m_V,iel]]
    r=0
    s=0
    for i in range(0,10):
        jcb=np.zeros((2,2),dtype=np.float64)
        rhs=np.zeros(2,dtype=np.float64)
        NNNV[0:9]=NNV(r,s)
        dNNNVdr[0:9]=dNNVdr(r,s)
        dNNNVds[0:9]=dNNVds(r,s)
        rhs[0]=-(sum(NNNV[:]*x[:])-xM)
        rhs[1]=-(sum(NNNV[:]*y[:])-yM)
        for k in range(0,m_V):
            jcb[0,0] += dNNNVdr[k]*x_V[icon_V[k,iel]]
            jcb[0,1] += dNNNVdr[k]*y_V[icon_V[k,iel]]
            jcb[1,0] += dNNNVds[k]*x_V[icon_V[k,iel]]
            jcb[1,1] += dNNNVds[k]*y_V[icon_V[k,iel]]
        #end for 
        jcbi=np.linalg.inv(jcb)
        deltar=jcbi[0,0]*rhs[0]+jcbi[0,1]*rhs[1]
        deltas=jcbi[1,0]*rhs[0]+jcbi[1,1]*rhs[1]
        r+=deltar
        s+=deltas
        if abs(deltar)<1e-6 and abs(deltas)<1e-6:
           break
    #end for
    return r,s

###############################################################################

cm=0.01
year=365.25*24.*3600.
sqrt2=np.sqrt(2)
eps=1.e-10
eps2=1.e-6

###############################################################################

print("*******************************")
print("********** stone 064 **********")
print("*******************************")

ndim=2
m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 50
   nely = 50
   visu = 1

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

gx=0.

debug=False

#######################################
# benchmark 11,12: maxwell body
# benchmark 2: slab Gerya book 
# benchmark 3: bending beam
# benchmark 4: flexure Choi
# benchmark 5: ice sheet load

benchmark=2

if benchmark==11 or benchmark==12: # maxwell body
   nelx=16
   nely=16
   Lx=100e3  
   Ly=100e3  
   dt=100*year
   rho1=0
   rho2=0
   mu1=1e10
   eta1=1e21
   etaeff1=eta1*dt/(dt+eta1/mu1)
   Z1=etaeff1/mu1/dt
   nstep=200
   nparticle_per_dim=10
   every=1
   eta_ref=1e23
   pnormalise=True
   use_ALE=False
   advection='RK1'

if benchmark==2: # slab Gerya book
   nelx=70
   nely=70
   Lx=1000e3 
   Ly=1000e3 
   dt=200*year
   rho1=4000
   rho2=1
   eta1=1e27
   eta2=1e21
   mu1=1e10
   mu2=1e20
   nstep=301
   nparticle_per_dim=8
   every=10
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   eta_ref=1e23
   pnormalise=False
   use_ALE=False
   advection='RK1'

if benchmark==3: # bending beam Keller et al 2013 
   Lx=7500
   Ly=5000
   nelx=75
   nely=50
   dt=100*year
   rho1=1000
   rho2=1500
   eta1=1e18
   eta2=1e24
   mu1=1e11
   mu2=1e10
   nstep=1000
   nparticle_per_dim=10
   every=1
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   eta_ref=1e23
   pnormalise=True
   use_ALE=False

if benchmark==4: # flexure Choi et al 2013 
   Lx=50e3
   Ly=17.5e3
   nelx=100
   nely=35
   dt=5*year
   rho1=2700
   rho2=1890
   rho3=2700
   eta1=1e25
   eta2=1e25
   eta3=1e17
   mu1=30e9
   mu2=30e9
   mu3=1e50
   nstep=100
   nparticle_per_dim=7
   every=5
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   etaeff3=eta3*dt/(dt+eta3/mu3)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   Z3=etaeff3/mu3/dt
   eta_ref=1e23
   pnormalise=False
   use_ALE=True

if benchmark==5: # ice sheet load 
   Lx=500e3
   Ly=500e3
   nelx=50
   nely=50
   dt=10*year
   rho1=3300
   eta1=3e20
   mu1=1e10
   nstep=10
   nparticle_per_dim=10
   every=1
   etaeff1=eta1*dt/(dt+eta1/mu1)
   Z1=etaeff1/mu1/dt
   eta_ref=1e23
   rhoi=900
   g0=9.8
   H0=1000
   t0=1000*year
   t1=1000*year
   pnormalise=False
   use_ALE=True
  

#1: nodal average
#2: c->n
computeLmethod=1

nnx=2*nelx+1             # number of Vnodes, x direction
nny=2*nely+1             # number of Vnodes, y direction
nn_V=nnx*nny             # number of Vnodes
nel=nelx*nely            # number of elements
Nfem_V=nn_V*ndof_V       # number of velocity dofs
Nfem_P=(nelx+1)*(nely+1) # number of pressure dofs
Nfem=Nfem_V+Nfem_P       # total number of dofs
Nfem_T=nn_V              # number of field dofs 
hx=Lx/nelx
hy=Ly/nely
nq=nel*nq_per_dim**ndim

scaling_coeff=eta_ref/Ly
   
r_V=[-1,1,1,-1,0,1,0,-1,0]
s_V=[-1,-1,1,1,-1,0,1,0,0]

time=0.

nparticle=int(nel*nparticle_per_dim**ndim)
   
#True: use shape fcts for node->qpt
#False: use Bernstein poly for node->qpt             
use_ss=False

###############################################################################

stats_exx_file=open('stats_exx.ascii',"w")
stats_eyy_file=open('stats_eyy.ascii',"w")
stats_exy_file=open('stats_exy.ascii',"w")
stats_wxy_file=open('stats_wxy.ascii',"w")
stats_tauxx_file=open('stats_tauxx.ascii',"w")
stats_tauyy_file=open('stats_tauyy.ascii',"w")
stats_tauxy_file=open('stats_tauxy.ascii',"w")
stats_u_file=open('stats_u.ascii',"w")
stats_v_file=open('stats_v.ascii',"w")
stats_Z_file=open('stats_Z.ascii',"w")
stats_vel_file=open('stats_vel.ascii',"w")
stats_etaeff_file=open('stats_etaeff.ascii',"w")
stats_Jxx_file=open('stats_Jxx.ascii',"w")
stats_Jyy_file=open('stats_Jyy.ascii',"w")
stats_Jxy_file=open('stats_Jxy.ascii',"w")
stats_m_tauxx_file=open('stats_m_tauxx.ascii',"w")
stats_m_tauyy_file=open('stats_m_tauyy.ascii',"w")
stats_m_tauxy_file=open('stats_m_tauxy.ascii',"w")
flagged_file=open('flagged_particles.ascii',"w")
if use_ALE: stats_topo_file=open('stats_topo.ascii',"w")

###############################################################################

print("benchmark=",benchmark)
print("nelx=",nelx)
print("nely=",nely)
print("nel=",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)
print("nparticle=",nparticle)

if benchmark==11 or benchmark==12 or benchmark==5:
   print("etaeff1=",etaeff1)
   print("Z1=",Z1)

if benchmark==2 or benchmark==3:
   print("etaeff1=",etaeff1)
   print("etaeff2=",etaeff2)
   print("Z1=",Z1)
   print("Z2=",Z2)
   print("t_M 1=",eta1/mu1/year,"yr")
   print("t_M 2=",eta2/mu2/year,"yr")

if benchmark==4:
   print("etaeff1=",etaeff1)
   print("etaeff2=",etaeff2)
   print("etaeff3=",etaeff3)
   print("Z1=",Z1)
   print("Z2=",Z2)
   print("Z3=",Z3)
   print("t_M 1=",eta1/mu1/year,"yr")
   print("t_M 2=",eta2/mu2/year,"yr")
   print("t_M 3=",eta3/mu3/year,"yr")

print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

match benchmark:
 case 11:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = -1*cm/year
       #end if
       if x_V[i]>(Lx-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = +1*cm/year
       #end if
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = +1*cm/year
       #end if
       if y_V[i]>(Ly-eps):
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = -1*cm/year
       #end if
   #end for

 case 12:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 
       #end if
       if x_V[i]>(Lx-eps):
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 
       #end if
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = -1*cm/year
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 
       #end if
       if y_V[i]>(Ly-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = +1*cm/year
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
       #end if
   #end for

 case 2 | 3:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       #end if
       if x_V[i]>(Lx-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
       #end if
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       #end if
       if y_V[i]>(Ly-eps):
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
       #end if
   #end for

 case 4:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

 case 5:
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0.
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# flag nodes at the surface
###############################################################################

surface=np.zeros(nn_V,dtype=bool)  

for i in range(0,nn_V):
    if y_V[i]/Ly>(1-eps):
       surface[i] = True 

###############################################################################
# particles layout.
# pseudo-randomly generated, not too close from domain sides
###############################################################################
start=clock.time()

swarm_x=np.zeros(nparticle,dtype=np.float64)  
swarm_y=np.zeros(nparticle,dtype=np.float64)  
swarm_u=np.zeros(nparticle,dtype=np.float64)  
swarm_v=np.zeros(nparticle,dtype=np.float64)  
swarm_Z=np.zeros(nparticle,dtype=np.float64)  
swarm_etaeff=np.zeros(nparticle,dtype=np.float64)  
swarm_rho=np.zeros(nparticle,dtype=np.float64)  
swarm_iel=np.zeros(nparticle,dtype=np.int32)  
swarm_r=np.zeros(nparticle,dtype=np.float64)  
swarm_s=np.zeros(nparticle,dtype=np.float64)  
swarm_tauxx=np.zeros(nparticle,dtype=np.float64)  
swarm_tauyy=np.zeros(nparticle,dtype=np.float64)  
swarm_tauxy=np.zeros(nparticle,dtype=np.float64)  
swarm_mat=np.zeros(nparticle,dtype=np.int32)  

counter=0
for iel in range(0,nel):
    for j in range(0,nparticle_per_dim):
        for i in range(0,nparticle_per_dim):
            r = -1.0 + i * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
            s = -1.0 + j * 2.0 / nparticle_per_dim + 1.0 / nparticle_per_dim
            r += random.uniform(-0.2, +0.2) * (2 / nparticle_per_dim)
            s += random.uniform(-0.2, +0.2) * (2 / nparticle_per_dim)
            N = basis_functions_V(r,s)
            swarm_r[counter] = r
            swarm_s[counter] = s
            swarm_x[counter] = np.dot(N[:], x_V[icon_V[:, iel]])
            swarm_y[counter] = np.dot(N[:], y_V[icon_V[:, iel]])
            swarm_iel[counter] = iel 
            counter += 1
        # end for
    # end for
# end for


match benchmark:
 case 11 | 12 | 5:
   for ip in range(0,nparticle):
       swarm_rho[ip]=rho1
       swarm_etaeff[ip]=etaeff1
       swarm_Z[ip]=Z1
       swarm_mat[ip]=1
 case 2:
   for ip in range(0,nparticle):
       if swarm_x[ip]<=800e3 and np.abs(swarm_y[ip]-Ly/2)<=300e3:
          swarm_rho[ip]=rho1
          swarm_etaeff[ip]=etaeff1
          swarm_Z[ip]=Z1
          swarm_mat[ip]=3
       else:
          swarm_rho[ip]=rho2
          swarm_etaeff[ip]=etaeff2
          swarm_Z[ip]=Z2
       #end if
   #end for
 case 3:
   for ip in range(0,nparticle):
       swarm_rho[ip]=rho1
       swarm_etaeff[ip]=etaeff1
       swarm_Z[ip]=Z1
       swarm_mat[ip]=1
       if swarm_y[ip]>2200 and swarm_y[ip]<2800 and swarm_x[ip]<4500: 
          swarm_rho[ip]=rho2
          swarm_etaeff[ip]=etaeff2
          swarm_Z[ip]=Z2
          swarm_mat[ip]=4
       #end if
       if (swarm_x[ip]-4500)**2+(swarm_y[ip]-Ly/2.)**2<300**2:
          swarm_rho[ip]=rho2
          swarm_etaeff[ip]=etaeff2
          swarm_Z[ip]=Z2
          swarm_mat[ip]=4
       #end if
   #end for
 case 4:
   for ip in range(0,nparticle):
       if swarm_y[ip]>Ly-5e3:
          swarm_rho[ip]=rho1
          swarm_etaeff[ip]=etaeff1
          swarm_Z[ip]=Z1
          swarm_mat[ip]=1
       else:
          swarm_rho[ip]=rho3
          swarm_etaeff[ip]=etaeff3
          swarm_Z[ip]=Z3
          swarm_mat[ip]=7
       #end if
       if swarm_x[ip]>Lx-5000 and  swarm_y[ip]<Ly-5e3 and  swarm_y[ip]>7.5e3:
          swarm_rho[ip]=rho2
          swarm_etaeff[ip]=etaeff2
          swarm_Z[ip]=Z2
          swarm_mat[ip]=4
       #end if
   #end for

if debug: np.savetxt('particles_init.ascii',np.array([swarm_x,swarm_y,swarm_rho,swarm_Z,swarm_etaeff]).T)

print("material layout: %.3f s" % (clock.time()-start))

###############################################################################

for ip in range(0,nparticle):
    if ip%11111==0: flagged_file.write("%d %e %e %e \n" %(ip,time,swarm_x[ip],swarm_y[ip])) 

###############################################################################
# painting particles
# this could probably a bit more clever/elegant.
###############################################################################
start=clock.time()

if benchmark==11 or benchmark==12:
   for i in [0,2,4]:
       dx=Lx/5
       for ip in range (0,nparticle):
           if swarm_x[ip]>i*dx and swarm_x[ip]<(i+1)*dx: swarm_mat[ip]+=1
   for i in [0,2,4]:
       dy=Ly/5
       for ip in range (0,nparticle):
           if swarm_y[ip]>i*dy and swarm_y[ip]<(i+1)*dy: swarm_mat[ip]+=1

if benchmark==2 or benchmark==5:
   for i in [0,2,4,6,8,10,12,14,16,18]:
       dx=Lx/20
       for ip in range (0,nparticle):
           if swarm_x[ip]>i*dx and swarm_x[ip]<(i+1)*dx: swarm_mat[ip]+=1
   for i in [0,2,4,6,8,10,12,14,16,18]:
       dy=Ly/20
       for ip in range (0,nparticle):
           if swarm_y[ip]>i*dy and swarm_y[ip]<(i+1)*dy: swarm_mat[ip]+=1

if benchmark==3:
   for i in [0,2,4]:
       dx=Lx/5
       for ip in range (0,nparticle):
           if swarm_x[ip]>i*dx and swarm_x[ip]<(i+1)*dx: swarm_mat[ip]+=1
   for i in [0,2,4,6,8,10,12,14,16,18,20,22,24]:
       dy=Ly/25
       for ip in range (0,nparticle):
           if swarm_y[ip]>i*dy and swarm_y[ip]<(i+1)*dy: swarm_mat[ip]+=1

if benchmark==4:
   for i in [0,2,4,6,8,10]:
       dy=2.5e3
       for ip in range (0,nparticle):
           if swarm_y[ip]>i*dy and swarm_y[ip]<(i+1)*dy: swarm_mat[ip]+=1

print("paint particles: %.3f s" % (clock.time()-start))

###############################################################################
# compute element center coordinates
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    x_e[iel]=np.sum(x_V[icon_V[:,iel]])/9
    y_e[iel]=np.sum(y_V[icon_V[:,iel]])/9

print("compute element center coords: %.3f s" % (clock.time()-start))
 
###############################################################################
# project particles onto elements
###############################################################################
start=clock.time()

Z=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  
etaeff=np.zeros(nel,dtype=np.float64)  
count=np.zeros(nel,dtype=np.float64)  

for ip in range(0,nparticle):
    iel=swarm_iel[ip]
    rho[iel]+=swarm_rho[ip]
    Z[iel]+=swarm_Z[ip]
    etaeff[iel]+=swarm_etaeff[ip]
    count[iel]+=1
#end for

Z/=count
rho/=count
etaeff/=count

if debug: np.savetxt('elemental_values.ascii',np.array([x_e,y_e,rho,Z,etaeff]).T,header='# x,y,rho,Z,eta_eff')

print("project Z, rho, etaeff onto elements: %.3f s" % (clock.time()-start))

###############################################################################
# initialise nodal fields 
###############################################################################

Jxx =np.zeros(nel,dtype=np.float64)  
Jyy =np.zeros(nel,dtype=np.float64)  
Jxy =np.zeros(nel,dtype=np.float64)  
tauxx =np.zeros(nel,dtype=np.float64)  
tauyy =np.zeros(nel,dtype=np.float64)  
tauxy =np.zeros(nel,dtype=np.float64)  
tauxxmem =np.zeros(nel,dtype=np.float64)  
tauyymem =np.zeros(nel,dtype=np.float64)  
tauxymem =np.zeros(nel,dtype=np.float64)  

#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================

u = np.zeros(nn_V,dtype=np.float64)
v = np.zeros(nn_V,dtype=np.float64)
q_x = np.zeros(nq,dtype=np.float64)    
q_y = np.zeros(nq,dtype=np.float64)   
q_Z = np.zeros(nq,dtype=np.float64)   
q_rho = np.zeros(nq,dtype=np.float64)   
q_etaeff = np.zeros(nq,dtype=np.float64)   
R = np.zeros(3,dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for istep in range(0,nstep):
    print("----------------------------------------")
    print("istep= ",istep,'/',nstep-1)
    print("----------------------------------------")

    #filename = 'quadrature_points_values_{:04d}.ascii'.format(istep)
    #qpts_file=open(filename,"w")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix 
    N_mat= np.zeros((3,m_P),dtype=np.float64) # matrix  
    jcb=np.zeros((2,2),dtype=np.float64)

    counterq=0
    for iel in range(0,nel):

        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)
        N_N_NP= np.zeros(m_P,dtype=np.float64)   

        R[0]=Z[iel]*(tauxx[iel]+dt*Jxx[iel])
        R[1]=Z[iel]*(tauyy[iel]+dt*Jyy[iel])
        R[2]=Z[iel]*(tauxy[iel]+dt*Jxy[iel])

        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                q_x[counterq]=np.dot(N_V,x_V[icon_V[:,iel]])
                q_y[counterq]=np.dot(N_V,y_V[icon_V[:,iel]])
                q_Z[counterq]=Z[iel]
                q_rho[counterq]=rho[iel]
                q_etaeff[counterq]=etaeff[iel]

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                K_el+=B.T.dot(C.dot(B))*q_etaeff[counterq]*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i  ]+=N_V[i]*JxWq*q_rho[counterq]*gx
                    f_el[ndof_V*i+1]+=N_V[i]*JxWq*q_rho[counterq]*gy(time)

                #compute elastic rhs
                f_el-=B.T.dot(R)*JxWq

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=0.
                #end for 

                G_el-=B.T.dot(N_mat)*JxWq

                N_N_NP[:]+=N_P[:]*JxWq

                counterq+=1
            #end for jq
        #end for iq

        G_el*=scaling_coeff

        # impose traction bc

        if benchmark==5:
           if surface[icon_V[2,iel]] and x_V[icon_V[2,iel]]<=100e3:
              if time<t0:
                 traction=-g0*H0*rhoi
              elif time<t1:
                 traction=-g0*H0*rhoi*(1-(time-t0)/t1)
              else:
                 traction=0.
              print (traction)
              #end if
              f_el[ndof_V*2+1]+=traction*hx/2.*(1./3.)
              f_el[ndof_V*6+1]+=traction*hx/2.*(4./3.)
              f_el[ndof_V*3+1]+=traction*hx/2.*(1./3.)
           #end if
        #end if

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
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

        # assemble matrix and rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_fem[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                    A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                #end for
                rhs[m1]+=f_el[ikk]
            #end for
        #end for

        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            rhs[Nfem_V+m2]+=h_el[k2]
        #end for

    #end for iel

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),rhs)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*scaling_coeff

    vel=np.sqrt(u**2+v**2)

    print("     -> u (m,M) %.3e %.3e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.3e %.3e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.3e %.3e " %(np.min(p),np.max(p)))

    stats_u_file.write("%e %e %e %e\n" %(time,np.min(u),np.max(u),time/year)) ; stats_u_file.flush()
    stats_v_file.write("%e %e %e %e\n" %(time,np.min(v),np.max(v),time/year)) ; stats_v_file.flush()
    stats_vel_file.write("%e %e %e %e\n" %(time,np.min(vel),np.max(vel),time/year)) ; stats_vel_file.flush()

    if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    ####### TEST/DEBUG ############
    #for i in range(0,NV):
    #    xx=xV[i]/Lx
    #    yy=yV[i]/Ly
    #    u[i]=(xx*xx*(1.-xx)**2*(2.*yy-6.*yy*yy+4*yy*yy*yy))*cm/year  *100
    #    v[i]=(-yy*yy*(1.-yy)**2*(2.*xx-6.*xx*xx+4*xx*xx*xx))*cm/year *100
    #CFL=0.25
    #dt=CFL*hx/np.max(np.sqrt(u**2+v**2))/order

    ###########################################################################
    # compute timestep
    ###########################################################################

    CFL=dt/min(hx,hy)*np.max(np.sqrt(u**2+v**2))

    print('     -> dt = %f year, corresponds to CFL= %f' %(dt/year,CFL))

    ###########################################################################
    # compute elementa velocity gradient 
    ###########################################################################
    start=clock.time()
    
    Lxx=np.zeros(nel,dtype=np.float64)  
    Lxy=np.zeros(nel,dtype=np.float64)  
    Lyx=np.zeros(nel,dtype=np.float64)  
    Lyy=np.zeros(nel,dtype=np.float64)  

    #u[:]=x_V[:]**2
    #v[:]=y_V[:]**2

    for iel in range(0,nel):
        rq=0.0
        sq=0.0
        N_V=basis_functions_V(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        Lxx[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
        Lxy[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])
        Lyx[iel]=np.dot(dNdy_V,u[icon_V[:,iel]])
        Lyy[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])

    print("     -> Lxx (m,M) %.3e %.3e " %(np.min(Lxx),np.max(Lxx)))
    print("     -> Lxy (m,M) %.3e %.3e " %(np.min(Lxy),np.max(Lxy)))
    print("     -> Lyx (m,M) %.3e %.3e " %(np.min(Lyx),np.max(Lyx)))
    print("     -> Lyy (m,M) %.3e %.3e " %(np.min(Lyy),np.max(Lyy)))

    if debug:
       np.savetxt('velgradient.ascii',np.array([x_e,y_e,Lxx,Lyy,Lxy,Lyx]).T,header='# x,y,Lxx,Lyy,Lxy,Lyx')

    print("compute nodal p and L: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute elemental fields
    # could/should probably be merged with previous block
    ###########################################################################
    start=clock.time()

    exx=np.copy(Lxx)
    eyy=np.copy(Lyy)
    exy=0.5*(Lxy+Lyx)
    wxy=0.5*(Lxy-Lyx)
    tauxx=2*etaeff*exx+Z*tauxx+Z*dt*Jxx
    tauyy=2*etaeff*eyy+Z*tauyy+Z*dt*Jyy
    tauxy=2*etaeff*exy+Z*tauxy+Z*dt*Jxy

    print("     -> exx   (m,M) %.3e %.3e " %(np.min(exx),np.max(exx)))
    print("     -> eyy   (m,M) %.3e %.3e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy   (m,M) %.3e %.3e " %(np.min(exy),np.max(exy)))
    print("     -> wxy   (m,M) %.3e %.3e " %(np.min(wxy),np.max(wxy)))
    print("     -> tauxx (m,M) %.3e %.3e " %(np.min(tauxx),np.max(tauxx)))
    print("     -> tauyy (m,M) %.3e %.3e " %(np.min(tauyy),np.max(tauyy)))
    print("     -> tauxy (m,M) %.3e %.3e " %(np.min(tauxy),np.max(tauxy)))

    stats_exx_file.write("%e %e %e \n" %(time,np.min(exx),np.max(exx))) ; stats_exx_file.flush()
    stats_eyy_file.write("%e %e %e \n" %(time,np.min(eyy),np.max(eyy))) ; stats_eyy_file.flush()
    stats_exy_file.write("%e %e %e \n" %(time,np.min(exy),np.max(exy))) ; stats_exy_file.flush()
    stats_wxy_file.write("%e %e %e \n" %(time,np.min(wxy),np.max(wxy))) ; stats_wxy_file.flush()
    stats_tauxx_file.write("%e %e %e \n" %(time,np.min(tauxx),np.max(tauxx))) ; stats_tauxx_file.flush()
    stats_tauyy_file.write("%e %e %e \n" %(time,np.min(tauyy),np.max(tauyy))) ; stats_tauyy_file.flush()
    stats_tauxy_file.write("%e %e %e \n" %(time,np.min(tauxy),np.max(tauxy))) ; stats_tauxy_file.flush()

    print("compute sr, rr and J: %.3f s" % (clock.time()-start))

    ###########################################################################

    time+=dt

    ###########################################################################
    # interpolate dev stress difference (increment) onto particles
    # and add it to the existing value on the particles (more accurate)
    ###########################################################################
    start=clock.time()

    for ip in range(0,nparticle):
        rm=swarm_r[ip]
        sm=swarm_s[ip]
        iel=swarm_iel[ip]
        swarm_tauxx[ip]+=tauxx[iel]-tauxxmem[iel]
        swarm_tauyy[ip]+=tauyy[iel]-tauyymem[iel]
        swarm_tauxy[ip]+=tauxy[iel]-tauxymem[iel]

    print("     -> swarm_tauxx (m,M) %.6e %.6e " %(np.min(swarm_tauxx),np.max(swarm_tauxx)))
    print("     -> swarm_tauyy (m,M) %.6e %.6e " %(np.min(swarm_tauyy),np.max(swarm_tauyy)))
    print("     -> swarm_tauxy (m,M) %.6e %.6e " %(np.min(swarm_tauxy),np.max(swarm_tauxy)))

    stats_m_tauxx_file.write("%e %e %e \n" %(time,np.min(swarm_tauxx),np.max(swarm_tauxx))) ;stats_m_tauxx_file.flush()
    stats_m_tauyy_file.write("%e %e %e \n" %(time,np.min(swarm_tauyy),np.max(swarm_tauyy))) ;stats_m_tauyy_file.flush()
    stats_m_tauxy_file.write("%e %e %e \n" %(time,np.min(swarm_tauxy),np.max(swarm_tauxy))) ;stats_m_tauxy_file.flush()

    print("interp. diff stress onto particles: %.3f s" % (clock.time()-start))

    ###########################################################################
    # advect particles. this is a very simple algo 
    ###########################################################################
    start=clock.time()

    if advection=='RK1':
       for ip in range(0,nparticle):
           rm=swarm_r[ip]
           sm=swarm_s[ip]
           iel=swarm_iel[ip]
           N_V=basis_functions_V(rm,sm)
           swarm_u[ip]=np.dot(N_V,u[icon_V[:,iel]]) 
           swarm_v[ip]=np.dot(N_V,v[icon_V[:,iel]])
           swarm_x[ip]+=swarm_u[ip]*dt 
           swarm_y[ip]+=swarm_v[ip]*dt 
           swarm_x[ip]=min((1-eps2)*Lx,swarm_x[ip])
           swarm_y[ip]=min((1-eps2)*Ly,swarm_y[ip])
           swarm_x[ip]=max(eps2*Lx,swarm_x[ip])
           swarm_y[ip]=max(eps2*Ly,swarm_y[ip])
       #end for

    if advection=='RK2':
       for ip in range(0,nparticle):
           # first step
           xA=swarm_x[ip]
           yA=swarm_y[ip]
           rm=swarm_r[ip]
           sm=swarm_s[ip]
           iel=swarm_iel[ip]
           N_V=basis_functions_V(rm,sm)
           uA=np.dot(N_V,u[icon_V[:,iel]]) 
           vA=np.dot(N_V,v[icon_V[:,iel]])
           xB = xA + uA * dt / 2.0
           yB = yA + vA * dt / 2.0
           xB=min((1-eps2)*Lx,xB)
           yB=min((1-eps2)*Ly,yB)
           xB=max(eps2*Lx,xB)
           yB=max(eps2*Ly,yB)

           # second step
           ielx=int(xB/hx)
           iely=int(yB/hy)
           iel=iely*nelx+ielx
           rm=((xB-x_V[icon_V[0,iel]])/hx-0.5)*2.
           sm=((yB-y_V[icon_V[0,iel]])/hy-0.5)*2.
           N_V=basis_functions_V(rm,sm)
           uB=np.dot(N_V,u[icon_V[:,iel]]) 
           vB=np.dot(N_V,v[icon_V[:,iel]])
           swarm_x[ip]= xA + uB * dt
           swarm_y[ip]= yA + vB * dt
           swarm_u[ip]=uB
           swarm_v[ip]=vB
           swarm_x[ip]=min((1-eps2)*Lx,swarm_x[ip])
           swarm_y[ip]=min((1-eps2)*Ly,swarm_y[ip])
           swarm_x[ip]=max(eps2*Lx,swarm_x[ip])
           swarm_y[ip]=max(eps2*Ly,swarm_y[ip])
       #end for

    print("     -> swarm_x (m,M) %.6e %.6e " %(np.min(swarm_x),np.max(swarm_x)))
    print("     -> swarm_y (m,M) %.6e %.6e " %(np.min(swarm_y),np.max(swarm_y)))

    print("advect particles: %.3f s" % (clock.time()-start))

    ###########################################################################

    for ip in range(0,nparticle):
        if ip%11111==0: flagged_file.write("%d %e %e \n" %(time,swarm_x[ip],swarm_y[ip]))
    flagged_file.flush()

    ###########################################################################
    # deform mesh
    ###########################################################################
    start=clock.time()

    if use_ALE:
       for i in range(0,nn_V): 
           if surface[i]:
              y_V[i]+=v[i]*dt
           #end if
       #end for
       filename = 'surface_{:04d}.ascii'.format(istep)
       np.savetxt(filename,np.array([x_V[surface],y_V[surface]-Ly,v[surface]]).T,header='# x,y')
       ymin=np.min(y_V[surface])
       ymax=np.max(y_V[surface])
       stats_topo_file.write("%e %e %e %e\n" %(time,ymin-Ly,ymax-Ly,time/year)) ; stats_topo_file.flush()
       print("     -> topo (m,M) %.6e %.6e " %(ymin,ymax))

       print("deform mesh: %.3f s" % (clock.time()-start))

       for iel in range(0,nel):
           x_e[iel]=np.sum(x_V[icon_V[:,iel]])/9
           y_e[iel]=np.sum(y_V[icon_V[:,iel]])/9

    ###########################################################################
    # re-locate them
    # if ALE is used, we here assume that only the top row of elements is
    # accomodating the deformation (dy<hy) so that all particles above
    # (nely-1)*hy are definitely inside the top row of elements and their iely is nely-1
    ###########################################################################
    start=clock.time()

    if use_ALE:
       for ip in range(0,nparticle):
           ielx=int(m_x[ip]/hx)
           m_r[ip]=((m_x[ip]-x_V[icon_V[0,ielx]])/hx-0.5)*2.
           if ielx<0:
              exit("ielx<0")
           if ielx>nelx-1:
              exit("ielx>nelx-1")
           if m_r[ip]<-1:
              exit("r<-1")
           if m_r[ip]>1:
              print(m_x[ip],m_y[ip])
              exit("r>1")

           if m_y[ip]>(nely-1)*hy:
              iely=nely-1
              m_iel[ip]=iely*nelx+ielx
              r,s=compute_rs(m_x[ip],m_y[ip],m_iel[ip])
              m_s[ip]=s 
           else:
              iely=int(m_y[ip]/Ly*nely)
              m_iel[ip]=iely*nelx+ielx
              m_s[ip]=((m_y[ip]-y_V[icon_V[0,m_iel[ip]]])/hy-0.5)*2.
       #end for
    else:
       for ip in range(0,nparticle):
           ielx=int(swarm_x[ip]/hx)
           iely=int(swarm_y[ip]/hy)
           if debug:
              if ielx<0: exit("ielx<0")
              if ielx>nelx-1: exit("ielx>nelx-1")
              if iely<0: exit("iely<0")
              if iely>nely-1: exit("iely>nely-1")
           swarm_iel[ip]=iely*nelx+ielx
           swarm_r[ip]=((swarm_x[ip]-x_V[icon_V[0,swarm_iel[ip]]])/hx-0.5)*2.
           swarm_s[ip]=((swarm_y[ip]-y_V[icon_V[0,swarm_iel[ip]]])/hy-0.5)*2.
       #end for
    #end if

    print("     -> swarm_iel (m,M) %d %d " %(np.min(swarm_iel),np.max(swarm_iel)))
    print("     -> swarm_r   (m,M) %.6e %.6e " %(np.min(swarm_r),np.max(swarm_r)))
    print("     -> swarm_s   (m,M) %.6e %.6e " %(np.min(swarm_s),np.max(swarm_s)))

    print("locate particles: %.3f s" % (clock.time()-start))

    ###########################################################################
    # project onto elements 
    ###########################################################################
    start=clock.time()

    Z=np.zeros(nel,dtype=np.float64)  
    rho=np.zeros(nel,dtype=np.float64)  
    etaeff=np.zeros(nel,dtype=np.float64)  
    count=np.zeros(nel,dtype=np.float64)  
    tauxx=np.zeros(nel,dtype=np.float64)  
    tauyy=np.zeros(nel,dtype=np.float64)  
    tauxy=np.zeros(nel,dtype=np.float64)  

    for ip in range(0,nparticle):
        iel=swarm_iel[ip]
        rho[iel]+=swarm_rho[ip]
        Z[iel]+=swarm_Z[ip]
        etaeff[iel]+=swarm_etaeff[ip]
        tauxx[iel]+=swarm_tauxx[ip]
        tauyy[iel]+=swarm_tauyy[ip]
        tauxy[iel]+=swarm_tauxy[ip]
        count[iel]+=1

    Z/=count
    rho/=count
    etaeff/=count
    tauxx/=count
    tauyy/=count
    tauxy/=count

    if debug: np.savetxt('elemental_values.ascii',np.array([x_e,y_e,rho,Z,etaeff]).T,header='# x,y,rho,Z,eta_eff')

    print("     -> Z     (m,M) %.6e %.6e " %(np.min(Z),np.max(R)))
    print("     -> rho   (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))
    print("     -> etaeff(m,M) %.6e %.6e " %(np.min(etaeff),np.max(etaeff)))
    print("     -> tauxx (m,M) %.6e %.6e " %(np.min(tauxx),np.max(tauxx)))
    print("     -> tauyy (m,M) %.6e %.6e " %(np.min(tauyy),np.max(tauyy)))
    print("     -> tauxy (m,M) %.6e %.6e " %(np.min(tauxy),np.max(tauxy)))

    stats_etaeff_file.write("%e %e %e \n" %(time,np.min(etaeff),np.max(etaeff))) ;stats_etaeff_file.flush()
    stats_Z_file.write("%e %e %e \n" %(time,np.min(Z),np.max(Z))) ;stats_Z_file.flush()

    print("project particles onto nodes: %.3f s" % (clock.time()-start))

    ###########################################################################
    start=clock.time()

    Jxx=2*tauxx*wxy
    Jyy=-2*tauxy*wxy
    Jxy=(tauyy-tauxx)*wxy

    stats_Jxx_file.write("%e %e %e \n" %(time,np.min(Jxx),np.max(Jxx))) ; stats_Jxx_file.flush()
    stats_Jyy_file.write("%e %e %e \n" %(time,np.min(Jyy),np.max(Jyy))) ; stats_Jyy_file.flush()
    stats_Jxy_file.write("%e %e %e \n" %(time,np.min(Jxy),np.max(Jxy))) ; stats_Jxy_file.flush()

    print("     -> Jxx (m,M) %.6e %.6e " %(np.min(Jxx),np.max(Jxx)))
    print("     -> Jyy (m,M) %.6e %.6e " %(np.min(Jyy),np.max(Jyy)))
    print("     -> Jxy (m,M) %.6e %.6e " %(np.min(Jxy),np.max(Jxy)))

    print("compute nodal J: %.3f s" % (clock.time()-start))

    ###########################################################################

    tauxxmem=np.copy(tauxx)
    tauyymem=np.copy(tauyy)
    tauxymem=np.copy(tauxy)

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if istep%every==0:
       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
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
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       #q.tofile(vtufile, sep=" ", format="%.4e")
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       rho.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
       etaeff.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Z' Format='ascii'> \n")
       Z.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
       exx.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
       eyy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
       exy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='omega_xy' Format='ascii'> \n")
       wxy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       tauxx.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
       tauyy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
       tauxy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],
                                                           icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],
                                                           icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*9))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       #-------------------------------

       filename = 'particles_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for ip in range(0,nparticle):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[ip],swarm_y[ip],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for ip in range(0,nparticle):
           vtufile.write("%e %e %e \n" %(swarm_u[ip],swarm_v[ip],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
       for ip in range(0,nparticle):
           vtufile.write("%e %e %e \n" %(swarm_u[ip]/cm*year,swarm_v[ip]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'> \n")
       swarm_mat.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       swarm_tauxx.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
       swarm_tauyy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
       swarm_tauxy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
       swarm_r.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='s' Format='ascii'> \n")
       swarm_s.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
       swarm_etaeff.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for ip in range (0,nparticle):
           vtufile.write("%d\n" % ip)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for ip in range (0,nparticle):
           vtufile.write("%d \n" % (ip+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for ip in range (0,nparticle):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       #-------------------------------

       if False: # deactivated since all fields are elemental

          filename = 'qpts_{:04d}.vtu'.format(istep) 
          vtufile=open(filename,"w")
          vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
          vtufile.write("<UnstructuredGrid> \n")
          vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
          #####
          vtufile.write("<Points> \n")
          #--
          vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
          for iq in range(0,nq):
              vtufile.write("%10e %10e %10e \n" %(q_x[iq],q_y[iq],0.))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("</Points> \n")
          #--
          vtufile.write("<PointData Scalars='scalars'>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
          q_etaeff.tofile(vtufile, sep=" ", format="%.4e")
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='Z' Format='ascii'> \n")
          q_Z.tofile(vtufile, sep=" ", format="%.4e")
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
          q_rho.tofile(vtufile, sep=" ", format="%.4e")
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("</PointData>\n")
          #####
          vtufile.write("<Cells>\n")
          #--
          vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
          for iq in range (0,nq):
              vtufile.write("%d\n" % iq )
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
          for iq in range (0,nq):
              vtufile.write("%d \n" % (iq+1) )
          vtufile.write("</DataArray>\n")
          #--
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

    print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
