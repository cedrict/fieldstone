import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
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
    if benchmark==11 or benchmark==12:
       val=0 

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

    if benchmark==4:
       val=-9.81

    if benchmark==5:
       val=-g0

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

order=2
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
   nmarker_per_element=100
   every=1
   eta_ref=1e23
   pnormalise=True
   use_ALE=False

if benchmark==2: # slab Gerya book
   nelx=50
   nely=50
   Lx=1000e3 
   Ly=1000e3 
   dt=200*year
   rho1=4000
   rho2=1
   eta1=1e27
   eta2=1e21
   mu1=1e10
   mu2=1e20
   nstep=5 # 251
   nmarker_per_element=64
   every=1
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   eta_ref=1e23
   pnormalise=True
   use_ALE=False

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
   nmarker_per_element=100
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
   nmarker_per_element=50
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
   nmarker_per_element=100
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

nmarker=nel*nmarker_per_element
   
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
stats_topo_file=open('stats_topo.ascii',"w")
stats_etaeff_file=open('stats_etaeff.ascii',"w")
stats_Jxx_file=open('stats_Jxx.ascii',"w")
stats_Jyy_file=open('stats_Jyy.ascii',"w")
stats_Jxy_file=open('stats_Jxy.ascii',"w")
stats_m_tauxx_file=open('stats_m_tauxx.ascii',"w")
stats_m_tauyy_file=open('stats_m_tauyy.ascii',"w")
stats_m_tauxy_file=open('stats_m_tauxy.ascii',"w")

###############################################################################

print("benchmark=",benchmark)
print("nelx=",nelx)
print("nely=",nely)
print("nel=",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)

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

#connectivity array for plotting
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
# markers layout
# randomly generated, not too close from domain sides
###############################################################################
start=clock.time()

m_x=np.zeros(nmarker,dtype=np.float64)  
m_y=np.zeros(nmarker,dtype=np.float64)  
m_u=np.zeros(nmarker,dtype=np.float64)  
m_v=np.zeros(nmarker,dtype=np.float64)  
m_Z=np.zeros(nmarker,dtype=np.float64)  
m_etaeff=np.zeros(nmarker,dtype=np.float64)  
m_rho=np.zeros(nmarker,dtype=np.float64)  
m_iel=np.zeros(nmarker,dtype=np.int32)  
m_r=np.zeros(nmarker,dtype=np.float64)  
m_s=np.zeros(nmarker,dtype=np.float64)  
m_tauxx=np.zeros(nmarker,dtype=np.float64)  
m_tauyy=np.zeros(nmarker,dtype=np.float64)  
m_tauxy=np.zeros(nmarker,dtype=np.float64)  
m_mat=np.zeros(nmarker,dtype=np.int32)  

counter=0
for iel in range(0,nel):
    for im in range(0,nmarker_per_element):
        aa=random.uniform(0,+1)
        bb=random.uniform(0,+1)
        #aa=0.5
        #bb=0.5
        m_x[counter]=x_V[icon_V[0,iel]]+aa*hx
        m_y[counter]=y_V[icon_V[0,iel]]+bb*hy 
        m_x[counter]=min((1-eps2)*Lx,m_x[counter])
        m_y[counter]=min((1-eps2)*Ly,m_y[counter])
        m_x[counter]=max(eps2*Lx,m_x[counter])
        m_y[counter]=max(eps2*Ly,m_y[counter])
        counter+=1
    #end for
#end for

match benchmark:
 case 11 | 12 | 5:
   for im in range(0,nmarker):
       m_rho[im]=rho1
       m_etaeff[im]=etaeff1
       m_Z[im]=Z1
       m_mat[im]=1
 case 2:
   for im in range(0,nmarker):
       if m_x[im]<=800e3 and np.abs(m_y[im]-Ly/2)<=300e3:
          m_rho[im]=rho1
          m_etaeff[im]=etaeff1
          m_Z[im]=Z1
          m_mat[im]=3
       else:
          m_rho[im]=rho2
          m_etaeff[im]=etaeff2
          m_Z[im]=Z2
       #end if
   #end for
 case 3:
   for im in range(0,nmarker):
       m_rho[im]=rho1
       m_etaeff[im]=etaeff1
       m_Z[im]=Z1
       m_mat[im]=1
       if m_y[im]>2200 and m_y[im]<2800 and m_x[im]<4500: 
          m_rho[im]=rho2
          m_etaeff[im]=etaeff2
          m_Z[im]=Z2
          m_mat[im]=4
       #end if
       if (m_x[im]-4500)**2+(m_y[im]-Ly/2.)**2<300**2:
          m_rho[im]=rho2
          m_etaeff[im]=etaeff2
          m_Z[im]=Z2
          m_mat[im]=4
       #end if
   #end for
 case 4:
   for im in range(0,nmarker):
       if m_y[im]>Ly-5e3:
          m_rho[im]=rho1
          m_etaeff[im]=etaeff1
          m_Z[im]=Z1
          m_mat[im]=1
       else:
          m_rho[im]=rho3
          m_etaeff[im]=etaeff3
          m_Z[im]=Z3
          m_mat[im]=7
       #end if
       if m_x[im]>Lx-5000 and  m_y[im]<Ly-5e3 and  m_y[im]>7.5e3:
          m_rho[im]=rho2
          m_etaeff[im]=etaeff2
          m_Z[im]=Z2
          m_mat[im]=4
       #end if
   #end for

if debug: np.savetxt('markers_init.ascii',np.array([m_x,m_y,m_rho,m_Z,m_etaeff]).T)

print("material layout: %.3f s" % (clock.time()-start))

###############################################################################
# marker paint
###############################################################################
start=clock.time()

if benchmark==11 or benchmark==12:
   for i in [0,2,4]:
       dx=Lx/5
       for im in range (0,nmarker):
           if m_x[im]>i*dx and m_x[im]<(i+1)*dx:
              m_mat[im]+=1
   for i in [0,2,4]:
       dy=Ly/5
       for im in range (0,nmarker):
           if m_y[im]>i*dy and m_y[im]<(i+1)*dy:
              m_mat[im]+=1

if benchmark==2 or benchmark==5:
   for i in [0,2,4,6,8,10,12,14,16,18]:
       dx=Lx/20
       for im in range (0,nmarker):
           if m_x[im]>i*dx and m_x[im]<(i+1)*dx:
              m_mat[im]+=1
   for i in [0,2,4,6,8,10,12,14,16,18]:
       dy=Ly/20
       for im in range (0,nmarker):
           if m_y[im]>i*dy and m_y[im]<(i+1)*dy:
              m_mat[im]+=1

if benchmark==3:
   for i in [0,2,4]:
       dx=Lx/5
       for im in range (0,nmarker):
           if m_x[im]>i*dx and m_x[im]<(i+1)*dx:
              m_mat[im]+=1
   for i in [0,2,4,6,8,10,12,14,16,18,20,22,24]:
       dy=Ly/25
       for im in range (0,nmarker):
           if m_y[im]>i*dy and m_y[im]<(i+1)*dy:
              m_mat[im]+=1

if benchmark==4:
   for i in [0,2,4,6,8,10]:
       dy=2.5e3
       for im in range (0,nmarker):
           if m_y[im]>i*dy and m_y[im]<(i+1)*dy:
              m_mat[im]+=1

print("paint markers: %.3f s" % (clock.time()-start))
 
#################################################################
# locate markers
#################################################################
start=clock.time()

for im in range(0,nmarker):
    ielx=int(m_x[im]/Lx*nelx)
    iely=int(m_y[im]/Ly*nely)
    m_iel[im]=iely*nelx+ielx
    m_r[im]=((m_x[im]-x_V[icon_V[0,m_iel[im]]])/hx-0.5)*2.
    m_s[im]=((m_y[im]-y_V[icon_V[0,m_iel[im]]])/hy-0.5)*2.

print("     -> m_iel (m,M) %d %d " %(np.min(m_iel),np.max(m_iel)))
print("     -> m_iel (m,M) %e %e " %(np.min(m_r),np.max(m_r)))
print("     -> m_iel (m,M) %e %e " %(np.min(m_s),np.max(m_s)))

print("locate markers: %.3f s" % (clock.time()-start))

#################################################################
# project markers onto Vnodes 
#################################################################
start=clock.time()

Z=np.zeros(nn_V,dtype=np.float64)  
rho=np.zeros(nn_V,dtype=np.float64)  
etaeff=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for im in range(0,nmarker):
    BP=BernsteinPolynomial(m_r[im],m_s[im])
    for i in range(0,m_V):
        inode=icon_V[i,m_iel[im]]
        rho[inode]+=m_rho[im]*BP[i]
        Z[inode]+=m_Z[im]*BP[i]
        etaeff[inode]+=m_etaeff[im]*BP[i]
        count[inode]+=BP[i]
    #end for
#end for

Z/=count
rho/=count
etaeff/=count

if debug: np.savetxt('nodes.ascii',np.array([x_V,y_V,rho,Z,etaeff]).T,header='# x,y')

print("project Z, rho, etaeff onto V nodes: %.3f s" % (clock.time()-start))

#################################################################
# initialise nodal fields 
#################################################################

Jxx =np.zeros(nn_V,dtype=np.float64)  
Jyy =np.zeros(nn_V,dtype=np.float64)  
Jxy =np.zeros(nn_V,dtype=np.float64)  
tauxx =np.zeros(nn_V,dtype=np.float64)  
tauyy =np.zeros(nn_V,dtype=np.float64)  
tauxy =np.zeros(nn_V,dtype=np.float64)  
tauxxmem =np.zeros(nn_V,dtype=np.float64)  
tauyymem =np.zeros(nn_V,dtype=np.float64)  
tauxymem =np.zeros(nn_V,dtype=np.float64)  

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

    K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
    G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(Nfem_V,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(Nfem_P,dtype=np.float64)         # right hand side h 
    constr= np.zeros(Nfem_P,dtype=np.float64)         # constraint matrix/vector

    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix 
    N_mat   = np.zeros((3,m_P),dtype=np.float64) # matrix  
    BBB     = np.zeros(m_V,dtype=np.float64)           # shape functions V
    jcb=np.zeros((2,2),dtype=np.float64)

    counterq=0
    for iel in range(0,nel):

        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)
        N_N_NP= np.zeros(m_P,dtype=np.float64)   

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


                # compute dNdx & dNdy
                if use_ss:
                   xq=np.dot(N_V,x_V[icon_V[:,iel]])
                   yq=np.dot(N_V,y_V[icon_V[:,iel]])
                   Zq=np.dot(N_V,Z[icon_V[:,iel]])
                   rhoq=np.dot(N_V,rho[icon_V[:,iel]])
                   Jxxq=np.dot(N_V,Jxx[icon_V[:,iel]])
                   Jyyq=np.dot(N_V,Jyy[icon_V[:,iel]])
                   Jxyq=np.dot(N_V,Jxy[icon_V[:,iel]])
                   tauxxq=np.dot(N_V,tauxx[icon_V[:,iel]])
                   tauyyq=np.dot(N_V,tauyy[icon_V[:,iel]])
                   tauxyq=np.dot(N_V,tauxy[icon_V[:,iel]])
                   etaeffq=np.dot(N_V,etaeff[icon_V[:,iel]])
                else:
                   BP=BernsteinPolynomial(rq,sq)
                   xq=np.dot(BP,x_V[icon_V[:,iel]])
                   yq=np.dot(BP,y_V[icon_V[:,iel]])
                   Zq=np.dot(BP,Z[icon_V[:,iel]])
                   rhoq=np.dot(BP,rho[icon_V[:,iel]])
                   Jxxq=np.dot(BP,Jxx[icon_V[:,iel]])
                   Jyyq=np.dot(BP,Jyy[icon_V[:,iel]])
                   Jxyq=np.dot(BP,Jxy[icon_V[:,iel]])
                   tauxxq=np.dot(BP,tauxx[icon_V[:,iel]])
                   tauyyq=np.dot(BP,tauyy[icon_V[:,iel]])
                   tauxyq=np.dot(BP,tauxy[icon_V[:,iel]])
                   etaeffq=np.dot(BP,etaeff[icon_V[:,iel]])
                #end if

                q_x[counterq]=xq
                q_y[counterq]=yq
                q_Z[counterq]=Zq
                q_rho[counterq]=rhoq
                q_etaeff[counterq]=etaeffq

                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                #if etaeffq<0:
                #   exit("etaeffq<0")
                #if rhoq<0:
                #   exit("rhoq<0")

                #qpts_file.write("%e %e %e %e %e %e %e %e \n"\
                #                 %(xq,yq,rhoq,etaeffq,Zq,tauxxq,tauyyq,tauxyq))

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                K_el+=B.T.dot(C.dot(B))*etaeffq*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i  ]+=N_V[i]*JxWq*rhoq*gx
                    f_el[ndof_V*i+1]+=N_V[i]*JxWq*rhoq*gy(time)

                #compute elastic rhs
                R[0]=Zq*(tauxxq+dt*Jxxq)
                R[1]=Zq*(tauyyq+dt*Jyyq)
                R[2]=Zq*(tauxyq+dt*Jxyq)
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
        h_el*=scaling_coeff

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

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    G_mat[m1,m2]+=G_el[ikk,jkk]
                #end for 
                f_rhs[m1]+=f_el[ikk]
            #end for 
        #end for 
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            h_rhs[m2]+=h_el[k2]
            constr[m2]+=N_N_NP[k2]
        #end for 

    #end for iel

    print("     -> K_mat (m,M) %.3e %.3e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G_mat (m,M) %.3e %.3e " %(np.min(G_mat),np.max(G_mat)))

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ###########################################################################
    # assemble K, G, GT, f, h into A and rhs
    ###########################################################################
    start=clock.time()

    if pnormalise:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
       a_mat[0:Nfem_V,0:Nfem_V]=K_mat
       a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
       a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
       a_mat[Nfem,Nfem_V:Nfem]=constr
       a_mat[Nfem_V:Nfem,Nfem]=constr
    else:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
       rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
       a_mat[0:Nfem_V,0:Nfem_V]=K_mat
       a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
       a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
    #end if

    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

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

    if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

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
    # compute nodal velocity gradient 
    ###########################################################################
    start=clock.time()
    
    count = np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)
    Lxx = np.zeros(nn_V,dtype=np.float64)  
    Lxy = np.zeros(nn_V,dtype=np.float64)  
    Lyx = np.zeros(nn_V,dtype=np.float64)  
    Lyy = np.zeros(nn_V,dtype=np.float64)  

    #u[:]=xV[:]**2
    #v[:]=yV[:]**2

    if computeLmethod==1:
        for iel in range(0,nel):
            for i in range(0,m_V):
                inode=icon_V[i,iel]
                rq=r_V[i]
                sq=s_V[i]
                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                Lxx[inode]+=np.dot(dNdx_V,u[icon_V[:,iel]])
                Lxy[inode]+=np.dot(dNdx_V,v[icon_V[:,iel]])
                Lyx[inode]+=np.dot(dNdy_V,u[icon_V[:,iel]])
                Lyy[inode]+=np.dot(dNdy_V,v[icon_V[:,iel]])
                q[inode]+=np.dot(N_P,p[icon_P[:,iel]])
                count[inode]+=1
            #end for
        #end for
        Lxx/=count
        Lxy/=count
        Lyx/=count
        Lyy/=count
        q/=count
    #end if

    if computeLmethod==2:
        for iel in range(0,nel):
            rq=0.0
            sq=0.0
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            L_xx=np.dot(dNdx_V,u[icon_V[:,iel]])
            L_xy=np.dot(dNdx_V,v[icon_V[:,iel]])
            L_yx=np.dot(dNdy_V,u[icon_V[:,iel]])
            L_yy=np.dot(dNdy_V,v[icon_V[:,iel]])
            qq=np.dot(N_P,p[icon_P[:,iel]])
            for i in range(0,m_V):
                inode=icon_V[i,iel]
                Lxx[inode]+=L_xx
                Lxy[inode]+=L_xy
                Lyx[inode]+=L_yx
                Lyy[inode]+=L_yy
                q[inode]+=qq
                count[inode]+=1
            #end for
        #end for
        Lxx/=count
        Lxy/=count
        Lyx/=count
        Lyy/=count
        q/=count
    #end if

    print("     -> Lxx (m,M) %.3e %.3e " %(np.min(Lxx),np.max(Lxx)))
    print("     -> Lxy (m,M) %.3e %.3e " %(np.min(Lxy),np.max(Lxy)))
    print("     -> Lyx (m,M) %.3e %.3e " %(np.min(Lyx),np.max(Lyx)))
    print("     -> Lyy (m,M) %.3e %.3e " %(np.min(Lyy),np.max(Lyy)))

    if debug:
       np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
       np.savetxt('velgradient.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T,header='# x,y,Lxx,Lyy,Lxy,Lyx')

    print("compute nodal p and L: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal fields
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
    # interpolate dev stress difference (increment) onto markers
    # and add it to the existing value on the markers
    ###########################################################################
    start=clock.time()

    for im in range(0,nmarker):
        rm=m_r[im]
        sm=m_s[im]
        iel=m_iel[im]
        N_V=basis_functions_V(rm,sm) ## SHOULD I Bernstein?!
        m_tauxx[im]+=np.dot(N_V,tauxx[icon_V[:,iel]]-tauxxmem[icon_V[:,iel]])
        m_tauyy[im]+=np.dot(N_V,tauyy[icon_V[:,iel]]-tauyymem[icon_V[:,iel]])
        m_tauxy[im]+=np.dot(N_V,tauxy[icon_V[:,iel]]-tauxymem[icon_V[:,iel]])

    print("     -> m_tauxx (m,M) %.6e %.6e " %(np.min(m_tauxx),np.max(m_tauxx)))
    print("     -> m_tauyy (m,M) %.6e %.6e " %(np.min(m_tauyy),np.max(m_tauyy)))
    print("     -> m_tauxy (m,M) %.6e %.6e " %(np.min(m_tauxy),np.max(m_tauxy)))

    stats_m_tauxx_file.write("%e %e %e \n" %(time,np.min(m_tauxx),np.max(m_tauxx))) ;stats_m_tauxx_file.flush()
    stats_m_tauyy_file.write("%e %e %e \n" %(time,np.min(m_tauyy),np.max(m_tauyy))) ;stats_m_tauyy_file.flush()
    stats_m_tauxy_file.write("%e %e %e \n" %(time,np.min(m_tauxy),np.max(m_tauxy))) ;stats_m_tauxy_file.flush()

    print("interp. diff stress onto markers: %.3f s" % (clock.time()-start))

    ###########################################################################
    # advect markers
    # this is a very simple  
    ###########################################################################
    start=clock.time()

    for im in range(0,nmarker):
        rm=m_r[im]
        sm=m_s[im]
        iel=m_iel[im]
        N_V=basis_functions_V(rm,sm) ## SHOULD I Bernstein?!
        m_u[im]=np.dot(N_V,u[icon_V[:,iel]]) 
        m_v[im]=np.dot(N_V,v[icon_V[:,iel]])
        m_x[im]+=m_u[im]*dt 
        m_y[im]+=m_v[im]*dt 
        if benchmark==11 or benchmark==12:
           m_x[im]=min((1-eps2)*Lx,m_x[im])
           m_y[im]=min((1-eps2)*Ly,m_y[im])
           m_x[im]=max(eps2*Lx,m_x[im])
           m_y[im]=max(eps2*Ly,m_y[im])
    #end for

    print("     -> m_x (m,M) %.6e %.6e " %(np.min(m_x),np.max(m_x)))
    print("     -> m_y (m,M) %.6e %.6e " %(np.min(m_y),np.max(m_y)))

    print("advect markers: %.3f s" % (clock.time()-start))

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

    ###########################################################################
    # re-locate them
    # if ALE is used, we here assume that only the top row of elements is
    # accomodating the deformation (dy<hy) so that all markers above
    # (nely-1)*hy are definitely inside the top row of elements and their iely is nely-1
    ###########################################################################
    start=clock.time()

    if use_ALE:
       for im in range(0,nmarker):
           ielx=int(m_x[im]/hx)
           m_r[im]=((m_x[im]-x_V[icon_V[0,ielx]])/hx-0.5)*2.
           if ielx<0:
              exit("ielx<0")
           if ielx>nelx-1:
              exit("ielx>nelx-1")
           if m_r[im]<-1:
              exit("r<-1")
           if m_r[im]>1:
              print(m_x[im],m_y[im])
              exit("r>1")

           if m_y[im]>(nely-1)*hy:
              iely=nely-1
              m_iel[im]=iely*nelx+ielx
              r,s=compute_rs(m_x[im],m_y[im],m_iel[im])
              m_s[im]=s 
           else:
              iely=int(m_y[im]/Ly*nely)
              m_iel[im]=iely*nelx+ielx
              m_s[im]=((m_y[im]-y_V[icon_V[0,m_iel[im]]])/hy-0.5)*2.
       #end for
    else:
       for im in range(0,nmarker):
           ielx=int(m_x[im]/hx)
           iely=int(m_y[im]/hy)
           #if ielx<0:
           #   exit("ielx<0")
           #if ielx>nelx-1:
           #   exit("ielx>nelx-1")
           #if iely<0:
           #   exit("iely<0")
           #if iely>nely-1:
           #   exit("iely>nely-1")
           m_iel[im]=iely*nelx+ielx
           m_r[im]=((m_x[im]-x_V[icon_V[0,m_iel[im]]])/hx-0.5)*2.
           m_s[im]=((m_y[im]-y_V[icon_V[0,m_iel[im]]])/hy-0.5)*2.
       #end for
    #end if

    print("     -> m_iel (m,M) %d %d " %(np.min(m_iel),np.max(m_iel)))
    print("     -> m_r   (m,M) %.6e %.6e " %(np.min(m_r),np.max(m_r)))
    print("     -> m_s   (m,M) %.6e %.6e " %(np.min(m_s),np.max(m_s)))

    print("locate markers: %.3f s" % (clock.time()-start))

    ###########################################################################
    # project onto nodes
    ###########################################################################
    start=clock.time()

    Z=np.zeros(nn_V,dtype=np.float64)  
    rho=np.zeros(nn_V,dtype=np.float64)  
    etaeff=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.float64)  
    tauxx=np.zeros(nn_V,dtype=np.float64)  
    tauyy=np.zeros(nn_V,dtype=np.float64)  
    tauxy=np.zeros(nn_V,dtype=np.float64)  

    for im in range(0,nmarker):
        BP=BernsteinPolynomial(m_r[im],m_s[im])
        for i in range(0,m_V):
            inode=icon_V[i,m_iel[im]]
            Z[inode]     +=m_Z[im]     *BP[i]
            rho[inode]   +=m_rho[im]   *BP[i]
            etaeff[inode]+=m_etaeff[im]*BP[i]
            tauxx[inode] +=m_tauxx[im] *BP[i]
            tauyy[inode] +=m_tauyy[im] *BP[i]
            tauxy[inode] +=m_tauxy[im] *BP[i]
            count[inode] +=             BP[i]
        #end for
    #end for

    Z/=count
    rho/=count
    etaeff/=count
    tauxx/=count
    tauyy/=count
    tauxy/=count

    print("     -> Z     (m,M) %.6e %.6e " %(np.min(Z),np.max(R)))
    print("     -> rho   (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))
    print("     -> etaeff(m,M) %.6e %.6e " %(np.min(etaeff),np.max(etaeff)))
    print("     -> tauxx (m,M) %.6e %.6e " %(np.min(tauxx),np.max(tauxx)))
    print("     -> tauyy (m,M) %.6e %.6e " %(np.min(tauyy),np.max(tauyy)))
    print("     -> tauxy (m,M) %.6e %.6e " %(np.min(tauxy),np.max(tauxy)))

    stats_etaeff_file.write("%e %e %e \n" %(time,np.min(etaeff),np.max(etaeff))) ;stats_etaeff_file.flush()
    stats_Z_file.write("%e %e %e \n" %(time,np.min(Z),np.max(Z))) ;stats_Z_file.flush()

    print("project markers onto nodes: %.3f s" % (clock.time()-start))

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
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel2))
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
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       q.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
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
       vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e \n" %(  np.sqrt((exx[i]**2+eyy[i]**2)+2*exy[i]**2) ))
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

       #-------------------------------

       filename = 'markers_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(m_x[im],m_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(m_u[im],m_v[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(m_u[im]/cm*year,m_v[im]/cm*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'> \n")
       m_mat.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       m_tauxx.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
       m_tauyy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
       m_tauxy.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
       m_r.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='s' Format='ascii'> \n")
       m_s.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
       m_etaeff.tofile(vtufile, sep=" ", format="%.4e")
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
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
