import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as timing
import random

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

def BB(rq,sq):
    BB_0= 0.25*(1-rq)**2 * 0.25*(1-sq)**2
    BB_1= 0.25*(1+rq)**2 * 0.25*(1-sq)**2
    BB_2= 0.25*(1+rq)**2 * 0.25*(1+sq)**2
    BB_3= 0.25*(1-rq)**2 * 0.25*(1+sq)**2
    BB_4= 0.5*(1-rq**2)  * 0.25*(1-sq)**2
    BB_5= 0.25*(1+rq)**2 * 0.5*(1-sq**2)
    BB_6= 0.5*(1-rq**2)  * 0.25*(1+sq)**2
    BB_7= 0.25*(1-rq)**2 * 0.5*(1-sq**2)
    BB_8= 0.5*(1-rq**2)  * 0.5*(1-sq**2)
    return BB_0,BB_1,BB_2,BB_3,BB_4,BB_5,BB_6,BB_7,BB_8

def gy(time):
    if benchmark==1:
       exit('bench 1 grav')

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

    return val

#------------------------------------------------------------------------------

order=2
cm=0.01
year=365.*24.*3600.
sqrt2=np.sqrt(2)
eps=1.e-10
eps2=1.e-6

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 50
   nely = 50
   visu = 1

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]


gx=0.

benchmark=4

if benchmark==1:
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
   etaeff2=0
   nstep=200
   eta_ref=1e23
   pnormalise=True

if benchmark==2:
   nelx=50
   nely=50
   Lx=1000e3  # horizontal extent of the domain 
   Ly=1000e3  # vertical extent of the domain 
   dt=200*year
   rho1=4000
   rho2=1
   eta1=1e27
   eta2=1e21
   mu1=1e10
   mu2=1e20
   nstep=251
   nmarker_per_element=64
   every=5
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   eta_ref=1e23
   pnormalise=True

if benchmark==3:
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

if benchmark==4:
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
   nstep=1
   nmarker_per_element=100
   every=1
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   etaeff3=eta3*dt/(dt+eta3/mu3)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   Z3=etaeff3/mu3/dt
   eta_ref=1e23
   pnormalise=False

#1: nodal average
#2: c->n
computeLmethod=1

nnx=2*nelx+1                  # number of Vnodes, x direction
nny=2*nely+1                  # number of Vnodes, y direction
NV=nnx*nny                    # number of Vnodes
nel=nelx*nely                 # number of elements
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
NfemT=NV                      # number of field dofs 
hx=Lx/nelx
hy=Ly/nely
nq=nel*nqperdim**ndim

scaling_coeff=eta_ref/Ly
   
rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

alpha=0.5

time=0.

nmarker=nel*nmarker_per_element
                
use_ss=False

#################################################################

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
stats_etaeff_file=open('stats_etaeff.ascii',"w")
stats_Jxx_file=open('stats_Jxx.ascii',"w")
stats_Jyy_file=open('stats_Jyy.ascii',"w")
stats_Jxy_file=open('stats_Jxy.ascii',"w")
stats_m_tauxx_file=open('stats_m_tauxx.ascii',"w")
stats_m_tauyy_file=open('stats_m_tauyy.ascii',"w")
stats_m_tauxy_file=open('stats_m_tauxy.ascii',"w")

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)

if benchmark==1:
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

#################################################################
# grid point setup
#################################################################
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

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid points: %.3f s" % (timing.time() - start))

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

iconV=np.zeros((mV,nel),dtype=np.int16)
iconP=np.zeros((mP,nel),dtype=np.int16)

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

#connectivity array for plotting
nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int16)
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

print("connectivity: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if benchmark==1:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -1*cm/year
       #end if
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +1*cm/year
       #end if
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = +1*cm/year
       #end if
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -1*cm/year
       #end if
   #end for

if benchmark==2 or benchmark==3:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #end if
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
   #end for

if benchmark==4:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# markers layout
# randomly generated, not too close from domain sides
#################################################################
start = timing.time()

m_x=np.zeros(nmarker,dtype=np.float64)  
m_y=np.zeros(nmarker,dtype=np.float64)  
m_u=np.zeros(nmarker,dtype=np.float64)  
m_v=np.zeros(nmarker,dtype=np.float64)  
m_Z=np.zeros(nmarker,dtype=np.float64)  
m_etaeff=np.zeros(nmarker,dtype=np.float64)  
m_rho=np.zeros(nmarker,dtype=np.float64)  
m_iel=np.zeros(nmarker,dtype=np.int16)  
m_r=np.zeros(nmarker,dtype=np.float64)  
m_s=np.zeros(nmarker,dtype=np.float64)  
m_tauxx=np.zeros(nmarker,dtype=np.float64)  
m_tauyy=np.zeros(nmarker,dtype=np.float64)  
m_tauxy=np.zeros(nmarker,dtype=np.float64)  
m_mat=np.zeros(nmarker,dtype=np.int16)  

counter=0
for iel in range(0,nel):
    for im in range(0,nmarker_per_element):
        eta=random.uniform(0,+1)
        xi=random.uniform(0,+1)
        m_x[counter]=xV[iconV[0,iel]]+eta*hx
        m_y[counter]=yV[iconV[0,iel]]+xi *hy 
        m_x[counter]=min((1-eps2)*Lx,m_x[counter])
        m_y[counter]=min((1-eps2)*Ly,m_y[counter])
        m_x[counter]=max(eps2*Lx,m_x[counter])
        m_y[counter]=max(eps2*Ly,m_y[counter])
        counter+=1
    #end for
#end for


#if benchmark==1:

if benchmark==2:
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

if benchmark==3:
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

if benchmark==4:
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


#np.savetxt('markers_init.ascii',np.array([m_x,m_y,m_rho,m_Z,m_etaeff]).T,header='# x,y')

print("material layout: %.3f s" % (timing.time() - start))

#################################################################
# marker paint
#################################################################

if benchmark==2:
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



 
#################################################################
# locate markers
#################################################################
start = timing.time()

for im in range(0,nmarker):
    ielx=int(m_x[im]/Lx*nelx)
    iely=int(m_y[im]/Ly*nely)
    m_iel[im]=iely*nelx+ielx
    m_r[im]=( (m_x[im]-xV[iconV[0,m_iel[im]]])/hx-0.5)*2.
    m_s[im]=( (m_y[im]-yV[iconV[0,m_iel[im]]])/hy-0.5)*2.
#end for

print("     -> m_iel (m,M) %d %d " %(np.min(m_iel),np.max(m_iel)))
print("     -> m_iel (m,M) %e %e " %(np.min(m_r),np.max(m_r)))
print("     -> m_iel (m,M) %e %e " %(np.min(m_s),np.max(m_s)))

print("locate markers: %.3f s" % (timing.time() - start))

#################################################################
# project markers onto Vnodes 
#################################################################

Z     =np.zeros(NV,dtype=np.float64)  
rho   =np.zeros(NV,dtype=np.float64)  
etaeff=np.zeros(NV,dtype=np.float64)  
count =np.zeros(NV,dtype=np.float64)  
BBB   = np.zeros(mV,dtype=np.float64) 

for im in range(0,nmarker):
    rm=m_r[im]
    sm=m_s[im]
    BBB[0:9]=BB(rm,sm)
    for i in range(0,mV):
        inode=iconV[i,m_iel[im]]
        rho[inode]+=m_rho[im]*BBB[i]
        Z[inode]+=m_Z[im]*BBB[i]
        etaeff[inode]+=m_etaeff[im]*BBB[i]
        count[inode]+=BBB[i]
    #end for
#end for

Z/=count
rho/=count
etaeff/=count

#np.savetxt('nodes.ascii',np.array([xV,yV,rho,Z,etaeff]).T,header='# x,y')

#################################################################
# initialise nodal fields 
#################################################################

Jxx =np.zeros(NV,dtype=np.float64)  
Jyy =np.zeros(NV,dtype=np.float64)  
Jxy =np.zeros(NV,dtype=np.float64)  
tauxx =np.zeros(NV,dtype=np.float64)  
tauyy =np.zeros(NV,dtype=np.float64)  
tauxy =np.zeros(NV,dtype=np.float64)  
tauxxmem =np.zeros(NV,dtype=np.float64)  
tauyymem =np.zeros(NV,dtype=np.float64)  
tauxymem =np.zeros(NV,dtype=np.float64)  
exx = np.zeros(NV,dtype=np.float64)  
eyy = np.zeros(NV,dtype=np.float64)  
exy = np.zeros(NV,dtype=np.float64)  
wxy = np.zeros(NV,dtype=np.float64)  

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================

q_x      = np.zeros(nq,dtype=np.float64)    
q_y      = np.zeros(nq,dtype=np.float64)   
q_Z      = np.zeros(nq,dtype=np.float64)   
q_rho    = np.zeros(nq,dtype=np.float64)   
q_etaeff = np.zeros(nq,dtype=np.float64)   
u = np.zeros(NV,dtype=np.float64)          # x-component velocity
v = np.zeros(NV,dtype=np.float64)          # y-component velocity
R = np.zeros(3,dtype=np.float64)           # shape functions V
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 


for istep in range(0,nstep):
    print("----------------------------------------")
    print("istep= ",istep,'/',nstep-1)
    print("----------------------------------------")

    #filename = 'quadrature_points_values_{:04d}.ascii'.format(istep)
    #qpts_file=open(filename,"w")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

    b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdy = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    BBB     = np.zeros(mV,dtype=np.float64)           # shape functions V

    counterq=0
    for iel in range(0,nel):

        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        N_N_NP= np.zeros(mP*ndofP,dtype=np.float64)   

        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for 
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                if use_ss:
                   xq=np.sum(NNNV[:]*xV[iconV[:,iel]])
                   yq=np.sum(NNNV[:]*yV[iconV[:,iel]])
                   Zq=np.sum(NNNV[:]*Z[iconV[:,iel]])
                   rhoq=np.sum(NNNV[:]*rho[iconV[:,iel]])
                   Jxxq=np.sum(NNNV[:]*Jxx[iconV[:,iel]])
                   Jyyq=np.sum(NNNV[:]*Jyy[iconV[:,iel]])
                   Jxyq=np.sum(NNNV[:]*Jxy[iconV[:,iel]])
                   tauxxq=np.sum(NNNV[:]*tauxx[iconV[:,iel]])
                   tauyyq=np.sum(NNNV[:]*tauyy[iconV[:,iel]])
                   tauxyq=np.sum(NNNV[:]*tauxy[iconV[:,iel]])
                   etaeffq=np.sum(NNNV[:]*etaeff[iconV[:,iel]])
                else:
                   BBB[0:9]=BB(rq,sq)
                   xq=np.sum(BBB[:]*xV[iconV[:,iel]])
                   yq=np.sum(BBB[:]*yV[iconV[:,iel]])
                   Zq=np.sum(BBB[:]*Z[iconV[:,iel]])
                   rhoq=np.sum(BBB[:]*rho[iconV[:,iel]])
                   Jxxq=np.sum(BBB[:]*Jxx[iconV[:,iel]])
                   Jyyq=np.sum(BBB[:]*Jyy[iconV[:,iel]])
                   Jxyq=np.sum(BBB[:]*Jxy[iconV[:,iel]])
                   tauxxq=np.sum(BBB[:]*tauxx[iconV[:,iel]])
                   tauyyq=np.sum(BBB[:]*tauyy[iconV[:,iel]])
                   tauxyq=np.sum(BBB[:]*tauxy[iconV[:,iel]])
                   etaeffq=np.sum(BBB[:]*etaeff[iconV[:,iel]])
                #end if

                q_x[counterq]=xq
                q_y[counterq]=yq
                q_Z[counterq]=Zq
                q_rho[counterq]=rhoq
                q_etaeff[counterq]=etaeffq

                # compute dNdx & dNdy
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for 

                #if etaeffq<0:
                #   exit("etaeffq<0")
                #if rhoq<0:
                #   exit("rhoq<0")

                #qpts_file.write("%e %e %e %e %e %e %e %e \n"\
                #                 %(xq,yq,rhoq,etaeffq,Zq,tauxxq,tauyyq,tauxyq))

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.      ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]
                #end for 

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaeffq*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rhoq*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq*gy(time)
                #end for 

                #compute elastic rhs
                R[0]=Zq*(tauxxq+dt*Jxxq)
                R[1]=Zq*(tauyyq+dt*Jyyq)
                R[2]=Zq*(tauxyq+dt*Jxyq)
                f_el-=b_mat.T.dot(R)*weightq*jcob

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.
                #end for 

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                N_N_NP[:]+=NNNP[:]*jcob*weightq

                counterq+=1
            #end for jq
        #end for iq

        G_el*=scaling_coeff
        h_el*=scaling_coeff

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
                   #end for 
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0
                #end if 
            #end for 
        #end for 

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    G_mat[m1,m2]+=G_el[ikk,jkk]
                #end for 
                f_rhs[m1]+=f_el[ikk]
            #end for 
        #end for 
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
            constr[m2]+=N_N_NP[k2]
        #end for 

    #end for iel

    print("     -> K_mat (m,M) %.3e %.3e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G_mat (m,M) %.3e %.3e " %(np.min(G_mat),np.max(G_mat)))

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    if pnormalise:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
       a_mat[Nfem,NfemV:Nfem]=constr
       a_mat[NfemV:Nfem,Nfem]=constr
    else:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
       rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #end if

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*scaling_coeff

    print("     -> u (m,M) %.3e %.3e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.3e %.3e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.3e %.3e " %(np.min(p),np.max(p)))

    stats_u_file.write("%e %e %e \n" %(time,np.min(u),np.max(u))) ; stats_u_file.flush()
    stats_v_file.write("%e %e %e \n" %(time,np.min(v),np.max(v))) ; stats_v_file.flush()

    #np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    ####### TEST/DEBUG ############
    #for i in range(0,NV):
    #    xx=xV[i]/Lx
    #    yy=yV[i]/Ly
    #    u[i]=(xx*xx*(1.-xx)**2*(2.*yy-6.*yy*yy+4*yy*yy*yy))*cm/year  *100
    #    v[i]=(-yy*yy*(1.-yy)**2*(2.*xx-6.*xx*xx+4*xx*xx*xx))*cm/year *100
    #CFL=0.25
    #dt=CFL*hx/np.max(np.sqrt(u**2+v**2))/order

    #################################################################
    # compute timestep
    #################################################################

    CFL=dt/((sqrt2*Lx/nelx)/np.max(np.sqrt(u**2+v**2))/order)

    print('     -> dt = %f year, corresponds to %f' %(dt/year,CFL))

    #####################################################################
    # compute nodal velocity gradient 
    #####################################################################
    start = timing.time()
    
    count = np.zeros(NV,dtype=np.int16)  
    q=np.zeros(NV,dtype=np.float64)
    Lxx = np.zeros(NV,dtype=np.float64)  
    Lxy = np.zeros(NV,dtype=np.float64)  
    Lyx = np.zeros(NV,dtype=np.float64)  
    Lyy = np.zeros(NV,dtype=np.float64)  

    #u[:]=xV[:]**2
    #v[:]=yV[:]**2

    if computeLmethod==1:
        for iel in range(0,nel):
            for i in range(0,mV):
                inode=iconV[i,iel]
                rq=rVnodes[i]
                sq=sVnodes[i]
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
                #end for
                jcbi=np.linalg.inv(jcb)
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for
                L_xx=0.
                L_xy=0.
                L_yx=0.
                L_yy=0.
                for k in range(0,mV):
                    L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
                    L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
                    L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
                    L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
                #end for
                Lxx[inode]+=L_xx
                Lxy[inode]+=L_xy
                Lyx[inode]+=L_yx
                Lyy[inode]+=L_yy
                q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
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
            rq = 0.0
            sq = 0.0
            wq = 2.0 * 2.0
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
            #end for
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            L_xx=0.
            L_xy=0.
            L_yx=0.
            L_yy=0.
            for k in range(0,mV):
                L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
                L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
                L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
                L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
            #end for
            for i in range(0,mV):
                inode=iconV[i,iel]
                Lxx[inode]+=L_xx
                Lxy[inode]+=L_xy
                Lyx[inode]+=L_yx
                Lyy[inode]+=L_yy
                q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
                count[inode]+=1
            #end for
        #end for
        Lxx/=count
        Lxy/=count
        Lyx/=count
        Lyy/=count
        q/=count
    #end if

    print("     -> Lxx (m,M)   %.3e %.3e " %(np.min(Lxx),np.max(Lxx)))
    print("     -> Lxy (m,M)   %.3e %.3e " %(np.min(Lxy),np.max(Lxy)))
    print("     -> Lyx (m,M)   %.3e %.3e " %(np.min(Lyx),np.max(Lyx)))
    print("     -> Lyy (m,M)   %.3e %.3e " %(np.min(Lyy),np.max(Lyy)))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
    #np.savetxt('velgradient.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T,header='# x,y,Lxx,Lyy,Lxy,Lyx')

    print("compute nodal p and L: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal fields
    #####################################################################
    start = timing.time()

    exx[:]=Lxx[:]
    eyy[:]=Lyy[:]
    exy[:]=0.5*(Lxy[:]+Lyx[:])
    wxy[:]=0.5*(Lxy[:]-Lyx[:])
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

    #tauxx[:]=xV[:]
    #np.savetxt('tau.ascii',np.array([xV,yV,tauxx,tauyy,tauxy]).T,header='# x,y')

    print("compute sr, rr and J: %.3f s" % (timing.time() - start))

    #####################################################################
    time+=dt

    #####################################################################
    # interpolate dev stress difference (increment) onto markers
    # and add it to the existing value on the markers
    #####################################################################
    start = timing.time()

    for im in range(0,nmarker):
        rm=m_r[im]
        sm=m_s[im]
        NNNV[0:mV]=NNV(rm,sm)
        m_tauxx[im]+=np.sum(NNNV[:]*(tauxx[iconV[:,m_iel[im]]]-tauxxmem[iconV[:,m_iel[im]]]))
        m_tauyy[im]+=np.sum(NNNV[:]*(tauyy[iconV[:,m_iel[im]]]-tauyymem[iconV[:,m_iel[im]]]))
        m_tauxy[im]+=np.sum(NNNV[:]*(tauxy[iconV[:,m_iel[im]]]-tauxymem[iconV[:,m_iel[im]]]))
    #end for

    print("     -> m_tauxx (m,M) %.6e %.6e " %(np.min(m_tauxx),np.max(m_tauxx)))
    print("     -> m_tauyy (m,M) %.6e %.6e " %(np.min(m_tauyy),np.max(m_tauyy)))
    print("     -> m_tauxy (m,M) %.6e %.6e " %(np.min(m_tauxy),np.max(m_tauxy)))

    stats_m_tauxx_file.write("%e %e %e \n" %(time,np.min(m_tauxx),np.max(m_tauxx))) ;stats_m_tauxx_file.flush()
    stats_m_tauyy_file.write("%e %e %e \n" %(time,np.min(m_tauyy),np.max(m_tauyy))) ;stats_m_tauyy_file.flush()
    stats_m_tauxy_file.write("%e %e %e \n" %(time,np.min(m_tauxy),np.max(m_tauxy))) ;stats_m_tauxy_file.flush()

    #np.savetxt('m_tau.ascii',np.array([m_x,m_y,m_tauxx,m_tauyy,m_tauxy]).T,header='# x,y')


    print("interp. diff stress onto markers: %.3f s" % (timing.time() - start))

    #####################################################################
    # advect markers and re-locate them
    #####################################################################
    start = timing.time()

    for im in range(0,nmarker):
        rm=m_r[im]
        sm=m_s[im]
        NNNV[0:mV]=NNV(rm,sm)
        m_u[im]=np.sum(NNNV[:]*u[iconV[:,m_iel[im]]]) 
        m_v[im]=np.sum(NNNV[:]*v[iconV[:,m_iel[im]]])
        m_x[im]+=m_u[im]*dt 
        m_y[im]+=m_v[im]*dt 
        m_x[im]=min((1-eps2)*Lx,m_x[im])
        m_y[im]=min((1-eps2)*Ly,m_y[im])
        m_x[im]=max(eps2*Lx,m_x[im])
        m_y[im]=max(eps2*Ly,m_y[im])
        ielx=int(m_x[im]/Lx*nelx)
        if ielx<0:
           exit("ielx<0")
        if ielx>nelx-1:
           exit("ielx>nelx-1")
        iely=int(m_y[im]/Ly*nely)
        if iely<0:
           exit("iely<0")
        if iely>nely-1:
           exit("iely>nely-1")
        m_iel[im]=iely*nelx+ielx
        m_r[im]=( (m_x[im]-xV[iconV[0,m_iel[im]]])/hx-0.5)*2.
        m_s[im]=( (m_y[im]-yV[iconV[0,m_iel[im]]])/hy-0.5)*2.
    #end for

    print("     -> m_x (m,M) %.6e %.6e " %(np.min(m_x),np.max(m_x)))
    print("     -> m_y (m,M) %.6e %.6e " %(np.min(m_y),np.max(m_y)))

    print("advect markers: %.3f s" % (timing.time() - start))

    #####################################################################
    # project onto nodes
    #####################################################################
    start = timing.time()

    count =np.zeros(NV,dtype=np.float64)  
    Z[:]=0
    rho[:]=0
    etaeff[:]=0
    tauxx[:]=0
    tauyy[:]=0
    tauxy[:]=0

    for im in range(0,nmarker):
        rm=m_r[im]
        sm=m_s[im]
        NNNV[0:9]=BB(rm,sm)
        for i in range(0,mV):
            inode=iconV[i,m_iel[im]]
            Z[inode]     +=m_Z[im]     *NNNV[i]
            rho[inode]   +=m_rho[im]   *NNNV[i]
            etaeff[inode]+=m_etaeff[im]*NNNV[i]
            tauxx[inode] +=m_tauxx[im] *NNNV[i]
            tauyy[inode] +=m_tauyy[im] *NNNV[i]
            tauxy[inode] +=m_tauxy[im] *NNNV[i]
            count[inode] +=             NNNV[i]
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

    print("project markers onto nodes: %.3f s" % (timing.time() - start))

    #####################################################################
    start = timing.time()

    Jxx[:]=2*tauxx[:]*wxy[:]
    Jyy[:]=-2*tauxy[:]*wxy[:]
    Jxy[:]=(tauyy[:]-tauxx[:])*wxy[:]

    stats_Jxx_file.write("%e %e %e \n" %(time,np.min(Jxx),np.max(Jxx))) ; stats_Jxx_file.flush()
    stats_Jyy_file.write("%e %e %e \n" %(time,np.min(Jyy),np.max(Jyy))) ; stats_Jyy_file.flush()
    stats_Jxy_file.write("%e %e %e \n" %(time,np.min(Jxy),np.max(Jxy))) ; stats_Jxy_file.flush()

    print("     -> Jxx (m,M) %.6e %.6e " %(np.min(Jxx),np.max(Jxx)))
    print("     -> Jyy (m,M) %.6e %.6e " %(np.min(Jyy),np.max(Jyy)))
    print("     -> Jxy (m,M) %.6e %.6e " %(np.min(Jxy),np.max(Jxy)))

    print("compute nodal J: %.3f s" % (timing.time() - start))

    #####################################################################

    tauxxmem=tauxx
    tauyymem=tauyy
    tauxymem=tauxy

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

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
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/yr)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %rho[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='mu' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e \n" %(C1[i]*mu1+C2[i]*mu2))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e \n" %(C1[i]*eta1+C2[i]*eta2))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %etaeff[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='Z' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %Z[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(exx[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(eyy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(exy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='omega_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(wxy[i]))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauxx[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauyy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauxy[i]))
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



    if istep%every==0:
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
       for im in range(0,nmarker):
           vtufile.write("%e \n" % m_mat[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%15e \n" % m_tauxx[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%15e \n" % m_tauyy[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%15e \n" % m_tauxy[im])
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e \n" % m_r[im])
       #vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='s' Format='ascii'> \n")
       #for im in range(0,nmarker):
       #    vtufile.write("%10e \n" % m_s[im])
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % m_etaeff[im])
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
       for iq in range(0,nq):
           vtufile.write("%10e \n" % q_etaeff[iq])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Z' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % q_Z[iq])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % q_rho[iq])
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






    print("export to vtu: %.3f s" % (timing.time() - start))




print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
