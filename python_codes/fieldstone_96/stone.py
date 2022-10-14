import numpy as np
import numpy.ma as ma
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import lil_matrix
import triangle as tr
import os 

###############################################################################

use_numba=True

if use_numba:
    from tools_numba import *
    from compute_gravity_at_point_numba import *
    from basis_functions_numba import *
else:
    from tools import *
    from compute_gravity_at_point import *
    from basis_functions import *

###############################################################################

Ggrav = 6.67430e-11
year=365.25*3600*24
cm=0.01
km=1000

###############################################################################
###############################################################################
###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

CR=False

if CR:
   mV=7     # number of velocity nodes making up an element
else:
   mV=6

mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

###############################################################################
############# Parameters ######################################################
###############################################################################

R_outer=3.397e6
R_inner=R_outer-1600e3

# main parameter which controls resolution
# shound be 1,2,3,4, or 5
res=2
nnr=res*16+1            #vertical boundary resolutions
nnt=res*100              #sphere boundary resolutions

#-------------------------------------
# 'Moho' setup
#-------------------------------------
R_moho = R_outer-500e3 #500km below the surface reside's the moho

#-------------------------------------
# viscosity model
#-------------------------------------
# 1: isoviscous
# 2: steinberger data
# 3: three layer model

viscosity_model = 2

rho_crust=3300
eta_crust=1e25

rho_lith=3300
eta_lith=1e21

rho_mantle=3300
eta_mantle=6e20

eta0=6e20 # isoviscous case
rho0=3300

eta_core=1e25
rho_core=0 #7200

#rho_crust+=3700
#rho_lith+=3700

eta_max=1e25

#-------------------------------------
# blob setup 
#-------------------------------------
np_blob=res*20          #blob resolution
R_blob=300e3            #radius of blob
z_blob=R_outer-1000e3   #starting depth
rho_blob=rho_mantle-200
eta_blob=6e20

#-------------------------------------
#boundary conditions at planet surface
#-------------------------------------
#0: no-slip
#1: free-slip
#2: free (only top surface)

surface_bc=1

cmb_bc=1

#-------------------------------------
# gravity acceleration
#-------------------------------------

use_isog=True
g0=3.72

#-------------------------------------
#do not change

gravity_method=2
np_grav=50
nel_phi=200

height=10e3

eta_ref=1e22

nstep=1

###############################################################################

print('R_inner=',R_inner)
print('R_outer=',R_outer)
print('Volume=',4*np.pi/3*(R_outer**3-R_inner**3))
print('CR=',CR)
print('height=',height)
print("-----------------------------")

###############################################################################
# 6 point integration coeffs and weights 
###############################################################################

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=np.array([nb1,nb2,nb2,nb4,nb3,nb4],dtype=np.float64) 
qcoords_s=np.array([nb2,nb1,nb2,nb3,nb4,nb4],dtype=np.float64) 
qweights =np.array([nb5,nb5,nb5,nb6,nb6,nb6],dtype=np.float64) 

###############################################################################
#############  Defining the nodes and vertices ################################
###############################################################################
start = timing.time()

#------------------------------------------------------------------------------
# inner boundary counterclockwise
#------------------------------------------------------------------------------
theta = np.linspace(-np.pi*0.5, 0.5*np.pi,nnt, endpoint=False)          #half inner sphere in the x-positive domain
pts_ib = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R_inner     #nnt-points on inner boundary
seg_ib = np.stack([np.arange(nnt), np.arange(nnt) + 1], axis=1)         #vertices on innerboundary (and the last vertices to the upper wall)
for i in range(0,nnt):                                                  #first point must be exactly on the y-axis
    if i==0:
       pts_ib[i,0]=0

#------------------------------------------------------------------------------
# *top vertical (left) wall 
#------------------------------------------------------------------------------
topw_z = np.linspace(R_inner,R_outer,nnr,endpoint=False)                #vertical boundary wall from inner to outer boundary sphere
pts_topw = np.stack([np.zeros(nnr),topw_z],axis=1)                      #nnr-points on vertical wall
seg_topw = np.stack([nnt+np.arange(nnr),nnt+np.arange(nnr)+1], axis=1)  #vertices on vertical wall, starts where inner boundary vertices stopped

#------------------------------------------------------------------------------
# outer boundary clockwise
#------------------------------------------------------------------------------
theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=False)            #half outer sphere in the x-positive domain
pts_ob = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_outer        #nnt-points on outer boundary
seg_ob = np.stack([nnr+nnt+np.arange(nnt), nnr+nnt+np.arange(nnt)+1], axis=1) #vertices on outerboundary, starts where top wall vertices stopped
for i in range(0,nnt):                                                  #first point must be exactly on the y-axis
    if i==0:
       pts_ob[i,0]=0

#------------------------------------------------------------------------------
# bottom vertical wall
#------------------------------------------------------------------------------
botw_z = np.linspace(-R_outer,-R_inner,nnr,endpoint=False)              #vertical boundary wall from outer to inner boundary sphere
pts_botw = np.stack([np.zeros(nnr),botw_z],axis=1)                      #nnr-points on vertical wall
seg_botw = np.stack([2*nnt+nnr+np.arange(nnr),2*nnt+nnr+np.arange(nnr)+1], axis=1) #vertices on bottem vertical wall, starts where outerboundary vertices stopped
seg_botw[-1,1]=0                                                        #stitch last point to first point with last vertice

#------------------------------------------------------------------------------
# blob 
#------------------------------------------------------------------------------
theta_bl = np.linspace(-np.pi/2,np.pi-np.pi/2,num=np_blob,endpoint=True) #half-sphere in the x and y positive domain
pts_bl = np.stack([R_blob*np.cos(theta_bl),z_blob+R_blob*np.sin(theta_bl)], axis=1) #points on blob outersurface 
seg_bl = np.stack([2*nnt+2*nnr+np.arange(np_blob-1), 2*nnt+2*nnr+np.arange(np_blob-1)+1], axis=1) #vertices on outersurface blob, numbering starts after last bottemwall node)
for i in range(0,np_blob):                                              #first and last point must be exactly on the y-axis.
    if i==0 or i==np_blob-1:
       pts_bl[i,0]=0

#------------------------------------------------------------------------------
# Moho
#------------------------------------------------------------------------------
theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
pts_mo = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_moho        #nnt-points on outer boundary
seg_mo = np.stack([2*nnt+2*nnr+np_blob+np.arange(nnt-1), 2*nnt+2*nnr++np_blob+np.arange(nnt-1)+1], axis=1) #vertices on moho, numbering starts after last blob node)
for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
    if i==0 or i==nnt-1:
       pts_mo[i,0]=0    
       
# Stacking the nodes and vertices 

seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_bl,seg_mo])
pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_bl,pts_mo]) 

#put all segments and nodes in a dictionary

dict_nodes = dict(vertices=pts, segments=seg,holes=[[0,0]]) #no core so we add a hole at x=0,y=0

print("setup: generate nodes: %.3f s" % (timing.time() - start))

###############################################################################
#############  Create the P1 and P2 mesh ######################################
###############################################################################
#
#  P_2^+           P_-1
#
#  02              02
#  ||\\            ||\\
#  || \\           || \\
#  ||  \\          ||  \\
#  05   04         ||   \\
#  || 06 \\        ||    \\
#  ||     \\       ||     \\
#  00==03==01      00======01
#
# note also that triangle returns nodes 0-5, but not 6.
###############################################################################
start = timing.time()

dict_mesh = tr.triangulate(dict_nodes,'pqa50000000000')
#compare mesh to node and vertice plot
#tr.compare(plt, dict_nodes, dict_mesh)
#plt.axis
#plt.show()

print("setup: call mesher: %.3f s" % (timing.time() - start))
start = timing.time()

## define icon, x and z for P1 mesh
iconP1=dict_mesh['triangles'] ; iconP1=iconP1.T
xP1=dict_mesh['vertices'][:,0]
zP1=dict_mesh['vertices'][:,1]
NP1=np.size(xP1)
mP,nel=np.shape(iconP1)
#export_elements_to_vtu(xP1,zP1,iconP1,'meshP1.vtu')

print("setup: make P1 mesh: %.3f s" % (timing.time() - start))
start = timing.time()

NV0,xP2,zP2,iconP2=mesh_P1_to_P2(NP1,nel,xP1,zP1,iconP1)

print("setup: make P2 mesh: %.3f s" % (timing.time() - start))

#export_elements_to_vtuP2(xP2,zP2,iconP2,'meshP2.vtu')

#print("setup: generate P1 & P2 meshes: %.3f s" % (timing.time() - start))

###############################################################################
# compute NP, NV, NfemV, NfemP, Nfem for both element pairs
# and build xV,zV,iconV,xP,zP,iconP
###############################################################################
start = timing.time()

if CR:
   NP=3*nel*ndofP
   NV=NV0+nel
   NfemV=NV*ndofV                       # number of velocity dofs
   NfemP=NP*ndofP                       # number of pressure dofs
   #-----
   #iconV
   #-----
   iconV=np.zeros((mV,nel),dtype=np.int32)
   iconV[0,:]=iconP2[0,:]
   iconV[1,:]=iconP2[1,:]
   iconV[2,:]=iconP2[2,:]
   iconV[3,:]=iconP2[3,:]
   iconV[4,:]=iconP2[4,:]
   iconV[5,:]=iconP2[5,:]
   for iel in range (0,nel):
       iconV[6,iel]=NV0+iel
   #-----
   #xV,zV
   #-----
   xV=np.zeros(NV,dtype=np.float64)     # x coordinates
   zV=np.zeros(NV,dtype=np.float64)     # y coordinates
   xV[0:NV0]=xP2[0:NV0]
   zV[0:NV0]=zP2[0:NV0]
   for iel in range (0,nel):
       xV[NV0+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
       zV[NV0+iel]=(zV[iconV[0,iel]]+zV[iconV[1,iel]]+zV[iconV[2,iel]])/3.
   #-----------
   #iconP,xP,zP
   #-----------
   xP=np.empty(NP,dtype=np.float64)  # x coordinates
   zP=np.empty(NP,dtype=np.float64)  # y coordinates
   rP=np.empty(NP,dtype=np.float64)  # x coordinates
   iconP=np.zeros((mP,nel),dtype=np.int32)
   counter=0
   for iel in range(0,nel):
       xP[counter]=xP1[iconP1[0,iel]]
       zP[counter]=zP1[iconP1[0,iel]]
       iconP[0,iel]=counter
       counter+=1
       xP[counter]=xP1[iconP1[1,iel]]
       zP[counter]=zP1[iconP1[1,iel]]
       iconP[1,iel]=counter
       counter+=1
       xP[counter]=xP1[iconP1[2,iel]]
       zP[counter]=zP1[iconP1[2,iel]]
       iconP[2,iel]=counter
       rP[counter]=np.sqrt(xP[counter]**2+zP[counter]**2)
       counter+=1

else:
   NP=NP1
   NV=NV0 #replace by NP2
   NfemV=NV*ndofV    # number of velocity dofs
   NfemP=NP*ndofP    # number of pressure dofs
   #-----
   #iconV
   #-----
   iconV=np.zeros((mV,nel),dtype=np.int32)
   iconV[0,:]=iconP2[0,:]
   iconV[1,:]=iconP2[1,:]
   iconV[2,:]=iconP2[2,:]
   iconV[3,:]=iconP2[3,:]
   iconV[4,:]=iconP2[4,:]
   iconV[5,:]=iconP2[5,:]
   #-----
   #xV,zV
   #-----
   xV=np.zeros(NV,dtype=np.float64)     # x coordinates
   zV=np.zeros(NV,dtype=np.float64)     # y coordinates
   xV[0:NV0]=xP2[0:NV0]
   zV[0:NV0]=zP2[0:NV0]
   #-----
   #iconP
   #-----
   iconP=np.zeros((mP,nel),dtype=np.int32)
   iconP[0,:]=iconP1[0,:]
   iconP[1,:]=iconP1[1,:]
   iconP[2,:]=iconP1[2,:]
   #-----------
   #xP,zP
   #-----------
   xP=np.empty(NfemP,dtype=np.float64)  # x coordinates
   zP=np.empty(NfemP,dtype=np.float64)  # y coordinates
   rP=np.empty(NfemP,dtype=np.float64)  # y coordinates
   xP[0:NP]=xP1[0:NV]
   zP[0:NP]=zP1[0:NV]
   rP[:]=np.sqrt(xP[:]**2+zP[:]**2)


Nfem=NfemV+NfemP  # total number of dofs

print('     -> nel', nel)
print('     -> NV', NV)
print('     -> NfemV', NfemV)
print('     -> NfemP', NfemP)
print('     -> Nfem', Nfem)
print("     -> xV (min/max): %.4f %.4f" %(np.min(xV),np.max(xV)))
print("     -> zV (min/max): %.4f %.4f" %(np.min(zV),np.max(zV)))
print("     -> xP (min/max): %.4f %.4f" %(np.min(xP),np.max(xP)))
print("     -> zP (min/max): %.4f %.4f" %(np.min(zP),np.max(zP)))

print("setup: generate FE meshes: %.3f s" % (timing.time() - start))

###############################################################################
# read profiles 
###############################################################################
start = timing.time()

profile_eta=np.empty(1968,dtype=np.float64) 
profile_depth=np.empty(1968,dtype=np.float64) 
#profile_eta,profile_depth=np.loadtxt('data/eta.ascii',unpack=True,usecols=[0,1])

profile_rho=np.empty(3390,dtype=np.float64) 
profile_depth=np.empty(3390,dtype=np.float64) 
#profile_rho,profile_depth=np.loadtxt('data/rho.ascii',unpack=True,usecols=[0,1])

# gets current file path as string 
cfilepath = os.path.dirname(os.path.abspath(__file__)) 
# changes the current working directory to current file path    
os.chdir(cfilepath)
profile_eta,profile_depth=np.loadtxt(cfilepath + '/data/eta.ascii',unpack=True,usecols=[0,1]) 
profile_rho,profile_depth=np.loadtxt(cfilepath + '/data/rho.ascii',unpack=True,usecols=[0,1]) 

print("setup: read profiles: %.3f s" % (timing.time() - start))

###############################################################################
# from density profile build gravity profile
###############################################################################
start = timing.time()

profile_rad=np.empty(3390,dtype=np.float64) 
profile_grav=np.zeros(3390,dtype=np.float64) 
profile_mass=np.zeros(3390,dtype=np.float64) 

profile_grav[0]=0
for i in range(1,3390):
    profile_rad[i]=i*1000
    profile_mass[i]=profile_mass[i-1]+4*np.pi/3*(profile_rad[i]**3-profile_rad[i-1]**3)\
                                     *(profile_rho[3389-i]+profile_rho[3389-(i-1)])*1000/2
    profile_grav[i]=Ggrav*profile_mass[i]/profile_rad[i]**2
    
#np.savetxt('profile_grav.ascii',np.array([profile_rad,profile_mass,profile_grav]).T)

print("setup: build more profiles: %.3f s" % (timing.time() - start))

###############################################################################
# project mid-edge nodes onto circle
# and compute r,theta (spherical coordinates) for each node
###############################################################################
start = timing.time()

theta_nodal=np.zeros(NV,dtype=np.float64)
r_nodal=np.zeros(NV,dtype=np.float64)
surface_node=np.zeros(NV,dtype=np.bool) 
cmb_node=np.zeros(NV,dtype=np.bool) 

for i in range(0,NV):
    theta_nodal[i]=np.arctan2(xV[i],zV[i])
    r_nodal[i]=np.sqrt(xV[i]**2+zV[i]**2)
    if r_nodal[i]>0.999*R_outer:
       r_nodal[i]=R_outer*0.99999999
       xV[i]=r_nodal[i]*np.sin(theta_nodal[i])
       zV[i]=r_nodal[i]*np.cos(theta_nodal[i])
       surface_node[i]=True
    if r_nodal[i]<1.0001*R_inner:
       cmb_node[i]=True

print("     -> theta_nodal (m,M) %.6e %.6e " %(np.min(theta_nodal),np.max(theta_nodal)))
print("     -> r_nodal (m,M) %.6e %.6e "     %(np.min(r_nodal),np.max(r_nodal)))

#np.savetxt('gridV_after.ascii',np.array([xV,zV]).T,header='# xV,zV')

print("setup: flag cmb nodes: %.3f s" % (timing.time() - start))

###############################################################################
start = timing.time()

surface_Pnode=np.zeros(NP,dtype=np.bool) 
for i in range(0,NP):
    rP[i]=np.sqrt(xP[i]**2+zP[i]**2)
    if rP[i]>=0.999*R_outer: 
       surface_Pnode[i]=True

#np.savetxt('gridP.ascii',np.array([xP,zP,rP,surface_Pnode]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], zP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], zP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], zP[iconP[2][iel]])

print("setup: flag surface nodes: %.3f s" % (timing.time() - start))

###############################################################################
# assigning material properties to elements
# and assigning  density and viscosity 
###############################################################################
start = timing.time()

rho=np.zeros(nel,dtype=np.float64) 
eta=np.zeros(nel,dtype=np.float64) 
eta_nodal=np.zeros(NV,dtype=np.float64) 
rho_nodal=np.zeros(NV,dtype=np.float64) 

for iel in range(0,nel):
    x_c=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3
    z_c=(zV[iconV[0,iel]]+zV[iconV[1,iel]]+zV[iconV[2,iel]])/3
    r_c=np.sqrt(x_c**2+z_c**2)
    j=int((R_outer-r_c)/1000)

    if viscosity_model==1: # isoviscous
       eta[iel]=eta0
       rho[iel]=rho0
    elif viscosity_model==2: # steinberger 
       if r_c>R_inner:
          eta[iel]=10**profile_eta[j]
       else:
          eta[iel]=eta_core
       #etaeff[iel]=mu*dt/(1+mu/eta[iel]*dt) 
       rho[iel]=profile_rho[j]*1000
    else: # 3 layer model
       if r_c>R_outer-100e3:
          eta[iel]=eta_crust
          rho[iel]=rho_crust
       elif r_c>R_outer-500e3:
          eta[iel]=eta_lith
          rho[iel]=rho_lith
       elif r_c>R_inner:
          eta[iel]=eta_mantle
          rho[iel]=rho_mantle
       else:
          eta[iel]=eta_core
          rho[iel]=rho_core

    if x_c**2+(z_c-z_blob)**2<R_blob**2:
       rho[iel]=rho_blob
       eta[iel]=eta_blob
    eta[iel]=min(eta_max,eta[iel])
#end for

for i in range(0,NV):
    r_c=np.sqrt(xV[i]**2+zV[i]**2)
    j=int((R_outer-r_c)/1000)

    if viscosity_model==1: # isoviscous
       eta_nodal[i]=eta0
       rho_nodal[i]=0
    elif viscosity_model==2: # steinberger 
       if r_c>R_inner*0.999:
          eta_nodal[i]=10**profile_eta[j]
       else:
          eta_nodal[i]=eta_core
       #etaeff_nodal[i]=mu*dt/(1+mu/eta_nodal[i]*dt) 
       #rho_nodal[i]=profile_rho[j]*1000
    else:
       if r_c>R_outer-100e3:
          eta_nodal[i]=eta_crust
       elif r_c>R_outer-500e3:
          eta_nodal[i]=eta_lith
       elif r_c>R_inner:
          eta_nodal[i]=eta_mantle
       else:
          eta_nodal[i]=eta_core

    if xV[i]**2+(zV[i]-z_blob)**2 < 1.001*R_blob**2:
       eta_nodal[i]=eta_blob
       rho_nodal[i]=rho_blob
    eta_nodal[i]=min(eta_max,eta_nodal[i])
#end for

print("     -> eta_elemental (m,M) %.6e %.6e " %(np.min(eta),np.max(eta)))
print("     -> eta_nodal     (m,M) %.6e %.6e " %(np.min(eta_nodal),np.max(eta_nodal)))
print("     -> rho_elemental (m,M) %.6e %.6e " %(np.min(rho),np.max(rho)))

print("material layout: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
on_surf=np.zeros(NV,dtype=np.bool)  # boundary condition, yes/no

for i in range(0, NV):
    #Left boundary  
    if xV[i]<0.000001*R_inner:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       #if abs(zV[i])<R_inner:
       #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

    #planet surface
    if surface_node[i] and surface_bc==0: #no-slip surface
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

    #core mantle boundary
    if cmb_node[i] and cmb_bc==0: #no-slip surface
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("define boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
arear=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNV(rq,sq,CR)
        dNNNVdr=dNNVdr(rq,sq,CR)
        dNNNVds=dNNVds(rq,sq,CR)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
        xq=NNNV.dot(xV[iconV[:,iel]])
        arear[iel]+=jcob*weightq*xq*2*np.pi

VOL=4*np.pi*(R_outer**3-R_inner**3)/3

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*(R_outer**2-R_inner**2)/2))
print("     -> total vol  (meas) %.6f " %(arear.sum()))
print("     -> total vol  (error) %.6f percent" %((abs(arear.sum()/VOL)-1)*100))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# compute normal to surface 
#################################################################

nx=np.zeros(NV,dtype=np.float64)     # x coordinates
nz=np.zeros(NV,dtype=np.float64)     # y coordinates

for i in range(0,NV):
    #Left boundary  
    if xV[i]<0.000001*R_inner:
       nx[i]=-1
       nz[i]=0.
    #planet surface
    if xV[i]**2+zV[i]**2>0.9999*R_outer**2:
       nx[i]=xV[i]/R_outer
       nz[i]=zV[i]/R_outer
#end for

print("compute surface normal vector: %.3f s" % (timing.time() - start))

#################################################################
# flag all elements with a node touching the surface r=R_outer
# or r=R_inner, used later for free slip b.c.
#################################################################

flag_top=np.zeros(nel,dtype=np.float64)  
flag_bot=np.zeros(nel,dtype=np.float64)  
for iel in range(0,nel):
    if surface_node[iconV[0,iel]] or surface_node[iconV[1,iel]] or\
       surface_node[iconV[2,iel]] or surface_node[iconV[3,iel]] or\
       surface_node[iconV[4,iel]] or surface_node[iconV[5,iel]]:
       flag_top[iel]=1
    if cmb_node[iconV[0,iel]] or cmb_node[iconV[1,iel]] or\
       cmb_node[iconV[2,iel]] or cmb_node[iconV[3,iel]] or\
       cmb_node[iconV[4,iel]] or cmb_node[iconV[5,iel]]:
       flag_bot[iel]=1

###############################################################################

u = np.zeros(NV,dtype=np.float64)           # x-component velocity
v = np.zeros(NV,dtype=np.float64)           # y-component velocity

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

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs      = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    b_mat    = np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat    = np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
    c_mat    = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 

    mass=0

    for iel in range(0,nel):

        if iel%1000==0:
           print(iel)

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        NNNP= np.zeros(mP*ndofP,dtype=np.float64)   

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            NNNV=NNV(rq,sq,CR)
            dNNNVdr=dNNVdr(rq,sq,CR)
            dNNNVds=dNNVds(rq,sq,CR)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            if jcob<0:
               exit("jacobian is negative - bad triangle")

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*zV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # compute etaq, rhoq
            etaq=eta[iel] 
            rhoq=rho[iel] 

            for i in range(0,mV):
                b_mat[0:4, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [NNNV[i]/xq,0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob * 2*np.pi*xq
            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=NNNP[i]
                N_mat[3,i]=0.
            G_el-=b_mat.T.dot(N_mat)*weightq*jcob * 2*np.pi*xq

            mass+=jcob*weightq*rhoq * 2*np.pi*xq

            #compute gx,gy
            radq=np.sqrt(xq**2+yq**2)
            if use_isog:
               grav=g0
            else:
               grav=profile_grav[int(radq/1000)]
            angle=np.arctan2(yq,xq)
            gx=grav*np.cos(angle)
            gy=grav*np.sin(angle)
            #print(xq,yq,gx,gy)

            for i in range(0,mV):
                f_el[ndofV*i  ]-=NNNV[i]*jcob*weightq*gx*rhoq * 2*np.pi*xq
                f_el[ndofV*i+1]-=NNNV[i]*jcob*weightq*gy*rhoq * 2*np.pi*xq
            #end for 

        #end for kq

        if (surface_bc==1 and flag_top[iel]) or (cmb_bc==1 and flag_bot[iel]):
           for k in range(0,mV):
               inode=iconV[k,iel]
               if surface_node[inode] or cmb_node[inode]:
                  RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                  for i in range(0,mV*ndofV):
                      RotMat[i,i]=1.
                  angle=np.pi/2-theta_nodal[inode]
                  #RotMat[2*k  ,2*k]= np.cos(theta_nodal[inode]) ; RotMat[2*k  ,2*k+1]=np.sin(theta_nodal[inode])  
                  #RotMat[2*k+1,2*k]=-np.sin(theta_nodal[inode]) ; RotMat[2*k+1,2*k+1]=np.cos(theta_nodal[inode])  
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
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0
                #end if
            #end for 
        #end for

        G_el*=eta_ref/R_outer
        h_el*=eta_ref/R_outer

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2] += K_el[ikk,jkk] 
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk] 
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk] 
                #end for
                f_rhs[m1]+=f_el[ikk] 
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]  
        #end for
    #end for

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print('     -> mass=',mass)

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol = np.zeros(Nfem,dtype=np.float64) 

    sparse_matrix=A_sparse.tocsr()

    #print(sparse_matrix.min(),sparse_matrix.max())

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*eta_ref/R_outer

    #np.savetxt('solution_velocity.ascii',np.array([xV,zV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('solution_pressure.ascii',np.array([xP,zP,p]).T,header='# x,y,p')

    print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute vrms 
    ######################################################################
    start = timing.time()

    vrms=0
    for iel in range(0,0):
        for kq in range (0,nqel):
            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]
            NNNV[0:mV]=NNV(rq,sq,CR)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
            dNNNVds[0:mV]=dNNVds(rq,sq,CR)
            NNNP[0:mP]=NNP(rq,sq)
            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*zV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            vrms+=(uq**2+vq**2)*jcob*weightq *xq*2*np.pi
        #end for
    #end for
    vrms=np.sqrt(vrms/(4/3*np.pi*R_outer**3))

    print("     -> nel= %6d ; vrms (m/year)= %e " %(nel,vrms*year))

    print("compute vrms: %.3f s" % (timing.time() - start))

    ######################################################################
    # if free slip  or no slip is used at the surface then there is 
    # a pressure nullspace which needs to be removed. 
    # I here make sure that the pressure is zero on the surface on average
    ######################################################################
    start = timing.time()

    if surface_bc==0 or surface_bc==1:

       #not accurate enough!!
       #avrg_p=0
       #counter=0
       #for i in range(NfemP):
       #    if rP[i]>0.99999*R_outer:
       #       avrg_p+=p[i]
       #       counter+=1
       #p-=(avrg_p/counter) 
       #print(avrg_p/counter)

       counter=0
       perim=0
       avrg_p=0
       for iel in range(0,nel):
           if surface_Pnode[iconP[0,iel]] and surface_Pnode[iconP[1,iel]]:
              xxp=(xP[iconP[0,iel]]+xP[iconP[1,iel]])/2
              zzp=(zP[iconP[0,iel]]+zP[iconP[1,iel]])/2
              ppp=( p[iconP[0,iel]]+ p[iconP[1,iel]])/2
              theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
              theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
              dtheta=abs(theta0-theta1)
              thetap=np.arctan2(xxp,zzp)
              dist=np.sqrt((xP[iconP[0,iel]]-xP[iconP[1,iel]])**2+(zP[iconP[0,iel]]-zP[iconP[1,iel]])**2)
              perim+=dist
              avrg_p+=ppp*dist
           if surface_Pnode[iconP[1,iel]] and surface_Pnode[iconP[2,iel]]:
              xxp=(xP[iconP[1,iel]]+xP[iconP[2,iel]])/2
              zzp=(zP[iconP[1,iel]]+zP[iconP[2,iel]])/2
              ppp=( p[iconP[1,iel]]+ p[iconP[2,iel]])/2
              theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
              theta2=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
              dtheta=abs(theta1-theta2)
              thetap=np.arctan2(xxp,zzp)
              dist=np.sqrt((xP[iconP[1,iel]]-xP[iconP[2,iel]])**2+(zP[iconP[1,iel]]-zP[iconP[2,iel]])**2)
              perim+=dist
              avrg_p+=ppp*dist
           if surface_Pnode[iconP[2,iel]] and surface_Pnode[iconP[0,iel]]:
              xxp=(xP[iconP[2,iel]]+xP[iconP[0,iel]])/2
              zzp=(zP[iconP[2,iel]]+zP[iconP[0,iel]])/2
              ppp=( p[iconP[2,iel]]+ p[iconP[0,iel]])/2
              theta2=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
              theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
              dtheta=abs(theta2-theta0)
              thetap=np.arctan2(xxp,zzp)
              dist=np.sqrt((xP[iconP[2,iel]]-xP[iconP[0,iel]])**2+(zP[iconP[2,iel]]-zP[iconP[0,iel]])**2)
              perim+=dist
              avrg_p+=ppp*dist

       p-=avrg_p/perim 

       print('     -> perim (meas) =',perim)
       print('     -> perim (anal) =',np.pi*R_outer)
       print('     -> perim (error)=',abs(perim-np.pi*R_outer)/(np.pi*R_outer)*100,'%')
       print('     -> p (m,M) %.6e %.6e ' %(np.min(p),np.max(p)))

       np.savetxt('solution_pressure_normalised.ascii',np.array([xP,zP,p,rP]).T)

    print("normalise pressure: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute elemental strainrate
    # although previously x and y have been used (instead of r and z) 
    # I compute the components of the strain rate and stress tensors
    # with r, theta, z indices for clarity. 
    ######################################################################
    start = timing.time()

    #u[:]=xV[:]
    #v[:]=zV[:]

    xc = np.zeros(nel,dtype=np.float64)  
    zc = np.zeros(nel,dtype=np.float64)  
    e_xx = np.zeros(nel,dtype=np.float64) 
    e_zz = np.zeros(nel,dtype=np.float64) 
    e_xz = np.zeros(nel,dtype=np.float64)  
    tau_xx = np.zeros(nel,dtype=np.float64)  
    tau_zz = np.zeros(nel,dtype=np.float64)  
    tau_xz = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):
        rq = 1./3
        sq = 1./3
        NNNV[0:mV]=NNV(rq,sq,CR)
        dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
        dNNNVds[0:mV]=dNNVds(rq,sq,CR)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*zV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*zV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        for k in range(0,mV):
            xc[iel] += NNNV[k]*xV[iconV[k,iel]]
            zc[iel] += NNNV[k]*zV[iconV[k,iel]]
            e_xx[iel] += dNNNVdx[k]*u[iconV[k,iel]]         # dv_r/dr
            e_zz[iel] += dNNNVdy[k]*v[iconV[k,iel]]         # dv_z/dz
            e_xz[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                         0.5*dNNNVdx[k]*v[iconV[k,iel]]     # 0.5 (dv_r/dz+dv_z/dr)
            tau_xx[iel] += dNNNVdx[k]*u[iconV[k,iel]]*2*eta[iel]        
            tau_zz[iel] += dNNNVdy[k]*v[iconV[k,iel]]*2*eta[iel]        
            tau_xz[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]*2*eta[iel]+\
                           0.5*dNNNVdx[k]*v[iconV[k,iel]]*2*eta[iel]           

    print("     -> xc     (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
    print("     -> zc     (m,M) %.6e %.6e " %(np.min(zc),np.max(zc)))
    print("     -> e_xx   (m,M) %.6e %.6e " %(np.min(e_xx),np.max(e_xx)))
    print("     -> e_zz   (m,M) %.6e %.6e " %(np.min(e_zz),np.max(e_zz)))
    print("     -> e_xz   (m,M) %.6e %.6e " %(np.min(e_xz),np.max(e_xz)))
    print("     -> tau_xx (m,M) %.6e %.6e " %(np.min(tau_xx),np.max(tau_xx)))
    print("     -> tau_zz (m,M) %.6e %.6e " %(np.min(tau_zz),np.max(tau_zz)))
    print("     -> tau_xz (m,M) %.6e %.6e " %(np.min(tau_xz),np.max(tau_xz)))

    #np.savetxt('centers.ascii',np.array([xc,zc]).T)

    print("compute sr and stress: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute nodal strainrate
    #  02              02
    #  ||\\            ||\\
    #  || \\           || \\
    #  ||  \\          ||  \\
    #  05   04         ||   \\
    #  || 06 \\        ||    \\
    #  ||     \\       ||     \\
    #  00==03==01      00======01
    ######################################################################
    start = timing.time()

    e_xx_nodal = np.zeros(NV,dtype=np.float64)  
    e_zz_nodal = np.zeros(NV,dtype=np.float64)  
    e_xz_nodal = np.zeros(NV,dtype=np.float64)  
    tau_xx_nodal = np.zeros(NV,dtype=np.float64)  
    tau_zz_nodal = np.zeros(NV,dtype=np.float64)  
    tau_xz_nodal = np.zeros(NV,dtype=np.float64)  
    cc         = np.zeros(NV,dtype=np.float64)
    q          = np.zeros(NV,dtype=np.float64)

    rVnodes=[0,1,0,0.5,0.5,0,1./3.] # valid for CR and P2P1
    sVnodes=[0,0,1,0,0.5,0.5,1./3.]

    #u[:]=xV[:]
    #v[:]=zV[:]

    for iel in range(0,nel):
        for kk in range(0,mV):
            inode=iconV[kk,iel]
            rq = rVnodes[kk]
            sq = sVnodes[kk]
            NNNV[0:mV]=NNV(rq,sq,CR)
            dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
            dNNNVds[0:mV]=dNNVds(rq,sq,CR)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*zV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*zV[iconV[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            for k in range(0,mV):
                tau_xx_nodal[inode] += dNNNVdx[k]*u[iconV[k,iel]]*2*(eta[iel]/eta_ref)        
                tau_zz_nodal[inode] += dNNNVdy[k]*v[iconV[k,iel]]*2*(eta[iel]/eta_ref)                
                tau_xz_nodal[inode] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]*2*(eta[iel]/eta_ref)+\
                                       0.5*dNNNVdx[k]*v[iconV[k,iel]]*2*(eta[iel]/eta_ref)                   
                e_xx_nodal[inode] += dNNNVdx[k]*u[iconV[k,iel]]
                e_zz_nodal[inode] += dNNNVdy[k]*v[iconV[k,iel]]
                e_xz_nodal[inode] += 0.5*dNNNVdy[k]*u[iconV[k,iel]] + 0.5*dNNNVdx[k]*v[iconV[k,iel]]
            for k in range(0,mP):
                q[inode]+= NNNP[k]*p[iconP[k,iel]]   
            #end for
            cc[inode]+=1
        #end for
    #end for
    e_xx_nodal/=cc
    e_zz_nodal/=cc
    e_xz_nodal/=cc
    tau_xx_nodal/=cc
    tau_zz_nodal/=cc
    tau_xz_nodal/=cc
    tau_xx_nodal*=eta_ref
    tau_zz_nodal*=eta_ref
    tau_xz_nodal*=eta_ref
    q[:]/=cc[:]

    #np.savetxt('solution_q.ascii',np.array([xV,zV,q]).T)
    #np.savetxt('solution_tau_cartesian.ascii',np.array([xV,zV,tau_xx_nodal,tau_zz_nodal,tau_xz_nodal]).T)

    print("     -> e_xx_nodal   (m,M) %.6e %.6e " %(np.min(e_xx_nodal),np.max(e_xx_nodal)))
    print("     -> e_zz_nodal   (m,M) %.6e %.6e " %(np.min(e_zz_nodal),np.max(e_zz_nodal)))
    print("     -> e_xz_nodal   (m,M) %.6e %.6e " %(np.min(e_xz_nodal),np.max(e_xz_nodal)))
    print("     -> tau_xx_nodal (m,M) %.6e %.6e " %(np.min(tau_xx_nodal),np.max(tau_xx_nodal)))
    print("     -> tau_zz_nodal (m,M) %.6e %.6e " %(np.min(tau_zz_nodal),np.max(tau_zz_nodal)))
    print("     -> tau_xz_nodal (m,M) %.6e %.6e " %(np.min(tau_xz_nodal),np.max(tau_xz_nodal)))

    print("compute sr and stress: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute theta / colatitude
    #####################################################################
    start = timing.time()

    theta=np.zeros(nel,dtype=np.float64)
    for iel in range(0,nel):
        theta[iel]=np.arctan2(xc[iel],zc[iel])

    print("     -> theta       (m,M) %.6e %.6e " %(np.min(theta),np.max(theta)))

    print("compute theta: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute stress tensor components
    #####################################################################
    start = timing.time()

    e_rr_nodal=np.zeros(NV,dtype=np.float64)  
    e_tt_nodal=np.zeros(NV,dtype=np.float64)  
    e_rt_nodal=np.zeros(NV,dtype=np.float64)  
    tau_rr_nodal=np.zeros(NV,dtype=np.float64)  
    tau_tt_nodal=np.zeros(NV,dtype=np.float64)  
    tau_rt_nodal=np.zeros(NV,dtype=np.float64)  

    tau_rr_nodal=tau_xx_nodal*np.sin(theta_nodal)**2+2*tau_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_zz_nodal*np.cos(theta_nodal)**2
    tau_tt_nodal=tau_xx_nodal*np.cos(theta_nodal)**2-2*tau_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_zz_nodal*np.sin(theta_nodal)**2
    tau_rt_nodal=(tau_xx_nodal-tau_zz_nodal)*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_xz_nodal*(-np.sin(theta_nodal)**2+np.cos(theta_nodal)**2)

    e_rr_nodal=e_xx_nodal*np.sin(theta_nodal)**2+2*e_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+e_zz_nodal*np.cos(theta_nodal)**2
    e_tt_nodal=e_xx_nodal*np.cos(theta_nodal)**2-2*e_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+e_zz_nodal*np.sin(theta_nodal)**2
    e_rt_nodal=(e_xx_nodal-e_zz_nodal)*np.sin(theta_nodal)*np.cos(theta_nodal)+e_xz_nodal*(-np.sin(theta_nodal)**2+np.cos(theta_nodal)**2)

    tau_rr=np.zeros(nel,dtype=np.float64)  
    tau_tt=np.zeros(nel,dtype=np.float64)  
    tau_rt=np.zeros(nel,dtype=np.float64)  

    tau_rr=tau_xx*np.sin(theta)**2+2*tau_xz*np.sin(theta)*np.cos(theta)+tau_zz*np.cos(theta)**2
    tau_tt=tau_xx*np.cos(theta)**2-2*tau_xz*np.sin(theta)*np.cos(theta)+tau_zz*np.sin(theta)**2
    tau_rt=(tau_xx-tau_zz)*np.sin(theta)*np.cos(theta)+tau_xz*(-np.sin(theta)**2+np.cos(theta)**2)

    sigma_rr_nodal=-q+tau_rr_nodal
    sigma_tt_nodal=-q+tau_tt_nodal
    sigma_rt_nodal=   tau_rr_nodal

    #np.savetxt('solution_tau_spherical.ascii',np.array([xV,zV,tau_rr_nodal,tau_tt_nodal,tau_rt_nodal]).T)

    print("rotate stresses: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute traction at surface and cmb
    #####################################################################
    start = timing.time()

    tracfile=open('surface_traction_nodal_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%e %e %e %e %e %e %e\n" \
                          %(theta_nodal[i],tau_rr_nodal[i]-q[i],xV[i],zV[i],tau_rr_nodal[i],q[i],e_rr_nodal[i]))
    tracfile.close()

    tracfile=open('surface_vr_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.sin(theta_nodal[i])+v[i]*np.cos(theta_nodal[i])  ))
    tracfile.close()

    tracfile=open('surface_vt_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.cos(theta_nodal[i])-v[i]*np.sin(theta_nodal[i]) ) )
    tracfile.close()

    tracfile=open('cmb_traction_nodal_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if cmb_node[i]: 
           tracfile.write("%e %e %e %e %e %e %e\n" \
                          %(theta_nodal[i],tau_rr_nodal[i]-q[i],xV[i],zV[i],tau_rr_nodal[i],q[i],e_rr_nodal[i]))
    tracfile.close()

    tracfile=open('cmb_vr_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if cmb_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.sin(theta_nodal[i])+v[i]*np.cos(theta_nodal[i])  ))
    tracfile.close()

    tracfile=open('cmb_vt_'+str(istep)+'.ascii',"w")
    for i in range(0,NV):
        if cmb_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.cos(theta_nodal[i])-v[i]*np.sin(theta_nodal[i]) ) )
    tracfile.close()
        
    print("compute surface tractions: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start = timing.time()

    #u[:]/=(cm/year)
    #v[:]/=(cm/year)

    filename = 'solution_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],0.,zV[i]))
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
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (rho[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % (eta[iel]))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='viscosity (effective)' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%7e\n" % (etaeff[iel]))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%7e\n" % ((p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='xc' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % xc[iel])
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='zc' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % zc[iel])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_xx)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e_xx[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_zz)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e_zz[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_xz)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (e_xz[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='flag_top' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (flag_top[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='flag_bot' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (flag_bot[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_xx)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_xx[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_zz)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_zz[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_xz)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_xz[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_rr)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_rr[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_tt)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_tt[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev stress (tau_rt)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (tau_rt[iel]))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='theta (sph.coords)' Format='ascii'> \n")
    #for iel in range(0,nel):
    #    vtufile.write("%e \n" %theta[iel])
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='theta_p(dev stress)' Format='ascii'> \n")
    #for iel in range(0,nel):
    #    theta_p=0.5*np.arctan(2*tauxy[iel]/(tauxx[iel]-tauyy[iel]))
    #    vtufile.write("%10e \n" % (theta_p/np.pi*180.))
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='tau_max' Format='ascii'> \n")
    #for iel in range(0,nel):
    #    tau_max=np.sqrt( (tauxx[iel]-tauyy[iel])**2/4 +tauxy[iel]**2 )
    #    vtufile.write("%10e \n" % tau_max)
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,0.,v[i]/cm*year))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32'  Name='vr (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (u[i]*np.sin(theta_nodal[i])+v[i]*np.cos(theta_nodal[i]) ) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32'  Name='vt (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (u[i]*np.cos(theta_nodal[i])-v[i]*np.sin(theta_nodal[i]) ) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='gravity vector (norm)' Format='ascii'> \n")
    for i in range(0,NV):
        rad=np.sqrt(xV[i]**2+zV[i]**2)
        if use_isog:
           grav=g0
        else:
           grav=profile_grav[int(rad/1000)]
        vtufile.write("%10e \n" %grav)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (nod)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %r_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='surface' Format='ascii'> \n")
    for i in range(0,NV):
        if surface_node[i]:
           vtufile.write("%d \n" %1)
        else:
           vtufile.write("%d \n" %0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='cmb' Format='ascii'> \n")
    for i in range(0,NV):
        if cmb_node[i]:
           vtufile.write("%d \n" %1)
        else:
           vtufile.write("%d \n" %0)
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Int32' Name='dyn topo' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if surface_node[i]:
    #       vtufile.write("%d \n" % (-(tau_rr_nodal[i]-q[i])/g0/rho_surf) ) # (-pI + tau).vec{n} / rho g0
    #    else:
    #       vtufile.write("%d \n" % 0)
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta (sph.coords)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %theta_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%e \n" %rho_nodal[i])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_xx)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_xx_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_zz)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_zz_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_xz)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_xz_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_rr)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_rr_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_tt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_tt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_rt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % tau_rt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_rr)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_rr_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_tt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_tt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_rt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_rt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_xx)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_xx_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_zz)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_zz_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (e_xz)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % e_xz_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='stress (sigma_rr)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % sigma_rr_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='stress (sigma_tt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % sigma_tt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='stress (sigma_rt)' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % sigma_rt_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='traction' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if surface_node[i]:
    #       vtufile.write("%10e %10e %10e \n" % (tau_rr_nodal[i],0.,tau_rt_nodal[i]))
    #    else:
    #       vtufile.write("%e %e %e \n" % (0,0,0))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%10e %10e %10e \n" % (nx[i],0.,nz[i]))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[3,iel],iconV[4,iel],iconV[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("write data: %.3fs" % (timing.time() - start))

    #####################################################################
    # compute gravity
    # phi goes from 0 to 2pi
    #####################################################################
    start = timing.time()

    xM=np.empty(np_grav,dtype=np.float64)     
    yM=np.empty(np_grav,dtype=np.float64)     
    zM=np.empty(np_grav,dtype=np.float64)     
    gvect_x=np.zeros(np_grav,dtype=np.float64)   
    gvect_y=np.zeros(np_grav,dtype=np.float64)   
    gvect_z=np.zeros(np_grav,dtype=np.float64)   
    angleM=np.zeros(np_grav,dtype=np.float64)   

    #-------------------

    dphi=2*np.pi/nel_phi
    for i in range(0,np_grav):
        angleM[i]=np.pi/2-np.pi/2/(np_grav-1)*i
        xM[i]=(R_outer+height)*np.cos(angleM[i])
        yM[i]=0
        zM[i]=(R_outer+height)*np.sin(angleM[i])

        if gravity_method==1:
           gvect_x[i],gvect_y[i],gvect_z[i]=compute_gravity_at_point1(xM[i],yM[i],zM[i],nel,xV,zV,iconV,rho,arear,dphi,nel_phi)
           print('point',i,'gx,gy,gz',gvect_x[i],gvect_y[i],gvect_z[i])

        if gravity_method==2:
           gvect_x[i],gvect_y[i],gvect_z[i]=compute_gravity_at_point2(xM[i],yM[i],zM[i],nel,xV,zV,iconV,rho,\
                                                                      dphi,nel_phi,qcoords_r,qcoords_s,qweights,CR,mV,nqel)
           print('point',i,'gx,gy,gz',gvect_x[i],gvect_y[i],gvect_z[i])

    #end for

    gvect=np.sqrt(gvect_x**2+gvect_y**2+gvect_z**2)
    rM=np.sqrt(xM**2+yM**2+zM**2)

    np.savetxt('gravity_'+str(istep)+'.ascii',np.array([xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect]).T,fmt='%.6e')

    print("compute gravity: %.3fs" % (timing.time() - start))

    #####################################################################
    # compute timestep
    #####################################################################
    start = timing.time()

    CFL_nb=0.5

    dt=CFL_nb*(np.min(np.sqrt(area)))/np.max(np.sqrt(u**2+v**2))
    print('     -> dt = %.6f yr' % (dt/year))

    print("compute dt: %.3fs" % (timing.time() - start))

    #####################################################################
    # evolve mesh
    #####################################################################
    start = timing.time()

    np.savetxt('meshV_bef_'+str(istep)+'.ascii',np.array([xV/km,zV/km,u,v]).T,header='# x,y')
    np.savetxt('meshP_bef_'+str(istep)+'.ascii',np.array([xP/km,zP/km]).T,header='# x,y')

    for i in range(0,NV):
        if not surface_node[i] and not cmb_node[i]:
           xV[i]+=u[i]*dt
           zV[i]+=v[i]*dt

    for iel in range(0,nel):
        xP[iconP[0,iel]]=xV[iconV[0,iel]]
        xP[iconP[1,iel]]=xV[iconV[1,iel]]
        xP[iconP[2,iel]]=xV[iconV[2,iel]]
        zP[iconP[0,iel]]=zV[iconV[0,iel]]
        zP[iconP[1,iel]]=zV[iconV[1,iel]]
        zP[iconP[2,iel]]=zV[iconV[2,iel]]

    np.savetxt('meshV_aft_'+str(istep)+'.ascii',np.array([xV/km,zV/km]).T,header='# x,y')
    np.savetxt('meshP_aft_'+str(istep)+'.ascii',np.array([xP/km,zP/km]).T,header='# x,y')

    print("evolve mesh: %.3fs" % (timing.time() - start))

#end istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
