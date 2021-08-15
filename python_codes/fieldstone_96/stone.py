import numpy as np
import numpy.ma as ma
import sys as sys
import scipy
import csv
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from scipy.sparse import lil_matrix
from parameters import *

Ggrav = 6.67430e-11

#------------------------------------------------------------------------------
# basis functions for Crouzeix-Raviart element and Taylor-Hood element
#------------------------------------------------------------------------------

def NNV(rq,sq):
    if CR:
       NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
       NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
       NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
       NV_3= 4.*(1.-rq-sq)*rq*(1.-3.*sq) 
       NV_4= 4.*rq*sq*(-2.+3.*rq+3.*sq)
       NV_5= 4.*(1.-rq-sq)*sq*(1.-3.*rq) 
       NV_6= 27*(1.-rq-sq)*rq*sq
       return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6
    else:
       NV_0= 1-3*rq-3*sq+2*rq**2+4*rq*sq+2*sq**2 
       NV_1= -rq+2*rq**2
       NV_2= -sq+2*sq**2
       NV_3= 4*rq-4*rq**2-4*rq*sq
       NV_4= 4*rq*sq 
       NV_5= 4*sq-4*rq*sq-4*sq**2
       return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5

def dNNVdr(rq,sq):
    if CR:
       dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
       dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
       dNVdr_2= 3*sq-6*rq*sq-3*sq**2
       dNVdr_3= -8*rq+24*rq*sq+4-16*sq+12*sq**2
       dNVdr_4= -8*sq+24*rq*sq+12*sq**2
       dNVdr_5= -16*sq+24*rq*sq+12*sq**2
       dNVdr_6= -54*rq*sq+27*sq-27*sq**2
       return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6
    else:
       dNVdr_0= -3+4*rq+4*sq 
       dNVdr_1= -1+4*rq
       dNVdr_2= 0
       dNVdr_3= 4-8*rq-4*sq
       dNVdr_4= 4*sq
       dNVdr_5= -4*sq
       return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5

def dNNVds(rq,sq):
    if CR:
       dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
       dNVds_1= rq*(3-3*rq-6*sq)
       dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
       dNVds_3= -16*rq+24*rq*sq+12*rq**2
       dNVds_4= -8*rq+12*rq**2+24*rq*sq
       dNVds_5= 4-16*rq-8*sq+24*rq*sq+12*rq**2
       dNVds_6= -54*rq*sq+27*rq-27*rq**2
       return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6
    else:
       dNVds_0= -3+4*rq+4*sq 
       dNVds_1= 0
       dNVds_2= -1+4*sq
       dNVds_3= -4*rq
       dNVds_4= +4*rq
       dNVds_5= 4-4*rq-8*sq
       return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

#------------------------------------------------------------------------------
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

#read nb of elements and nb of nodes from temp file 

counter=0
file=open("temp", "r")
for line in file:
    fields = line.strip().split()
    #print(fields[0], fields[1], fields[2])
    if counter==0:
       nel=int(fields[0])
    if counter==1:
       NV0=int(fields[0])
    counter+=1

if CR:
   NV=NV0+nel
else:
   NV=NV0

NfemV=NV*ndofV     # number of velocity dofs
if CR:
   NfemP=nel*3*ndofP   # number of pressure dofs

print ('nel', nel)
print ('NV0', NV0)
print ('NfemV', NfemV)

eta_ref=1e23

#---------------------------------------
# 6 point integration coeffs and weights 

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights =[nb5,nb5,nb5,nb6,nb6,nb6]

#################################################################
# read profiles 
#################################################################

profile_eta=np.empty(1968,dtype=np.float64) 
profile_depth=np.empty(1968,dtype=np.float64) 
profile_eta,profile_depth=np.loadtxt('data/eta.ascii',unpack=True,usecols=[0,1])

profile_rho=np.empty(3390,dtype=np.float64) 
profile_depth=np.empty(3390,dtype=np.float64) 
profile_rho,profile_depth=np.loadtxt('data/rho.ascii',unpack=True,usecols=[0,1])

#################################################################
# from density profile build gravity profile
#################################################################

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

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)     # x coordinates
zV=np.zeros(NV,dtype=np.float64)     # y coordinates

xV[0:NV0],zV[0:NV0]=np.loadtxt('mesh.1.node',unpack=True,usecols=[1,2],skiprows=1)

print("xV (min/max): %.4f %.4f" %(np.min(xV[0:NV0]),np.max(xV[0:NV0])))
print("zV (min/max): %.4f %.4f" %(np.min(zV[0:NV0]),np.max(zV[0:NV0])))

#np.savetxt('gridV0.ascii',np.array([xV,zV]).T,header='# xV,zV')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
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
# note that the ordering of nodes returned by triangle is different
# than mine: https://www.cs.cmu.edu/~quake/triangle.highorder.html.
# note also that triangle returns nodes 0-5, but not 6.
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

iconV[0,:],iconV[1,:],iconV[2,:],iconV[4,:],iconV[5,:],iconV[3,:]=\
np.loadtxt('mesh.1.ele',unpack=True, usecols=[1,2,3,4,5,6],skiprows=1)

iconV[0,:]-=1
iconV[1,:]-=1
iconV[2,:]-=1
iconV[3,:]-=1
iconV[4,:]-=1
iconV[5,:]-=1

if CR:
   for iel in range (0,nel):
       iconV[6,iel]=NV0+iel
   for iel in range (0,nel): #bubble nodes
       xV[NV0+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
       zV[NV0+iel]=(zV[iconV[0,iel]]+zV[iconV[1,iel]]+zV[iconV[2,iel]])/3.
else:
   # from this information I must now extract the number of nodes 
   # which make the P1 mesh for pressure.
   P1bool=np.zeros(NV,dtype=np.bool) 
   for iel in range(0,nel):
           P1bool[iconV[0,iel]]=True
           P1bool[iconV[1,iel]]=True
           P1bool[iconV[2,iel]]=True
   NfemP=np.count_nonzero(P1bool)

Nfem=NfemV+NfemP    # total number of dofs
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

#np.savetxt('gridV.ascii',np.array([xV,zV]).T,header='# xV,zV')

print("setup: connectivity V: %.3f s" % (timing.time() - start))

#################################################################
# project mid-edge nodes onto circle
# and compute r,theta (spherical coordinates) for each node
#################################################################

theta_nodal=np.zeros(NV,dtype=np.float64)
r_nodal=np.zeros(NV,dtype=np.float64)
surface_node=np.zeros(NV,dtype=np.bool) 

for i in range(0,NV):
    theta_nodal[i]=np.arctan2(xV[i],zV[i])
    r_nodal[i]=np.sqrt(xV[i]**2+zV[i]**2)

    if r_nodal[i]>0.9999*R_outer:
       r_nodal[i]=R_outer*0.99999999
       xV[i]=r_nodal[i]*np.sin(theta_nodal[i])
       zV[i]=r_nodal[i]*np.cos(theta_nodal[i])
       surface_node[i]=True

print("     -> theta_nodal (m,M) %.6e %.6e " %(np.min(theta_nodal),np.max(theta_nodal)))
print("     -> r_nodal (m,M) %.6e %.6e "     %(np.min(r_nodal),np.max(r_nodal)))

#np.savetxt('gridV_after.ascii',np.array([xV,zV]).T,header='# xV,zV')

#################################################################
# build pressure grid (nodes and icon)
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)
xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
yP=np.empty(NfemP,dtype=np.float64)     # y coordinates
rP=np.empty(NfemP,dtype=np.float64)     # r coordinates

if CR:
   counter=0
   for iel in range(0,nel):
       xP[counter]=xV[iconV[0,iel]]
       yP[counter]=zV[iconV[0,iel]]
       iconP[0,iel]=counter
       counter+=1
       xP[counter]=xV[iconV[1,iel]]
       yP[counter]=zV[iconV[1,iel]]
       iconP[1,iel]=counter
       counter+=1
       xP[counter]=xV[iconV[2,iel]]
       yP[counter]=zV[iconV[2,iel]]
       iconP[2,iel]=counter
       rP[counter]=np.sqrt(xP[counter]**2+yP[counter]**2)
       counter+=1
else:
   iconP[0,:]=iconV[0,:]
   iconP[1,:]=iconV[1,:]
   iconP[2,:]=iconV[2,:]
   for iel in range(0,nel):
       xP[iconP[0,iel]]=xV[iconP[0,iel]]
       xP[iconP[1,iel]]=xV[iconP[1,iel]]
       xP[iconP[2,iel]]=xV[iconP[2,iel]]
       yP[iconP[0,iel]]=zV[iconP[0,iel]]
       yP[iconP[1,iel]]=zV[iconP[1,iel]]
       yP[iconP[2,iel]]=zV[iconP[2,iel]]
   rP[:]=np.sqrt(xP[:]**2+yP[:]**2)


#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# assigning material properties to elements
# and assigning  density and viscosity 
#################################################################
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
       rho[iel]=0
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
          rho[iel]=5000

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

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
on_surf=np.zeros(NV,dtype=np.bool)  # boundary condition, yes/no

for i in range(0, NV):
    #Left boundary  
    if xV[i]<0.000001*R_inner:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       if abs(zV[i])<R_inner:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #bottom boundary  
    #if zV[i]<0.000001*R_inner:
    #   bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #   if abs(xV[i])<R_inner:
    #      bc_fix[i*ndofV] = True ; bc_val[i*ndofV] = 0.

    #planet surface
    if surface_node[i] and surface_bc==0: #no-slip surface
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

    #core mantle boundary
    if abs(np.sqrt(xV[i]**2+zV[i]**2)-R_inner)<0.001*R_inner:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("define boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
arear=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
        xq=0.
        for k in range(0,mV):
            xq+=NNNV[k]*xV[iconV[k,iel]]
        arear[iel]+=jcob*weightq*xq

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*R_outer**2/2))
print("     -> total vol  (meas) %.6f " %(arear.sum()))
print("     -> total vol  (anal) %.6f " %(4*np.pi*R_outer**3/3))

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
# used later for free slip b.c.
#################################################################

flag=np.zeros(nel,dtype=np.float64)  
for iel in range(0,nel):
    if surface_node[iconV[0,iel]] or surface_node[iconV[1,iel]] or\
       surface_node[iconV[2,iel]] or surface_node[iconV[3,iel]] or\
       surface_node[iconV[4,iel]] or surface_node[iconV[5,iel]]:
       flag[iel]=1

################################################################################################

u = np.zeros(NV,dtype=np.float64)           # x-component velocity
v = np.zeros(NV,dtype=np.float64)           # y-component velocity

for istep in range(0,1):

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
    NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    b_mat    = np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat    = np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
    c_mat    = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 

    mass=0

    for iel in range(0,nel):

        if iel%5000==0:
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

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
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

        if surface_bc==1 and flag[iel]:
           for k in range(0,mV):
               inode=iconV[k,iel]
               if surface_node[inode]:
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

    print('mass=',mass)

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

    #np.savetxt('p_solution.ascii',np.array([xP,yP,p]).T,header='# x,y')

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
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
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

    print("     -> nel= %6d ; vrms= %e " %(nel,vrms))

    print("compute vrms: %.3f s" % (timing.time() - start))

    ######################################################################
    # if free slip  or no slip is used at the surface then there is 
    # a pressure nullspace which needs to be removed. 
    # I here make sure that the pressure is zero on the surface on average
    ######################################################################
    start = timing.time()

    if surface_bc==0 or surface_bc==1:

       avrg_p=0
       counter=0
       for i in range(NfemP):
           if rP[i]>0.99999*R_outer:
              avrg_p+=p[i]
              counter+=1

       p-=(avrg_p/counter) 

       #np.savetxt('p_solution_normalised.ascii',np.array([xP,yP,p,rP]).T)

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
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
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

    tau_xx_nodal = np.zeros(NV,dtype=np.float64)  
    tau_zz_nodal = np.zeros(NV,dtype=np.float64)  
    tau_xz_nodal = np.zeros(NV,dtype=np.float64)  
    cc         = np.zeros(NV,dtype=np.float64)
    q          = np.zeros(NV,dtype=np.float64)

    rVnodes=[0,1,0,0.5,0.5,0,1./3.] # valid for CR and P2P1
    sVnodes=[0,0,1,0,0.5,0.5,1./3.]

    for iel in range(0,nel):
        for k in range(0,mV):
            inode=iconV[k,iel]
            rq = rVnodes[k]
            sq = sVnodes[k]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
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
                tau_xx_nodal[inode] += dNNNVdx[k]*u[iconV[k,iel]]*2*eta[iel]        
                tau_zz_nodal[inode] += dNNNVdy[k]*v[iconV[k,iel]]*2*eta[iel]        
                tau_xz_nodal[inode] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]*2*eta[iel]+\
                                       0.5*dNNNVdx[k]*v[iconV[k,iel]]*2*eta[iel]   
            for k in range(0,mP):
                q[inode]+= NNNP[k]*p[iconP[k,iel]]   
            #end for
            cc[inode]+=1
        #end for
    #end for
    tau_xx_nodal/=cc
    tau_zz_nodal/=cc
    tau_xz_nodal/=cc
    q[:]/=cc[:]

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

    tau_rr_nodal=np.zeros(NV,dtype=np.float64)  
    tau_tt_nodal=np.zeros(NV,dtype=np.float64)  
    tau_rt_nodal=np.zeros(NV,dtype=np.float64)  

    tau_rr_nodal=tau_xx_nodal*np.sin(theta_nodal)**2+2*tau_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_zz_nodal*np.cos(theta_nodal)**2
    tau_tt_nodal=tau_xx_nodal*np.cos(theta_nodal)**2-2*tau_xz_nodal*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_zz_nodal*np.sin(theta_nodal)**2
    tau_rt_nodal=(tau_xx_nodal-tau_zz_nodal)*np.sin(theta_nodal)*np.cos(theta_nodal)+tau_xz_nodal*(-np.sin(theta_nodal)**2+np.cos(theta_nodal)**2)

    tau_rr=np.zeros(nel,dtype=np.float64)  
    tau_tt=np.zeros(nel,dtype=np.float64)  
    tau_rt=np.zeros(nel,dtype=np.float64)  

    tau_rr=tau_xx*np.sin(theta)**2+2*tau_xz*np.sin(theta)*np.cos(theta)+tau_zz*np.cos(theta)**2
    tau_tt=tau_xx*np.cos(theta)**2-2*tau_xz*np.sin(theta)*np.cos(theta)+tau_zz*np.sin(theta)**2
    tau_rt=(tau_xx-tau_zz)*np.sin(theta)*np.cos(theta)+tau_xz*(-np.sin(theta)**2+np.cos(theta)**2)

    sigma_rr_nodal=-q+tau_rr_nodal
    sigma_tt_nodal=-q+tau_tt_nodal
    sigma_rt_nodal=   tau_rr_nodal


    print("rotate stresses: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute traction at surface
    #####################################################################
    start = timing.time()

    tracfile=open('surface_traction_nodal.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%10e %10e %10e %10e \n" \
                          %(theta_nodal[i],tau_rr_nodal[i]-q[i],xV[i],zV[i]))
    tracfile.close()

    tracfile=open('surface_traction_elemental.ascii',"w")
    for iel in range(0,nel):
        if surface_node[iconV[0,iel]] or surface_node[iconV[1,iel]] or\
           surface_node[iconV[2,iel]] or surface_node[iconV[3,iel]] or\
           surface_node[iconV[4,iel]] or surface_node[iconV[5,iel]]:
           pel=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.
           tracfile.write("%10e %10e %10e %10e \n" %(theta[iel],tau_rr[iel]-pel,xc[iel],zc[iel]))

    tracfile=open('surface_vr.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.sin(theta_nodal[i])+v[i]*np.cos(theta_nodal[i])  ))
    tracfile.close()

    tracfile=open('surface_vt.ascii',"w")
    for i in range(0,NV):
        if surface_node[i]: 
           tracfile.write("%10e %10e \n" \
                          %(theta_nodal[i],u[i]*np.cos(theta_nodal[i])-v[i]*np.sin(theta_nodal[i]) ) )
    tracfile.close()



#        if np.sqrt(xc[iel]**2+zc[iel]**2)>R_outer-dr:
#           if surface_node[iconV[3,iel]] and xV[iconV[3,iel]]>0:
#              flag[iel]=1
#              pel=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.
#              tracfile.write("%10e %10e %10e %10e \n" %(theta[iel],tau_rr[iel]-pel,xc[iel],zc[iel]))
#           if surface_node[iconV[4,iel]] and xV[iconV[4,iel]]>0:
#              flag[iel]=1
#              pel=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.
#              tracfile.write("%10e %10e %10e %10e \n" %(theta[iel],tau_rr[iel]-pel,xc[iel],zc[iel]))
#           if surface_node[iconV[5,iel]] and xV[iconV[5,iel]]>0:
#              flag[iel]=1
#              pel=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.
#              tracfile.write("%10e %10e %10e %10e \n" %(theta[iel],tau_rr[iel]-pel,xc[iel],zc[iel]))
    tracfile.close()
        
    print("compute surface tractions: %.3f s" % (timing.time() - start))

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start = timing.time()

    year=365.25*3600*24
    cm=0.01
    u[:]/=(cm/year)
    v[:]/=(cm/year)

    filename = 'solution.vtu'
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
    vtufile.write("<DataArray type='Float32' Name='flag' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (flag[iel]))
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
    vtufile.write("<DataArray type='Float32' Name='theta (sph.coords)' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%e \n" %theta[iel])
    vtufile.write("</DataArray>\n")
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
        vtufile.write("%10e %10e %10e \n" %(u[i],0.,v[i]))
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
    vtufile.write("<DataArray type='Int32' Name='dyn topo' Format='ascii'> \n")
    for i in range(0,NV):
        if surface_node[i]:
           vtufile.write("%d \n" % (-(tau_rr_nodal[i]-q[i])/g0/rho_surf) ) # (-pI + tau).vec{n} / rho g0
        else:
           vtufile.write("%d \n" % 0)
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='theta (sph.coords)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %theta_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%e \n" %eta_nodal[i])
    #vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='viscosity (effective)' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%e \n" %etaeff_nodal[i])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %rho_nodal[i])
    vtufile.write("</DataArray>\n")
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
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='traction' Format='ascii'> \n")
    for i in range(0,NV):
        if surface_node[i]:
           vtufile.write("%10e %10e %10e \n" % (tau_rr_nodal[i],0.,tau_rt_nodal[i]))
        else:
           vtufile.write("%e %e %e \n" % (0,0,0))
    vtufile.write("</DataArray>\n")
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
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel]))
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
    #####################################################################
    start = timing.time()

    xM=np.empty(np_grav,dtype=np.float64)     
    yM=np.empty(np_grav,dtype=np.float64)     
    zM=np.empty(np_grav,dtype=np.float64)     
    gvect_x=np.zeros(np_grav,dtype=np.float64)   
    gvect_y=np.zeros(np_grav,dtype=np.float64)   
    gvect_z=np.zeros(np_grav,dtype=np.float64)   
    angleM=np.zeros(np_grav,dtype=np.float64)   

    dphi=2*np.pi/nel_phi

    for i in range(0,np_grav):

        #angleM[i]=np.pi/2-np.pi/(np_grav-1)*i
        #xM[i]=R_outer*np.cos(angleM[i])
        #yM[i]=0
        #zM[i]=R_outer*np.sin(angleM[i])

        angleM[i]=np.pi/2 #np.pi/5
        xM[i]=(R_outer+i*R_outer/(np_grav-1))*np.cos(angleM[i])
        yM[i]=0
        zM[i]=(R_outer+i*R_outer/(np_grav-1))*np.sin(angleM[i])

        total_mass=0
        total_vol=0
        for iel in range(0,nel):
            r_c=np.sqrt(xc[iel]**2+zc[iel]**2)
            z_c=zc[iel] 
            theta=np.arccos(z_c/r_c)
            for jel in range(0,nel_phi):
                x_c=r_c*np.sin(theta)*np.cos((jel+0.5)*dphi)
                y_c=r_c*np.sin(theta)*np.sin((jel+0.5)*dphi)
                vol=arear[iel]*dphi
                mass=vol*rho[iel]
                total_vol+=vol
                total_mass+=mass
                dist=np.sqrt((xM[i]-x_c)**2 + (yM[i]-y_c)**2 + (zM[i]-z_c)**2)
                gvect_x[i]-= Ggrav/dist**3*mass*(xM[i]-x_c)
                gvect_y[i]-= Ggrav/dist**3*mass*(yM[i]-y_c)
                gvect_z[i]-= Ggrav/dist**3*mass*(zM[i]-z_c)
            #end for
        #end for
        print('meas. point',i,':','M=',total_mass,6.4171e23,' | V=',total_vol,1.333333*np.pi*R_outer**3)
    #end for

    gvect=np.sqrt(gvect_x**2+gvect_y**2+gvect_z**2)
    rM=np.sqrt(xM**2+yM**2+zM**2)

    #np.savetxt('gravity.ascii',np.array([xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect]).T)

    print("compute gravity: %.3fs" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
