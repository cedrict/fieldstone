import numpy as np
import sys as sys
import time as timing
import triangle as tr
import scipy.sparse as sps
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import *
from tools_numba import *
from basis_functions_numba import *
from compute_gravity_at_point_numba import *

###############################################################################

Ggrav = 6.67430e-11
year=365.25*3600*24
cm=1e-2
km=1e3
g_mars=3.72076 #https://en.wikipedia.org/wiki/Mars
g_earth=9.81 

###############################################################################

print("-----------------------------")
print("---------- stone 96 ---------")
print("-----------------------------")

ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1e22 # numerical parameter for FEM

###############################################################################
#
#  ----------------R_outer               blob
#    layer 1     
#  ----------------R_12                  -----
#    layer 2                            /     \
#  ----------------R_23                 |  6  | 
#    layer 3                            \     /
#  ----------------R_34                  -----
#    layer 4
#  ----------------R_45
#    layer 5
#  ----------------R_inner
#
###############################################################################

if int(len(sys.argv)) > 1:
   print(sys.argv)
 
   R_inner=float(sys.argv[1])
   R_outer=float(sys.argv[2])
   R_12=float(sys.argv[3])
   R_23=float(sys.argv[4])
   R_34=float(sys.argv[5])
   R_45=float(sys.argv[6])
   rho1=float(sys.argv[7])
   rho2=float(sys.argv[8])
   rho3=float(sys.argv[9])
   rho4=float(sys.argv[10])
   rho5=float(sys.argv[11])
   eta1=float(sys.argv[12])
   eta2=float(sys.argv[13])
   eta3=float(sys.argv[14])
   eta4=float(sys.argv[15])
   eta5=float(sys.argv[16])
   R_blob=float(sys.argv[17])
   eccentricity=float(sys.argv[18])
   rho_blob=float(sys.argv[19])
   eta_blob=float(sys.argv[20])
   depth_blob=float(sys.argv[21])
   solve_stokes=(int(sys.argv[22])==1)
   compute_gravity=(int(sys.argv[23])==1)
   nstep=int(sys.argv[24])
   hhh=float(sys.argv[25])
   elt=int(sys.argv[26])

   if elt==1: element='Q2Q1'
   if elt==2: element='P2P1'
   if elt==3: element='CR'
   if elt==4: element='Q1+Q1'

else: 

   R_inner=3000e3
   R_outer=6000e3

   R_12=R_outer-50e3
   R_23=R_outer-150e3
   R_34=R_outer-500e3
   R_45=R_outer-1000e3

   rho1=3000
   rho2=3000
   rho3=3000
   rho4=3000
   rho5=3000

   eta1=1e22
   eta2=1e22
   eta3=1e22
   eta4=1e22
   eta5=1e22

   R_blob=0e3
   eccentricity=0
   rho_blob=3000
   eta_blob=1e25
   depth_blob=1000e3

   solve_stokes=True
   compute_gravity=False

   nstep=1 # number of time steps

   hhh=100e3 # element size at the surface

   element='Q2Q1'
   #element='P2P1'
   #element='CR'
   #element='Q1+Q1'

###############################################################################

if element=='P2P1':
   mV=6
   mP=3
elif element=='CR':
   mV=7
   mP=3
elif element=='Q2Q1':
   mV=9
   mP=4
elif element=='Q1+Q1':
   mV=5
   mP=4
else:
   exit('element not implemented')

###############################################################################

dt_user=50*year
   
eta_max=1e25
   
density_above=0
density_below=5900

g0=10

np_grav=80       
nel_phi=int(10*np.pi*R_outer/hhh)
height=10e3
nqel_grav=12 # 6 or 12

fs_method=3

axisymm=True

###############################################################################

np_blob=int(2*np.pi*R_blob/hhh)*2
b_blob=R_blob
a_blob=b_blob*np.sqrt(1/(1-eccentricity**2))
R_blob=R_outer-depth_blob   

###############################################################################
#boundary conditions at planet surface
#0: no-slip
#1: free-slip
#2: free (only top surface)
###############################################################################

surface_bc=1

cmb_bc=1

###############################################################################
# gravity acceleration
# TODO: if use_isog is false we would then need to read in a gravity profile
###############################################################################

use_isog=True

###############################################################################
# store all 6 densities and viscosities in single array
# compute the minimum density (we kinda implicitely expect it to be at 
# the surface) and subtract from the array.
###############################################################################

rhos=np.array([rho1,rho2,rho3,rho4,rho5,rho_blob])
etas=np.array([eta1,eta2,eta3,eta4,eta5,eta_blob])

rho_min=0 #np.min(rhos)
rhos-=rho_min

print('rho_min=',rho_min)
print('rhos=',rhos)

###############################################################################
# refinement node layers below surface
#
# --------------- R_outer
#    ----RA
#    ----RB
#    ----RC
#
#
#    ----RF
#    ----RE
#    ----RD
# --------------- R_inner
#
###############################################################################

#RA=R_outer-10e3
#RB=R_outer-20e3
#RC=R_outer-30e3
#RD=R_inner+10e3
#RE=R_inner+20e3
#RF=R_inner+30e3

###############################################################################

print('element=',element)
print('R_inner=',R_inner)
print('R_outer=',R_outer)
print('Volume=',4*np.pi/3*(R_outer**3-R_inner**3))
print('height=',height)
print('nel_phi=',nel_phi)
print('hhh=',hhh)
print('np_blob=',np_blob)
print('a_blob=',a_blob)
print('b_blob=',b_blob)
print('depth_blob=',depth_blob)
print('rho_blob=',rho_blob)
print('axisymm=',axisymm)
print("-----------------------------")

###############################################################################
start = timing.time()

if element=='CR':
   rVnodes=[0,1,0,0.5,0.5,0,1./3.] 
   sVnodes=[0,0,1,0,0.5,0.5,1./3.]
elif element=='P2P1':
   rVnodes=[0,1,0,0.5,0.5,0] 
   sVnodes=[0,0,1,0,0.5,0.5]
elif element=='Q2Q1':
   rVnodes=[-1, 1, 1, -1, 0, 1, 0, -1, 0]
   sVnodes=[-1, -1, 1, 1, -1, 0, 1, 0, 0]
elif element=='Q1+Q1':
   rVnodes=[-1, 1, 1, -1, 0]
   sVnodes=[-1, -1, 1, 1, 0]

#for i in range(0,mV):
#    print(NNV(rVnodes[i],sVnodes[i],element))
#for i in range(0,mP):
#    print(NNP(rVnodes[i],sVnodes[i],element))
    
print("checking basis functions: %.3fs" % (timing.time() - start))

###############################################################################
# integration points coordinates and weights 
# I tested that nqperdim=4 is best in both cases axisymm and not-axisymm
###############################################################################

if element=='Q2Q1':

   if axisymm:

      #nqperdim=3
      #coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      #weights=[5./9.,8./9.,5./9.]

      nqperdim=4
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]

      #nqperdim=5
      #qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
      #qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
      #qc5c=0.
      #qw5a=(322.-13.*np.sqrt(70.))/900.
      #qw5b=(322.+13.*np.sqrt(70.))/900.
      #qw5c=128./225.
      #coords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
      #weights=[qw5a,qw5b,qw5c,qw5b,qw5a]

   else:
  
      #nqperdim=3
      #coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
      #weights=[5./9.,8./9.,5./9.]

      nqperdim=4
      qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
      qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
      qw4a=(18-np.sqrt(30.))/36.
      qw4b=(18+np.sqrt(30.))/36.
      coords=[-qc4a,-qc4b,qc4b,qc4a]
      weights=[qw4a,qw4b,qw4b,qw4a]

   #end if
      
   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   print('nqel=',nqel)

elif element=='P2P1' or element=='CR':

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

elif element=='Q1+Q1':

   #nqperdim=3
   #coords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   #weights=[5./9.,8./9.,5./9.]

   nqperdim=4
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   coords=[-qc4a,-qc4b,qc4b,qc4a]
   weights=[qw4a,qw4b,qw4b,qw4a]


   nqel=nqperdim**2
   qcoords_r=np.empty(nqel,dtype=np.float64)
   qcoords_s=np.empty(nqel,dtype=np.float64)
   qweights=np.empty(nqel,dtype=np.float64) 

   counterq=0
   for iq in range(0,nqperdim):
       for jq in range(0,nqperdim):
           qcoords_r[counterq]=coords[iq]
           qcoords_s[counterq]=coords[jq]
           qweights[counterq]=weights[iq]*weights[jq]
           counterq+=1

   print('nqel=',nqel)

###############################################################################
# 12 point integration coeffs and weights (only for gravity) 
###############################################################################

qcoords_r_grav=np.zeros(nqel_grav,dtype=np.float64)
qcoords_s_grav=np.zeros(nqel_grav,dtype=np.float64)
qweights_grav=np.zeros(nqel_grav,dtype=np.float64)

if nqel_grav==nqel:
   qcoords_r_grav=qcoords_r
   qcoords_s_grav=qcoords_s
   qweights_grav =qweights

if nqel_grav==12:
   qcoords_r_grav[ 0]=0.24928674517091 ; qcoords_s_grav[ 0]=0.24928674517091 ; qweights_grav[ 0]=0.11678627572638/2
   qcoords_r_grav[ 1]=0.24928674517091 ; qcoords_s_grav[ 1]=0.50142650965818 ; qweights_grav[ 1]=0.11678627572638/2
   qcoords_r_grav[ 2]=0.50142650965818 ; qcoords_s_grav[ 2]=0.24928674517091 ; qweights_grav[ 2]=0.11678627572638/2
   qcoords_r_grav[ 3]=0.06308901449150 ; qcoords_s_grav[ 3]=0.06308901449150 ; qweights_grav[ 3]=0.05084490637021/2
   qcoords_r_grav[ 4]=0.06308901449150 ; qcoords_s_grav[ 4]=0.87382197101700 ; qweights_grav[ 4]=0.05084490637021/2
   qcoords_r_grav[ 5]=0.87382197101700 ; qcoords_s_grav[ 5]=0.06308901449150 ; qweights_grav[ 5]=0.05084490637021/2
   qcoords_r_grav[ 6]=0.31035245103378 ; qcoords_s_grav[ 6]=0.63650249912140 ; qweights_grav[ 6]=0.08285107561837/2
   qcoords_r_grav[ 7]=0.63650249912140 ; qcoords_s_grav[ 7]=0.05314504984482 ; qweights_grav[ 7]=0.08285107561837/2
   qcoords_r_grav[ 8]=0.05314504984482 ; qcoords_s_grav[ 8]=0.31035245103378 ; qweights_grav[ 8]=0.08285107561837/2
   qcoords_r_grav[ 9]=0.63650249912140 ; qcoords_s_grav[ 9]=0.31035245103378 ; qweights_grav[ 9]=0.08285107561837/2
   qcoords_r_grav[10]=0.31035245103378 ; qcoords_s_grav[10]=0.05314504984482 ; qweights_grav[10]=0.08285107561837/2
   qcoords_r_grav[11]=0.05314504984482 ; qcoords_s_grav[11]=0.63650249912140 ; qweights_grav[11]=0.08285107561837/2

###############################################################################
# Defining the nodes and vertices for constrained Delaunay triangulation
###############################################################################
start = timing.time()

if element=='P2P1' or element=='CR':

   nnt=int(np.pi*R_outer/hhh) 
   nnr=int((R_outer-R_inner)/hhh)+1 
   nnt2=int(np.pi*R_outer/hhh*1.8) 

   print('nnt=',nnt)
   print('nnt2=',nnt2)
   print('nnr=',nnr)

   counter=0

   #------------------------------------------------------------------------------
   # inner boundary counterclockwise
   #------------------------------------------------------------------------------
   theta = np.linspace(-np.pi*0.5, 0.5*np.pi,nnt, endpoint=False)      #half inner sphere in the x-positive domain
   pts_ib = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R_inner #nnt-points on inner boundary
   seg_ib = np.stack([np.arange(nnt), np.arange(nnt) + 1], axis=1)     #vertices on innerboundary (and the last vertices to the upper wall)
   for i in range(0,nnt):                                              #first point must be exactly on the y-axis
       if i==0:
          pts_ib[i,0]=0

   counter+=nnt

   #------------------------------------------------------------------------------
   # top vertical (left) wall 
   #------------------------------------------------------------------------------
   topw_z = np.linspace(R_inner,R_outer,nnr,endpoint=False)                #vertical boundary wall from inner to outer boundary sphere
   pts_topw = np.stack([np.zeros(nnr),topw_z],axis=1)                      #nnr-points on vertical wall
   seg_topw = np.stack([counter+np.arange(nnr),counter+np.arange(nnr)+1], axis=1)  #vertices on vertical wall

   counter+=nnr

   #------------------------------------------------------------------------------
   # outer boundary clockwise
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2,-np.pi/2,num=nnt2,endpoint=False)            #half outer sphere in the x-positive domain
   pts_ob = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_outer        #nnt-points on outer boundary
   seg_ob = np.stack([counter+np.arange(nnt2), counter+np.arange(nnt2)+1], axis=1) #vertices on outerboundary
   #for i in range(0,nnt2):                                                  #first point must be exactly on the y-axis
   #    if i==0:
   #       pts_ob[i,0]=0 # enforce x=0
   pts_ob[0,0]=0 # enforce x=0

   counter+=nnt2

   #------------------------------------------------------------------------------
   # bottom vertical wall
   #------------------------------------------------------------------------------
   botw_z = np.linspace(-R_outer,-R_inner,nnr,endpoint=False)              #vertical boundary wall from outer to inner boundary sphere
   pts_botw = np.stack([np.zeros(nnr),botw_z],axis=1)                      #nnr-points on vertical wall
   seg_botw = np.stack([counter+np.arange(nnr),counter+np.arange(nnr)+1], axis=1) #vertices on bottem vertical wall
   seg_botw[-1,1]=0 # enforce x=0  

   counter+=nnr

   #------------------------------------------------------------------------------
   # 3 layers of nodes just below surface
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2*0.999,-np.pi/2*0.999,num=nnt2,endpoint=True)            #half outer sphere in the x-positive domain

   #pts_moA = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RA     
   #seg_moA = np.stack([counter+np.arange(nnt2-1),counter+np.arange(nnt2-1)+1], axis=1) 
   #counter+=nnt2

   #pts_moB = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RB     
   #seg_moB = np.stack([counter+np.arange(nnt2-1),counter+np.arange(nnt2-1)+1], axis=1) 
   #counter+=nnt2

   #pts_moC = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RC     
   #seg_moC = np.stack([counter+np.arange(nnt2-1),counter+np.arange(nnt2-1)+1], axis=1) 
   #counter+=nnt2

   #------------------------------------------------------------------------------
   # 3 layers of nodes just above the CMB
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2*0.999,-np.pi/2*0.999,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain

   #pts_moD = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RD     
   #seg_moD = np.stack([counter+np.arange(nnt-1),counter+np.arange(nnt-1)+1], axis=1) 
   #counter+=nnt

   #pts_moE = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RE     
   #seg_moE = np.stack([counter+np.arange(nnt-1),counter+np.arange(nnt-1)+1], axis=1) 
   #counter+=nnt

   #pts_moF = np.stack([np.cos(theta),np.sin(theta)],axis=1)*RF     
   #seg_moF = np.stack([counter+np.arange(nnt-1),counter+np.arange(nnt-1)+1], axis=1) 
   #counter+=nnt

   #------------------------------------------------------------------------------
   # discontinuity #1
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
   pts_mo1 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_12        #nnt-points on outer boundary
   seg_mo1 = np.stack([counter+np.arange(nnt-1), counter+np.arange(nnt-1)+1], axis=1) #vertices on disc1
   for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
       if i==0 or i==nnt-1:
          pts_mo1[i,0]=0    

   counter+=nnt

   #------------------------------------------------------------------------------
   # discontinuity #2
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
   pts_mo2 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_23        #nnt-points on outer boundary
   seg_mo2 = np.stack([counter+np.arange(nnt-1), counter+np.arange(nnt-1)+1], axis=1) #vertices on disc2
   for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
       if i==0 or i==nnt-1:
          pts_mo2[i,0]=0    

   counter+=nnt

   #------------------------------------------------------------------------------
   # discontinuity #3
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
   pts_mo3 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_34        #nnt-points on outer boundary
   seg_mo3 = np.stack([counter+np.arange(nnt-1), counter+np.arange(nnt-1)+1], axis=1) #vertices on disc3
   for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
       if i==0 or i==nnt-1:
          pts_mo3[i,0]=0    

   counter+=nnt

   #------------------------------------------------------------------------------
   # discontinuity #4
   #------------------------------------------------------------------------------
   theta = np.linspace(np.pi/2,-np.pi/2,num=nnt,endpoint=True)            #half outer sphere in the x-positive domain
   pts_mo4 = np.stack([np.cos(theta),np.sin(theta)], axis=1)*R_45        #nnt-points on outer boundary
   seg_mo4 = np.stack([counter+np.arange(nnt-1), counter+np.arange(nnt-1)+1], axis=1) #vertices on disc3
   for i in range(0,nnt):                                                 #first and last point must be exactly on the y-axis
       if i==0 or i==nnt-1:
          pts_mo4[i,0]=0    

   counter+=nnt

   if np_blob>0:
      #------------------------------------------------------------------------------
      # blob 
      #------------------------------------------------------------------------------
      theta_bl = np.linspace(-np.pi/2,np.pi/2,num=np_blob,endpoint=True,dtype=np.float64) #half-sphere in the x and y positive domain
      pts_bl = np.stack([a_blob*np.cos(theta_bl),R_blob+b_blob*np.sin(theta_bl)], axis=1) #points on blob outersurface 
      seg_bl = np.stack([counter+np.arange(np_blob-1), counter+np.arange(np_blob-1)+1], axis=1) #vertices on outersurface blob
      for i in range(0,np_blob):                                              #first and last point must be exactly on the y-axis.
          #print(pts_bl[i,0],pts_bl[i,1])
          if i==0 or i==np_blob-1:
             pts_bl[i,0]=0
             print('corrected:',pts_bl[i,0],pts_bl[i,1])

      # Stacking the nodes and vertices 
      seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_mo1,seg_mo2,seg_mo3,seg_mo4,seg_bl])
      pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_mo1,pts_mo2,pts_mo3,pts_mo4,pts_bl]) 

      #seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_moA,seg_moB,seg_moC,seg_moD,seg_moE,seg_moF,seg_mo1,seg_mo2,seg_mo3,seg_mo4,seg_bl])
      #pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_moA,pts_moB,pts_moC,pts_moD,pts_moE,pts_moF,pts_mo1,pts_mo2,pts_mo3,pts_mo4,pts_bl]) 

   else:
      # Stacking the nodes and vertices 
      #seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_moA,seg_moB,seg_moC,seg_moD,seg_moE,seg_moF,seg_mo1,seg_mo2,seg_mo3,seg_mo4])
      #pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_moA,pts_moB,pts_moC,pts_moD,pts_moE,pts_moF,pts_mo1,pts_mo2,pts_mo3,pts_mo4]) 

      seg = np.vstack([seg_ib,seg_topw,seg_ob,seg_botw,seg_mo1,seg_mo2,seg_mo3,seg_mo4])
      pts = np.vstack([pts_ib,pts_topw,pts_ob,pts_botw,pts_mo1,pts_mo2,pts_mo3,pts_mo4]) 

   #print(seg)

   #put all segments and nodes in a dictionary

   dict_nodes = dict(vertices=pts, segments=seg,holes=[[0,0]]) #no core so we add a hole at x=0,y=0

   print("generate nodes: %.3f s" % (timing.time() - start))

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

   target_area=str(int(hhh**2/2))
   print('target_area=',target_area)

   dict_mesh = tr.triangulate(dict_nodes,'pqa'+target_area)

   #tr.compare(plt, dict_nodes, dict_mesh)
   #plt.axis
   #plt.show()
   #exit()

   print("call mesher: %.3f s" % (timing.time() - start))
   start = timing.time()

   ## define icon, x and z for P1 mesh
   iconP1=dict_mesh['triangles'] ; iconP1=iconP1.T
   xP1=dict_mesh['vertices'][:,0]
   zP1=dict_mesh['vertices'][:,1]
   NP1=np.size(xP1)
   mP,nel=np.shape(iconP1)
   export_elements_to_vtu(xP1,zP1,iconP1,'meshP1.vtu')

   print('NP1=',NP1)
   print('nel=',nel)
   #exit()

   print("make P1 mesh: %.3f s" % (timing.time() - start))
   start = timing.time()

   NV0,xP2,zP2,iconP2=mesh_P1_to_P2(NP1,nel,xP1,zP1,iconP1)

   print("make P2 mesh: %.3f s" % (timing.time() - start))

   export_elements_to_vtuP2(xP2,zP2,iconP2,'meshP2.vtu')
   #exit()

###############################################################################
# compute NP, NV, NfemV, NfemP, Nfem for both element pairs
# and build xV,zV,iconV,xP,zP,iconP
###############################################################################
start = timing.time()

if element=='CR':
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
   radiusP=np.empty(NP,dtype=np.float64)  # x coordinates
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
       radiusP[counter]=np.sqrt(xP[counter]**2+zP[counter]**2)
       counter+=1

   radiusV=np.empty(NV,dtype=np.float64)  
   thetaV=np.empty(NV,dtype=np.float64)  
   for i in range(0,NV):
       thetaV[i]=np.arctan2(xV[i],zV[i])
       radiusV[i]=np.sqrt(xV[i]**2+zV[i]**2)

elif element=='P2P1':
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
   radiusP=np.empty(NfemP,dtype=np.float64)  # y coordinates
   xP[0:NP]=xP1[0:NV]
   zP[0:NP]=zP1[0:NV]
   radiusP[:]=np.sqrt(xP[:]**2+zP[:]**2)

   radiusV=np.empty(NV,dtype=np.float64)  
   thetaV=np.empty(NV,dtype=np.float64)  
   for i in range(0,NV):
       thetaV[i]=np.arctan2(xV[i],zV[i])
       radiusV[i]=np.sqrt(xV[i]**2+zV[i]**2)



elif element=='Q2Q1':

   nelt=int(np.pi*R_outer/hhh)
   nelr=int((R_outer-R_inner)/hhh)
   print('nelt=',nelt)
   print('nelr=',nelr)
   nel=nelt*nelr

   nnx=2*nelt+1
   nny=2*nelr+1
   NV=nnx*nny
   NP=(nelr+1)*(nelt+1)
   NfemV=NV*ndofV    # number of velocity dofs
   NfemP=NP*ndofP    # number of pressure dofs

   xV=np.empty(NV,dtype=np.float64)  # x coordinates
   zV=np.empty(NV,dtype=np.float64)  # y coordinates
   radiusV=np.empty(NV,dtype=np.float64)  
   thetaV=np.empty(NV,dtype=np.float64)  

   counter = 0 
   for j in range(0,nny):
       for i in range(0,nnx):
           thetaV[counter]=np.pi/2-i*(np.pi/nelt)/2.
           radiusV[counter]=R_inner+j*((R_outer-R_inner)/nelr)/2.
           xV[counter]=radiusV[counter]*np.cos(thetaV[counter])
           zV[counter]=radiusV[counter]*np.sin(thetaV[counter])
           counter += 1
       #end for
   #end for

   #redefine thetaV as colatitude [0,pi]
   thetaV=np.empty(NV,dtype=np.float64)  
   for i in range(0,NV):
       thetaV[i]=np.arctan2(xV[i],zV[i])

   iconV=np.zeros((mV,nel),dtype=np.int32)
   iconP=np.zeros((mP,nel),dtype=np.int32)

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
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
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconP[0,counter]=i+j*(nelt+1)
           iconP[1,counter]=i+1+j*(nelt+1)
           iconP[2,counter]=i+1+(j+1)*(nelt+1)
           iconP[3,counter]=i+(j+1)*(nelt+1)
           counter += 1
       #end for
   #end for

   xP=np.empty(NP,dtype=np.float64)  # x coordinates
   zP=np.empty(NP,dtype=np.float64)  # y coordinates
   radiusP=np.empty(NP,dtype=np.float64)  

   counter=0
   for iel in range(0,nel):
       for k in range(0,4):
           xP[iconP[k,iel]]=xV[iconV[k,iel]]
           zP[iconP[k,iel]]=zV[iconV[k,iel]]
           radiusP[counter]=np.sqrt(xP[counter]**2+zP[counter]**2)

   #np.savetxt('gridV.ascii',np.array([xV,zV]).T,header='# xV,zV')
   #np.savetxt('gridP.ascii',np.array([xP,zP]).T,header='# xP,zP')


elif element=='Q1+Q1':

   nelt=int(np.pi*R_outer/hhh)
   nelr=int((R_outer-R_inner)/hhh)
   print('nelt=',nelt)
   print('nelr=',nelr)
   nel=nelt*nelr

   nnx=nelt+1
   nny=nelr+1
   NV=nnx*nny+nel
   NP=nnx*nny
   NfemV=NV*ndofV    # number of velocity dofs
   NfemP=NP*ndofP    # number of pressure dofs

   xV=np.empty(NV,dtype=np.float64)  # x coordinates
   zV=np.empty(NV,dtype=np.float64)  # y coordinates
   radiusV=np.empty(NV,dtype=np.float64)  
   thetaV=np.empty(NV,dtype=np.float64)  

   counter = 0 
   for j in range(0,nny):
       for i in range(0,nnx):
           thetaV[counter]=np.pi/2-i*(np.pi/nelt)
           radiusV[counter]=R_inner+j*((R_outer-R_inner)/nelr)
           xV[counter]=radiusV[counter]*np.cos(thetaV[counter])
           zV[counter]=radiusV[counter]*np.sin(thetaV[counter])
           counter += 1
       #end for
   #end for

   #redefine thetaV as colatitude [0,pi]
   thetaV=np.empty(NV,dtype=np.float64)  
   for i in range(0,NV):
       thetaV[i]=np.arctan2(xV[i],zV[i])

   iconV=np.zeros((mV,nel),dtype=np.int32)
   iconP=np.zeros((mP,nel),dtype=np.int32)

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconV[0,counter]=i+j*(nelt+1)
           iconV[1,counter]=i+1+j*(nelt+1)
           iconV[2,counter]=i+1+(j+1)*(nelt+1)
           iconV[3,counter]=i+(j+1)*(nelt+1)
           iconV[4,counter]=nnx*nny+counter
           counter += 1
       #end for
   #end for

   counter = 0
   for j in range(0,nelr):
       for i in range(0,nelt):
           iconP[0,counter]=i+j*(nelt+1)
           iconP[1,counter]=i+1+j*(nelt+1)
           iconP[2,counter]=i+1+(j+1)*(nelt+1)
           iconP[3,counter]=i+(j+1)*(nelt+1)
           counter += 1
       #end for
   #end for

   dr=(R_outer-R_inner)/nelr
   dtheta=np.pi/nelt
   for iel in range(0,nel):
       radiusV[iconV[4,iel]]=radiusV[iconV[0,iel]]+dr/2
       thetaV[iconV[4,iel]]=thetaV[iconV[0,iel]]+dtheta/2
       xV[iconV[4,iel]]=radiusV[iconV[4,iel]]*np.sin(thetaV[iconV[4,iel]])
       zV[iconV[4,iel]]=radiusV[iconV[4,iel]]*np.cos(thetaV[iconV[4,iel]])

   xP=np.empty(NP,dtype=np.float64)  # x coordinates
   zP=np.empty(NP,dtype=np.float64)  # y coordinates
   radiusP=np.empty(NP,dtype=np.float64)  

   counter=0
   for iel in range(0,nel):
       for k in range(0,4):
           xP[iconP[k,iel]]=xV[iconV[k,iel]]
           zP[iconP[k,iel]]=zV[iconV[k,iel]]
           radiusP[counter]=np.sqrt(xP[counter]**2+zP[counter]**2)

   np.savetxt('gridV.ascii',np.array([xV,zV]).T,header='# xV,zV')
   np.savetxt('gridP.ascii',np.array([xP,zP]).T,header='# xP,zP')


else:
   exit('unknown element!')

print('     -> nel', nel)
print('     -> NV', NV)
print("     -> xV (min/max): %.4f %.4f" %(np.min(xV),np.max(xV)))
print("     -> zV (min/max): %.4f %.4f" %(np.min(zV),np.max(zV)))
print("     -> xP (min/max): %.4f %.4f" %(np.min(xP),np.max(xP)))
print("     -> zP (min/max): %.4f %.4f" %(np.min(zP),np.max(zP)))
print("     -> thetaV (m,M)  %.6e %.6e" %(np.min(thetaV),np.max(thetaV)))
print("     -> radiusV (m,M) %.6e %.6e" %(np.min(radiusV),np.max(radiusV)))

print("generate FE meshes: %.3f s" % (timing.time() - start))

###############################################################################
# read profiles 
###############################################################################

# TODO


###############################################################################
# from density profile build total mass and moment of inertia
###############################################################################

# TODO


###############################################################################
# compute 
###############################################################################
start = timing.time()

surface_node=np.zeros(NV,dtype=bool) 
cmb_node=np.zeros(NV,dtype=bool) 

for i in range(0,NV):
    if radiusV[i]/R_outer>0.999:
       surface_node[i]=True
    if radiusV[i]/R_inner<1.001:
       cmb_node[i]=True

print("flag surface & cmb V nodes: %.3f s" % (timing.time() - start))

###############################################################################
# if fs_method==3 we use Lagrange multipliers so matrix needs to be bigger
###############################################################################

Nlm=0
if fs_method==3:
   if surface_bc==1:
      Nlm+=np.sum(surface_node)
   if cmb_bc==1:
      Nlm+=np.sum(cmb_node)

Nfem=NfemV+NfemP+Nlm  # total number of dofs

print('     -> NfemV', NfemV)
print('     -> NfemP', NfemP)
print('     -> Nlm', Nlm)
print('     -> Nfem', Nfem)

###############################################################################
# just to be sure that mid-egde nodes are indeed in the middle
###############################################################################
start = timing.time()

for i in range(0,NV):
    if abs(xV[i])/R_outer<1e-6:
       xV[i]=0

for i in range(0,NP):
    if abs(xP[i])/R_outer<1e-6:
       xP[i]=0

#if element=='P2P1':
#   for iel in range(0,nel):
#       xV[iconV[3,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
#       zV[iconV[3,iel]]=0.5*(zV[iconV[0,iel]]+zV[iconV[1,iel]])
#       xV[iconV[4,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
#       zV[iconV[4,iel]]=0.5*(zV[iconV[1,iel]]+zV[iconV[2,iel]])
#       xV[iconV[5,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[0,iel]])
#       zV[iconV[5,iel]]=0.5*(zV[iconV[2,iel]]+zV[iconV[0,iel]])
       #print(xV[iconV[0,iel]]-xP[iconP[0,iel]],zV[iconV[0,iel]]-zP[iconP[0,iel]])
       #print(xV[iconV[1,iel]]-xP[iconP[1,iel]],zV[iconV[1,iel]]-zP[iconP[1,iel]])
       #print(xV[iconV[2,iel]]-xP[iconP[2,iel]],zV[iconV[2,iel]]-zP[iconP[2,iel]])

#if element=='CR':
#   for iel in range(0,nel):
#       xV[iconV[3,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
#       zV[iconV[3,iel]]=0.5*(zV[iconV[0,iel]]+zV[iconV[1,iel]])
#       xV[iconV[4,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
#       zV[iconV[4,iel]]=0.5*(zV[iconV[1,iel]]+zV[iconV[2,iel]])
#       xV[iconV[5,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[0,iel]])
#       zV[iconV[5,iel]]=0.5*(zV[iconV[2,iel]]+zV[iconV[0,iel]])
#       xV[iconV[6,iel]]=np.sum(xV[iconV[0:3,iel]])/3
#       zV[iconV[6,iel]]=np.sum(zV[iconV[0:3,iel]])/3

if element=='Q2Q1':
   for iel in range(0,nel):
       xV[iconV[4,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
       zV[iconV[4,iel]]=0.5*(zV[iconV[0,iel]]+zV[iconV[1,iel]])
       xV[iconV[5,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
       zV[iconV[5,iel]]=0.5*(zV[iconV[1,iel]]+zV[iconV[2,iel]])
       xV[iconV[6,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[3,iel]])
       zV[iconV[6,iel]]=0.5*(zV[iconV[2,iel]]+zV[iconV[3,iel]])
       xV[iconV[7,iel]]=0.5*(xV[iconV[3,iel]]+xV[iconV[0,iel]])
       zV[iconV[7,iel]]=0.5*(zV[iconV[3,iel]]+zV[iconV[0,iel]])
       xV[iconV[8,iel]]=np.sum(xV[iconV[0:4,iel]])/4
       zV[iconV[8,iel]]=np.sum(zV[iconV[0:4,iel]])/4

#np.savetxt('gridV_after.ascii',np.array([xV,zV]).T,header='# xV,zV')

print("straightening element edges: %.3f s" % (timing.time() - start))

###############################################################################
# flag surface P nodes (needed for p normalisation)
###############################################################################
start = timing.time()

thetaP=np.zeros(NP,dtype=np.float64)
surface_Pnode=np.zeros(NP,dtype=bool) 
cmb_Pnode=np.zeros(NP,dtype=bool) 
for i in range(0,NP):
    thetaP[i]=np.arctan2(xP[i],zP[i])
    radiusP[i]=np.sqrt(xP[i]**2+zP[i]**2)
    if radiusP[i]>=0.999*R_outer: 
       surface_Pnode[i]=True
    if radiusP[i]<=1.001*R_inner: 
       cmb_Pnode[i]=True

#np.savetxt('surface_Pnode.ascii',np.array([xP,zP,surface_Pnode,radiusP]).T,header='# x,y,flag,r')
#np.savetxt('cmb_Pnode.ascii',np.array([xP,zP,cmb_Pnode,radiusP]).T,header='# x,y,flag,r')

print("flag surface &cmb P nodes: %.3f s" % (timing.time() - start))

######################################################################
# compute normal vector to surfaces
######################################################################
start = timing.time()

nx=np.zeros(NV,dtype=np.float64)  
nz=np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    if surface_node[i]:
       nx[i]=xV[i]/radiusV[i]
       nz[i]=zV[i]/radiusV[i]
    if cmb_node[i]:
       nx[i]=-xV[i]/radiusV[i]
       nz[i]=-zV[i]/radiusV[i]
    if xV[i]<0.000001*R_inner and (not surface_node[i]) and (not cmb_node[i]):
       nx[i]=-1
       nz[i]=0

print("compute normals: %.3f s" % (timing.time() - start))

######################################################################
# compute element center coordinates
######################################################################
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)  
zc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]= np.sum(xV[iconV[0:mV,iel]])/mV
    zc[iel]= np.sum(zV[iconV[0:mV,iel]])/mV
    #print(xc[iel],xV[iconV[4,iel]])
    #print(zc[iel],zV[iconV[4,iel]])

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> zc (m,M) %.6e %.6e " %(np.min(zc),np.max(zc)))

#np.savetxt('centers.ascii',np.array([xc,zc]).T)

print("compute element center coords: %.3f s" % (timing.time() - start))

###############################################################################
# flag elements inside blob and layers
###############################################################################
start = timing.time()

blob=np.zeros(nel,dtype=bool)
material=np.zeros(nel,dtype=np.int16)
 
for iel in range(0,nel):
    if np_blob>0 :
       if xc[iel]**2/a_blob**2+(zc[iel]-R_blob)**2/b_blob**2<1:
          material[iel]=6
          blob[iel]=True

    radiusc=np.sqrt(xc[iel]**2+zc[iel]**2)

    if radiusc>R_12 and not blob[iel]:
       material[iel]=1
    elif radiusc>R_23 and not blob[iel]:
       material[iel]=2
    elif radiusc>R_34 and not blob[iel]:
       material[iel]=3
    elif radiusc>R_45 and not blob[iel]:
       material[iel]=4
    elif not blob[iel]:
       material[iel]=5

print("flag elts in blob & layers: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

xq=np.zeros(nel*nqel,dtype=np.float64) 
zq=np.zeros(nel*nqel,dtype=np.float64) 
jcobq=np.zeros(nel*nqel,dtype=np.float64) 
radiusq=np.zeros(nel*nqel,dtype=np.float64) 
thetaq=np.zeros(nel*nqel,dtype=np.float64) 

area=np.zeros(nel,dtype=np.float64) 
arear=np.zeros(nel,dtype=np.float64) 
vol_blob=0

counterq=0
for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNV(rq,sq,element)
        dNNNVdr=dNNVdr(rq,sq,element)
        dNNNVds=dNNVds(rq,sq,element)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
        jcobq[counterq]=np.linalg.det(jcb)
        area[iel]+=jcobq[counterq]*weightq
        xq[counterq]=NNNV.dot(xV[iconV[:,iel]])
        zq[counterq]=NNNV.dot(zV[iconV[:,iel]])
        arear[iel]+=jcobq[counterq]*weightq*xq[counterq]*2*np.pi
        counterq+=1
    if blob[iel]:
       vol_blob+=arear[iel]

VOL=4*np.pi/3*(R_outer**3-R_inner**3)
VOL_blob=4/3*np.pi*a_blob**2*b_blob
        
radiusq=np.sqrt(xq**2+zq**2)
thetaq=np.arccos(zq/radiusq)
        
#np.savetxt('qpoints.ascii',np.array([xq,zq,radiusq,thetaq]).T,header='# x,y,u,v')

print("     -> area (m,M) %e %e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %e " %(area.sum()))
print("     -> total area (anal) %e " %(np.pi*(R_outer**2-R_inner**2)/2))
print("     -> total vol  (meas) %e " %(arear.sum()))
print("     -> total vol  (error) %e percent | nel= %d" %((abs(arear.sum()/VOL-1))*100,nel))
if VOL_blob>0:
   print("     -> blob vol (meas) %e " %(vol_blob))
   print("     -> blob vol (anal) %e " %(VOL_blob))
   print("     -> blob vol (error) %.6f percent" % (abs(vol_blob/VOL_blob-1)*100))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
# at the moment the code does free slip on ALL boundaries, nothing else
###############################################################################
#start = timing.time()

left_node=np.zeros(NV,dtype=bool) 
bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    #Left boundary  
    if xV[i]<0.000001*R_inner: 
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV  ] = 0 # vx=0
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # vx=0
       left_node[i]=True

#planet surface
    if surface_node[i] and surface_bc==0: #no-slip surface
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
#core mantle boundary
    if cmb_node[i] and cmb_bc==0: #no-slip surface
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
#print("define boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# flag all elements with a node touching the surface r=R_outer
# or r=R_inner, used later for free slip b.c.
###############################################################################
start = timing.time()

flag_top=np.zeros(nel,dtype=np.float64)  
flag_bot=np.zeros(nel,dtype=np.float64)  

if element=='Q1+Q1':
   for iel in range(0,nel):
       if surface_node[iconV[0,iel]] or surface_node[iconV[1,iel]] or\
          surface_node[iconV[2,iel]] or surface_node[iconV[3,iel]]:
          flag_top[iel]=1
       if cmb_node[iconV[0,iel]] or cmb_node[iconV[1,iel]] or\
          cmb_node[iconV[2,iel]] or cmb_node[iconV[3,iel]]:
          flag_bot[iel]=1


else:
   for iel in range(0,nel):
       if surface_node[iconV[0,iel]] or surface_node[iconV[1,iel]] or\
          surface_node[iconV[2,iel]] or surface_node[iconV[3,iel]] or\
          surface_node[iconV[4,iel]] or surface_node[iconV[5,iel]]:
          flag_top[iel]=1
       if cmb_node[iconV[0,iel]] or cmb_node[iconV[1,iel]] or\
          cmb_node[iconV[2,iel]] or cmb_node[iconV[3,iel]] or\
          cmb_node[iconV[4,iel]] or cmb_node[iconV[5,iel]]:
          flag_bot[iel]=1

print("flag elts on boundaries: %.3f s" % (timing.time() - start))

###############################################################################
# assign density & viscosity to quadrature points
###############################################################################
start = timing.time()
    
etaq = np.zeros(nel*nqel,dtype=np.float64) 
rhoq = np.zeros(nel*nqel,dtype=np.float64) 

counter=0
for iel in range(0,nel):
    for kq in range (0,nqel):
        etaq[counter]=etas[material[iel]-1]
        rhoq[counter]=rhos[material[iel]-1]
        counter+=1

print("     -> rhoq (m,M) %.6e %.6e " %(np.min(rhoq),np.max(rhoq)))
print("     -> etaq (m,M) %.6e %.6e " %(np.min(etaq),np.max(etaq)))

print("compute etaq,rhoq: %.3f s" % (timing.time() - start))

###############################################################################
###############################################################################
# time stepping
###############################################################################
###############################################################################

for istep in range(0,nstep):

    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [ G 0 ][p] [h]
    ###########################################################################
    start = timing.time()

    A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs      = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
    if axisymm:
       c_mat=np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
       b_mat= np.zeros((4,ndofV*mV),dtype=np.float64) # gradient matrix B 
       N_mat= np.zeros((4,ndofP*mP),dtype=np.float64) # matrix  
    else:
       c_mat=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       b_mat= np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
       N_mat= np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

    mass=0

    if solve_stokes:

        counterq=0
        for iel in range(0,nel):

            if iel%2000==0:
               print(iel,'/',nel)

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

                NNNV=NNV(rq,sq,element)
                dNNNVdr=dNNVdr(rq,sq,element)
                dNNNVds=dNNVds(rq,sq,element)
                NNNP[0:mP]=NNP(rq,sq,element)

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
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

                # compute etaq, rhoq
                #etaq,rhoq=material_model(xq,yq,eta_blob,rho_blob,z_blob,a_blob,b_blob,npt_rho,\
                #                         npt_eta,profile_rho,profile_eta,blobtype)

                if axisymm:
                   coeffq=weightq*jcob*2*np.pi*xq[counterq]
                else:
                   coeffq=weightq*jcob

                if axisymm:
                   for i in range(0,mV):
                       b_mat[0:4,2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                               [NNNV[i]/xq[counterq],0.],
                                               [0.        ,dNNNVdy[i]],
                                               [dNNNVdy[i],dNNNVdx[i]]]
                   for i in range(0,mP):
                       N_mat[0,i]=NNNP[i]
                       N_mat[1,i]=NNNP[i]
                       N_mat[2,i]=NNNP[i]
                else:
                   for i in range(0,mV):
                       b_mat[0:3,2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                               [0.        ,dNNNVdy[i]],
                                               [dNNNVdy[i],dNNNVdx[i]]]
                   for i in range(0,mP):
                       N_mat[0,i]=NNNP[i]
                       N_mat[1,i]=NNNP[i]

                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*coeffq 

                G_el-=b_mat.T.dot(N_mat)*coeffq 

                mass+=rhoq[counterq]*coeffq 

                #compute gx,gy
                #radq=np.sqrt(xq[counterq]**2+zq[counterq]**2)
                if use_isog:
                   grav=g0
                else:
                   grav=profile_grav[int(radiusq[counterq]/1000)]
                angle=np.arctan2(zq[counterq],xq[counterq])

                for i in range(0,mV):
                    f_el[ndofV*i  ]-=NNNV[i]*rhoq[counterq]*coeffq*grav*np.cos(angle)
                    f_el[ndofV*i+1]-=NNNV[i]*rhoq[counterq]*coeffq*grav*np.sin(angle)
                #end for

                counterq+=1

            #end for kq

            #G_el*=eta_ref/R_outer

            if (fs_method==1 and surface_bc==1 and flag_top[iel]) or (cmb_bc==1 and flag_bot[iel]):
               for k in range(0,mV):
                   inode=iconV[k,iel]
                   if surface_node[inode] or cmb_node[inode]:
                      RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                      for i in range(0,mV*ndofV):
                          RotMat[i,i]=1.
                      angle=np.pi/2-thetaV[inode]
                      #RotMat[2*k  ,2*k]= np.cos(thetaV[inode]) ; RotMat[2*k  ,2*k+1]=np.sin(thetaV[inode])
                      #RotMat[2*k+1,2*k]=-np.sin(thetaV[inode]) ; RotMat[2*k+1,2*k+1]=np.cos(thetaV[inode])
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

            G_el*=eta_ref/R_outer

            if fs_method==2: #do not use!
               for k1 in range(0,mV):
                   inode=iconV[k1,iel]
                   if nx[inode]**2+nz[inode]**2>0:
                      #print(xV[inode],zV[inode])
                      if abs(nz[inode])<1e-8: #equator and left boundary -> u=0
                         ikk=ndofV*k1
                         K_ref=K_el[ikk,ikk]
                         K_el[ikk,:]=0
                         K_el[:,ikk]=0
                         #for jkk in range(0,mV*ndofV):
                         #    K_el[ikk,jkk]=0
                         #    K_el[jkk,ikk]=0
                         K_el[ikk,ikk]=K_ref
                         f_el[ikk]=0
                         G_el1[ikk,:]=0
                         G_el2[ikk,:]=0
                      elif abs(nx[inode])<1e-8: #poles -> u=0 AND w=0
                         #fix x component
                         ikk=ndofV*k1
                         K_ref=K_el[ikk,ikk]
                         for jkk in range(0,mV*ndofV):
                             K_el[ikk,jkk]=0
                             K_el[jkk,ikk]=0
                         K_el[ikk,ikk]=K_ref
                         f_el[ikk]=0
                         G_el1[ikk,:]=0
                         G_el2[ikk,:]=0
                         #fix y component
                         ikk=ndofV*k1+1
                         K_ref=K_el[ikk,ikk]
                         for jkk in range(0,mV*ndofV):
                             K_el[ikk,jkk]=0
                             K_el[jkk,ikk]=0
                         K_el[ikk,ikk]=K_ref
                         f_el[ikk]=0
                         G_el1[ikk,:]=0
                         G_el2[ikk,:]=0
                      else:
                         if abs(nx[inode])>=abs(nz[inode]):
                            print(nz[inode]/nx[inode])
                            ikk=ndofV*k1
                            K_ref=K_el[ikk,ikk]
                            K_el[ikk,:]=0
                            K_el[ikk,ikk]=K_ref
                            K_el[ikk,ikk+1]=K_ref*nz[inode]/nx[inode]
                            G_el1[ikk,:]=0
                            f_el[ikk]=0
                         else:
                            ikk=ndofV*k1+1
                            K_ref=K_el[ikk,ikk]
                            K_el[ikk,:]=0
                            K_el[ikk,ikk-1]=K_ref*nx[inode]/nz[inode]
                            K_el[ikk,ikk  ]=K_ref
                            G_el1[ikk,:]=0
                            f_el[ikk]=0
                         #end if
                      #end if
                   #end if
               #end for
            #end if

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

            #G_el*=eta_ref/R_outer
            #h_el*=eta_ref/R_outer

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
                        #end for
                    #end for
                    for k2 in range(0,mP):
                        jkk=k2
                        m2 =iconP[k2,iel]
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
        rhs[NfemV:NfemV+NfemP]=h_rhs

        print('     -> mass=',mass,VOL*3000,abs(mass-VOL*3000)/(VOL*3000))

        print("build FE matrix: %.3f s" % (timing.time() - start))

        ###############################################################################
        # Lagrange multipliers business
        ###############################################################################

        if fs_method==3:

           start = timing.time()

           counter=NfemV+NfemP

           if surface_bc==1:
              for i in range(0,NV):
                  if surface_node[i]:
                     # we need nx[i]*u[i]+nz[i]*w[i]=0
                     A_sparse[counter,2*i  ]=nx[i]*eta_ref#/R_outer
                     A_sparse[counter,2*i+1]=nz[i]*eta_ref#/R_outer
                     A_sparse[2*i  ,counter]=nx[i]*eta_ref#/R_outer
                     A_sparse[2*i+1,counter]=nz[i]*eta_ref#/R_outer
                     counter+=1

           if cmb_bc==1:
              for i in range(0,NV):
                  if cmb_node[i]:
                     # we need nx[i]*u[i]+nz[i]*w[i]=0
                     A_sparse[counter,2*i  ]=nx[i]*eta_ref#/R_outer
                     A_sparse[counter,2*i+1]=nz[i]*eta_ref#/R_outer
                     A_sparse[2*i  ,counter]=nx[i]*eta_ref#/R_outer
                     A_sparse[2*i+1,counter]=nz[i]*eta_ref#/R_outer
                     counter+=1

           print("build L block (%.3fs)" % (timing.time() - start))

        ######################################################################
        # solve system
        ######################################################################
        start = timing.time()

        sparse_matrix=A_sparse.tocsr()

        sol=sps.linalg.spsolve(sparse_matrix,rhs)

        u,v=np.reshape(sol[0:NfemV],(NV,2)).T
        p=sol[NfemV:NfemV+NfemP]*eta_ref/R_outer

        if fs_method==3: 
           l=sol[NfemV+NfemP:Nfem]
           print("     -> l (m,M) %.4f %.4f " %(np.min(l),np.max(l)))
           np.savetxt('solution_lambda.ascii',np.array([l]).T)

        np.savetxt('solution_velocity.ascii',np.array([xV,zV,u/cm*year,v/cm*year,radiusV]).T,header='# x,y,u,v,r')
        np.savetxt('solution_pressure.ascii',np.array([xP,zP,p,radiusP]).T,header='# x,y,p,r')

        print("     -> u (m,M) %.6e %.6e cm/year" %(np.min(u/cm*year),np.max(u/cm*year)))
        print("     -> v (m,M) %.6e %.6e cm/year" %(np.min(v/cm*year),np.max(v/cm*year)))
        print("     -> p (m,M) %.6e %.6e Mpa" %(np.min(p/1e6),np.max(p/1e6)))

        print("solve time: %.3f s" % (timing.time() - start))

        ######################################################################
        # compute cylindrical coordinates velocity components
        ######################################################################

        vr = np.zeros(NV,dtype=np.float64)
        vt = np.zeros(NV,dtype=np.float64)
        vel= np.zeros(NV,dtype=np.float64)

        vel[:]=np.sqrt(u[:]**2+v[:]**2)

        vr[:]=u[:]*np.sin(thetaV[:])+v[:]*np.cos(thetaV[:])
        vt[:]=u[:]*np.cos(thetaV[:])-v[:]*np.sin(thetaV[:])

        print("     -> vr (m,M) %.6e %.6e cm/year" %(np.min(vr/cm*year),np.max(vr/cm*year)))
        print("     -> vt (m,M) %.6e %.6e cm/year" %(np.min(vt/cm*year),np.max(vt/cm*year)))

        ######################################################################
        # compute vrms 
        ######################################################################
        start = timing.time()

        vrms=0
        counter=0
        for iel in range(0,nel):
            for kq in range (0,nqel):
                # position & weight of quad. point
                rq=qcoords_r[kq]
                sq=qcoords_s[kq]
                weightq=qweights[kq]
                NNNV[0:mV]=NNV(rq,sq,element)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
                dNNNVds[0:mV]=dNNVds(rq,sq,element)
                NNNP[0:mP]=NNP(rq,sq,element)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*zV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*zV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                uq=NNNV[0:mV].dot(u[iconV[0:mV,iel]])/cm*year
                vq=NNNV[0:mV].dot(v[iconV[0:mV,iel]])/cm*year
                vrms+=(uq**2+vq**2)*jcob*weightq *xq[counter]*2*np.pi
                counter+=1
            #end for
        #end for
        vrms=np.sqrt(vrms/(4/3*np.pi*(R_outer**3-R_inner**3)))

        print("     -> nel= %6d ; vrms (cm/year)= %.7e " %(nel,vrms))

        print("compute vrms: %.3f s" % (timing.time() - start))

        ######################################################################
        # if free slip  or no slip is used at the surface then there is 
        # a pressure nullspace which needs to be removed. 
        # I here make sure that the pressure is zero on the surface on average
        ######################################################################
        start = timing.time()

        if surface_bc==0 or surface_bc==1: # assuming bottom is closed


           if element=='CR' or element=='P2P1':
              perim_surface=0
              avrg_p_surf=0
              for iel in range(0,nel):
                  if surface_Pnode[iconP[0,iel]] and surface_Pnode[iconP[1,iel]]:
                     ppp=(p[iconP[0,iel]]+ p[iconP[1,iel]])/2
                     theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
                     theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
                     dist=abs(np.cos(theta0)-np.cos(theta1))
                     perim_surface+=dist
                     avrg_p_surf+=ppp*dist
                  if surface_Pnode[iconP[1,iel]] and surface_Pnode[iconP[2,iel]]:
                     ppp=(p[iconP[1,iel]]+ p[iconP[2,iel]])/2
                     theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
                     theta2=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
                     dist=abs(np.cos(theta2)-np.cos(theta1))
                     perim_surface+=dist
                     avrg_p_surf+=ppp*dist
                  if surface_Pnode[iconP[2,iel]] and surface_Pnode[iconP[0,iel]]:
                     ppp=(p[iconP[2,iel]]+ p[iconP[0,iel]])/2
                     theta2=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
                     theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
                     dist=abs(np.cos(theta2)-np.cos(theta0))
                     perim_surface+=dist
                     avrg_p_surf+=ppp*dist

           if element=='Q2Q1' or element=='Q1+Q1':
              perim_surface=0
              avrg_p_surf=0
              for iel in range(0,nel):
                  if surface_Pnode[iconP[3,iel]] and surface_Pnode[iconP[2,iel]]:
                     ppp=(p[iconP[2,iel]]+ p[iconP[3,iel]])/2
                     theta0=np.arctan2(xP[iconP[3,iel]],zP[iconP[3,iel]])
                     theta1=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
                     dist=abs(np.cos(theta0)-np.cos(theta1))
                     perim_surface+=dist
                     avrg_p_surf+=ppp*dist

           p-=avrg_p_surf/perim_surface

           print('     -> perim_surface (meas) =',perim_surface)
           print('     -> perim_surface (anal) =',2)
           print('     -> perim_surface (error)=',abs(perim_surface-2)/(2)*100,'%')
           print('     -> p (m,M) %.6e %.6e ' %(np.min(p),np.max(p)))

           if element=='CR' or element=='P2P1':
              perim_cmb=0
              avrg_p_cmb=0
              for iel in range(0,nel):
                  if cmb_Pnode[iconP[0,iel]] and cmb_Pnode[iconP[1,iel]]:
                     ppp=(p[iconP[0,iel]]+ p[iconP[1,iel]])/2
                     theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
                     theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
                     dist=abs(np.cos(theta0)-np.cos(theta1))
                     perim_cmb+=dist
                     avrg_p_cmb+=ppp*dist
                  if cmb_Pnode[iconP[1,iel]] and cmb_Pnode[iconP[2,iel]]:
                     ppp=(p[iconP[1,iel]]+ p[iconP[2,iel]])/2
                     theta0=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
                     theta1=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
                     dist=abs(np.cos(theta0)-np.cos(theta1))
                     perim_cmb+=dist
                     avrg_p_cmb+=ppp*dist
                  if cmb_Pnode[iconP[2,iel]] and cmb_Pnode[iconP[0,iel]]:
                     ppp=(p[iconP[2,iel]]+ p[iconP[0,iel]])/2
                     theta2=np.arctan2(xP[iconP[2,iel]],zP[iconP[2,iel]])
                     theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
                     dist=abs(np.cos(theta2)-np.cos(theta0))
                     perim_cmb+=dist
                     avrg_p_cmb+=ppp*dist

           if element=='Q2Q1' or element=='Q1+Q1':
              perim_cmb=0
              avrg_p_cmb=0
              for iel in range(0,nel):
                  if cmb_Pnode[iconP[0,iel]] and cmb_Pnode[iconP[1,iel]]:
                     ppp=(p[iconP[0,iel]]+ p[iconP[1,iel]])/2
                     theta0=np.arctan2(xP[iconP[0,iel]],zP[iconP[0,iel]])
                     theta1=np.arctan2(xP[iconP[1,iel]],zP[iconP[1,iel]])
                     dist=abs(np.cos(theta0)-np.cos(theta1))
                     perim_cmb+=dist
                     avrg_p_cmb+=ppp*dist

           avrg_p_cmb/=perim_cmb

           print('     -> cmb_surface (meas) =',perim_cmb)
           print('     -> cmb_surface (anal) =',2)
           print('     -> cmb_surface (error)=',abs(perim_cmb-2)/(2)*100,'%')
           print('     -> avrg_p_cmb =',avrg_p_cmb,nel)

           #np.savetxt('solution_pressure_normalised.ascii',np.array([xP,zP,p,radiusP]).T)

        print("normalise pressure: %.3f s" % (timing.time() - start))

        ######################################################################

        print("     -> stats", nel,np.min(u/cm*year),np.max(u/cm*year),\
                                   np.min(v/cm*year),np.max(v/cm*year),\
                                   np.min(vr/cm*year),np.max(vr/cm*year),\
                                   np.min(vt/cm*year),np.max(vt/cm*year),\
                                   np.min(p),np.max(p))

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

        e_nodal   = np.zeros(NV,dtype=np.float64)  
        e_xx      = np.zeros(NV,dtype=np.float64)  
        e_zz      = np.zeros(NV,dtype=np.float64)  
        e_xz      = np.zeros(NV,dtype=np.float64)  
        tau_xx    = np.zeros(NV,dtype=np.float64)  
        tau_zz    = np.zeros(NV,dtype=np.float64)  
        tau_xz    = np.zeros(NV,dtype=np.float64)  
        eta_nodal = np.zeros(NV,dtype=np.float64) 
        rho_nodal = np.zeros(NV,dtype=np.float64) 
        cc        = np.zeros(NV,dtype=np.float64)
        q         = np.zeros(NV,dtype=np.float64)


        for iel in range(0,nel):
            eta_el=etas[material[iel]-1]/eta_ref
            for kk in range(0,mV):
                inode=iconV[kk,iel]
                rq=rVnodes[kk]
                sq=sVnodes[kk]
                NNNV[0:mV]=NNV(rq,sq,element)
                dNNNVdr[0:mV]=dNNVdr(rq,sq,element)
                dNNNVds[0:mV]=dNNVds(rq,sq,element)
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
                    tau_xx[inode] += dNNNVdx[k]*u[iconV[k,iel]]*2*eta_el
                    tau_zz[inode] += dNNNVdy[k]*v[iconV[k,iel]]*2*eta_el
                    tau_xz[inode] += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]]) *2*eta_el 
                    e_xx[inode] += dNNNVdx[k]*u[iconV[k,iel]]
                    e_zz[inode] += dNNNVdy[k]*v[iconV[k,iel]]
                    e_xz[inode] += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
                #end for

                NNNP[0:mP]=NNP(rq,sq,element)
                q[inode]+=np.dot(NNNP[:],p[iconP[:,iel]])

                eta_nodal[inode]+=etas[material[iel]-1]
                rho_nodal[inode]+=rhos[material[iel]-1]
                cc[inode]+=1
            #end for
        #end for
        q/=cc
        rho_nodal/=cc
        eta_nodal/=cc
        e_xx/=cc
        e_zz/=cc
        e_xz/=cc
        tau_xx/=cc
        tau_zz/=cc
        tau_xz/=cc
        tau_xx*=eta_ref
        tau_zz*=eta_ref
        tau_xz*=eta_ref

        e_nodal=np.sqrt(0.5*(e_xx**2+e_zz**2)+e_xz**2)

        rho_nodal+=rho_min

        #np.savetxt('solution_q.ascii',np.array([xV,zV,q,rV]).T)
        #np.savetxt('solution_tau_cartesian.ascii',np.array([xV,zV,tau_xx,tau_zz,tau_xz]).T)
        #np.savetxt('strainrate_cartesian.ascii',np.array([xV,zV,e_xx,e_zz,e_xz,e_nodal,cc]).T)

        print("     -> e_xx   (m,M) %.6e %.6e " %(np.min(e_xx),np.max(e_xx)))
        print("     -> e_zz   (m,M) %.6e %.6e " %(np.min(e_zz),np.max(e_zz)))
        print("     -> e_xz   (m,M) %.6e %.6e " %(np.min(e_xz),np.max(e_xz)))
        print("     -> tau_xx (m,M) %.6e %.6e " %(np.min(tau_xx),np.max(tau_xx)))
        print("     -> tau_zz (m,M) %.6e %.6e " %(np.min(tau_zz),np.max(tau_zz)))
        print("     -> tau_xz (m,M) %.6e %.6e " %(np.min(tau_xz),np.max(tau_xz)))

        print("compute sr and stress: %.3f s" % (timing.time() - start))

        #####################################################################
        # compute stress tensor components
        #####################################################################
        start = timing.time()

        e_rr=np.zeros(NV,dtype=np.float64)  
        e_tt=np.zeros(NV,dtype=np.float64)  
        e_rt=np.zeros(NV,dtype=np.float64)  
        tau_rr=np.zeros(NV,dtype=np.float64)  
        tau_tt=np.zeros(NV,dtype=np.float64)  
        tau_rt=np.zeros(NV,dtype=np.float64)  

        tau_rr[:]=tau_xx[:]*np.sin(thetaV)**2+2*tau_xz[:]*np.sin(thetaV)*np.cos(thetaV)+tau_zz[:]*np.cos(thetaV)**2
        tau_tt[:]=tau_xx[:]*np.cos(thetaV)**2-2*tau_xz[:]*np.sin(thetaV)*np.cos(thetaV)+tau_zz[:]*np.sin(thetaV)**2
        tau_rt[:]=(tau_xx[:]-tau_zz[:])*np.sin(thetaV)*np.cos(thetaV)+tau_xz[:]*(-np.sin(thetaV)**2+np.cos(thetaV)**2)
 
        e_rr=e_xx*np.sin(thetaV)**2+2*e_xz*np.sin(thetaV)*np.cos(thetaV)+e_zz*np.cos(thetaV)**2
        e_tt=e_xx*np.cos(thetaV)**2-2*e_xz*np.sin(thetaV)*np.cos(thetaV)+e_zz*np.sin(thetaV)**2
        e_rt=(e_xx-e_zz)*np.sin(thetaV)*np.cos(thetaV)+e_xz*(-np.sin(thetaV)**2+np.cos(thetaV)**2)

        e_nodal=np.sqrt(0.5*(e_rr**2+e_tt**2)+e_rt**2) #check it is same as e_nodal before!

        sigma_rr=-q+tau_rr
        sigma_tt=-q+tau_tt
        sigma_rt=   tau_rt

        #np.savetxt('sol_sr_spherical.ascii',np.array([xV,zV,e_rr,e_tt,e_rt,e_nodal]).T)
        #np.savetxt('solution_tau_cartesian.ascii',np.array([xV,zV,tau_xx,tau_zz,tau_xz]).T)
        #np.savetxt('solution_tau_spherical.ascii',np.array([xV,zV,tau_rr,tau_tt,tau_rt]).T)

        print("rotate stresses: %.3f s" % (timing.time() - start))

        #####################################################################
        # compute dynamic topography
        # in order to be able to compare with aspect we need to 
        # subtract the average cmb pressure from sigma_rr
        ##################################################################### 
        start = timing.time()

        dyn_topo=np.zeros(NV,dtype=np.float64)
        for i in range(0,NV):
            if surface_node[i] and rho_nodal[i]>0:
               dyn_topo[i]= -(tau_rr[i]-q[i])/((rho_nodal[i]-density_above)*g0) 
            if cmb_node[i]:
               dyn_topo[i]= -(tau_rr[i]-q[i])/((density_below-rho_nodal[i])*g0) 

        print("compute dynamic topography: %.3f s" % (timing.time() - start))

        #####################################################################
        # compute traction at surface and cmb
        #####################################################################
        start = timing.time()

        np.savetxt('surface_r_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],radiusV[surface_node]]).T)
        np.savetxt('surface_vt_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],vt[surface_node]/cm*year]).T)
        np.savetxt('surface_vr_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],vr[surface_node]/cm*year]).T)
        np.savetxt('surface_vel_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],vel[surface_node]/cm*year]).T)
        np.savetxt('surface_q_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],q[surface_node]]).T)
        np.savetxt('surface_tau_rr_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],tau_rr[surface_node]]).T)
        np.savetxt('surface_tau_tt_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],tau_tt[surface_node]]).T)
        np.savetxt('surface_tau_rt_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],tau_rt[surface_node]]).T)
        np.savetxt('surface_normal_{:04d}.ascii'.format(istep),np.array([xV[surface_node],zV[surface_node],nx[surface_node],nz[surface_node]]).T)
        np.savetxt('surface_traction_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],tau_rr[surface_node]-q[surface_node]]).T)
        np.savetxt('surface_uv_{:04d}.ascii'.format(istep),np.array([xV[surface_node],zV[surface_node],u[surface_node]/cm*year,v[surface_node]/cm*year]).T)
        np.savetxt('surface_sr_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],e_nodal[surface_node]]).T)
        np.savetxt('surface_dyn_topo_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],dyn_topo[surface_node]]).T)
        np.savetxt('surface_p_{:04d}.ascii'.format(istep),np.array([thetaP[surface_Pnode],p[surface_Pnode]]).T)

        np.savetxt('cmb_r_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],radiusV[cmb_node]]).T)
        np.savetxt('cmb_vt_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],vt[cmb_node]/cm*year]).T)
        np.savetxt('cmb_vr_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],vr[cmb_node]/cm*year]).T)
        np.savetxt('cmb_vel_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],vel[cmb_node]/cm*year]).T)
        np.savetxt('cmb_q_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],q[cmb_node]]).T)
        np.savetxt('cmb_tau_rr_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],tau_rr[cmb_node]]).T)
        np.savetxt('cmb_normal_{:04d}.ascii'.format(istep),np.array([xV[cmb_node],zV[cmb_node],nx[cmb_node],nz[cmb_node]]).T)
        np.savetxt('cmb_traction_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],tau_rr[cmb_node]-q[cmb_node]]).T)
        np.savetxt('cmb_sr_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],e_nodal[cmb_node]]).T)
        np.savetxt('cmb_dyn_topo_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],dyn_topo[cmb_node]]).T)
        np.savetxt('cmb_p_{:04d}.ascii'.format(istep),np.array([thetaP[cmb_Pnode],p[cmb_Pnode]]).T)

        if fs_method==3:
           if surface_bc==1 and cmb_bc != 1:
              np.savetxt('surface_l_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],l[0:int(Nlm)]]).T)
           if surface_bc !=1 and cmb_bc == 1:
              np.savetxt('cmb_l_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],l[0:int(Nlm)]]).T)
           if surface_bc==1 and cmb_bc == 1:
              np.savetxt('surface_l_{:04d}.ascii'.format(istep),np.array([thetaV[surface_node],l[0:int(Nlm/2)]]).T)
              np.savetxt('cmb_l_{:04d}.ascii'.format(istep),np.array([thetaV[cmb_node],l[int(Nlm/2):Nlm]]).T)

        print("compute surface tractions: %.3f s" % (timing.time() - start))

    else:

        print("****no Stokes solve*****")

        u = np.zeros(NV,dtype=np.float64)
        v = np.zeros(NV,dtype=np.float64)
        p = np.zeros(NP,dtype=np.float64)
        q = np.zeros(NV,dtype=np.float64)
        eta_nodal= np.zeros(NV,dtype=np.float64) 
        rho_nodal= np.zeros(NV,dtype=np.float64) 
        vr = np.zeros(NV,dtype=np.float64)
        vt = np.zeros(NV,dtype=np.float64)

    print("-----------------------------")

    #####################################################################
    # plot of solution
    # the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
    #####################################################################
    start = timing.time()

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
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rhos[material[iel]-1]+rho_min))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (etas[material[iel]-1]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='flag_top' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (flag_top[iel]))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Int32' Name='blob' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    if blob[iel]:
    #       value=1
    #    else:
    #       value=0
    #    vtufile.write("%d\n" % (value))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='material' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d\n" % (material[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='flag_bot' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (flag_bot[iel]))
    vtufile.write("</DataArray>\n")
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
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(nx[i],0.,nz[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32'  Name='vr (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (u[i]/cm*year*np.sin(thetaV[i])+v[i]/cm*year*np.cos(thetaV[i]) ) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32'  Name='vt (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (u[i]/cm*year*np.cos(thetaV[i])-v[i]/cm*year*np.sin(thetaV[i]) ) )
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
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %radiusV[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %rho_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %eta_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    if solve_stokes:
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%e \n" % (e_nodal[i]))
       vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='cc' Format='ascii'> \n")
    #for i in range(0,NV):
    #    vtufile.write("%e \n" % cc[i])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='flag_surface' Format='ascii'> \n")
    for i in range(0,NV):
        if surface_node[i]:
           vtufile.write("%d \n" %1)
        else:
           vtufile.write("%d \n" %0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='flag_cmb' Format='ascii'> \n")
    for i in range(0,NV):
        if cmb_node[i]:
           vtufile.write("%d \n" %1)
        else:
           vtufile.write("%d \n" %0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='flag_left' Format='ascii'> \n")
    for i in range(0,NV):
        if left_node[i]:
           vtufile.write("%d \n" %1)
        else:
           vtufile.write("%d \n" %0)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='theta (sph.coords)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%e \n" %thetaV[i])
    vtufile.write("</DataArray>\n")
    #--
    if solve_stokes:
       vtufile.write("<DataArray type='Int32' Name='dyn topo' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%d \n" % (dyn_topo[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_xx)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_xx[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_zz)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_zz[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_xz)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_xz[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_rr)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_rr[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_tt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_tt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dev. stress (tau_rt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % tau_rt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_rr)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % e_rr[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_tt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % e_tt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_rt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % e_rt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_xx)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % (e_xx[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_zz)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % e_zz[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate (e_xz)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % e_xz[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='stress (sigma_rr)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % sigma_rr[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='stress (sigma_tt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % sigma_tt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='stress (sigma_rt)' Format='ascii'> \n")
       for i in range (0,NV):
           vtufile.write("%10e\n" % sigma_rt[i])
       vtufile.write("</DataArray>\n")
       #--
    #end if solve_stokes:
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
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    if element=='P2P1' or element=='CR':
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                 iconV[3,iel],iconV[4,iel],iconV[5,iel]))
       vtufile.write("</DataArray>\n")
    if element=='Q2Q1':
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                       iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
       vtufile.write("</DataArray>\n")

    if element=='Q1+Q1':
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    if element=='P2P1' or element=='CR':
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*6))
       vtufile.write("</DataArray>\n")
    if element=='Q2Q1':
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
    if element=='Q1+Q1':
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    if element=='P2P1' or element=='CR':
       for iel in range (0,nel):
           vtufile.write("%d \n" %22)
       vtufile.write("</DataArray>\n")
    if element=='Q2Q1':
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
    if element=='Q1+Q1':
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")



    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


    filename = 'solutionP_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mP*nel,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for iel in range(0,nel):
        for i in range(0,mP):
            vtufile.write("%10e %10e %10e \n" %(xP[iconP[i,iel]],0.,zP[iconP[i,iel]]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####

    if solve_stokes:
       vtufile.write("<PointData Scalars='scalars'>\n")

       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range(0,nel):
           for i in range(0,mP):
               vtufile.write("%10e \n" %(p[iconP[i,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='tau_rr' Format='ascii'> \n")
       for iel in range(0,nel):
           for i in range(0,mP):
               vtufile.write("%10e \n" %(tau_rr[iconV[i,iel]]))
       vtufile.write("</DataArray>\n")

       vtufile.write("<DataArray type='Float32' Name='e_rr' Format='ascii'> \n")
       for iel in range(0,nel):
           for i in range(0,mP):
               vtufile.write("%10e \n" %(e_rr[iconV[i,iel]]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for iel in range(0,nel):
           for i in range(0,mP):
               vtufile.write("%10e %10e %10e \n" %(u[iconV[i,iel]]/cm*year,0.,v[iconV[i,iel]]/cm*year))
       vtufile.write("</DataArray>\n")

       vtufile.write("</PointData>\n")

    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    if element=='P2P1' or element=='CR':
       for iel in range (0,nel):
           vtufile.write("%d %d %d \n" %(iel*3,iel*3+1,iel*3+2))
       vtufile.write("</DataArray>\n")
    if element=='Q2Q1':
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(iel*4,iel*4+1,iel*4+2,iel*4+3))
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*mP))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    if element=='P2P1' or element=='CR':
       for iel in range (0,nel):
           vtufile.write("%d \n" %5)
       vtufile.write("</DataArray>\n")
    if element=='Q2Q1':
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


    print("export to vtu: %.3fs" % (timing.time() - start))

    print("-----------------------------")

    #####################################################################
    # compute gravity. phi goes from 0 to 2pi
    #####################################################################
    start = timing.time()

    if compute_gravity:

       xM=np.zeros(np_grav,dtype=np.float64)     
       yM=np.zeros(np_grav,dtype=np.float64)     
       zM=np.zeros(np_grav,dtype=np.float64)     
       gvect_x=np.zeros(np_grav,dtype=np.float64)   
       gvect_y=np.zeros(np_grav,dtype=np.float64)   
       gvect_z=np.zeros(np_grav,dtype=np.float64)   
       gvect_x_0=np.zeros(np_grav,dtype=np.float64)   
       gvect_y_0=np.zeros(np_grav,dtype=np.float64)   
       gvect_z_0=np.zeros(np_grav,dtype=np.float64)   
       gvect_0=np.zeros(np_grav,dtype=np.float64)   
       angleM=np.zeros(np_grav,dtype=np.float64)   

       for i in range(0,np_grav):
           angleM[i]=np.pi/2-np.pi/(np_grav-1)*i
           xM[i]=(R_outer+height)*np.cos(angleM[i])
           zM[i]=(R_outer+height)*np.sin(angleM[i])
           gvect_x[i],gvect_y[i],gvect_z[i]=compute_gravity_at_point3(xM[i],yM[i],zM[i],\
                                                                      nel,nqel,qcoords_r,qcoords_s,qweights,\
                                                                      rhoq,jcobq,xq,zq,nel_phi,thetaq,radiusq)

           print('point',i,'gx,gy,gz',gvect_x[i],gvect_y[i],gvect_z[i])

       #end for

       gvect=np.sqrt(gvect_x**2+gvect_y**2+gvect_z**2)
       gvect_avrg=np.sum(gvect)/np_grav
       rM=np.sqrt(xM**2+yM**2+zM**2)

       gravfile1 = 'gravity_{:04d}.ascii'.format(istep)
       np.savetxt(gravfile1,np.array([xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect,gvect-gvect_avrg]).T,fmt='%.6e',\
                  header='# xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect,gvect-avrg')

       gmean = Ggrav*mass/(R_outer+height)**2
       gravfile2 = 'gravityanomaly_wp_{:04d}.ascii'.format(istep)
       np.savetxt(gravfile2,np.array([xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect-gmean]).T,fmt='%.6e',\
                  header='# xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect-gmean,gmean=%.6e'%(gmean))

       #gmean = Ggrav*Prof_masse/(R_outer+height)**2
       #gravfile3 = 'gravityanomaly_np_{:04d}.ascii'.format(istep)
       #np.savetxt(gravfile3,np.array([xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect-gmean]).T,fmt='%.6e',\
       #           header='# xM,yM,zM,rM,angleM,gvect_x,gvect_y,gvect_z,gvect-gmean,gmean=%0.6e'%(gmean))

       if istep>0:
          gravfile4 = 'gravity_diff_{:04d}.ascii'.format(istep)
          np.savetxt(gravfile4,np.array([xM,yM,zM,rM,angleM,gvect_x-gvect_x_0,\
                                                            gvect_y-gvect_y_0,\
                                                            gvect_z-gvect_z_0,\
                                                            gvect-gvect_0]).T,fmt='%.6e')
       else:
          gvect_x_0[:]=gvect_x[:]
          gvect_y_0[:]=gvect_y[:]
          gvect_z_0[:]=gvect_z[:]
          gvect_0[:]=gvect[:]

       print("compute gravity: %.3fs" % (timing.time() - start))

    #####################################################################
    # compute timestep
    #####################################################################
    start = timing.time()

    if solve_stokes:

       #CFL_nb=1
       #dt=CFL_nb*(np.min(np.sqrt(area)))/np.max(np.sqrt(u**2+v**2))
       #dt=min(dt,dt_user)
       dt=dt_user
       print('     -> dt = %.6f yr' % (dt/year))

    else:

       dt=0

    print("compute dt: %.3fs" % (timing.time() - start))

    #####################################################################
    # evolve mesh
    #####################################################################
    start = timing.time()

    #np.savetxt('meshV_bef_'+str(istep)+'.ascii',np.array([xV/km,zV/km,u,v]).T,header='# x,y')
    #np.savetxt('meshP_bef_'+str(istep)+'.ascii',np.array([xP/km,zP/km]).T,header='# x,y')

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

    #np.savetxt('meshV_aft_'+str(istep)+'.ascii',np.array([xV/km,zV/km]).T,header='# x,y')
    #np.savetxt('meshP_aft_'+str(istep)+'.ascii',np.array([xP/km,zP/km]).T,header='# x,y')

    print("evolve mesh: %.3fs" % (timing.time() - start))

#end istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
