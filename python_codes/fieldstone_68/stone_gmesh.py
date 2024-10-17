import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.special import erf
import velocity

#------------------------------------------------------------------------------
# defining velocity and pressure shape functions
#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= (1.-rq-sq)*(1.-2.*rq-2.*sq+ 3.*rq*sq)
    NV_1= rq*(2.*rq -1. + 3.*sq-3.*rq*sq-3.*sq**2 )
    NV_2= sq*(2.*sq -1. + 3.*rq-3.*rq**2-3.*rq*sq )
    NV_3= 4.*rq*sq*(-2.+3.*rq+3.*sq)
    NV_4= 4.*(1.-rq-sq)*sq*(1.-3.*rq)
    NV_5= 4.*(1.-rq-sq)*rq*(1.-3.*sq)
    NV_6= 27*(1.-rq-sq)*rq*sq
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6

def dNNVdr(rq,sq):
    dNVdr_0= -3+4*rq+7*sq-6*rq*sq-3*sq**2
    dNVdr_1= 4*rq-1+3*sq-6*rq*sq-3*sq**2
    dNVdr_2= 3*sq-6*rq*sq-3*sq**2
    dNVdr_3= -8*sq+24*rq*sq+12*sq**2
    dNVdr_4= -16*sq+24*rq*sq+12*sq**2
    dNVdr_5= -8*rq+24*rq*sq+4-16*sq+12*sq**2
    dNVdr_6= -54*rq*sq+27*sq-27*sq**2
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6

def dNNVds(rq,sq):
    dNVds_0= -3+7*rq+4*sq-6*rq*sq-3*rq**2
    dNVds_1= rq*(3-3*rq-6*sq)
    dNVds_2= 4*sq-1+3*rq-3*rq**2-6*rq*sq
    dNVds_3= -8*rq+12*rq**2+24*rq*sq
    dNVds_4= 4-16*rq-8*sq+24*rq*sq+12*rq**2
    dNVds_5= -16*rq+24*rq*sq+12*rq**2
    dNVds_6= -54*rq*sq+27*rq-27*rq**2
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6

#------------------------------------------------------------------------------
# standard P1 shape functions for pressure
#------------------------------------------------------------------------------

def NNP(rq,sq):
    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return NP_0,NP_1,NP_2

#------------------------------------------------------------------------------
# standard P2 shape functions for temperature
#------------------------------------------------------------------------------

def NNT(r,s):
    N_0=1-3*r-3*s+2*r**2+4*r*s+2*s**2
    N_1=-r+2*r**2
    N_2=-s+2*s**2
    N_5=4*r-4*r**2-4*r*s
    N_3=4*r*s
    N_4=4*s-4*r*s-4*s**2
    return N_0,N_1,N_2,N_3,N_4,N_5

def dNNTdr(r,s):
    dNdr_0=-3+4*r+4*s
    dNdr_1=-1+4*r
    dNdr_2=0
    dNdr_5=4-8*r-4*s
    dNdr_3=4*s
    dNdr_4=-4*s
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5

def dNNTds(r,s):
    dNds_0=-3+4*r+4*s
    dNds_1=0
    dNds_2=-1+4*s
    dNds_5=-4*r
    dNds_3=4*r
    dNds_4=4-4*r-8*s
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5

#------------------------------------------------------------------------------
# Crouzeix-Raviart elements for Stokes, P_2 for Temp.
# the internal numbering of the nodes is as follows:
#
#  P_2^+            P_-1             P_2 
#
#  02           #   02           #   02
#  ||\\         #   ||\\         #   ||\\
#  || \\        #   || \\        #   || \\
#  ||  \\       #   ||  \\       #   ||  \\
#  04   03      #   ||   \\      #   04   03
#  || 06 \\     #   ||    \\     #   ||    \\
#  ||     \\    #   ||     \\    #   ||     \\
#  00==05==01   #   00======01   #   00==05==01

rVnodes=[0,1,0,0.5,0.0,0.5,1./3.]
sVnodes=[0,0,1,0.5,0.5,0.0,1./3.]

rTnodes=[0,1,0,0.5,0.0,0.5]
sTnodes=[0,0,1,0.5,0.5,0.0]

#------------------------------------------------------------------------------
# material model
#------------------------------------------------------------------------------

def viscosity(exx,eyy,exy,T,imat):
    if case=='1a' or case=='1b' or case=='1c':
       val=1e21
    elif case=='2a': #diffusion creep 
       if (imat==4 or imat==2) and iter>0:
          eta_diff=A_diff*np.exp(Q_diff/Rgas/T)
          val=1/(1/eta_max+1/eta_diff)
       else:
          val=1e21
    else: #dislocation creep 
       if (imat==4 or imat==2) and iter>0:
          e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
          eta_disl=A_disl*np.exp(Q_disl/n_disl/Rgas/T)*e**(-1+1./n_disl)
          val=1/(1/eta_max+1/eta_disl)
       else:
          val=1e21
    return val

#------------------------------------------------------------------------------
# useful constants
#------------------------------------------------------------------------------

cm=0.01
year=365.25*3600.*24.
Kelvin=273

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2   # number of physical dimensions
mV=7     # number of velocity nodes making up an element
mP=3     # number of pressure nodes making up an element
mT=6     # number of temperature nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1  # number of temperature degrees of freedom 

##########################################################
# input parameters

filename='vankeken2_4_0.msh'

Lx=660e3
Ly=600e3

hcapa=1250 # vack08
hcond=3    # vack08
rho0=3300  # vack08
kappa=hcond/rho0/hcapa

eps=1e-9

l1=1000.e3
l2=50.e3
l3=0.e3
vel=5*cm/year
angle=45./180.*np.pi  

Rgas=8.3145

eta_max=1e26

rheology=1

if rheology==1: #vack08
   Q_diff=335e3
   A_diff=1.32043e9
   Q_disl=540e3
   A_disl=28968.6
   n_disl=3.5

if rheology==2: #hiko03 wet
   n_disl=3.5
   Q_diff=375e3
   Q_disl=520e3
   A_diff=0
   A_disl=0
   
if rheology==2: #hiko03 disl
   n_disl=3.5
   Q_diff=375e3
   Q_disl=520e3
   A_diff=0
   A_disl=0



niter=100

#case='1a'
#case='1b'
#case='1c'
case='2a'

if case=='1a':
   niter=1

do_post_processing=True

relax=0.99

tol=1e-4

print('case=',case)
print('relax=',relax)

##########################################################
# checking that all velocity shape functions are 1 on 
# their node and zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i]))
#exit()
# checking that all velocity temperature functions are 1 on 
# their node and zero elsewhere
#for i in range(0,mT):
#   print ('node',i,':',NNT(rTnodes[i],sTnodes[i]))
#exit()
##########################################################
#find NV and nel as indicated in msh file
##########################################################

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter==4:
       NV=int(columns[0])
       print ('read: NV=',NV)
    counter+=1
#end for

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter==NV+7:
       nel=int(columns[0])
       print ('read: nel=',nel)
    counter+=1
#end for

##########################################################
# find exact number of real triangles
# i.e. removing the interface related lines which 
# fall under triangles in input file but which aren't.
# The lines in question count 7 numbers per line 
# instead of 11 for real triangles.
# Once this number is found it should be removed from 
# the read-in number of elements and added to NV (the 
# C-R element has a 7th node in the middle of each triangle).  
##########################################################

f = open(filename,'r')
counter=0
counter2=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NV+7 and counter<NV+8+nel:
       l=len(columns)
       if l<11:
          counter2+=1
    counter+=1
#end for

print('nb of interface lines in msh file:',counter2)

nel-=counter2

NVold=NV

NV+=nel #adding 7th node in the middle of each element

print ('new nel=',nel)
print ('new NV=',NV)

##########################################################
# read in coordinates of nodes
##########################################################

xV=np.zeros(NV,dtype=np.float64)     # x coordinates
yV=np.zeros(NV,dtype=np.float64)     # y coordinates

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>4 and counter<NVold+5:
       xV[counter-5]=np.float64(columns[1])
       yV[counter-5]=np.float64(columns[2])
    counter+=1
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T)

##########################################################
# read in connectivity array
##########################################################

iconV=np.zeros((mV,nel),dtype=np.int64)
mat=np.zeros(nel,dtype=np.int64)
    
#print (NVold+7+counter2,NVold+8+nel+counter2)

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NVold+7+counter2 and counter<NVold+8+nel+counter2:
       l=len(columns)
       iconV[0,counter-NVold-8-counter2]=np.int64(columns[l-6])-1
       iconV[1,counter-NVold-8-counter2]=np.int64(columns[l-5])-1
       iconV[2,counter-NVold-8-counter2]=np.int64(columns[l-4])-1
       iconV[5,counter-NVold-8-counter2]=np.int64(columns[l-3])-1
       iconV[3,counter-NVold-8-counter2]=np.int64(columns[l-2])-1
       iconV[4,counter-NVold-8-counter2]=np.int64(columns[l-1])-1
       mat[counter-NVold-8-counter2]=np.int64(columns[4])/1e6
    #end if
    counter+=1
#end for

#print (iconV[:,0])
#exit()

##########################################################
# compute coordinate of middle node
# The 7th node of the C-R element is at the barycenter 
# of the element. This node needs to be added to the 
# connectivity array too.
##########################################################

for iel in range(0,nel):
    iconV[6,iel]=NVold+iel
    xV[iconV[6,iel]]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
    yV[iconV[6,iel]]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
#end for

#np.savetxt('gridV2.ascii',np.array([xV,yV]).T)

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconV[0,iel],"at pos.",xV[iconV[0][iel]], yV[iconV[0][iel]])
#    print ("node 1",iconV[1,iel],"at pos.",xV[iconV[1][iel]], yV[iconV[1][iel]])
#    print ("node 2",iconV[2,iel],"at pos.",xV[iconV[2][iel]], yV[iconV[2][iel]])
#    print ("node 3",iconV[3,iel],"at pos.",xV[iconV[3][iel]], yV[iconV[3][iel]])
#    print ("node 4",iconV[4,iel],"at pos.",xV[iconV[4][iel]], yV[iconV[4][iel]])
#    print ("node 5",iconV[5,iel],"at pos.",xV[iconV[5][iel]], yV[iconV[5][iel]])
#    print ("node 6",iconV[6,iel],"at pos.",xV[iconV[6][iel]], yV[iconV[6][iel]])

##########################################################
##########################################################

NfemV=NV*ndofV     # number of velocity dofs
NfemP=nel*3*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP   # total number of dofs

print ('NfemV', NfemV)
print ('NfemP', NfemP)
print ('Nfem ', Nfem)

pressure_scaling=1e22/1000e3

##########################################################
# 6 point integration coeffs and weights 
##########################################################

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=[nb1,nb2,nb2,nb4,nb3,nb4]
qcoords_s=[nb2,nb1,nb2,nb3,nb4,nb4]
qweights=[nb5,nb5,nb5,nb6,nb6,nb6]

nq=nqel*nel
print ('nq=', nq)
print("-----------------------------")

#################################################################
# build pressure grid (nodes and icon)
# because the pressure is dicontinuous (P_-1) then the total 
# number of pressure dofs is nel*3 (see above)
# We use the iconV connectivity to build the iconP array
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int64)
xP=np.empty(NfemP,dtype=np.float64)     # x coordinates
yP=np.empty(NfemP,dtype=np.float64)     # y coordinates

counter=0
for iel in range(0,nel):
    xP[counter]=xV[iconV[0,iel]]
    yP[counter]=yV[iconV[0,iel]]
    iconP[0,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[1,iel]]
    yP[counter]=yV[iconV[1,iel]]
    iconP[1,iel]=counter
    counter+=1
    xP[counter]=xV[iconV[2,iel]]
    yP[counter]=yV[iconV[2,iel]]
    iconP[2,iel]=counter
    counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 0",iconP[0,iel],"at pos.",xP[iconP[0][iel]], yP[iconP[0][iel]])
#    print ("node 1",iconP[1,iel],"at pos.",xP[iconP[1][iel]], yP[iconP[1][iel]])
#    print ("node 2",iconP[2,iel],"at pos.",xP[iconP[2][iel]], yP[iconP[2][iel]])

print("setup: connectivity P: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
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
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    if area[iel]<0: 
       for k in range(0,mV):
           print (xV[iconV[k,iel]],yV[iconV[k,iel]])
   #    print(iel,iconV[:,iel])
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# read from msh file the identity of the nodes on the slab interface
#################################################################
start = timing.time()

interface=np.zeros(NV,dtype=np.int32)  

f = open(filename,'r')
counter=0
for line in f:
    line=line.strip()
    columns=line.split()
    if counter>NVold+7 and counter<NVold+8+nel:
       l=len(columns)
       if l==8:
          interface[np.int64(columns[5])-1]=columns[3]
          interface[np.int64(columns[6])-1]=columns[3]
          interface[np.int64(columns[7])-1]=columns[3]
       #end if
    #end if
    counter+=1
#end for

print("read in slab interface: %.3f s" % (timing.time() - start))

#################################################################
# compute normal vector along the interface
# Loop over all triangles, find edges on the interface, 
# compute their normal, add it to all three nodes on the edge. 
# In the end renormalise the normal vectors. 
#################################################################
start = timing.time()

nx=np.zeros(NV,dtype=np.float64) 
ny=np.zeros(NV,dtype=np.float64) 

for iel in range(0,nel):
    inode0=iconV[0,iel]
    inode1=iconV[1,iel]
    inode2=iconV[2,iel]
    x0=xV[inode0]
    x1=xV[inode1]
    x2=xV[inode2]
    y0=yV[inode0]
    y1=yV[inode1]
    y2=yV[inode2]
    if (interface[inode0]==101 and interface[inode1]==101):
       vx=abs(x1-x0)
       vy=abs(y1-y0)
       vnorm=np.sqrt(vx**2+vy**2)
       ax= vy/vnorm
       ay= vx/vnorm
       nx[inode0]+=ax
       ny[inode0]+=ay
       nx[iconV[5,iel]]=ax
       ny[iconV[5,iel]]=ay
       nx[inode1]+=ax
       ny[inode1]+=ay
    #end if
    if (interface[inode0]==101 and interface[inode2]==101):
       vx=abs(x2-x0)
       vy=abs(y2-y0)
       vnorm=np.sqrt(vx**2+vy**2)
       ax= vy/vnorm
       ay= vx/vnorm
       nx[inode0]+=ax
       ny[inode0]+=ay
       nx[iconV[4,iel]]=ax
       ny[iconV[4,iel]]=ay
       nx[inode2]+=ax
       ny[inode2]+=ay
    #end if
    if (interface[inode1]==101 and interface[inode2]==101):
       vx=abs(x2-x1)
       vy=abs(y2-y1)
       vnorm=np.sqrt(vx**2+vy**2)
       ax= vy/vnorm
       ay= vx/vnorm
       nx[inode1]+=ax
       ny[inode1]+=ay
       nx[iconV[3,iel]]=ax
       ny[iconV[3,iel]]=ay
       nx[inode2]+=ax
       ny[inode2]+=ay
    #end if
#end for
    
for i in range(0,NV):
    norm=np.sqrt(nx[i]**2+ny[i]**2)
    if norm>1e-6:
       nx[i]/=norm
       ny[i]/=norm
    #end if
#end for

print("compute normals: %.3f s" % (timing.time() - start))

#################################################################
# use normal vector to impose velocity bcs: velocity on the interface 
# should be perpendicular to normal.
# We wish to fix both components of the velocity in the over-riding 
# plate. Through trial and error, I found out that one cannot 
# fix all dofs, so I leave some free, which is not too important since 
# these elements are pretty much bound by no slip b.c. all around 
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
 
    #top
    if interface[i]==111:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end if   

    #right
    if interface[i]==112:
       if yV[i]>-l2:
          bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       elif case=='1b':
          ui,vi=velocity.compute_corner_flow_velocity(xV[i],yV[i]+Ly,l1,l2,l3,angle,vel,Lx,Ly)
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ui
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi
       #end if   
    #end if   

    #bottom right
    if case=='1b' and interface[i]==113 and xV[i]>602e3:
       ui,vi=velocity.compute_corner_flow_velocity(xV[i],yV[i]+Ly,l1,l2,l3,angle,vel,Lx,Ly)
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ui
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi
    #end if   

    # bottom overriding plate
    if interface[i]==104: 
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    #end if   

    # plate contact
    if interface[i]==102:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
    #end if   

    #top of slab ramp
    if interface[i]==101:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ny[i]*vel 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[i]*vel
    #end if   

#fixing nodes in thin zone between 101 and 111
for iel in range(0,nel):
    if interface[iconV[0,iel]]==101 and interface[iconV[1,iel]]==104:
       print('101--104a') 
       i=iconV[5,iel]
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[0,iel]]*vel/2 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[0,iel]]*vel/2
    if interface[iconV[1,iel]]==101 and interface[iconV[2,iel]]==104:
       print('101--104b') 
       i=iconV[3,iel]
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[0,iel]]*vel/2 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[0,iel]]*vel/2
    if interface[iconV[2,iel]]==101 and interface[iconV[0,iel]]==104:
       print('101--104c') 
       i=iconV[4,iel]
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[0,iel]]*vel/2 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[0,iel]]*vel/2

    if interface[iconV[0,iel]]==101 and interface[iconV[1,iel]]==102:
       print('101--102a') 
       i=iconV[5,iel]
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[0,iel]]*vel/2 
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[0,iel]]*vel/2
#    if interface[iconV[1,iel]]==101 and interface[iconV[2,iel]]==102:
#       print('101--102b') 
#       i=iconV[3,iel]
#       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[1,iel]]*vel/2 
#       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[1,iel]]*vel/2
#    if interface[iconV[2,iel]]==101 and interface[iconV[0,iel]]==102:
#       print('101--102c') 
#       i=iconV[4,iel]
#       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[2,iel]]*vel/2 
#       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[2,iel]]*vel/2

#    if interface[iconV[0,iel]]==101 and interface[iconV[1,iel]]==111:
#       print('101--111') 
#       i=iconV[5,iel]
#       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] =  ny[iconV[0,iel]]*vel/2 
#       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -nx[iconV[0,iel]]*vel/2

#unfixing a few nodes top left
for i in range(0,NV):
    if xV[i]>0 and xV[i]<5e3 and yV[i]>-1:
       print('CCC') 
       bc_fix[i*ndofV  ] = False 
       bc_fix[i*ndofV+1] = False 



print("setup boundary conditions: %.3f s" % (timing.time() - start))

######################################################################
# temperature nodes and connectivity 
######################################################################

NT=NVold
NfemT=NT*ndofT

xT=np.zeros(NT,dtype=np.float64)     # x coordinates
yT=np.zeros(NT,dtype=np.float64)     # y coordinates

xT[0:NT]=xV[0:NT]
yT[0:NT]=yV[0:NT]

#np.savetxt('gridT.ascii',np.array([xT,yT]).T,header='# x,y')

iconT=np.zeros((mT,nel),dtype=np.int64)

iconT[0:mT,0:nel]=iconV[0:mT,0:nel]

#################################################################

u      = np.zeros(NV,dtype=np.float64)           # x-component velocity
v      = np.zeros(NV,dtype=np.float64)           # y-component velocity
p      = np.zeros(NfemP,dtype=np.float64)        # pressure field
T      = np.zeros(NT,dtype=np.float64)           # temperature field 
NNNV   = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP   = np.zeros(mP,dtype=np.float64)           # shape functions P
NNNT   = np.zeros(mT,dtype=np.float64)           # shape functions T 
dNNNVdx= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds= np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNTdx= np.zeros(mT,dtype=np.float64)           # shape functions derivatives
dNNNTdy= np.zeros(mT,dtype=np.float64)           # shape functions derivatives
dNNNTdr= np.zeros(mT,dtype=np.float64)           # shape functions derivatives
dNNNTds= np.zeros(mT,dtype=np.float64)           # shape functions derivatives
u_old  = np.zeros(NV,dtype=np.float64)           # x-component velocity
v_old  = np.zeros(NV,dtype=np.float64)           # y-component velocity
T_old  = np.zeros(NT,dtype=np.float64)           # temperature field 
c_mat  = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
x_q    = np.zeros(nq,dtype=np.float64)           # x-component velocity
y_q    = np.zeros(nq,dtype=np.float64)           # y-component velocity
eta_q  = np.zeros(nq,dtype=np.float64)           # y-component velocity
T_q    = np.zeros(nq,dtype=np.float64)           # y-component velocity
exx_q  = np.zeros(nq,dtype=np.float64)           # y-component velocity
eyy_q  = np.zeros(nq,dtype=np.float64)           # y-component velocity
exy_q  = np.zeros(nq,dtype=np.float64)           # y-component velocity

################################################################################################
################################################################################################
# ITERATIONS
################################################################################################
################################################################################################
convfile=open('conv.ascii',"w")

for iter in range(0,niter):

    print('*******************')
    print('***** iter=',iter,'*****')
    print('*******************')

    #################################################################
    # build FE matrix
    # [ K   G ][u]=[f]
    # [ G^T 0 ][p] [h]
    #################################################################
    start = timing.time()

    A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs    = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs  = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs  = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat  = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat  = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix to build G_el 

    counterq=0

    for iel in range(0,nel):

        if iel%2000==0:
           print('iel=',iel)

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)

        for kq in range (0,nqel):

            # position & weight of quad. point
            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:4]=NNP(rq,sq)
            NNNT[0:mT]=NNT(rq,sq)

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
            xq=0.0
            yq=0.0
            exxq=0.0
            eyyq=0.0
            exyq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
            x_q[counterq]=xq 
            y_q[counterq]=yq 
            exx_q[counterq]=exxq 
            eyy_q[counterq]=eyyq 
            exy_q[counterq]=exyq 

            Tq=0.
            for k in range(0,mT):
                Tq+=NNNT[k]*T[iconT[k,iel]]
            T_q[counterq]=Tq

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            eta_q[counterq]=viscosity(exxq,eyyq,exyq,Tq,mat[iel])
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*weightq*jcob*eta_q[counterq]

            # compute elemental rhs vector
            #for i in range(0,mV):
            #    f_el[ndofV*i  ]+=NV[i]*jcob*weightq*gx(xq,yq,grav)*rhoq
            #    f_el[ndofV*i+1]+=NV[i]*jcob*weightq*gy(xq,yq,grav)*rhoq

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            counterq+=1

        #end for kq

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
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    #G_mat[m1,m2]+=G_el[ikk,jkk]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*pressure_scaling
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*pressure_scaling
                #end for
                f_rhs[m1]+=f_el[ikk]
            #end for
        #end for
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]*pressure_scaling
        #end for

    #end for

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    if case=='1a':
       for i in range(0,NV):
           u[i],v[i]=velocity.compute_corner_flow_velocity(xV[i],yV[i]+Ly,l1,l2,l3,angle,vel,Lx,Ly)
       #end for
    else:
       sol = np.zeros(Nfem,dtype=np.float64) 
       sparse_matrix=A_sparse.tocsr()
       sol=sps.linalg.spsolve(sparse_matrix,rhs)
       u,v=np.reshape(sol[0:NfemV],(NV,2)).T
       p=sol[NfemV:Nfem]*pressure_scaling

    print("     -> u (m,M) %.6e %.6e (cm/yr)" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("     -> v (m,M) %.6e %.6e (cm/yr)" %(np.min(v)/cm*year,np.max(v)/cm*year))
    print("     -> p (m,M) %.6e %.6e (MPa)" %(np.min(p)/1e6,np.max(p)/1e6))
    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("solve time: %.3f s" % (timing.time() - start))

    #####################################################################
    # relaxation step
    #####################################################################

    if iter>0:
       u=relax*u+(1-relax)*u_old
       v=relax*v+(1-relax)*v_old

    ######################################################################
    # temperature boundary conditions
    # which depend on velocity 
    ######################################################################

    bc_fixT=np.zeros(NfemT,dtype=bool)  # boundary condition, yes/no
    bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

    for i in range(0,NT):
        # top boundary - vack08
        if yT[i]/Ly>-eps: #
           bc_fixT[i]=True ; bc_valT[i]=Kelvin
        # left boundary 
        if xT[i]/Lx<eps:
           bc_fixT[i]=True ; bc_valT[i]=Kelvin+(1573-Kelvin)*erf((-yT[i])/(2*np.sqrt(kappa*50e6*year)))
        # right boundary 
        if xT[i]/Lx>1-eps:
           if yT[i]>-l2:
              bc_fixT[i]=True ; bc_valT[i]=(-yT[i])/l2*1300+Kelvin
           elif u[i]<0:
              bc_fixT[i]=True ; bc_valT[i]=1300.+Kelvin

    ######################################################################
    # build FE matrix for temperature 
    ######################################################################
    start = timing.time()

    A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64)# FE matrix
    rhs   = np.zeros(NfemT,dtype=np.float64)          # FE rhs 
    B_mat = np.zeros((ndim,ndofT*mT),dtype=np.float64)# gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(mT*ndofT,dtype=np.float64)
        a_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for kq in range(0,nqel):

            rq=qcoords_r[kq]
            sq=qcoords_s[kq]
            weightq=qweights[kq]

            N_mat[0:mT,0]=NNT(rq,sq)
            dNNNTdr[0:mT]=dNNTdr(rq,sq)
            dNNNTds[0:mT]=dNNTds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
                jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
                jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
                jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]

            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            # compute dNdx & dNdy
            vel[0,0]=0.
            vel[0,1]=0.
            xq=0.
            yq=0.
            for k in range(0,mT):
                vel[0,0]+=N_mat[k,0]*u[iconT[k,iel]]
                vel[0,1]+=N_mat[k,0]*v[iconT[k,iel]]
                xq+=N_mat[k,0]*xT[iconT[k,iel]]
                yq+=N_mat[k,0]*yT[iconT[k,iel]]
                dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                B_mat[0,k]=dNNNTdx[k]
                B_mat[1,k]=dNNNTdy[k]

            # compute diffusion matrix
            Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

            # compute advection matrix
            Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

        # end for kq

        a_el=Ka+Kd

        # apply boundary conditions

        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mT):
                   m2=iconT[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
        # end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            for k2 in range(0,mT):
                m2=iconT[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            rhs[m1]+=b_el[k1]
        # end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system 
    ######################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T: %.3f s" % (timing.time() - start))

    #################################################################
    # relax
    #################################################################

    if iter>0:
       T=relax*T+(1-relax)*T_old

    ######################################################################
    # checking for convergence
    ######################################################################

    xi_u=np.linalg.norm(u-u_old,2)/np.linalg.norm(u+u_old,2)
    xi_v=np.linalg.norm(v-v_old,2)/np.linalg.norm(v+v_old,2)
    xi_T=np.linalg.norm(T-T_old,2)/np.linalg.norm(T+T_old,2)

    print("conv: u,v,T: %.6f %.6f %.6f | tol= %.6f" %(xi_u,xi_v,xi_T,tol))

    convfile.write("%3d %10e %10e %10e %10e \n" %(iter,xi_u,xi_v,xi_T,tol)) 
    convfile.flush()

    if xi_u<tol and xi_v<tol and xi_T<tol:
       break
    else:
       u_old[:]=u[:]
       v_old[:]=v[:]
       T_old[:]=T[:]

    ######################################################################
    ######################################################################

    filename = 'Vnodes_{:04d}.vtu'.format(iter)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,NV))
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='vel' Format='ascii'>\n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='fix' Format='ascii'>\n")
    for i in range(0,NV):
        if bc_fix[i*ndofV]:
           vtufile.write("%10e %10e %10e \n" %(bc_val[i*ndofV],bc_val[i*ndofV+1],0.))
        else:
           vtufile.write("%10e %10e %10e \n" %(0,0.,0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,NV):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    ######################################################################
    ######################################################################

    filename = 'qpoints_{:04d}.vtu'.format(iter)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
    vtufile.write("<PointData Scalars='scalars'>\n")
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e \n" %eta_q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e \n" %T_q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e \n" %exx_q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e \n" %eyy_q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e \n" %exy_q[i])
    vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%10e %10e %10e \n" %(x_q[i],y_q[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nq):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nq):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nq):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

# end for iter

################################################################################################
################################################################################################

######################################################################
# compute nodal strainrate and pressure  
# Pressure is discontinous from element to element. I then compute
# its value on the nodes of the element and add the value in the q
# array, while keeping track how many times a given node has received 
# a value, so that I can compute the average after. 
# Same for strain rate components.
######################################################################
start = timing.time()

exx_n = np.zeros(NV,dtype=np.float64)  
eyy_n = np.zeros(NV,dtype=np.float64)  
exy_n = np.zeros(NV,dtype=np.float64)  
count = np.zeros(NV,dtype=np.int32)  
q=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,mV):
        rq=rVnodes[i]
        sq=sVnodes[i]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:mP]=NNP(rq,sq)
        jcb=np.zeros((2,2),dtype=np.float64)
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
        inode=iconV[i,iel]
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        exx_n[inode]+=np.dot(dNNNVdx[0:mV],u[iconV[0:mV,iel]])
        eyy_n[inode]+=np.dot(dNNNVdy[0:mV],v[iconV[0:mV,iel]])
        exy_n[inode]+=0.5*np.dot(dNNNVdx[0:mV],v[iconV[0:mV,iel]])+\
                      0.5*np.dot(dNNNVdy[0:mV],u[iconV[0:mV,iel]])
        count[inode]+=1
    #end for
#end for
 
exx_n/=count
eyy_n/=count
exy_n/=count
q/=count

print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
print("     -> q (m,M) %.6e %.6e (MPa)" %(np.min(q)/1e6,np.max(q)/1e6))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute nodal press & sr: %.3f s" % (timing.time() - start))

#################################################################
# compute nodal heat flux 
#################################################################
start = timing.time()

qx_n=np.zeros(NT,dtype=np.float64)
qy_n=np.zeros(NT,dtype=np.float64)
count=np.zeros(NT,dtype=np.int32)  

for iel in range(0,nel):
    for i in range(0,mT):
        rq=rTnodes[i]
        sq=sTnodes[i]
        dNNNTdr[0:mT]=dNNTdr(rq,sq)
        dNNNTds[0:mT]=dNNTds(rq,sq)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
            jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
            jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
            jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]
        #end for
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mT):
            dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
            dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
        #end for
        inode=iconT[i,iel]
        qx_n[inode]+=np.dot(dNNNTdx[0:mT],T[iconT[0:mT,iel]])
        qy_n[inode]+=np.dot(dNNNTdy[0:mT],T[iconT[0:mT,iel]])
        count[inode]+=1
    #end for
#end for
 
qx_n/=count
qy_n/=count

print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))

print("compute nodal heat flux: %.3f s" % (timing.time() - start))

#################################################################
# post-processing
#################################################################

#measuring T_{11,11}
for i in range(0,NT):
    if abs(xT[i]-60e3)<1 and abs(yT[i]+Ly-540e3)<1:
       print ('result1:',xT[i],yT[i]+Ly,T[i]-Kelvin)

diagfile=open('tempdiag.ascii',"w")
for i in range(0,NT):
    if abs(xT[i] + yT[i] ) <1: 
       diagfile.write("%10e %10e %10e \n " %(xT[i],yT[i],T[i]-Kelvin))
diagfile.close()

# compute average temperature

Tavrg=0.
for iel in range(0,nel):
    for kq in range(0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        N_mat[0:mT,0]=NNT(rq,sq)
        dNNNTdr[0:mT]=dNNTdr(rq,sq)
        dNNNTds[0:mT]=dNNTds(rq,sq)
        # calculate jacobian matrix
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mT):
            jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
            jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
            jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
            jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]
        jcob=np.linalg.det(jcb)
        Tq=0.
        for k in range(0,mT):
            Tq+=N_mat[k,0]*T[iconT[k,iel]]
        Tavrg+=Tq*weightq*jcob
    #end for
#end for
Tavrg/=(Lx*Ly)

print ('Tavrg=',Tavrg)

#################################################################
#equidistant grid with 6 km spacing, which is a 111 × 101 matrix
#stored row-wise starting in the top left corner.
#We need to localise every point of this processing grid in the 
#FE mesh, which is now done not so efficiently at all.
#grid_T1 is the temperature obtained from P1 shape functions 
#while grid_T2 is the temperature obtained from P2 shape functions. 
#################################################################
start = timing.time()

if do_post_processing:

   nnnx=111
   nnny=101
   M=nnnx*nnny
   grid_x = np.empty(M,dtype=np.float64) 
   grid_y = np.empty(M,dtype=np.float64) 
   grid_T1 = np.empty(M,dtype=np.float64)  
   grid_T2 = np.empty(M,dtype=np.float64)  
   grid_corner = np.zeros(M,dtype=np.int)  
   grid_slab = np.zeros(M,dtype=np.int)  
   grid_wedge = np.zeros(M,dtype=np.int)  

   counter = 0
   for j in range(0,nnny):
       for i in range(0,nnnx):
           grid_x[counter]=i*Lx/float(nnnx-1)
           grid_y[counter]=-j*Ly/float(nnny-1)
           #making sure nodes are stricly in the domain
           grid_x[counter]=min(grid_x[counter],(1-eps)*Lx)
           grid_x[counter]=max(grid_x[counter],eps*Lx)
           grid_y[counter]=min(grid_y[counter],-eps*Ly)
           grid_y[counter]=max(grid_y[counter],-(1-eps)*Ly)
           counter += 1
       #end for
   #end for

   for i in range(0,M):
       #print ('grid point ',i,x[i],y[i])
       for iel in range(0,nel):
           x1=xT[iconT[0,iel]] ; y1=yT[iconT[0,iel]]
           x2=xT[iconT[1,iel]] ; y2=yT[iconT[1,iel]]
           x3=xT[iconT[2,iel]] ; y3=yT[iconT[2,iel]]
           denom=((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
           N0 = ((y2-y3)*(grid_x[i]-x3) + (x3-x2)*(grid_y[i]-y3)) / denom 
           N1 = ((y3-y1)*(grid_x[i]-x3) + (x1-x3)*(grid_y[i]-y3)) / denom
           N2 = 1-N0-N1
           if 0<=N0 and N0<=1 and 0<=N1 and N1<=1 and 0<=N2 and N2<=1: #inside test 
              r=N1
              s=N2
              N_mat[0:mT,0]=NNT(r,s)
              grid_T1[i]=N0*T[iconT[0,iel]]+N1*T[iconT[1,iel]]+N2*T[iconT[2,iel]]
              grid_T2[i]=np.sum(N_mat[0:mT,0]*T[iconT[0:mT,iel]])
              #print('     -> iel=',iel,'T=',grid_T1[i],grid_T2[i])
              break
           #end if
       #end for
   #end for

   grid_T1-=Kelvin
   grid_T2-=Kelvin

   #temperature $T(11,11)$ which is the 111+11=122th point
   inode=111*10+11-1
   grid_corner[inode]=1
   print('     -> Tcorner=',grid_T1[inode],grid_T2[inode],grid_x[inode],grid_y[inode])

   #equation 17 in vack08
   Tslab=0.
   counter = 0
   for j in range(0,nnny):
       for i in range(0,nnnx):
           if i==j and i<=35:
              Tslab+=grid_T2[counter]**2
              grid_slab[counter]=1
           counter += 1
       #end for
   #end for
   Tslab=np.sqrt(Tslab/36)
   print('     -> Tslab=',Tslab)

   #equation 18 in vack08
   Twedge=0.
   counter = 0
   for j in range(0,nnny):
       for i in range(0,nnnx):
           if  9<=i and i<=20:
               if 9<=j and j<=i:
                  Twedge+=grid_T2[counter]**2
                  grid_wedge[counter]=1
           counter += 1
   #    #end for
   #end for
   Twedge=np.sqrt(Twedge/78)
   print('     -> Twedge=',Twedge)

   np.savetxt('grid.ascii',np.array([grid_x,grid_y,grid_T1,grid_T2,grid_corner,grid_slab,grid_wedge]).T)


   filename = 'meas_grid.vtu'
   vtufile=open(filename,"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(M,M))
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='T1' Format='ascii'>\n")
   for i in range(0,M):
        vtufile.write("%10e \n" %grid_T1[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='T2' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%10e \n" %grid_T2[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='corner' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%10e \n" %grid_corner[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='slab' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%10e \n" %grid_slab[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='wedge' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%10e \n" %grid_wedge[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%10e %10e %10e \n" %(grid_x[i],grid_y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nq):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,M):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,M):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   print("post processing on grid: %.3f s" % (timing.time() - start))

#################################################################
start = timing.time()

if True:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NVold,nel))
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
    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (mat[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='iel' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % iel)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range (0,nel):
        if mat[iel]==4:
           vtufile.write("%10e\n" % ((p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3.))
        else:
           vtufile.write("%10e\n" % 0.) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='tau' Format='ascii'> \n")
    for iel in range (0,nel):
        if mat[iel]==4:
           exx_c=(exx_n[iconV[0,iel]]+exx_n[iconV[1,iel]]+exx_n[iconV[2,iel]])/3.
           eyy_c=(eyy_n[iconV[0,iel]]+eyy_n[iconV[1,iel]]+eyy_n[iconV[2,iel]])/3.
           exy_c=(exy_n[iconV[0,iel]]+exy_n[iconV[1,iel]]+exy_n[iconV[2,iel]])/3.
           T_c  =(    T[iconT[0,iel]]+    T[iconT[1,iel]]+    T[iconT[2,iel]])/3.
           eta_c=viscosity(exx_c,eyy_c,exy_c,T_c,mat[iel])
           tau_xx=2.*eta_c*exx_c 
           tau_yy=2.*eta_c*eyy_c 
           tau_xy=2.*eta_c*exy_c 
           vtufile.write("%10e %10e %10e\n" % (tau_xx,tau_yy,tau_xy))
        else:
           vtufile.write("%10e %10e %10e\n" % (0.,0.,0.)) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(nx[i],ny[i],0))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (diff)' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %((u[i]-u_old[i])/cm*year,(v[i]-v_old[i])/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(qx_n[i],qy_n[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" %(T[i]-Kelvin))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='T (diff)' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" %(T[i]-T_old[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for i in range(0,NT):
        eta=viscosity(exx_n[i],eyy_n[i],exy_n[i],T[i],mat[iel])
        vtufile.write("%10e \n" %eta)
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
    vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (np.sqrt(0.5*(exx_n[i]**2+eyy_n[i]**2)+exy_n[i]**2)))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='interface' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%d \n" %interface[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u (bool)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v (bool)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_u (value)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2]:
           val=bc_val[i*2]/cm*year
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_v (value)' Format='ascii'> \n")
    for i in range(0,NV):
        if bc_fix[i*2+1]:
           val=bc_val[i*2+1]/cm*year
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_T (bool)' Format='ascii'> \n")
    for i in range(0,NT):
        if bc_fixT[i]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='fix_T (value)' Format='ascii'> \n")
    for i in range(0,NT):
        if bc_fixT[i]:
           val=bc_valT[i]-Kelvin
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
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[5,iel],iconV[3,iel],iconV[4,iel])    )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22) #5
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    #############################################

    vtufile=open('gridP.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NfemP,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NfemP):
        vtufile.write("%10e %10e %10e \n" %(xP[i],yP[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d\n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*3))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %5) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("produce vtu files: %.3f s" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

