import numpy as np
import sys as sys
import time as time
from tools import *
import velocity
from scipy.special import erf
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

###############################################################################
# Q1 basis functions in 2D - temperature equation
###############################################################################

def NNT(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def dNNTdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def dNNTds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###############################################################################
# MINI basis functions
###############################################################################

bubble=2
beta=0.25

def B(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)

def NNV(r,s):
    NV_0=0.25*(1-r)*(1-s) - 0.25*B(r,s)
    NV_1=0.25*(1+r)*(1-s) - 0.25*B(r,s)
    NV_2=0.25*(1+r)*(1+s) - 0.25*B(r,s)
    NV_3=0.25*(1-r)*(1+s) - 0.25*B(r,s)
    NV_4=B(r,s)
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4],dtype=np.float64)

def dNNVdr(r,s):
    dNVdr_0=-0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_1=+0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_2=+0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_3=-0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_4=dBdr(r,s) 
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4],dtype=np.float64)

def dNNVds(r,s):
    dNVds_0=-0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_1=-0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_2=+0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_3=+0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_4=dBds(r,s) 
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4],dtype=np.float64)

def NNP(r,s):
    NP_0=0.25*(1-r)*(1-s)
    NP_1=0.25*(1+r)*(1-s)
    NP_2=0.25*(1+r)*(1+s)
    NP_3=0.25*(1-r)*(1+s)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################
# this function receives the coordinates of the corners of a block
# as well as its desired resolution and returns the position of nodes.
# nodes on the boundary are flagged.
###############################################################################

def laypts4(x1,y1,x2,y2,x3,y3,x4,y4,x,y,hull,level):
    counter=0
    for j in range(0,level+1):
        for i in range(0,level+1):
            #equidistant
            r=-1.+2./level*i
            s=-1.+2./level*j
            N1=0.25*(1.-r)*(1.-s)
            N2=0.25*(1.+r)*(1.-s)
            N3=0.25*(1.+r)*(1.+s)
            N4=0.25*(1.-r)*(1.+s)
            x[counter]=x1*N1+x2*N2+x3*N3+x4*N4
            y[counter]=y1*N1+y2*N2+y3*N3+y4*N4
            if i==0 or i==level: hull[counter]=True
            if j==0 or j==level: hull[counter]=True
            counter+=1
        #end for
    #end for

###############################################################################

def eta(x,y):
    return 1e21

###############################################################################
#
# 12---------------13
# | \       7       |
# |  10------------11 
# |   | \           | 
# |   |  \     6    | 
# |   |   \         | 
# |   |    \        | 
# 5---6     8-------9 
# |   | \  / \      | 
# | 0 |  \7   \  3  | 
# |   | 1  \   \    | 
# 0---1-----2----3--4
#
###############################################################################

print("-----------------------------")
print("--------- stone 149 ---------")
print("-----------------------------")

m=4   # number of nodes per element
nel=8 # number of elements
NV=14 # number of nodes

x=np.empty(NV,dtype=np.float64) 
y=np.empty(NV,dtype=np.float64) 
icon =np.zeros((m,nel),dtype=np.int32)
hull=np.zeros(14,dtype=bool)

x[ 0]=0   ; y[ 0]=0
x[ 1]=50  ; y[ 1]=0
x[ 2]=330 ; y[ 2]=0
x[ 3]=600 ; y[ 3]=0
x[ 4]=660 ; y[ 4]=0
x[ 5]=0   ; y[ 5]=300
x[ 6]=50  ; y[ 6]=300
x[ 8]=330 ; y[ 8]=270
x[ 9]=660 ; y[ 9]=270 
x[10]=50  ; y[10]=550
x[11]=660 ; y[11]=550
x[12]=0   ; y[12]=600
x[13]=660 ; y[13]=600

x[7]=(x[1]+x[3]+x[10])/3    
y[7]=(y[1]+y[3]+y[10])/3    

icon[0:m,0]=[0,1,6,5]
icon[0:m,1]=[1,2,7,6]
icon[0:m,2]=[2,3,8,7]
icon[0:m,3]=[3,4,9,8]
icon[0:m,4]=[5,6,10,12]
icon[0:m,5]=[6,7,8,10]
icon[0:m,6]=[8,9,11,10]
icon[0:m,7]=[10,11,13,12]

export_to_vtu('initial.vtu',x,y,icon,hull)

###############################################################################
# assigning level (resolution) of each block
###############################################################################

if int(len(sys.argv) == 2):
   level=int(sys.argv[1])
else:
   level=48

nelx=level
nely=level
nel=nelx*nely

nnx=level+1
nny=level+1
NV=nnx*nny

print('level=',level)
print('nnx=nny=',nnx)

###############################################################################
# build generic connectivity array for a block
###############################################################################

block_icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        block_icon[0,counter]=i+j*(nelx+1)
        block_icon[1,counter]=i+1+j*(nelx+1)
        block_icon[2,counter]=i+1+(j+1)*(nelx+1)
        block_icon[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

#################################################################
# build each individual block
#################################################################

block0_x=np.empty(NV,dtype=np.float64) 
block0_y=np.empty(NV,dtype=np.float64) 
block0_icon =np.zeros((m, nel),dtype=np.int32)
block0_hull=np.zeros(NV,dtype=bool)
block0_icon[:,:]=block_icon[:,:]
laypts4(x[0],y[0],x[1],y[1],x[6],y[6],x[5],y[5],block0_x,block0_y,block0_hull,level)
export_to_vtu('block0.vtu',block0_x,block0_y,block0_icon,block0_hull)

block1_x=np.empty(NV,dtype=np.float64) 
block1_y=np.empty(NV,dtype=np.float64) 
block1_icon =np.zeros((m, nel),dtype=np.int32)
block1_hull=np.zeros(NV,dtype=bool)
block1_icon[:,:]=block_icon[:,:]
laypts4(x[1],y[1],x[2],y[2],x[7],y[7],x[6],y[6],block1_x,block1_y,block1_hull,level)
export_to_vtu('block1.vtu',block1_x,block1_y,block1_icon,block1_hull)

block2_x=np.empty(NV,dtype=np.float64) 
block2_y=np.empty(NV,dtype=np.float64) 
block2_icon =np.zeros((m, nel),dtype=np.int32)
block2_hull=np.zeros(NV,dtype=bool)
block2_icon[:,:]=block_icon[:,:]
laypts4(x[2],y[2],x[3],y[3],x[8],y[8],x[7],y[7],block2_x,block2_y,block2_hull,level)
export_to_vtu('block2.vtu',block2_x,block2_y,block2_icon,block2_hull)

block3_x=np.empty(NV,dtype=np.float64) 
block3_y=np.empty(NV,dtype=np.float64) 
block3_icon =np.zeros((m, nel),dtype=np.int32)
block3_hull=np.zeros(NV,dtype=bool)
block3_icon[:,:]=block_icon[:,:]
laypts4(x[3],y[3],x[4],y[4],x[9],y[9],x[8],y[8],block3_x,block3_y,block3_hull,level)
export_to_vtu('block3.vtu',block3_x,block3_y,block3_icon,block3_hull)

block4_x=np.empty(NV,dtype=np.float64) 
block4_y=np.empty(NV,dtype=np.float64) 
block4_icon =np.zeros((m, nel),dtype=np.int32)
block4_hull=np.zeros(NV,dtype=bool)
block4_icon[:,:]=block_icon[:,:]
laypts4(x[5],y[5],x[6],y[6],x[10],y[10],x[12],y[12],block4_x,block4_y,block4_hull,level)
export_to_vtu('block4.vtu',block4_x,block4_y,block4_icon,block4_hull)

block5_x=np.empty(NV,dtype=np.float64) 
block5_y=np.empty(NV,dtype=np.float64) 
block5_icon =np.zeros((m, nel),dtype=np.int32)
block5_hull=np.zeros(NV,dtype=bool)
block5_icon[:,:]=block_icon[:,:]
laypts4(x[6],y[6],x[7],y[7],x[8],y[8],x[10],y[10],block5_x,block5_y,block5_hull,level)
export_to_vtu('block5.vtu',block5_x,block5_y,block5_icon,block5_hull)

block6_x=np.empty(NV,dtype=np.float64) 
block6_y=np.empty(NV,dtype=np.float64) 
block6_icon =np.zeros((m, nel),dtype=np.int32)
block6_hull=np.zeros(NV,dtype=bool)
block6_icon[:,:]=block_icon[:,:]
laypts4(x[8],y[8],x[9],y[9],x[11],y[11],x[10],y[10],block6_x,block6_y,block6_hull,level)
export_to_vtu('block6.vtu',block6_x,block6_y,block6_icon,block6_hull)

block7_x=np.empty(NV,dtype=np.float64) 
block7_y=np.empty(NV,dtype=np.float64) 
block7_icon =np.zeros((m, nel),dtype=np.int32)
block7_hull=np.zeros(NV,dtype=bool)
block7_icon[:,:]=block_icon[:,:]
laypts4(x[10],y[10],x[11],y[11],x[13],y[13],x[12],y[12],block7_x,block7_y,block7_hull,level)
export_to_vtu('block7.vtu',block7_x,block7_y,block7_icon,block7_hull)

###############################################################################
# assemble blocks into single mesh
###############################################################################

print('-----merging 0+1----------------')
x01,y01,icon01,hull01=merge_two_blocks(block0_x,block0_y,block0_icon,block0_hull,\
                                       block1_x,block1_y,block1_icon,block1_hull)

export_to_vtu('blocks_0-1.vtu',x01,y01,icon01,hull01)
print('produced blocks_0-1.vtu')

print('-----merging 0+1+2--------------')
x02,y02,icon02,hull02=merge_two_blocks(x01,y01,icon01,hull01,\
                                       block2_x,block2_y,block2_icon,block2_hull)


export_to_vtu('blocks_0-2.vtu',x02,y02,icon02,hull02)
print('produced blocks_0-2.vtu')

print('-----merging 0+1+2+3------------')

x03,y03,icon03,hull03=merge_two_blocks(x02,y02,icon02,hull02,\
                                       block3_x,block3_y,block3_icon,block3_hull)

export_to_vtu('blocks_0-3.vtu',x03,y03,icon03,hull03)
print('produced blocks_0-3.vtu')

print('-----merging 0+1+2+3+4----------')

x04,y04,icon04,hull04=merge_two_blocks(x03,y03,icon03,hull03,\
                                       block4_x,block4_y,block4_icon,block4_hull)

export_to_vtu('blocks_0-4.vtu',x04,y04,icon04,hull04)
print('produced blocks_0-4.vtu')

print('-----merging 0+1+2+3+4+5--------')

x05,y05,icon05,hull05=merge_two_blocks(x04,y04,icon04,hull04,\
                                       block5_x,block5_y,block5_icon,block5_hull)

export_to_vtu('blocks_0-5.vtu',x05,y05,icon05,hull05)
print('produced blocks_0-5.vtu')

print('-----merging 0+1+2+3+4+5+6------')

x06,y06,icon06,hull06=merge_two_blocks(x05,y05,icon05,hull05,\
                                       block6_x,block6_y,block6_icon,block6_hull)

export_to_vtu('blocks_0-6.vtu',x06,y06,icon06,hull06)
print('produced blocks_0-6.vtu')

print('-----merging 0+1+2+3+4+5+6+7----')

x07,y07,icon07,hull07=merge_two_blocks(x06,y06,icon06,hull06,\
                                       block7_x,block7_y,block7_icon,block7_hull)

export_to_vtu('blocks_0-7.vtu',x07,y07,icon07,hull07)
print('produced blocks_0-7.vtu')

print("------------------------------")
print(" Temperature meshing completed ")
print("------------------------------")

x36,y36,icon36,hull36=merge_two_blocks(block3_x,block3_y,block3_icon,block3_hull,\
                                       block6_x,block6_y,block6_icon,block6_hull)

export_to_vtu('blocks_3+6.vtu',x36,y36,icon36,hull36)

print("--------------------------------")
print(" Stokes domain meshing completed ")
print("--------------------------------")

###############################################################################
mV=5

ndofV=2
ndofP=1
ndofT=1

NT=np.size(x07)
mT,nelT=np.shape(icon07)

NP=np.size(x36)
mP,nelS=np.shape(icon36)

NV=NP+nelS

xT=np.empty(NT,dtype=np.float64)
yT=np.empty(NT,dtype=np.float64)
iconT=np.zeros((mT,nelT),dtype=np.int32)

xP=np.empty(NP,dtype=np.float64)
yP=np.empty(NP,dtype=np.float64)
iconP=np.zeros((mP,nelS),dtype=np.int32)

xV=np.zeros(NV,dtype=np.float64)
yV=np.zeros(NV,dtype=np.float64)
iconV=np.zeros((mV,nelS),dtype=np.int32)

xT[:]=x07*1e3
yT[:]=y07*1e3
iconT[:,:]=icon07[:,:]

xP[:]=x36*1e3
yP[:]=y36*1e3
iconP[:,:]=icon36[:,:]

xV[0:NP]=xP[0:NP]
yV[0:NP]=yP[0:NP]
iconV[0:mP,:]=iconP[0:mP,:]

for iel in range(0,nelS):
    xV[NP+iel]=np.sum(xP[iconP[:,iel]])*0.25
    yV[NP+iel]=np.sum(yP[iconP[:,iel]])*0.25
    iconV[4,iel]=NP+iel

NfemT=NT*ndofT
NfemP=NP*ndofP
NfemV=NV*ndofV
Nfem=NfemV+NfemP

print('mV=',mV)
print('mP=',mP)
print('mT=',mT)
print('nelT=',nelT)
print('NT=',NT)
print('nelS=',nelS)
print('NP=',NP)
print('NV=',NV)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('NfemT=',NfemT)
print('Nfem=',Nfem)
print("-----------------------------")

#np.savetxt('meshT.ascii',np.array([xT,yT]).T,header='# x,y')
#np.savetxt('meshP.ascii',np.array([xP,yP]).T,header='# x,y')
#np.savetxt('meshV.ascii',np.array([xV,yV]).T,header='# x,y')

###############################################################################
# constants
###############################################################################

Kelvin=273.15
ndim=2
sqrt3=np.sqrt(3.)
cm=0.01 
year=365.25*24*3600
eps=1e-4

###############################################################################

caase='1c'

Lx=660e3
Ly=600e3
hcond=3    # heat conductivity
hcapa=1250 # heat capacity
rho=3300   # density
l1=1000.e3
l2=50.e3
l3=0.e3
vel=5*cm/year
angle=45./180.*np.pi  

eta_ref=1e21

print('case=',caase)
print("-----------------------------")

###############################################################################
# quadrature rules
###############################################################################

nqperdim=2

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

###############################################################################
# mapping the Stokes domain (only Q1 nodes) onto T domain
# very inefficient algorithm!
###############################################################################
start = timing.time()

in_stokes_domain=np.zeros(NT,dtype=bool) 
for i in range(0,NT):
    if yT[i]>= Ly-xT[i]-1 and yT[i]<=Ly-50e3 +1:
       in_stokes_domain[i]=True

mapping =np.zeros(NP,dtype=np.int32)
for i in range(0,NP):
    for j in range(0,NT):
        if in_stokes_domain[j] and abs(xV[i]-xT[j])<eps and abs(yV[i]-yT[j])<eps:
           mapping[i]=j
           break

print("establish mapping S->T: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
# This is only valid for the Stokes mesh.
###############################################################################
start = timing.time()

area=np.zeros(nelS,dtype=np.float64) 

for iel in range(0,nelS):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq
        if area[iel]<0: 
           print('PB with iel=',iel) 
           for k in range(0,mV):
               print (xV[iconV[k,iel]],yV[iconV[k,iel]])
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %((610e3+60e3)/2*550e3))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# this is a steady state code, but I leave the loop so that 
# one could re-introduce time stepping or iterations between
# Stokes and temperature equations.
###############################################################################

T=np.empty(NT,dtype=np.float64) 

for iter in range(0,1):

    uu=np.zeros(NT,dtype=np.float64) # velocity x field on T mesh
    vv=np.zeros(NT,dtype=np.float64) # velocity y field on T mesh
    pp=np.zeros(NT,dtype=np.float64) # pressure field on T mesh

    #################################################################
    # assign i(case 1a) or compute (case 1b,1c) velocity to nodes
    # note that I have completely taken the buyoancy forces out
    #################################################################

    if not caase=='1a':

       bc_fix = np.zeros(NfemV, dtype=bool)  # boundary condition, yes/no
       bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value
       for i in range(0,NV):
           if abs(yV[i]-550e3)/Ly<eps: #top
              bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
              bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
           if abs(yV[i]-Ly+xV[i])/Lx<eps: #left
              bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = +vel/np.sqrt(2)
              bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -vel/np.sqrt(2)
           if caase=='1b' and  (abs(yV[i])/Ly<eps or abs(xV[i]-Lx)/Lx<eps): 
              uui,vvi=velocity.compute_corner_flow_velocity(xV[i],yV[i],l1,l2,l3,angle,vel,Lx,Ly)
              bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = uui
              bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vvi
       #end for

       A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
       c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
       dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
       b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
       N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
       f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
       h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 

       for iel in range(0,nelS):

           # set arrays to 0 every loop
           f_el =np.zeros((mV*ndofV),dtype=np.float64)
           K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
           G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
           h_el=np.zeros((mP*ndofP),dtype=np.float64)

           for iq in range(0,nqperdim):
               for jq in range(0,nqperdim):
                   rq=qcoords[iq]
                   sq=qcoords[jq]
                   weightq=qweights[iq]*qweights[jq]

                   NNNV=NNV(rq,sq)
                   dNNNVdr=dNNVdr(rq,sq)
                   dNNNVds=dNNVds(rq,sq)
                   NNNP=NNP(rq,sq)

                   # calculate jacobian matrix
                   jcb=np.zeros((ndim,ndim),dtype=np.float64)
                   for k in range(0,mV):
                       jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                       jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                       jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                       jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                   jcob=np.linalg.det(jcb)
                   jcbi=np.linalg.inv(jcb)

                   # compute dNdx & dNdy
                   xq=0.0
                   yq=0.0
                   for k in range(0,mV):
                       xq+=NNNV[k]*xV[iconV[k,iel]]
                       yq+=NNNV[k]*yV[iconV[k,iel]]
                       dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                       dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

                   # construct 3x8 b_mat matrix
                   for i in range(0,mV):
                       b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                                [0.        ,dNNNVdy[i]],
                                                [dNNNVdy[i],dNNNVdx[i]]]

                   # compute elemental a_mat matrix
                   K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

                   for i in range(0,mP):
                       N_mat[0,i]=NNNP[i]
                       N_mat[1,i]=NNNP[i]
                       N_mat[2,i]=0.

                   G_el-=b_mat.T.dot(N_mat)*weightq*jcob

               # end for jq
           # end for iq

           G_el*=eta_ref/Ly

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

       #end for iel

       print("build FE matrix: %.3fs" % (timing.time()-start))

       ############################################################## 
       # solving linear system
       ##############################################################
       start = timing.time()

       rhs=np.zeros(Nfem,dtype=np.float64)
       rhs[0:NfemV]=f_rhs
       rhs[NfemV:Nfem]=h_rhs
       sparse_matrix=A_sparse.tocsr()
       sol=sps.linalg.spsolve(sparse_matrix,rhs)

       u,v=np.reshape(sol[0:NfemV],(NV,2)).T
       p=sol[NfemV:Nfem]*(eta_ref/Ly)

       print("     -> u (m,M) %.4e %.4e " %(np.min(u/cm*year),np.max(u/cm*year)))
       print("     -> v (m,M) %.4e %.4e " %(np.min(v/cm*year),np.max(v/cm*year)))
       print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

       #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
       #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

       print("solve time: %.3f s" % (timing.time() - start))

       ##############################################################
       # export Stokes solution to vtu format
       ##############################################################
       start = timing.time()

       vtufile=open('solutionS_lvl'+str(level)+'.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nelS))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" %(xP[i],yP[i],0))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e %10e %10e \n" % (u[i]/cm*year,v[i]/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cornerflow)' Format='ascii'> \n")
       for i in range(0,NP):
           uui,vvi=velocity.compute_corner_flow_velocity(xP[i],yP[i],l1,l2,l3,angle,vel,Lx,Ly)
           vtufile.write("%10e %10e %10e \n" % (uui/cm*year,vvi/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32'   Name='pressure' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10e \n" % (p[i]))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='bc' Format='ascii'> \n")
       for i in range(0,NP):
           if bc_fix[2*i+0]: 
              uii=bc_val[2*i]/cm*year
           else:
              uii=0
           if bc_fix[2*i+1]: 
              vii=bc_val[2*i+1]/cm*year
           else:
              vii=0
           vtufile.write("%10e %10e %10e \n" % (uii,vii,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nelS):
           vtufile.write("%d %d %d %d \n" %(iconP[0,iel],iconP[1,iel],iconP[2,iel],iconP[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nelS):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nelS):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu: %.3f s" % (timing.time() - start))

       ##############################################################
       # project velocity on T mesh
       ##############################################################

       for i in range(0,NP):
           uu[mapping[i]]=u[i]
           vv[mapping[i]]=v[i]
           pp[mapping[i]]=p[i]

       for i in range(0,NT):
           if yT[i]<Ly-xT[i]:
              uu[i]=+vel/np.sqrt(2)
              vv[i]=-vel/np.sqrt(2)

    else: # not solving Stokes system

       for i in range(0,NT):
           uu[i],vv[i]=velocity.compute_corner_flow_velocity(xT[i],yT[i],l1,l2,l3,angle,vel,Lx,Ly)

    #################################################################
    # temperature boundary conditions which depend on velocity 
    #################################################################

    bc_fixT=np.zeros(NfemT,dtype=bool)  # boundary condition, yes/no
    bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

    kappa=hcond/rho/hcapa

    age=50e6

    for i in range(0,NT):
        # top boundary 
        if abs(yT[i]-Ly)/Ly<eps: #
           bc_fixT[i]=True ; bc_valT[i]=Kelvin
        # left boundary 
        if abs(xT[i]/Lx)<eps:
           bc_fixT[i]=True ; bc_valT[i]=Kelvin+(1573-Kelvin)*erf(((Ly-yT[i]))/(2*np.sqrt(kappa*age*year)))
        # right boundary 
        if abs(xT[i]-Lx)/Lx<eps:
           if yT[i]>=Ly-l2:
              bc_fixT[i]=True ; bc_valT[i]=((Ly-yT[i]))/l2*1300+Kelvin
           elif uu[i]<0:
              bc_fixT[i]=True ; bc_valT[i]=1300.+Kelvin

    #################################################################
    # build FE matrix for temperature 
    #################################################################
    start = timing.time()

    A_mat = lil_matrix((NfemT,NfemT),dtype=np.float64) # FE matrix
    rhs   = np.zeros(NfemT,dtype=np.float64)           # FE rhs 
    B_mat = np.zeros((ndim,mT),dtype=np.float64)       # gradient matrix B 
    N_mat = np.zeros((mT,1),dtype=np.float64)          # shape functions vector
    dNNNTdr = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTds = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdx = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdy = np.zeros(mT,dtype=np.float64)            # shape functions derivatives

    for iel in range (0,nelT):

        b_el=np.zeros(mT,dtype=np.float64)       # elemental rhs
        a_el=np.zeros((mT,mT),dtype=np.float64)  # elemental matrix
        Ka=np.zeros((mT,mT),dtype=np.float64)    # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)    # elemental diffusion matrix 
        velq=np.zeros((1,ndim),dtype=np.float64) # velocity at quad point

        # integrate matrix and rhs at 4 quadrature points
        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

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
                velq[0,0]=0.
                velq[0,1]=0.
                for k in range(0,mT):
                    velq[0,0]+=N_mat[k,0]*uu[iconT[k,iel]]
                    velq[0,1]+=N_mat[k,0]*vv[iconT[k,iel]]
                    dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                    dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
                    B_mat[0,k]=dNNNTdx[k]
                    B_mat[1,k]=dNNNTdy[k]

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(velq.dot(B_mat))*rho*hcapa*weightq*jcob

	    # end for jq
        # end for iq

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
               # end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            # end if
        # end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            for k2 in range(0,mT):
                m2=iconT[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            # end for
            rhs[m1]+=b_el[k1]
        # end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system 
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    #np.savetxt('T.ascii',np.array([xT,yT,T]).T,header='# x,y')

    print("     -> T (m,M) %.4f %.4f " %(np.min(T-Kelvin),np.max(T-Kelvin)))

    print("solve T: %.3f s" % (timing.time() - start))

    #################################################################
    # compute heat flux
    #################################################################
    start = timing.time()

    qx=np.zeros(NT,dtype=np.float64) 
    qy=np.zeros(NT,dtype=np.float64) 
    cc=np.zeros(NT,dtype=np.float64) 

    rTnodes=[-1,1,1,-1]
    sTnodes=[-1,-1,1,1]

    for iel in range(0,nelT):
        for kk in range(0,mT):
            inode=iconT[kk,iel]
            rq = rTnodes[kk]
            sq = sTnodes[kk]
            dNNNTdr=dNNTdr(rq,sq)
            dNNNTds=dNNTds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mT):
                jcb[0,0]+=dNNNTdr[k]*xT[iconT[k,iel]]
                jcb[0,1]+=dNNNTdr[k]*yT[iconT[k,iel]]
                jcb[1,0]+=dNNNTds[k]*xT[iconT[k,iel]]
                jcb[1,1]+=dNNNTds[k]*yT[iconT[k,iel]]
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mT):
                dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]
                dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]
            for k in range(0,mT):
                qx[inode]-=hcond*dNNNTdx[k]*T[iconT[k,iel]]
                qy[inode]-=hcond*dNNNTdy[k]*T[iconT[k,iel]]
            cc[inode]+=1
        #end for
    #end for
    qx/=cc
    qy/=cc

    print("     -> qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
    print("     -> qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))

    print("compute heat flux: %.3f s" % (timing.time() - start))

    #################################################################
    # export Temperature along given lines 
    #################################################################
    start = timing.time()

    diagT=np.zeros(NT,dtype=np.float64)  # size way too large
    diagP=np.zeros(NT,dtype=np.float64)  # size way too large
    dist=np.zeros(NT,dtype=np.float64)   # size way too large
    rightu=np.zeros(NT,dtype=np.float64) # size way too large
    rightv=np.zeros(NT,dtype=np.float64) # size way too large
    rightT=np.zeros(NT,dtype=np.float64) # size way too large
    rightP=np.zeros(NT,dtype=np.float64) # size way too large
    depth=np.zeros(NT,dtype=np.float64)  # size way too large
    topqx=np.zeros(NT,dtype=np.float64)  # size way too large
    topqy=np.zeros(NT,dtype=np.float64)  # size way too large
    topxx=np.zeros(NT,dtype=np.float64)  # size way too large

    counter=0
    for i in range(0,NT):
        if abs(yT[i]-Ly+xT[i])/Lx<eps:
           diagT[counter]=T[i]
           diagP[counter]=pp[i]
           dist[counter]=np.sqrt( (xT[i]-0)**2+(yT[i]-Ly)**2 )
           counter+=1

    np.savetxt('diag_lvl'+str(level)+'.ascii',np.array([dist[0:counter],diagT[0:counter],diagP[0:counter]]).T)

    counter=0
    for i in range(0,NT):
        if abs(xT[i]-Lx)/Lx<eps:
           rightT[counter]=T[i]
           rightP[counter]=pp[i]
           rightu[counter]=uu[i]
           rightv[counter]=vv[i]
           depth[counter]=Ly-yT[i]
           counter+=1

    np.savetxt('right_lvl'+str(level)+'.ascii',np.array([depth[0:counter],rightT[0:counter],rightP[0:counter],\
                                                                          rightu[0:counter],rightv[0:counter]]).T)

    counter=0
    for i in range(0,NT):
        if abs(yT[i]-Ly)/Ly<eps:
           topqx[counter]=qx[i]
           topqy[counter]=qy[i]
           topxx[counter]=xT[i]
           counter+=1

    np.savetxt('top_lvl'+str(level)+'.ascii',np.array([topxx[0:counter],topqx[0:counter],topqy[0:counter]]).T)

    print('     -> produced diag_lvl'+str(level)+'.ascii')
    print('     -> produced right_lvl'+str(level)+'.ascii')
    print('     -> produced top_lvl'+str(level)+'.ascii')

    print("carry out measurements: %.3f s" % (timing.time() - start))

    #################################################################
    # export solution to vtu file
    #################################################################
    start = timing.time()

    vtufile=open('solutionT_lvl'+str(level)+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nelT))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" %(xT[i],yT[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--  
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='velocity' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" % (uu[i]/cm*year,vv[i]/cm*year,0))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='heat flux' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e %10e %10e \n" % (qx[i],qy[i],0))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" % (pp[i]))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    for i in range(0,NT):
        vtufile.write("%10e \n" % (T[i]-Kelvin))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='stokes' Format='ascii'> \n")
    for i in range(0,NT):
        if in_stokes_domain[i]:  
           vtufile.write("%10e \n" % 1)
        else:
           vtufile.write("%10e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nelT):
        vtufile.write("%d %d %d %d \n" %(iconT[0,iel],iconT[1,iel],iconT[2,iel],iconT[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nelT):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nelT):
        vtufile.write("%d \n" %9)
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

###############################################################################
