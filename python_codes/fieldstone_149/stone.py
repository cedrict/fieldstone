import numpy as np
import sys as sys
import time as time
from tools import *
import velocity
from scipy.special import erf
import time as clock
from scipy.sparse import lil_matrix
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve

debug=False
caase='1c'

###############################################################################
# Q1 basis functions in 2D - temperature equation
###############################################################################

def basis_functions_T(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_T_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_T_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################
# MINI basis functions
###############################################################################

bubble=2
beta=0.25

def BB(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)

def basis_functions_V(r,s):
    N0=0.25*(1-r)*(1-s) - 0.25*BB(r,s)
    N1=0.25*(1+r)*(1-s) - 0.25*BB(r,s)
    N2=0.25*(1+r)*(1+s) - 0.25*BB(r,s)
    N3=0.25*(1-r)*(1+s) - 0.25*BB(r,s)
    N4=BB(r,s)
    return np.array([N0,N1,N2,N3,N4],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s) -0.25*dBBdr(r,s)
    dNdr1=+0.25*(1.-s) -0.25*dBBdr(r,s)
    dNdr2=+0.25*(1.+s) -0.25*dBBdr(r,s)
    dNdr3=-0.25*(1.+s) -0.25*dBBdr(r,s)
    dNdr4=dBBdr(r,s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r) -0.25*dBBds(r,s)
    dNds1=-0.25*(1.+r) -0.25*dBBds(r,s)
    dNds2=+0.25*(1.+r) -0.25*dBBds(r,s)
    dNds3=+0.25*(1.-r) -0.25*dBBds(r,s)
    dNds4=dBBds(r,s) 
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

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

print("*******************************")
print("********** stone 149 **********")
print("*******************************")

###############################################################################
start=clock.time()

m=4   # number of nodes per element
nel=8 # number of elements
nn_V=14 # number of nodes

x=np.zeros(nn_V,dtype=np.float64) 
y=np.zeros(nn_V,dtype=np.float64) 
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

if debug: export_to_vtu('initial.vtu',x,y,icon,hull)

print("initial block layout: %.3f s" % (clock.time()-start))

###############################################################################
# assigning level (resolution) of each block
###############################################################################
start=clock.time()

if int(len(sys.argv) == 2):
   level=int(sys.argv[1])
else:
   level=48

nelx=level
nely=level
nel=nelx*nely

nnx=level+1
nny=level+1
nn_V=nnx*nny

print('level=',level)
print('nnx=nny=',nnx)

print("assign level to blocks: %.3f s" % (clock.time()-start))

###############################################################################
# build generic connectivity array for a block
###############################################################################
start=clock.time()

block_icon=np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        block_icon[0,counter]=i+j*(nelx+1)
        block_icon[1,counter]=i+1+j*(nelx+1)
        block_icon[2,counter]=i+1+(j+1)*(nelx+1)
        block_icon[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("build generic block connectivity: %.3f s" % (clock.time()-start))

#################################################################
# build each individual block
#################################################################
start=clock.time()

block0_x=np.zeros(nn_V,dtype=np.float64) 
block0_y=np.zeros(nn_V,dtype=np.float64) 
block0_icon =np.zeros((m, nel),dtype=np.int32)
block0_hull=np.zeros(nn_V,dtype=bool)
block0_icon[:,:]=block_icon[:,:]
laypts4(x[0],y[0],x[1],y[1],x[6],y[6],x[5],y[5],block0_x,block0_y,block0_hull,level)
if debug: export_to_vtu('block0.vtu',block0_x,block0_y,block0_icon,block0_hull)

block1_x=np.zeros(nn_V,dtype=np.float64) 
block1_y=np.zeros(nn_V,dtype=np.float64) 
block1_icon =np.zeros((m, nel),dtype=np.int32)
block1_hull=np.zeros(nn_V,dtype=bool)
block1_icon[:,:]=block_icon[:,:]
laypts4(x[1],y[1],x[2],y[2],x[7],y[7],x[6],y[6],block1_x,block1_y,block1_hull,level)
if debug: export_to_vtu('block1.vtu',block1_x,block1_y,block1_icon,block1_hull)

block2_x=np.zeros(nn_V,dtype=np.float64) 
block2_y=np.zeros(nn_V,dtype=np.float64) 
block2_icon =np.zeros((m, nel),dtype=np.int32)
block2_hull=np.zeros(nn_V,dtype=bool)
block2_icon[:,:]=block_icon[:,:]
laypts4(x[2],y[2],x[3],y[3],x[8],y[8],x[7],y[7],block2_x,block2_y,block2_hull,level)
if debug: export_to_vtu('block2.vtu',block2_x,block2_y,block2_icon,block2_hull)

block3_x=np.zeros(nn_V,dtype=np.float64) 
block3_y=np.zeros(nn_V,dtype=np.float64) 
block3_icon =np.zeros((m, nel),dtype=np.int32)
block3_hull=np.zeros(nn_V,dtype=bool)
block3_icon[:,:]=block_icon[:,:]
laypts4(x[3],y[3],x[4],y[4],x[9],y[9],x[8],y[8],block3_x,block3_y,block3_hull,level)
if debug: export_to_vtu('block3.vtu',block3_x,block3_y,block3_icon,block3_hull)

block4_x=np.zeros(nn_V,dtype=np.float64) 
block4_y=np.zeros(nn_V,dtype=np.float64) 
block4_icon =np.zeros((m, nel),dtype=np.int32)
block4_hull=np.zeros(nn_V,dtype=bool)
block4_icon[:,:]=block_icon[:,:]
laypts4(x[5],y[5],x[6],y[6],x[10],y[10],x[12],y[12],block4_x,block4_y,block4_hull,level)
if debug: export_to_vtu('block4.vtu',block4_x,block4_y,block4_icon,block4_hull)

block5_x=np.zeros(nn_V,dtype=np.float64) 
block5_y=np.zeros(nn_V,dtype=np.float64) 
block5_icon =np.zeros((m, nel),dtype=np.int32)
block5_hull=np.zeros(nn_V,dtype=bool)
block5_icon[:,:]=block_icon[:,:]
laypts4(x[6],y[6],x[7],y[7],x[8],y[8],x[10],y[10],block5_x,block5_y,block5_hull,level)
if debug: export_to_vtu('block5.vtu',block5_x,block5_y,block5_icon,block5_hull)

block6_x=np.zeros(nn_V,dtype=np.float64) 
block6_y=np.zeros(nn_V,dtype=np.float64) 
block6_icon =np.zeros((m, nel),dtype=np.int32)
block6_hull=np.zeros(nn_V,dtype=bool)
block6_icon[:,:]=block_icon[:,:]
laypts4(x[8],y[8],x[9],y[9],x[11],y[11],x[10],y[10],block6_x,block6_y,block6_hull,level)
if debug: export_to_vtu('block6.vtu',block6_x,block6_y,block6_icon,block6_hull)

block7_x=np.zeros(nn_V,dtype=np.float64) 
block7_y=np.zeros(nn_V,dtype=np.float64) 
block7_icon =np.zeros((m, nel),dtype=np.int32)
block7_hull=np.zeros(nn_V,dtype=bool)
block7_icon[:,:]=block_icon[:,:]
laypts4(x[10],y[10],x[11],y[11],x[13],y[13],x[12],y[12],block7_x,block7_y,block7_hull,level)
if debug: export_to_vtu('block7.vtu',block7_x,block7_y,block7_icon,block7_hull)

print("build 8 blocks: %.3f s" % (clock.time()-start))

###############################################################################
# assemble blocks into single mesh
###############################################################################
start=clock.time()

print('     merging 0+1')
x01,y01,icon01,hull01=merge_two_blocks(block0_x,block0_y,block0_icon,block0_hull,\
                                       block1_x,block1_y,block1_icon,block1_hull)

if debug: export_to_vtu('blocks_0-1.vtu',x01,y01,icon01,hull01) ; print('     -> produced blocks_0-1.vtu')

print('     merging 0+1+2')
x02,y02,icon02,hull02=merge_two_blocks(x01,y01,icon01,hull01,\
                                       block2_x,block2_y,block2_icon,block2_hull)


if debug: export_to_vtu('blocks_0-2.vtu',x02,y02,icon02,hull02) ; print('     -> produced blocks_0-2.vtu')

print('     merging 0+1+2+3')

x03,y03,icon03,hull03=merge_two_blocks(x02,y02,icon02,hull02,\
                                       block3_x,block3_y,block3_icon,block3_hull)

if debug: export_to_vtu('blocks_0-3.vtu',x03,y03,icon03,hull03) ; print('     -> produced blocks_0-3.vtu')

print('     merging 0+1+2+3+4')

x04,y04,icon04,hull04=merge_two_blocks(x03,y03,icon03,hull03,\
                                       block4_x,block4_y,block4_icon,block4_hull)

if debug: export_to_vtu('blocks_0-4.vtu',x04,y04,icon04,hull04) ; print('     -> produced blocks_0-4.vtu')

print('     merging 0+1+2+3+4+5')

x05,y05,icon05,hull05=merge_two_blocks(x04,y04,icon04,hull04,\
                                       block5_x,block5_y,block5_icon,block5_hull)

if debug: export_to_vtu('blocks_0-5.vtu',x05,y05,icon05,hull05) ; print('     -> produced blocks_0-5.vtu')

print('     merging 0+1+2+3+4+5+6')

x06,y06,icon06,hull06=merge_two_blocks(x05,y05,icon05,hull05,\
                                       block6_x,block6_y,block6_icon,block6_hull)

if debug: export_to_vtu('blocks_0-6.vtu',x06,y06,icon06,hull06) ; print('     -> produced blocks_0-6.vtu')

print('     merging 0+1+2+3+4+5+6+7')

x07,y07,icon07,hull07=merge_two_blocks(x06,y06,icon06,hull06,\
                                       block7_x,block7_y,block7_icon,block7_hull)

if debug: export_to_vtu('blocks_0-7.vtu',x07,y07,icon07,hull07) ; print('     -> produced blocks_0-7.vtu')

print("     -> Temp domain meshing completed ")

x36,y36,icon36,hull36=merge_two_blocks(block3_x,block3_y,block3_icon,block3_hull,\
                                       block6_x,block6_y,block6_icon,block6_hull)

if debug: export_to_vtu('blocks_3+6.vtu',x36,y36,icon36,hull36) ; print('     -> produced blocks_3+6.vtu')

print("     -> Stokes domain meshing completed ")

print("merge blocks: %.3f s" % (clock.time()-start))

###############################################################################
# compute number of nodes and elts for each grid
###############################################################################
start=clock.time()

ndof_V=2

nn_T=np.size(x07)
m_T,nel_T=np.shape(icon07)

nn_P=np.size(x36)
m_P,nel_S=np.shape(icon36)

m_V=5
nn_V=nn_P+nel_S

x_T=np.zeros(nn_T,dtype=np.float64)
y_T=np.zeros(nn_T,dtype=np.float64)
icon_T=np.zeros((m_T,nel_T),dtype=np.int32)

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)
icon_P=np.zeros((m_P,nel_S),dtype=np.int32)

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
icon_V=np.zeros((m_V,nel_S),dtype=np.int32)

x_T[:]=x07*1e3
y_T[:]=y07*1e3
icon_T[:,:]=icon07[:,:]

x_P[:]=x36*1e3
y_P[:]=y36*1e3
icon_P[:,:]=icon36[:,:]

x_V[0:nn_P]=x_P[0:nn_P]
y_V[0:nn_P]=y_P[0:nn_P]
icon_V[0:m_P,:]=icon_P[0:m_P,:]

for iel in range(0,nel_S):
    x_V[nn_P+iel]=np.sum(x_P[icon_P[:,iel]])*0.25
    y_V[nn_P+iel]=np.sum(y_P[icon_P[:,iel]])*0.25
    icon_V[4,iel]=nn_P+iel

print(np.min(x_V),np.max(x_V))
print(np.min(y_V),np.max(y_V))

Nfem_T=nn_T
Nfem_P=nn_P
Nfem_V=nn_V*ndof_V
Nfem=Nfem_V+Nfem_P

print('m_V=',m_V)
print('m_P=',m_P)
print('m_T=',m_T)
print('nel_T=',nel_T)
print('nn_T=',nn_T)
print('nel_S=',nel_S)
print('nn_P=',nn_P)
print('nn_V=',nn_V)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('Nfem_T=',Nfem_T)
print('Nfem=',Nfem)
print('case=',caase)

if debug:
   np.savetxt('mesh_T.ascii',np.array([x_T,y_T]).T,header='# x,y')
   np.savetxt('mesh_P.ascii',np.array([x_P,y_P]).T,header='# x,y')
   np.savetxt('mesh_V.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("compute nb of nodes and elts: %.3f s" % (clock.time()-start))

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

###############################################################################
# quadrature rules
###############################################################################

nq_per_dim=2

if nq_per_dim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

###############################################################################
# mapping the Stokes domain (only Q1 nodes) onto T domain
# very inefficient algorithm!
###############################################################################
start=clock.time()

in_stokes_domain=np.zeros(nn_T,dtype=bool) 
for i in range(0,nn_T):
    if y_T[i]>= Ly-x_T[i]-1 and y_T[i]<=Ly-50e3 +1:
       in_stokes_domain[i]=True

mapping=np.zeros(nn_P,dtype=np.int32)
for i in range(0,nn_P):
    for j in range(0,nn_T):
        if in_stokes_domain[j] and abs(x_V[i]-x_T[j])<eps and abs(y_V[i]-y_T[j])<eps:
           mapping[i]=j
           break

print("establish mapping S->T: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
# This is only valid for the Stokes mesh.
###############################################################################
start = clock.time()

area=np.zeros(nel_S,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel_S):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
    if area[iel]<0: 
       print('PB with iel=',iel) 
       for k in range(0,m_V):
           print (xV[icon_V[k,iel]],yV[icon_V[k,iel]])
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %((610e3+60e3)/2*550e3))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# this is a steady state code, but I leave the loop so that 
# one could re-introduce time stepping or iterations between
# Stokes and temperature equations.
###############################################################################

T=np.zeros(nn_T,dtype=np.float64) 

for iter in range(0,1):

    uu=np.zeros(nn_T,dtype=np.float64) # velocity x field on T mesh
    vv=np.zeros(nn_T,dtype=np.float64) # velocity y field on T mesh
    pp=np.zeros(nn_T,dtype=np.float64) # pressure field on T mesh

    #################################################################
    # assign i(case 1a) or compute (case 1b,1c) velocity to nodes
    # note that I have completely taken the buyoancy forces out
    #################################################################

    if not caase=='1a':

       start=clock.time()

       bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
       bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

       for i in range(0,nn_V):
           if abs(y_V[i]-550e3)/Ly<eps: #top
              bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V]   = 0.
              bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
           if abs(y_V[i]-Ly+x_V[i])/Lx<eps: #left
              bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V]   = +vel/np.sqrt(2)
              bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = -vel/np.sqrt(2)
           if caase=='1b' and  (abs(y_V[i])/Ly<eps or abs(x_V[i]-Lx)/Lx<eps): 
              uui,vvi=velocity.compute_corner_flow_velocity(x_V[i],y_V[i],l1,l2,l3,angle,vel,Lx,Ly)
              bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V]   = uui
              bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vvi
       #end for

       print("stokes b.c.: %.3fs" % (clock.time()-start))

       ##############################################################
       start=clock.time()

       A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
       b_fem=np.zeros(Nfem,dtype=np.float64)
       C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
       B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
       N_mat=np.zeros((3,m_P),dtype=np.float64)

       for iel in range(0,nel_S):

           K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
           G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
           f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
           h_el=np.zeros((m_P),dtype=np.float64)

           for iq in range(0,nq_per_dim):
               for jq in range(0,nq_per_dim):
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
                   xq=np.dot(N_V,x_V[icon_V[:,iel]])
                   yq=np.dot(N_V,y_V[icon_V[:,iel]])
                   dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                   dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                   for i in range(0,m_V):
                       B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                         [0.       ,dNdy_V[i]],
                                         [dNdy_V[i],dNdx_V[i]]]

                   K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq

                   N_mat[0,:]=N_P[:]
                   N_mat[1,:]=N_P[:]
                   G_el-=B.T.dot(N_mat)*JxWq

               # end for jq
           # end for iq

           G_el*=eta_ref/Ly

           # impose b.c. 
           for k1 in range(0,m_V):
               for i1 in range(0,ndof_V):
                   ikk=ndof_V*k1          +i1
                   m1 =ndof_V*icon_V[k1,iel]+i1
                   if bc_fix_V[m1]:
                      K_ref=K_el[ikk,ikk] 
                      for jkk in range(0,m_V*ndof_V):
                          f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                          K_el[ikk,jkk]=0
                          K_el[jkk,ikk]=0
                      K_el[ikk,ikk]=K_ref
                      f_el[ikk]=K_ref*bc_val_V[m1]
                      h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
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
                           A_fem[m1,m2] += K_el[ikk,jkk]
                       #end for
                   #end for
                   for k2 in range(0,m_P):
                       jkk=k2
                       m2 =icon_P[k2,iel]
                       A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                       A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                   #end for
                   b_fem[m1]+=f_el[ikk]
               #end for
           #end for
           for k2 in range(0,m_P):
               m2=icon_P[k2,iel]
               b_fem[Nfem_V+m2]+=h_el[k2]
           #end for

       #end for iel

       print("build FE matrix: %.3fs" % (clock.time()-start))

       ############################################################## 
       # solving linear system
       ##############################################################
       start=clock.time()

       sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

       u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
       p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

       print("     -> u (m,M) %.4e %.4e " %(np.min(u/cm*year),np.max(u/cm*year)))
       print("     -> v (m,M) %.4e %.4e " %(np.min(v/cm*year),np.max(v/cm*year)))
       print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

       if debug:
          np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
          np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

       print("solve time: %.3f s" % (clock.time()-start))

       ##############################################################
       # export Stokes solution to vtu format
       ##############################################################
       start=clock.time()

       vtufile=open('solutionS_lvl'+str(level)+'.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_P,nel_S))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%e %e %e \n" %(x_P[i],y_P[i],0))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%e %e %e \n" % (u[i]/cm*year,v[i]/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cornerflow)' Format='ascii'> \n")
       for i in range(0,nn_P):
           uui,vvi=velocity.compute_corner_flow_velocity(x_P[i],y_P[i],l1,l2,l3,angle,vel,Lx,Ly)
           vtufile.write("%e %e %e \n" % (uui/cm*year,vvi/cm*year,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32'   Name='pressure' Format='ascii'> \n")
       for i in range(0,nn_P):
           vtufile.write("%e \n" % (p[i]))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='bc' Format='ascii'> \n")
       for i in range(0,nn_P):
           if bc_fix_V[2*i+0]: 
              uii=bc_val_V[2*i]/cm*year
           else:
              uii=0
           if bc_fix_V[2*i+1]: 
              vii=bc_val_V[2*i+1]/cm*year
           else:
              vii=0
           vtufile.write("%e %e %e \n" % (uii,vii,0))
       vtufile.write("</DataArray>\n")
       #--  
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel_S):
           vtufile.write("%d %d %d %d \n" %(icon_P[0,iel],icon_P[1,iel],icon_P[2,iel],icon_P[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel_S):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel_S):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu: %.3f s" % (clock.time() - start))

       ##############################################################
       # project velocity on T mesh
       ##############################################################
       start=clock.time()

       for i in range(0,nn_P):
           uu[mapping[i]]=u[i]
           vv[mapping[i]]=v[i]
           pp[mapping[i]]=p[i]

       for i in range(0,nn_T):
           if y_T[i]<Ly-x_T[i]:
              uu[i]=+vel/np.sqrt(2)
              vv[i]=-vel/np.sqrt(2)

       print("project vel on t mesh: %.3f s" % (clock.time() - start))

    else: # not solving Stokes system

       for i in range(0,nn_T):
           uu[i],vv[i]=velocity.compute_corner_flow_velocity(x_T[i],y_T[i],l1,l2,l3,angle,vel,Lx,Ly)

    #################################################################
    # temperature boundary conditions which depend on velocity 
    #################################################################
    start=clock.time()

    bc_fix_T=np.zeros(Nfem_T,dtype=bool)  # boundary condition, yes/no
    bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

    kappa=hcond/rho/hcapa

    age=50e6

    for i in range(0,nn_T):
        # top boundary 
        if abs(y_T[i]-Ly)/Ly<eps: #
           bc_fix_T[i]=True ; bc_val_T[i]=Kelvin
        # left boundary 
        if abs(x_T[i]/Lx)<eps:
           bc_fix_T[i]=True ; bc_val_T[i]=Kelvin+(1573-Kelvin)*erf(((Ly-y_T[i]))/(2*np.sqrt(kappa*age*year)))
        # right boundary 
        if abs(x_T[i]-Lx)/Lx<eps:
           if y_T[i]>=Ly-l2:
              bc_fix_T[i]=True ; bc_val_T[i]=((Ly-y_T[i]))/l2*1300+Kelvin
           elif uu[i]<0:
              bc_fix_T[i]=True ; bc_val_T[i]=1300.+Kelvin

    print("temp b.c.: %.3f s" % (clock.time() - start))

    #################################################################
    # build FE matrix for temperature 
    #################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix
    b_fem=np.zeros(Nfem_T,dtype=np.float64)            # FE rhs 
    B=np.zeros((ndim,m_T),dtype=np.float64)   
    N_mat=np.zeros((m_T,1),dtype=np.float64)  

    for iel in range (0,nel_T):

        b_el=np.zeros(m_T,dtype=np.float64)        # elemental rhs
        A_el=np.zeros((m_T,m_T),dtype=np.float64)  # elemental matrix
        Ka=np.zeros((m_T,m_T),dtype=np.float64)    # elemental advection matrix 
        Kd=np.zeros((m_T,m_T),dtype=np.float64)    # elemental diffusion matrix 
        velq=np.zeros((1,ndim),dtype=np.float64)   # velocity at quad point

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_mat[:,0]=basis_functions_T(rq,sq)
                dNdr_T=basis_functions_T_dr(rq,sq)
                dNds_T=basis_functions_T_ds(rq,sq)
                jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
                jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
                jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
                jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                velq[0,0]=np.dot(N_mat[:,0],uu[icon_T[:,iel]])
                velq[0,1]=np.dot(N_mat[:,0],vv[icon_T[:,iel]])
                dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
                dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
                B[0,:]=dNdx_T[:]
                B[1,:]=dNdy_T[:]
                Kd+=B.T.dot(B)*hcond*JxWq
                Ka+=N_mat.dot(velq.dot(B))*rho*hcapa*JxWq
	    # end for jq
        # end for iq

        A_el=Ka+Kd

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               # end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            # end if
        # end for

        # assemble matrix and right hand side 
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            # end for
            b_fem[m1]+=b_el[k1]
        # end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (clock.time() - start))

    #################################################################
    # solve system 
    #################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    if debug: np.savetxt('T.ascii',np.array([x_T,y_T,T]).T,header='# x,y')

    print("     -> T (m,M) %.4f %.4f " %(np.min(T-Kelvin),np.max(T-Kelvin)))

    print("solve T: %.3f s" % (clock.time()-start))

    #################################################################
    # compute heat flux
    #################################################################
    start=clock.time()

    qx=np.zeros(nn_T,dtype=np.float64) 
    qy=np.zeros(nn_T,dtype=np.float64) 
    cc=np.zeros(nn_T,dtype=np.float64) 

    r_T=[-1,1,1,-1]
    s_T=[-1,-1,1,1]

    for iel in range(0,nel_T):
        for kk in range(0,m_T):
            inode=icon_T[kk,iel]
            dNdr_T=basis_functions_T_dr(r_T[kk],s_T[kk])
            dNds_T=basis_functions_T_ds(r_T[kk],s_T[kk])
            jcb[0,0]=np.dot(dNdr_T,x_T[icon_T[:,iel]])
            jcb[0,1]=np.dot(dNdr_T,y_T[icon_T[:,iel]])
            jcb[1,0]=np.dot(dNds_T,x_T[icon_T[:,iel]])
            jcb[1,1]=np.dot(dNds_T,y_T[icon_T[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T
            dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T
            qx[inode]-=hcond*np.dot(dNdx_T,T[icon_T[:,iel]])
            qy[inode]-=hcond*np.dot(dNdy_T,T[icon_T[:,iel]])
            cc[inode]+=1
        #end for
    #end for
    qx/=cc
    qy/=cc

    print("     -> qx (m,M) %.4f %.4f " %(np.min(qx),np.max(qx)))
    print("     -> qy (m,M) %.4f %.4f " %(np.min(qy),np.max(qy)))

    print("compute heat flux: %.3f s" % (clock.time()-start))

    #################################################################
    # export Temperature along given lines 
    #################################################################
    start=clock.time()

    diagT=np.zeros(nn_T,dtype=np.float64)  # size way too large
    diagP=np.zeros(nn_T,dtype=np.float64)  # size way too large
    dist=np.zeros(nn_T,dtype=np.float64)   # size way too large
    rightu=np.zeros(nn_T,dtype=np.float64) # size way too large
    rightv=np.zeros(nn_T,dtype=np.float64) # size way too large
    rightT=np.zeros(nn_T,dtype=np.float64) # size way too large
    rightP=np.zeros(nn_T,dtype=np.float64) # size way too large
    depth=np.zeros(nn_T,dtype=np.float64)  # size way too large
    topqx=np.zeros(nn_T,dtype=np.float64)  # size way too large
    topqy=np.zeros(nn_T,dtype=np.float64)  # size way too large
    topxx=np.zeros(nn_T,dtype=np.float64)  # size way too large

    counter=0
    for i in range(0,nn_T):
        if abs(y_T[i]-Ly+x_T[i])/Lx<eps:
           diagT[counter]=T[i]
           diagP[counter]=pp[i]
           dist[counter]=np.sqrt( (x_T[i]-0)**2+(y_T[i]-Ly)**2 )
           counter+=1

    np.savetxt('diag_lvl'+str(level)+'.ascii',np.array([dist[0:counter],diagT[0:counter],diagP[0:counter]]).T)

    counter=0
    for i in range(0,nn_T):
        if abs(x_T[i]-Lx)/Lx<eps:
           rightT[counter]=T[i]
           rightP[counter]=pp[i]
           rightu[counter]=uu[i]
           rightv[counter]=vv[i]
           depth[counter]=Ly-y_T[i]
           counter+=1

    np.savetxt('right_lvl'+str(level)+'.ascii',np.array([depth[0:counter],\
                                               rightT[0:counter],rightP[0:counter],\
                                               rightu[0:counter],rightv[0:counter]]).T)

    counter=0
    for i in range(0,nn_T):
        if abs(y_T[i]-Ly)/Ly<eps:
           topqx[counter]=qx[i]
           topqy[counter]=qy[i]
           topxx[counter]=x_T[i]
           counter+=1

    np.savetxt('top_lvl'+str(level)+'.ascii',np.array([topxx[0:counter],\
                                                       topqx[0:counter],\
                                                       topqy[0:counter]]).T)

    print('     -> produced diag_lvl'+str(level)+'.ascii')
    print('     -> produced right_lvl'+str(level)+'.ascii')
    print('     -> produced top_lvl'+str(level)+'.ascii')

    print("carry out measurements: %.3f s" % (clock.time()-start))

    #################################################################
    # export solution to vtu file
    #################################################################
    start=clock.time()

    vtufile=open('solutionT_lvl'+str(level)+'.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel_T))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_T):
        vtufile.write("%e %e %e \n" %(x_T[i],y_T[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--  
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='velocity' Format='ascii'> \n")
    for i in range(0,nn_T):
        vtufile.write("%e %e %e \n" % (uu[i]/cm*year,vv[i]/cm*year,0))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'  Name='heat flux' Format='ascii'> \n")
    for i in range(0,nn_T):
        vtufile.write("%e %e %e \n" % (qx[i],qy[i],0))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,nn_T):
        vtufile.write("%e \n" % (pp[i]))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    for i in range(0,nn_T):
        vtufile.write("%e \n" % (T[i]-Kelvin))
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("<DataArray type='Float32' Name='stokes domain' Format='ascii'> \n")
    for i in range(0,nn_T):
        if in_stokes_domain[i]:  
           vtufile.write("%e \n" % 1)
        else:
           vtufile.write("%e \n" % 0)
    vtufile.write("</DataArray>\n")
    #--  
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel_T):
        vtufile.write("%d %d %d %d \n" %(icon_T[0,iel],icon_T[1,iel],icon_T[2,iel],icon_T[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel_T):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel_T):
        vtufile.write("%d \n" %9)
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
