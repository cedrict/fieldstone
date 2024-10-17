import numpy as np
import time 
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import math
import sys as sys

###############################################################################

def NNV(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1+r)*(1+s)
    N_3=0.25*(1-r)*(1+s)
    return N_0,N_1,N_2,N_3

def sigma_xx_th(x,y,r,theta,rad,sigma_bc):
    val=1-rad**2/r**2*(1.5*np.cos(2*theta)+np.cos(4*theta))+1.5*rad**4/r**4*np.cos(4*theta)
    val*=sigma_bc
    return val

def sigma_yy_th(x,y,r,theta,rad,sigma_bc):
    val=rad**2/r**2*(0.5*np.cos(2*theta)-np.cos(4*theta))-1.5*rad**4/r**4*np.cos(4*theta)
    val*=-sigma_bc
    return val

def sigma_xy_th(x,y,r,theta,rad,sigma_bc):
    val=rad**2/r**2*(0.5*np.sin(2*theta)+np.sin(4*theta))+1.5*rad**4/r**4*np.sin(4*theta)
    val*=-sigma_bc
    return val

###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

experiment=3

if experiment==1: #hassani
   E=1e10
   nu=0.25
   v_bc=1e-3
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   rad=0.1
   Lx=0.5
   Ly=0.5

if experiment==2: #Ramachandran
   nu=0.3
   E=3e7 
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   rad=1
   Lx=5
   Ly=5
   sigma_bc=10

if experiment==3: #Ramm et al 2003
   nu=0.29
   E=2.069e14/1e6
   mu=E/2/(1+nu)
   lambdaa=E*nu/(1+nu)/(1-2*nu)
   rad=0.01
   Lx=0.1
   Ly=0.1
   sigma_bc=1e8*4.5
   
if int(len(sys.argv) == 2):
   nelx   = int(sys.argv[1])
else:
   nelx=40

nely=int(nelx/4)
#nely=nelx

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)

eps=1e-8
distance=eps*Lx

hx=Lx/nelx
hy=Ly/nely

print('     -> experiment=',experiment)
print('     -> nelx=',nelx)
print('     -> nely=',nely)

###############################################################################
#  The mesh is composed of eight blocks. Each is first built and warped
# to fit the contour of the hole and they are subsequently merged.
#
#  G-----E-----C
#  |   3 | 2   |
#  |     F     |
#  | 4  H D  1 |
#  I---J   A---B
#  | 5  L P  8 |
#  |     N     |
#  |   6 | 7   |
#  K-----M-----O
#
###############################################################################
start = time.time()

b_nelx=nelx
b_nely=nely
b_nnx=nelx+1
b_nny=nely+1
b_NV=b_nnx*b_nny
b_nel=b_nelx*b_nely

b1_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b1_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b2_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b2_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b3_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b3_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b4_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b4_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b5_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b5_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b6_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b6_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b7_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b7_y=np.empty(b_NV,dtype=np.float64)  # y coordinates
b8_x=np.empty(b_NV,dtype=np.float64)  # x coordinates
b8_y=np.empty(b_NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        b1_x[counter]=i*Lx/float(b_nelx)
        b1_y[counter]=j*Ly/float(b_nely)
        counter += 1

b2_x[:]=b1_x[:]
b2_y[:]=b1_y[:]

b1_icon =np.zeros((m,b_nel),dtype=np.int32)
b2_icon =np.zeros((m,b_nel),dtype=np.int32)
b3_icon =np.zeros((m,b_nel),dtype=np.int32)
b4_icon =np.zeros((m,b_nel),dtype=np.int32)
b5_icon =np.zeros((m,b_nel),dtype=np.int32)
b6_icon =np.zeros((m,b_nel),dtype=np.int32)
b7_icon =np.zeros((m,b_nel),dtype=np.int32)
b8_icon =np.zeros((m,b_nel),dtype=np.int32)

counter = 0
for j in range(0,b_nely):
    for i in range(0,b_nelx):
        b1_icon[0, counter] = i + j * (b_nelx + 1)
        b1_icon[1, counter] = i + 1 + j * (b_nelx + 1)
        b1_icon[2, counter] = i + 1 + (j + 1) * (b_nelx + 1)
        b1_icon[3, counter] = i + (j + 1) * (b_nelx + 1)
        counter += 1

b2_icon[:,:]=b1_icon[:,:]
b3_icon[:,:]=b1_icon[:,:]
b4_icon[:,:]=b1_icon[:,:]
b5_icon[:,:]=b1_icon[:,:]
b6_icon[:,:]=b1_icon[:,:]
b7_icon[:,:]=b1_icon[:,:]
b8_icon[:,:]=b1_icon[:,:]

print("prepare arrays: %.3f s" % (time.time() - start))

###############################################################################
# map block 1
###############################################################################
start = time.time()

NNNV=np.zeros(m,dtype=np.float64)       

xA=rad
yA=0

xB=Lx
yB=0

xC=Lx
yC=Ly

xD=rad/sqrt2
yD=rad/sqrt2

# map to quadrilateral ABCD
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        r=2*(b1_x[counter]/Lx-0.5)
        s=2*(b1_y[counter]/Ly-0.5)
        NNNV[0:m]=NNV(r,s)
        b1_x[counter]=NNNV[0]*xA+NNNV[1]*xB+NNNV[2]*xC+NNNV[3]*xD
        b1_y[counter]=NNNV[0]*yA+NNNV[1]*yB+NNNV[2]*yC+NNNV[3]*yD
        counter+=1
   #end for
#end for

# bend AD side
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        if i==0:
           angle=np.arctan(b1_y[counter]/b1_x[counter])
           b1_x[counter]=rad*np.cos(angle)
           b1_y[counter]=rad*np.sin(angle)
        counter+=1
   #end for
#end for

# recompute position of inside nodes
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        if i!=0 and i!=b_nnx-1 and j!=0 and j!=b_nny-1:
           inode_beg=j*b_nnx
           inode_end=j*b_nnx+b_nnx-1
           b1_x[counter]=float(i)/b_nelx*(b1_x[inode_end]-b1_x[inode_beg]) + b1_x[inode_beg]
           b1_y[counter]=float(i)/b_nelx*(b1_y[inode_end]-b1_y[inode_beg]) + b1_y[inode_beg]
        counter+=1
   #end for
#end for

print("make block 1: %.3f s" % (time.time() - start))

###############################################################################
# map block 2
###############################################################################
start = time.time()

xC=Lx
yC=Ly

xD=rad/sqrt2
yD=rad/sqrt2

xF=0
yF=rad

xE=0
yE=Lx

# map to quadrilateral FDCE
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        r=2*(b2_x[counter]/Lx-0.5)
        s=2*(b2_y[counter]/Ly-0.5)
        NNNV[0:m]=NNV(r,s)
        b2_x[counter]=NNNV[0]*xD+NNNV[1]*xC+NNNV[2]*xE+NNNV[3]*xF
        b2_y[counter]=NNNV[0]*yD+NNNV[1]*yC+NNNV[2]*yE+NNNV[3]*yF
        counter+=1
   #end for
#end for

#bend FD side
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        if i==0 and j<b_nny-1:
           angle=np.arctan(b2_y[counter]/b2_x[counter])
           b2_x[counter]=rad*np.cos(angle)
           b2_y[counter]=rad*np.sin(angle)
        counter+=1
   #end for
#end for

# recompute position of inside nodes
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        if i!=0 and i!=b_nnx-1 and j!=0 and j!=b_nny-1:
           inode_beg=j*b_nnx
           inode_end=j*b_nnx+b_nnx-1
           b2_x[counter]=float(i)/b_nelx*(b2_x[inode_end]-b2_x[inode_beg]) + b2_x[inode_beg]
           b2_y[counter]=float(i)/b_nelx*(b2_y[inode_end]-b2_y[inode_beg]) + b2_y[inode_beg]
        counter+=1
   #end for
#end for

print("make block 2: %.3f s" % (time.time() - start))

###############################################################################
# make block 3 - it is a rotated block 1
# make block 4 - it is a rotated block 2
# make block 5 - it is a rotated block 3
# make block 6 - it is a rotated block 4
# make block 7 - it is a rotated block 5
# make block 8 - it is a rotated block 6
###############################################################################
start = time.time()

b3_x[:]=-b1_y[:]
b3_y[:]= b1_x[:]

b4_x[:]=-b2_y[:]
b4_y[:]= b2_x[:]

b5_x[:]=-b3_y[:]
b5_y[:]= b3_x[:]

b6_x[:]=-b4_y[:]
b6_y[:]= b4_x[:]

b7_x[:]=-b5_y[:]
b7_y[:]= b5_x[:]

b8_x[:]=-b6_y[:]
b8_y[:]= b6_x[:]

#np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
#np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
#np.savetxt('temp3.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
#np.savetxt('temp4.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
#np.savetxt('temp5.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
#np.savetxt('temp6.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
#np.savetxt('temp7.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
#np.savetxt('temp8.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

print("make block 3-8: %.3f s" % (time.time() - start))

###############################################################################
# merge blocks
###############################################################################
start = time.time()

nblock=8

tempx=np.empty(nblock*b_NV,dtype=np.float64)  # x coordinates
tempy=np.empty(nblock*b_NV,dtype=np.float64)  # y coordinates

tempx[0*b_NV:1*b_NV]=b1_x[:]
tempx[1*b_NV:2*b_NV]=b2_x[:]
tempx[2*b_NV:3*b_NV]=b3_x[:]
tempx[3*b_NV:4*b_NV]=b4_x[:]
tempx[4*b_NV:5*b_NV]=b5_x[:]
tempx[5*b_NV:6*b_NV]=b6_x[:]
tempx[6*b_NV:7*b_NV]=b7_x[:]
tempx[7*b_NV:8*b_NV]=b8_x[:]

tempy[0*b_NV:1*b_NV]=b1_y[:]
tempy[1*b_NV:2*b_NV]=b2_y[:]
tempy[2*b_NV:3*b_NV]=b3_y[:]
tempy[3*b_NV:4*b_NV]=b4_y[:]
tempy[4*b_NV:5*b_NV]=b5_y[:]
tempy[5*b_NV:6*b_NV]=b6_y[:]
tempy[6*b_NV:7*b_NV]=b7_y[:]
tempy[7*b_NV:8*b_NV]=b8_y[:]

doubble=np.zeros(nblock*b_NV,dtype=bool)  # boundary condition, yes/no
pointto=np.zeros(nblock*b_NV,dtype=np.int32)

for i in range(0,nblock*b_NV):
    pointto[i]=i

counter=0
for i in range(1,nblock*b_NV):
   gxi=tempx[i]
   gyi=tempy[i]
   for j in range(0,i):
      if abs(gxi-tempx[j])<distance and abs(gyi-tempy[j])<distance:
          doubble[i]=True
          pointto[i]=j
          break
      #end if
   #end do
#end do


NV=nblock*b_NV-sum(doubble)
nel=nblock*b_nel
Nfem=NV*ndof

print('     -> doubles=',sum(doubble))
print('     -> NV=',NV)
print('     -> nel=',nel)
print('     -> Nfem=',Nfem)

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates
icon =np.zeros((m,nel),dtype=np.int32)

counter=0
for i in range(0,nblock*b_NV):
    if not doubble[i]: 
       x[counter]=tempx[i]+Lx
       y[counter]=tempy[i]+Ly
       counter+=1

icon[0:m,0*b_nel:1*b_nel]=b1_icon[0:m,0:b_nel]+0*b_NV
icon[0:m,1*b_nel:2*b_nel]=b2_icon[0:m,0:b_nel]+1*b_NV
icon[0:m,2*b_nel:3*b_nel]=b2_icon[0:m,0:b_nel]+2*b_NV
icon[0:m,3*b_nel:4*b_nel]=b2_icon[0:m,0:b_nel]+3*b_NV
icon[0:m,4*b_nel:5*b_nel]=b2_icon[0:m,0:b_nel]+4*b_NV
icon[0:m,5*b_nel:6*b_nel]=b2_icon[0:m,0:b_nel]+5*b_NV
icon[0:m,6*b_nel:7*b_nel]=b2_icon[0:m,0:b_nel]+6*b_NV
icon[0:m,7*b_nel:8*b_nel]=b2_icon[0:m,0:b_nel]+7*b_NV

for iel in range(0,nel):
    for i in range(0,m):
        icon[i,iel]=pointto[icon[i,iel]]

compact=np.zeros(nblock*b_NV,dtype=np.int32)

counter=0
for i in range(0,nblock*b_NV):
    if not doubble[i]:
       compact[i]=counter
       counter=counter+1

for iel in range(0,nel):
   for i in range(0,m):
      icon[i,iel]=compact[icon[i,iel]]

#np.savetxt('mesh.ascii',np.array([x,y]).T,header='# x,y')

Lx*=2
Ly*=2

print("assemble blocks: %.3f s" % (time.time() - start))

###############################################################################
# compute Cartesian and polar coordinates of element centers
###############################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m):
        xc[iel]+=x[icon[k,iel]]*0.25
        yc[iel]+=y[icon[k,iel]]*0.25

rr = np.zeros(nel,dtype=np.float64)  
theta = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rr[iel]=np.sqrt((xc[iel]-Lx/2)**2+(yc[iel]-Ly/2)**2)
    theta[iel]=math.atan2((yc[iel]-Ly/2),(xc[iel]-Lx/2))

print("compute coords.: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0, NV):
       if abs(y[i])<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +v_bc
          #remove null space
          if abs(x[i]-Lx/2)/Lx<eps:
             bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0
             
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -v_bc
       #if np.sqrt((x[i]-0.5*Lx)**2+(y[i]-0.5*Ly)**2)<rad*(1+eps):
       #   bc_fix[i*ndof]   = True ; bc_val[i*ndof+0] = 0
       #   bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

if experiment==2:
   for i in range(0, NV):
       if np.sqrt((x[i]-0.5*Lx)**2+(y[i]-0.5*Ly)**2)<rad*(1+eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof+0] = 0
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

if experiment==3:
   for i in range(0, NV):

       if abs(y[i]-(Ly/2-rad))/Ly<eps and abs(x[i]-Lx/2)/Lx<eps: #bottom hole
             bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0

       if abs(y[i]-(Ly/2+rad))/Ly<eps and abs(x[i]-Lx/2)/Lx<eps: #top hole
             bc_fix[i*ndof] = True ; bc_val[i*ndof] = 0

       if abs(y[i]-Ly/2)/Ly<eps and abs(x[i])/Lx<eps: #left
             bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

       if abs(y[i]-Ly/2)/Ly<eps and abs(x[i]-Lx)/Lx<eps: #right
             bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

       #if np.sqrt((x[i]-0.5*Lx)**2+(y[i]-0.5*Ly)**2)<rad*(1+eps):
       #   bc_fix[i*ndof]   = True ; bc_val[i*ndof+0] = 0
       #   bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0



print("define boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)

b_mat = np.zeros((3,ndof*m),dtype=np.float64)  # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)        # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)           # shape functions
dNdx  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)           # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component displacement 
v     = np.zeros(NV,dtype=np.float64)          # y-component displacement 
c_mat = np.array([[2*mu+lambdaa,lambdaa,0],[lambdaa,2*mu+lambdaa,0],[0,0,mu]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m*ndof)
    a_el = np.zeros((m*ndof,m*ndof), dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*wq*jcob

            # compute elemental rhs vector
            #for i in range(0, m):
            #    b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
            #    b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

        #end for
    #end for

    if experiment==2:
       if abs(x[icon[0,iel]])/Lx<eps and abs(x[icon[1,iel]])/Lx<eps: #left wall
          b_el[0]-=sigma_bc*hy*0.5
          b_el[2]-=sigma_bc*hy*0.5
       if abs(x[icon[1,iel]])/Lx<eps and abs(x[icon[2,iel]])/Lx<eps: #left wall
          b_el[2]-=sigma_bc*hy*0.5
          b_el[4]-=sigma_bc*hy*0.5
       if abs(x[icon[2,iel]])/Lx<eps and abs(x[icon[3,iel]])/Lx<eps: #left wall
          b_el[4]-=sigma_bc*hy*0.5
          b_el[6]-=sigma_bc*hy*0.5
       if abs(x[icon[3,iel]])/Lx<eps and abs(x[icon[0,iel]])/Lx<eps: #left wall
          b_el[6]-=sigma_bc*hy*0.5
          b_el[0]-=sigma_bc*hy*0.5
       if abs(x[icon[0,iel]]-Lx)/Lx<eps and abs(x[icon[1,iel]]-Lx)/Lx<eps: #right wall
          b_el[0]+=sigma_bc*hy*0.5
          b_el[2]+=sigma_bc*hy*0.5
       if abs(x[icon[1,iel]]-Lx)/Lx<eps and abs(x[icon[2,iel]]-Lx)/Lx<eps: #right wall
          b_el[2]+=sigma_bc*hy*0.5
          b_el[4]+=sigma_bc*hy*0.5
       if abs(x[icon[2,iel]]-Lx)/Lx<eps and abs(x[icon[3,iel]]-Lx)/Lx<eps: #right wall
          b_el[4]+=sigma_bc*hy*0.5
          b_el[6]+=sigma_bc*hy*0.5
       if abs(x[icon[3,iel]]-Lx)/Lx<eps and abs(x[icon[0,iel]]-Lx)/Lx<eps: #right wall
          b_el[6]+=sigma_bc*hy*0.5
          b_el[0]+=sigma_bc*hy*0.5

    if experiment==3:
       if abs(y[icon[0,iel]])/Ly<eps and abs(y[icon[1,iel]])/Ly<eps: #bottom wall
          b_el[1]-=sigma_bc*hy*0.5
          b_el[3]-=sigma_bc*hy*0.5
       if abs(y[icon[1,iel]])/Ly<eps and abs(y[icon[2,iel]])/Ly<eps: #bottom wall
          b_el[3]-=sigma_bc*hy*0.5
          b_el[5]-=sigma_bc*hy*0.5
       if abs(y[icon[2,iel]])/Ly<eps and abs(y[icon[3,iel]])/Ly<eps: #bottom wall
          b_el[5]-=sigma_bc*hy*0.5
          b_el[7]-=sigma_bc*hy*0.5
       if abs(y[icon[3,iel]])/Ly<eps and abs(y[icon[0,iel]])/Ly<eps: #bottom wall
          b_el[7]-=sigma_bc*hy*0.5
          b_el[1]-=sigma_bc*hy*0.5
       if abs(y[icon[0,iel]]-Ly)/Ly<eps and abs(y[icon[1,iel]]-Ly)/Ly<eps: #top wall
          b_el[1]+=sigma_bc*hy*0.5
          b_el[3]+=sigma_bc*hy*0.5
       if abs(y[icon[1,iel]]-Ly)/Ly<eps and abs(y[icon[2,iel]]-Ly)/Ly<eps: #top wall
          b_el[3]+=sigma_bc*hy*0.5
          b_el[5]+=sigma_bc*hy*0.5
       if abs(y[icon[2,iel]]-Ly)/Ly<eps and abs(y[icon[3,iel]]-Ly)/Ly<eps: #top wall
          b_el[5]+=sigma_bc*hy*0.5
          b_el[7]+=sigma_bc*hy*0.5
       if abs(y[icon[3,iel]]-Ly)/Ly<eps and abs(y[icon[0,iel]]-Ly)/Ly<eps: #top wall
          b_el[7]+=sigma_bc*hy*0.5
          b_el[1]+=sigma_bc*hy*0.5





    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndof):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               #end for
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            ikk=ndof*k1          +i1
            m1 =ndof*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndof):
                    jkk=ndof*k2          +i2
                    m2 =ndof*icon[k2,iel]+i2
                    a_mat[m1,m2]+=a_el[ikk,jkk]
                #end for
            #end for
            rhs[m1]+=b_el[ikk]
        #end for
    #end for

#end for

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

#np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure, stress, strain, ...
#####################################################################
start = time.time()

q  = np.zeros(NV,dtype=np.float64)  
cc = np.zeros(NV,dtype=np.float64)  
e_n = np.zeros(NV,dtype=np.float64)  
tau_n = np.zeros(NV,dtype=np.float64)  
e_xx_n = np.zeros(NV,dtype=np.float64)  
e_yy_n = np.zeros(NV,dtype=np.float64)  
e_xy_n = np.zeros(NV,dtype=np.float64)  
tau_xx_n = np.zeros(NV,dtype=np.float64)  
tau_yy_n = np.zeros(NV,dtype=np.float64)  
tau_zz_n = np.zeros(NV,dtype=np.float64)  
tau_xy_n = np.zeros(NV,dtype=np.float64)  
sigma_n = np.zeros(NV,dtype=np.float64)  
sigma_xx_n = np.zeros(NV,dtype=np.float64)  
sigma_yy_n = np.zeros(NV,dtype=np.float64)  
sigma_zz_n = np.zeros(NV,dtype=np.float64)  
sigma_xy_n = np.zeros(NV,dtype=np.float64)  

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

for iel in range(0,nel):
    for i in range(0,m):
        rq = rVnodes[i]
        sq = sVnodes[i]

        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,m):
            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNds[k]*y[icon[k,iel]]

        jcbi=np.linalg.inv(jcb)

        for k in range(0,m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

        exx=0
        eyy=0
        exy=0
        for k in range(0,m):
            exx += dNdx[k]*u[icon[k,iel]]
            eyy += dNdy[k]*v[icon[k,iel]]
            exy += 0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]

        e_xx_n[icon[i,iel]]+=exx
        e_yy_n[icon[i,iel]]+=eyy
        e_xy_n[icon[i,iel]]+=exy
        cc[icon[i,iel]]+=1.

    #end for
#end for

e_xx_n[:]/=cc[:]
e_yy_n[:]/=cc[:]
e_xy_n[:]/=cc[:]

q[:]=-(lambdaa+2./3.*mu)*(e_xx_n[:]+e_yy_n[:])

e_n[:]=np.sqrt(0.5*(e_xx_n[:]**2+e_yy_n[:]**2)+e_xy_n[:]**2)

sigma_xx_n[:]=lambdaa*(e_xx_n[:]+e_yy_n[:])+2*mu*e_xx_n[:]
sigma_yy_n[:]=lambdaa*(e_xx_n[:]+e_yy_n[:])+2*mu*e_yy_n[:]
sigma_xy_n[:]=2*mu*e_xy_n[:]
sigma_zz_n[:]=nu*(sigma_xx_n[:]+sigma_yy_n[:])

sigma_n[:]=np.sqrt(0.5*(sigma_xx_n[:]**2+sigma_yy_n[:]**2+sigma_zz_n[:]**2)+sigma_xy_n[:]**2)

tau_xx_n[:]=sigma_xx_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_yy_n[:]=sigma_yy_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_zz_n[:]=sigma_zz_n[:]-(1+nu)/3*(sigma_xx_n[:]+sigma_yy_n[:])
tau_xy_n[:]=sigma_xy_n[:]

tau_n[:]=np.sqrt(0.5*(tau_xx_n[:]**2+tau_yy_n[:]**2+tau_zz_n[:]**2)+tau_xy_n[:]**2)

print("     -> exx_n (m,M) %e %e " %(np.min(e_xx_n),np.max(e_xx_n)))
print("     -> eyy_n (m,M) %e %e " %(np.min(e_yy_n),np.max(e_yy_n)))
print("     -> exy_n (m,M) %e %e " %(np.min(e_xy_n),np.max(e_xy_n)))
print("     -> q (m,M) %e %e " %(np.min(q),np.max(q)))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,e_xx,e_yy,e_xy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & strain: %.3f s" % (time.time() - start))

###############################################################################

if experiment==1:
   for i in range(0,NV):
       if abs(y[i])/Ly<eps and abs(x[i])/Lx<eps: #lower left corner
          print('corner: ',u[i],v[i],sigma_xx_n[i],sigma_yy_n[i],sigma_xy_n[i],q[i],Nfem)
       if abs(y[i]-Ly/2+rad)/Ly<eps and abs(x[i]-Lx/2)/Lx<eps: 
          print('south: ',u[i],v[i],sigma_xx_n[i],sigma_yy_n[i],sigma_xy_n[i],q[i],Nfem)
       if abs(y[i]-Ly/2)/Ly<eps and abs(x[i]-Lx/2+rad)/Lx<eps: 
          print('west: ',u[i],v[i],sigma_xx_n[i],sigma_yy_n[i],sigma_xy_n[i],q[i],Nfem)

if experiment==3:
   for i in range(0,NV):
       if abs(y[i]-Ly/2)/Ly<eps and abs(x[i]-(Lx/2-rad))/Lx<eps: #point2
          print('point_2: ux: ',u[i],'m','| sigma_yy:',sigma_yy_n[i]/1e6,'N/mm^2', Nfem)
       if abs(y[i]-Ly)/Ly<eps and abs(x[i]-Lx/2)/Lx<eps: #point4
          print('point_4: uy: ',v[i],'m', Nfem)
       if abs(y[i]-Ly)/Ly<eps and abs(x[i])/Lx<eps: #point5
          print('point_5: ux: ',u[i],'m', Nfem)

###############################################################################
# plot of solution
###############################################################################
       
if True: 
       vtufile=open('solution.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %e_xx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %e_yy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %e_xy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %sigma_xx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %sigma_yy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %sigma_xy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %sigma_zz_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %sigma_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e  \n" %e_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % rr[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % theta[iel])
       vtufile.write("</DataArray>\n")
       #--
       if experiment==2:
          vtufile.write("<DataArray type='Float32' Name='sigma_xx (th)' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e\n" % sigma_xx_th(xc[iel],yc[iel],rr[iel],theta[iel],rad,sigma_bc))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigma_yy (th)' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e\n" % sigma_yy_th(xc[iel],yc[iel],rr[iel],theta[iel],rad,sigma_bc))
          vtufile.write("</DataArray>\n")
          #--
          vtufile.write("<DataArray type='Float32' Name='sigma_xy (th)' Format='ascii'> \n")
          for iel in range (0,nel):
              vtufile.write("%e\n" % sigma_xy_th(xc[iel],yc[iel],rr[iel],theta[iel],rad,sigma_bc))
          vtufile.write("</DataArray>\n")
          #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
