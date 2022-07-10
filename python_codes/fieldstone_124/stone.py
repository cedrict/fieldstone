import numpy as np
import time 
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

def NNV(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1+r)*(1+s)
    N_3=0.25*(1-r)*(1+s)
    return N_0,N_1,N_2,N_3

###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

experiment=2

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

nelx=20
nely=nelx

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)

eps=1e-8
distance=eps*Lx

hx=Lx/nelx
hy=Ly/nely

###############################################################################
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

#np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')

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
        b2_x[counter]=NNNV[0]*xF+NNNV[1]*xD+NNNV[2]*xC+NNNV[3]*xE
        b2_y[counter]=NNNV[0]*yF+NNNV[1]*yD+NNNV[2]*yC+NNNV[3]*yE
        counter+=1
   #end for
#end for

#bend FD side
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        if j==0:
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
           inode_beg=i
           inode_end=(b_nny-1)*b_nnx+i
           b2_x[counter]=float(j)/b_nely*(b2_x[inode_end]-b2_x[inode_beg]) + b2_x[inode_beg]
           b2_y[counter]=float(j)/b_nely*(b2_y[inode_end]-b2_y[inode_beg]) + b2_y[inode_beg]
        counter+=1
   #end for
#end for

#np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')

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

#np.savetxt('temps.ascii',np.array([tempx,tempy]).T,header='# x,y')

doubble=np.zeros(nblock*b_NV,dtype=np.bool)  # boundary condition, yes/no
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

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m):
        xc[iel]+=x[icon[k,iel]]*0.25
        yc[iel]+=y[icon[k,iel]]*0.25

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0, NV):
       if y[i]<eps:
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = +v_bc
       if y[i]>(Ly-eps):
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -v_bc
       if np.sqrt((x[i]-0.5*Lx)**2+(y[i]-0.5*Ly)**2)<rad*(1+eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof+0] = 0
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0

if experiment==2:
   for i in range(0, NV):
       if np.sqrt((x[i]-0.5*Lx)**2+(y[i]-0.5*Ly)**2)<rad*(1+eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof+0] = 0
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0



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

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('displacement.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

#####################################################################
# retrieve pressure
#####################################################################
start = time.time()

p  = np.zeros(nel,dtype=np.float64)  
e_xx = np.zeros(nel,dtype=np.float64)  
e_yy = np.zeros(nel,dtype=np.float64)  
e_xy = np.zeros(nel,dtype=np.float64)  
e  = np.zeros(nel,dtype=np.float64)  
sigma_xx = np.zeros(nel,dtype=np.float64)  
sigma_yy = np.zeros(nel,dtype=np.float64)  
sigma_xy = np.zeros(nel,dtype=np.float64)  
sigma  = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

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

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0,m):
        e_xx[iel] += dNdx[k]*u[icon[k,iel]]
        e_yy[iel] += dNdy[k]*v[icon[k,iel]]
        e_xy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+0.5*dNdx[k]*v[icon[k,iel]]

    e[iel]=np.sqrt(0.5*(e_xx[iel]**2+e_yy[iel]**2)+e_xy[iel]**2)

    p[iel]=-(lambdaa+2./3.*mu)*(e_xx[iel]+e_yy[iel])

    sigma_xx[iel]=lambdaa*(e_xx[iel]+e_yy[iel])+2*mu*e_xx[iel]
    sigma_yy[iel]=lambdaa*(e_xx[iel]+e_yy[iel])+2*mu*e_yy[iel]
    sigma_xy[iel]=                              2*mu*e_xy[iel]

    sigma[iel]=np.sqrt(0.5*(sigma_xx[iel]**2+sigma_yy[iel]**2)+sigma_xy[iel]**2)
   
#end for 

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(e_xx),np.max(e_xx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(e_yy),np.max(e_yy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(e_xy),np.max(e_xy)))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,e_xx,e_yy,e_xy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & strain: %.3f s" % (time.time() - start))

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
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % e_xx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % e_yy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % e_xy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % e[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigma_xx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigma_yy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigma_xy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % sigma[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div(u)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (e_xx[iel]+e_yy[iel]))
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
