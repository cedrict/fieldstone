import numpy as np
import time as clock
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import sys as sys

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)
eps=1e-8

print("*******************************")
print("********** stone 186 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

nelx=64

E=5e10
nu=0.35
mu=E/2/(1+nu)
lambdaa=E*nu/(1+nu)/(1-2*nu)
rad=2e3
Lx=7e3
Ly=7e3

sigma_bc=5e6

factor=7

nely=nelx

distance=eps*Lx

print('     -> nelx=',nelx)
print('     -> nely=',nely)
print('     -> mu=',mu)
print('     -> lambdaa=',lambdaa)

debug=False

###############################################################################
#  The mesh is composed of eight blocks. Each is first built and warped
# to fit the contour of the hole and they are subsequently merged.
# all blocks have the same topology and number of elements.
#
#   E-----C-------------G
#   | 2   |             |
#   F     |      3      |
#    D  1 |             |
#     A---B-------------H
#    P  8 |             |
#   N     |      4      |
#   | 7   |             |
#   M-----O-------------I
#   |     |             |
#   |  6  |      5      |
#   |     |             |
#   L-----K-------------J
#
###############################################################################
start=clock.time()


xA=rad       ; yA=0
xB=Lx        ; yB=0
xC=Lx        ; yC=Ly
xD=rad/sqrt2 ; yD=rad/sqrt2
xE=0         ; yE=Lx
xF=0         ; yF=rad
xG=factor*Lx ; yG=Ly
xO=Lx        ; yO=-Ly
xL=0         ; yL=-(factor-1)*Ly
xM=0         ; yM=-Ly
xK=Lx        ; yK=yL

b_nelx=nelx
b_nely=nely
b_nnx=nelx+1
b_nny=nely+1
b_NV=b_nnx*b_nny
b_nel=b_nelx*b_nely

b1_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b1_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b2_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b2_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b3_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b3_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b4_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b4_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b5_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b5_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b6_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b6_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b7_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b7_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates
b8_x=np.zeros(b_NV,dtype=np.float64)  # x coordinates
b8_y=np.zeros(b_NV,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        b1_x[counter]=i*Lx/float(b_nelx)
        b1_y[counter]=j*Ly/float(b_nely)
        counter+=1

b2_x[:]=b1_x[:]
b2_y[:]=b1_y[:]

b1_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b2_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b3_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b4_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b5_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b6_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b7_icon =np.zeros((m_V,b_nel),dtype=np.int32)
b8_icon =np.zeros((m_V,b_nel),dtype=np.int32)

counter=0
for j in range(0,b_nely):
    for i in range(0,b_nelx):
        b1_icon[0, counter] = i + j * (b_nelx + 1)
        b1_icon[1, counter] = i + 1 + j * (b_nelx + 1)
        b1_icon[2, counter] = i + 1 + (j + 1) * (b_nelx + 1)
        b1_icon[3, counter] = i + (j + 1) * (b_nelx + 1)
        counter+=1

b2_icon[:,:]=b1_icon[:,:]
b3_icon[:,:]=b1_icon[:,:]
b4_icon[:,:]=b1_icon[:,:]
b5_icon[:,:]=b1_icon[:,:]
b6_icon[:,:]=b1_icon[:,:]
b7_icon[:,:]=b1_icon[:,:]
b8_icon[:,:]=b1_icon[:,:]

print("prepare arrays: %.3f s" % (clock.time()-start))

###############################################################################
# map block 1 (ABCD)
###############################################################################
start=clock.time()

# map to quadrilateral ABCD
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        r=2*(b1_x[counter]/Lx-0.5)
        s=2*(b1_y[counter]/Ly-0.5)
        N_V=basis_functions_V(r,s)
        b1_x[counter]=N_V[0]*xA+N_V[1]*xB+N_V[2]*xC+N_V[3]*xD
        b1_y[counter]=N_V[0]*yA+N_V[1]*yB+N_V[2]*yC+N_V[3]*yD
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

print("make block 1: %.3f s" % (clock.time()-start))

###############################################################################
# map block 2
###############################################################################
start=clock.time()

# map to quadrilateral FDCE
counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        r=2*(b2_x[counter]/Lx-0.5)
        s=2*(b2_y[counter]/Ly-0.5)
        N_V=basis_functions_V(r,s)
        b2_x[counter]=N_V[0]*xD+N_V[1]*xC+N_V[2]*xE+N_V[3]*xF
        b2_y[counter]=N_V[0]*yD+N_V[1]*yC+N_V[2]*yE+N_V[3]*yF
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

print("make block 2: %.3f s" % (clock.time()-start))

###############################################################################
# make block 3,4,5,6 
###############################################################################
start=clock.time()

counter=0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        b3_x[counter]=xB+i*(xG-xB)/float(b_nelx)
        b3_y[counter]=yB+j*Ly/float(b_nely)
        b4_x[counter]=xO+i*(xG-xO)/float(b_nelx)
        b4_y[counter]=yO+j*Ly/float(b_nely)
        b5_x[counter]=xK+i*(xG-xK)/float(b_nelx)
        b5_y[counter]=yK+j*(yM-yL)/float(b_nely)
        b6_x[counter]=xL+i*Lx/float(b_nelx)
        b6_y[counter]=yL+j*(yM-yL)/float(b_nely)
        counter+=1

print("make blocks 3,4,5,6: %.3f s" % (clock.time()-start))

###############################################################################
# make block 7 - it is a rotated block 5
# make block 8 - it is a rotated block 6
###############################################################################
start=clock.time()

b7_x[:]= b1_y[:]
b7_y[:]=-b1_x[:]

b8_x[:]= b2_y[:]
b8_y[:]= -b2_x[:]

print("make blocks 7,8: %.3f s" % (clock.time()-start))

###############################################################################

if debug:
   np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
   np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
   np.savetxt('temp3.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
   np.savetxt('temp4.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
   np.savetxt('temp5.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
   np.savetxt('temp6.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
   np.savetxt('temp7.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
   np.savetxt('temp8.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

###############################################################################
# merge blocks
###############################################################################
start=clock.time()

nblock=8

tempx=np.zeros(nblock*b_NV,dtype=np.float64)  # x coordinates
tempy=np.zeros(nblock*b_NV,dtype=np.float64)  # y coordinates

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

nn_V=nblock*b_NV-sum(doubble)
nel=nblock*b_nel
Nfem=nn_V*ndof_V

print('     -> doubles=',sum(doubble))
print('     -> nn_V=',nn_V)
print('     -> nel=',nel)
print('     -> Nfem=',Nfem)

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for i in range(0,nblock*b_NV):
    if not doubble[i]: 
       x_V[counter]=tempx[i]#+Lx
       y_V[counter]=tempy[i]#+Ly
       counter+=1

icon_V[0:m_V,0*b_nel:1*b_nel]=b1_icon[0:m_V,0:b_nel]+0*b_NV
icon_V[0:m_V,1*b_nel:2*b_nel]=b2_icon[0:m_V,0:b_nel]+1*b_NV
icon_V[0:m_V,2*b_nel:3*b_nel]=b2_icon[0:m_V,0:b_nel]+2*b_NV
icon_V[0:m_V,3*b_nel:4*b_nel]=b2_icon[0:m_V,0:b_nel]+3*b_NV
icon_V[0:m_V,4*b_nel:5*b_nel]=b2_icon[0:m_V,0:b_nel]+4*b_NV
icon_V[0:m_V,5*b_nel:6*b_nel]=b2_icon[0:m_V,0:b_nel]+5*b_NV
icon_V[0:m_V,6*b_nel:7*b_nel]=b2_icon[0:m_V,0:b_nel]+6*b_NV
icon_V[0:m_V,7*b_nel:8*b_nel]=b2_icon[0:m_V,0:b_nel]+7*b_NV

for iel in range(0,nel):
    for i in range(0,m_V):
        icon_V[i,iel]=pointto[icon_V[i,iel]]

compact=np.zeros(nblock*b_NV,dtype=np.int32)

counter=0
for i in range(0,nblock*b_NV):
    if not doubble[i]:
       compact[i]=counter
       counter=counter+1

for iel in range(0,nel):
   for i in range(0,m_V):
      icon_V[i,iel]=compact[icon_V[i,iel]]

if debug: np.savetxt('mesh.ascii',np.array([x_V,y_V]).T,header='# x,y')

Lx=2*Lx  #### CHECK!!!!!!
Ly=3*Ly

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# flag boundary elements
###############################################################################
start=clock.time()

left=np.zeros(nn_V,dtype=bool)  
right=np.zeros(nn_V,dtype=bool)  
bottom=np.zeros(nn_V,dtype=bool)  
top=np.zeros(nn_V,dtype=bool)  
circle=np.zeros(nn_V,dtype=bool)  

for i in range(0,nn_V):
    if abs(x_V[i])/Lx<eps:       left[i]=True  
    if abs(x_V[i]-xG)/Lx<eps:    right[i]=True  
    if abs(y_V[i]-yK)/Ly<eps:    bottom[i]=True  
    if abs(y_V[i]-yC)/Ly<eps:    top[i]=True  
    if np.sqrt(x_V[i]**2+y_V[i]**2)<(1+eps)*rad:  circle[i]=True  

print("flag boundary nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute Cartesian and polar coordinates of element centers
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
r_e=np.zeros(nel,dtype=np.float64)  
theta_e=np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    for k in range(0,m_V):
        x_e[iel]+=x_V[icon_V[k,iel]]*0.25
        y_e[iel]+=y_V[icon_V[k,iel]]*0.25
    r_e[iel]=np.sqrt(x_e[iel]**2+y_e[iel]**2)
    theta_e[iel]=np.arctan2(y_e[iel],x_e[iel])
   
if debug: np.savetxt('centers.ascii',np.array([x_e,y_e,r_e,theta_e]).T)

print("compute el coords.: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if left[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
    if bottom[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    if right[i]:
       bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = 0
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0
    #if circle[i]:
    #   bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = x_V[i]/100
    #   bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = y_V[i]/100

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
jcb=np.zeros((2,2),dtype=np.float64)
C=np.array([[2*mu+lambdaa,lambdaa,0],
            [lambdaa,2*mu+lambdaa,0],
            [0,0,mu]],dtype=np.float64) 

for iel in range(0,nel):

    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

    for iq in [-1,1]:
        for jq in [-1,1]:

            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            N_V=basis_functions_V(rq,sq)
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

            # construct 3x8 b_mat matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental A_fem matrix
            A_el+=B.T.dot(C.dot(B))*JxWq

            # compute elemental rhs vector
            #for i in range(0, m):
            #    b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
            #    b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

        #end for
    #end for

    #if circle[icon_V[0,iel]] and circle[icon_V[1,iel]]:
    #   print('aaaa')
    #if circle[icon_V[1,iel]] and circle[icon_V[2,iel]]:
    #   print('bbbb')
    #if circle[icon_V[2,iel]] and circle[icon_V[3,iel]]:
    #   print('cccc')
    if circle[icon_V[3,iel]] and circle[icon_V[0,iel]]:
       dist=np.sqrt((x_V[icon_V[3,iel]]-x_V[icon_V[0,iel]])**2+\
                    (y_V[icon_V[3,iel]]-y_V[icon_V[0,iel]])**2 )
       xm=(x_V[icon_V[3,iel]]+x_V[icon_V[0,iel]])/2
       ym=(y_V[icon_V[3,iel]]+y_V[icon_V[0,iel]])/2
       theta=np.arctan2(ym,xm)
       b_el[0]+=sigma_bc*dist*np.cos(theta)/2
       b_el[1]+=sigma_bc*dist*np.sin(theta)/2
       b_el[6]+=sigma_bc*dist*np.cos(theta)/2
       b_el[7]+=sigma_bc*dist*np.sin(theta)/2

    # apply boundary conditions
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof_V*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m_V*ndof_V):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               #end for
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix A_fem and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate u,v arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

if debug: np.savetxt('displacement.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure, stress, strain, ...
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
cc=np.zeros(nn_V,dtype=np.float64)  
e_n=np.zeros(nn_V,dtype=np.float64)  
tau_n=np.zeros(nn_V,dtype=np.float64)  
e_xx_n=np.zeros(nn_V,dtype=np.float64)  
e_yy_n=np.zeros(nn_V,dtype=np.float64)  
e_xy_n=np.zeros(nn_V,dtype=np.float64)  
tau_xx_n=np.zeros(nn_V,dtype=np.float64)  
tau_yy_n=np.zeros(nn_V,dtype=np.float64)  
tau_zz_n=np.zeros(nn_V,dtype=np.float64)  
tau_xy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xx_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_zz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xy_n=np.zeros(nn_V,dtype=np.float64)  

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

for iel in range(0,nel):
    for i in range(0,m_V):
        rq = rVnodes[i]
        sq = sVnodes[i]
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exx=np.dot(dNdx_V,u[icon_V[:,iel]])
        eyy=np.dot(dNdy_V,v[icon_V[:,iel]])
        exy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
           +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
        e_xx_n[icon_V[i,iel]]+=exx
        e_yy_n[icon_V[i,iel]]+=eyy
        e_xy_n[icon_V[i,iel]]+=exy
        cc[icon_V[i,iel]]+=1.
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

if debug:
   np.savetxt('pressure.ascii',np.array([x_V,y_V,q]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_V,y_V,e_xx_n,e_yy_n,e_xy_n]).T,header='# x,y,exx,eyy,exy')

print("compute press & strain: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################


np.savetxt('top_solution.ascii',np.array([x_V[top],u[top],v[top],
                                          sigma_xx_n[top],sigma_yy_n[top],sigma_xy_n[top]]).T,header='# x,u,v')


###############################################################################
# plot of solution
###############################################################################
       
vtufile=open('solution.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
e_xx_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
e_yy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
e_xy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
e_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")


vtufile.write("<DataArray type='Float32' Name='circle' Format='ascii'> \n")
circle.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='left' Format='ascii'> \n")
left.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='right' Format='ascii'> \n")
right.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='bottom' Format='ascii'> \n")
bottom.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='top' Format='ascii'> \n")
top.tofile(vtufile,sep=' ',format='%d')
vtufile.write("</DataArray>\n")


#--
vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
sigma_xx_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
sigma_yy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
sigma_xy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
sigma_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
r_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
theta_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################






