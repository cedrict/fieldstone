import numpy as np
import time as clock
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import sys as sys

###############################################################################

def basis_functions_2d(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

def basis_functions_V(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t) 
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t) 
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def basis_functions_V_dt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s) 
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

###############################################################################

sqrt2=np.sqrt(2.)
sqrt3=np.sqrt(3.)
eps=1e-8

print("*******************************")
print("********** stone 185 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=0.05 # 50% of desired Lx
Ly=0.05 # 50% of desired Ly
Lz=0.1
rad=0.0125

E1=3e6   ; G1=1.8e6
E2=2.5e6 ; G2=1e6
sigma_bc=9e3

nelx=12
nely=nelx
nelz=12   #must be even number

distance=eps*Lx

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

print('     -> nelx=',nelx)
print('     -> nely=',nely)
print('     -> nelz=',nelz)

debug=False

###############################################################################
# The mesh in the xy plane is composed of eight blocks. Each is first built 
# and warped to fit the contour of the hole and they are subsequently merged.
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
start=clock.time()

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

counter = 0
for j in range(0,b_nny):
    for i in range(0,b_nnx):
        b1_x[counter]=i*Lx/float(b_nelx)
        b1_y[counter]=j*Ly/float(b_nely)
        counter += 1

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

print("prepare arrays: %.3f s" % (clock.time()-start))

###############################################################################
# map block 1
###############################################################################
start=clock.time()

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
        N_V=basis_functions_2d(r,s)
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
        N_V=basis_functions_2d(r,s)
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
# make block 3 - it is a rotated block 1
# make block 4 - it is a rotated block 2
# make block 5 - it is a rotated block 3
# make block 6 - it is a rotated block 4
# make block 7 - it is a rotated block 5
# make block 8 - it is a rotated block 6
###############################################################################
start=clock.time()

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

if debug:
   np.savetxt('temp1.ascii',np.array([b1_x,b1_y]).T,header='# x,y')
   np.savetxt('temp2.ascii',np.array([b2_x,b2_y]).T,header='# x,y')
   np.savetxt('temp3.ascii',np.array([b3_x,b3_y]).T,header='# x,y')
   np.savetxt('temp4.ascii',np.array([b4_x,b4_y]).T,header='# x,y')
   np.savetxt('temp5.ascii',np.array([b5_x,b5_y]).T,header='# x,y')
   np.savetxt('temp6.ascii',np.array([b6_x,b6_y]).T,header='# x,y')
   np.savetxt('temp7.ascii',np.array([b7_x,b7_y]).T,header='# x,y')
   np.savetxt('temp8.ascii',np.array([b8_x,b8_y]).T,header='# x,y')

print("make block 3-8: %.3f s" % (clock.time()-start))

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
       x_V[counter]=tempx[i]+Lx
       y_V[counter]=tempy[i]+Ly
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

Lx*=2
Ly*=2

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################

vtufile=open('mesh2d.vtu',"w")
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

###############################################################################
# extrude 2d mesh to make 3d volume
###############################################################################
start=clock.time()

x_2d=np.copy(x_V)
y_2d=np.copy(y_V)
icon_2d=np.copy(icon_V)
nn_2d=nn_V
nel_2d=nel

ndof_V=3
m_V=8
nel=nel_2d*nelz
nn_V=nn_2d*(nelz+1)
Nfem=nn_V*ndof_V

print('     -> nn_V=',nn_V)
print('     -> nel=',nel)
print('     -> Nfem=',Nfem)

x_V=np.zeros(nn_V,dtype=np.float64) 
y_V=np.zeros(nn_V,dtype=np.float64) 
z_V=np.zeros(nn_V,dtype=np.float64) 
icon_V=np.zeros((m_V,nel),dtype=np.int32)

for i in range(0,nelz+1):
    #print(i,i*nn_2d,(i+1)*nn_2d)
    x_V[i*nn_2d:(i+1)*nn_2d]=x_2d[:]
    y_V[i*nn_2d:(i+1)*nn_2d]=y_2d[:]
    z_V[i*nn_2d:(i+1)*nn_2d]=i*hz

if debug: np.savetxt('nodes.ascii',np.array([x_V,y_V,z_V]).T,header='# x,y,z')

counter=0
for i in range(0,nel_2d):
    for k in range(0,nelz):
        icon_V[0:4,counter]=icon_2d[0:4,i]+nn_2d*k
        icon_V[4:8,counter]=icon_2d[0:4,i]+nn_2d*(k+1)
        counter+=1

print("extrude 2d mesh to 3d: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental volume / sanity check
###############################################################################
start=clock.time()

jcb=np.zeros((3,3),dtype=np.float64)
volume=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.
                N_V=basis_functions_V(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                volume[iel]+=JxWq
            #end for
        #end for
    #end for
#end for

volume_analytical=Lx*Ly*Lz-(np.pi*rad**2*Lz)

print("     -> volume (m,M) %.6e %.6e " %(np.min(volume),np.max(volume)))
print("     -> total area meas %.8e " %(volume.sum()))
print("     -> total area anal %.8e " %(volume_analytical))

print("compute elements volume: %.3f s" % (clock.time()-start))

###############################################################################
# flag elements on sides
###############################################################################
start=clock.time()

face1=np.zeros(nel,dtype=bool)
face2=np.zeros(nel,dtype=bool)
face3=np.zeros(nel,dtype=bool)
face4=np.zeros(nel,dtype=bool)
face5=np.zeros(nel,dtype=bool)
face6=np.zeros(nel,dtype=bool)

for iel in range(0,nel):
    for k in range(m_V):
        #x=0 side
        if abs((x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])*0.5) < eps: face1[iel]=True
        #x=Lx side
        if abs((x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])*0.5-Lx) < eps: face2[iel]=True
        #y=0 side
        if abs((y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])*0.5) < eps: face3[iel]=True
        #y=Ly side
        if abs((y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])*0.5-Ly) < eps: face4[iel]=True
        #z=0 side
        if abs((z_V[icon_V[0,iel]]+z_V[icon_V[3,iel]])*0.5) < eps: face5[iel]=True
        #z=Lz side
        if abs((z_V[icon_V[4,iel]]+z_V[icon_V[7,iel]])*0.5-Lz) < eps: face6[iel]=True

print("flag elements: %.3f s" % (clock.time()-start))

###############################################################################
# export mesh to vtu
###############################################################################
start=clock.time()

vtufile=open('mesh3d.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % volume[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='face1' Format='ascii'> \n")
for iel in range (0,nel):
    if face1[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='face2' Format='ascii'> \n")
for iel in range (0,nel):
    if face2[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='face3' Format='ascii'> \n")
for iel in range (0,nel):
    if face3[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='face4' Format='ascii'> \n")
for iel in range (0,nel):
    if face4[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='face5' Format='ascii'> \n")
for iel in range (0,nel):
    if face5[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='face6' Format='ascii'> \n")
for iel in range (0,nel):
    if face6[iel]:
       vtufile.write("%d\n" % 1)
    else:
       vtufile.write("%d\n" % 0)
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],\
                                                icon_V[2,iel],icon_V[3,iel],\
                                                icon_V[4,iel],icon_V[5,iel],\
                                                icon_V[6,iel],icon_V[7,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %12)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export mesh to vtu: %.3f s" % (clock.time()-start))

###############################################################################
# compute Cartesian and polar coordinates of element centers
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
z_e=np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    z_e[iel]=np.dot(N_V,z_V[icon_V[:,iel]])

rr=np.zeros(nel,dtype=np.float64)  
theta=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rr[iel]=np.sqrt((x_e[iel]-Lx/2)**2+(y_e[iel]-Ly/2)**2)
    theta[iel]=np.arctan2((y_e[iel]-Ly/2),(x_e[iel]-Lx/2))

print("compute coords.: %.3f s" % (clock.time()-start))

###############################################################################
# assign elements physical properties
###############################################################################
start=clock.time()

E=np.zeros(nel,dtype=np.float64)       # Young's modulus
G=np.zeros(nel,dtype=np.float64)       # shear modulus
lambdaa=np.zeros(nel,dtype=np.float64) # Lame's first parameter 

for iel in range(0,nel):
    if z_e[iel]<Lz/3:
       G[iel]=G1
       E[iel]=E1
    elif z_e[iel]<2*Lz/3:
       G[iel]=G2
       E[iel]=E2
    else:
       G[iel]=G1
       E[iel]=E1
    
lambdaa=G*(E-2*G)/(3*G-E) # Table 4.1 of Sadd's book, wiki

nu=(E-2*G)/(2*G) # Table 4.1 of Sadd's book
nu=E/(2*G)-1     # https://en.wikipedia.org/wiki/Elastic_modulus, identical


print("     -> Young's modulus: E (m,M) %e %e " %(np.min(E),np.max(E)))
print("     -> Shear modulus: G (m,M) %e %e " %(np.min(G),np.max(G)))
print("     -> Poisson's ratio: nu (m,M) %e %e " %(np.min(nu),np.max(nu)))
print("     -> Lame's 1st parameter: lambdaa (m,M) %e %e " %(np.min(lambdaa),np.max(lambdaa)))

print("compute coords.: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
# there are 3 translational nullspaces so they need to be dealt with 
# by means of Dirichlet boundary conditions at specific nodes:
# point A (Lx/2,0,Lz/2) -> (u=0,_,w=0)
# point B (Lx/2,Ly,Lz/2) -> (u=0,_,w=0)
# point C (0,Ly/2,Lz/2) -> (_,v=0,w=0)
# point D (Lx,Ly/2,Lz/2) -> (_,v=0,w=0)
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):

    # point A
    if abs(x_V[i]-Lx/2)/Lx<eps and \
       abs(y_V[i]-   0)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = 0 # fix u
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # point B
    if abs(x_V[i]-Lx/2)/Lx<eps and \
       abs(y_V[i]-  Ly)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = 0 # fix u
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # point C
    if abs(x_V[i]-   0)/Lx<eps and \
       abs(y_V[i]-Ly/2)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # point D
    if abs(x_V[i]-  Lx)/Lx<eps and \
       abs(y_V[i]-Ly/2)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w


print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# section 22.1 in fieldstone
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)

for iel in range(0,nel):

    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V)

    C=np.array([[2*G[iel]+lambdaa[iel],         lambdaa[iel],         lambdaa[iel],     0,     0,     0],
                [         lambdaa[iel],2*G[iel]+lambdaa[iel],         lambdaa[iel],     0,     0,     0],
                [         lambdaa[iel],         lambdaa[iel],2*G[iel]+lambdaa[iel],     0,     0,     0],
                [                    0,                    0,                    0,G[iel],     0,     0],
                [                    0,                    0,                    0,     0,G[iel],     0],
                [                    0,                    0,                    0,     0,     0,G[iel]]],dtype=np.float64) 

    for iq in [-1,1]:
        for jq in [-1,1]:
            for kq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                tq=kq/sqrt3
                weightq=1.*1.*1.
                N_V=basis_functions_V(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                for i in range(0,m_V):
                    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                        [0.       ,dNdy_V[i],0.       ],
                                        [0.       ,0.       ,dNdz_V[i]],
                                        [dNdy_V[i],dNdx_V[i],0.       ],
                                        [dNdz_V[i],0.       ,dNdx_V[i]],
                                        [0.       ,dNdz_V[i],dNdy_V[i]]]

                # compute elemental A_fem matrix
                A_el+=B.T.dot(C.dot(B))*JxWq

                # compute elemental rhs vector
                #for i in range(0, m):
                #    b_el[2*i  ]-=N[i]*jcob*wq*gx*rho
                #    b_el[2*i+1]-=N[i]*jcob*wq*gy*rho

            #end for
        #end for
    #end for

    # impose stress boundary conditions
    if face1[iel]: # face is nodes 1-2-5-6
          surf=hz*abs(y_V[icon_V[1,iel]]-y_V[icon_V[2,iel]])
          b_el[ndof_V*1]-=sigma_bc*surf*0.25
          b_el[ndof_V*2]-=sigma_bc*surf*0.25
          b_el[ndof_V*5]-=sigma_bc*surf*0.25
          b_el[ndof_V*6]-=sigma_bc*surf*0.25
    if face2[iel]: # face is nodes 1-2-5-6
          surf=hz*abs(y_V[icon_V[1,iel]]-y_V[icon_V[2,iel]])
          b_el[ndof_V*1]+=sigma_bc*surf*0.25
          b_el[ndof_V*2]+=sigma_bc*surf*0.25
          b_el[ndof_V*5]+=sigma_bc*surf*0.25
          b_el[ndof_V*6]+=sigma_bc*surf*0.25

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

#sol=np.zeros(Nfem,dtype=np.float64) 

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate u,v,w arrays
###############################################################################
start=clock.time()

u,v,w=np.reshape(sol,(nn_V,3)).T

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %e %e " %(np.min(w),np.max(w)))

if debug: 
   np.savetxt('displacement.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve nodal strain tensor components 
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
cc=np.zeros(nn_V,dtype=np.float64)  
e_n=np.zeros(nn_V,dtype=np.float64)  
e_xx_n=np.zeros(nn_V,dtype=np.float64)  
e_yy_n=np.zeros(nn_V,dtype=np.float64)  
e_zz_n=np.zeros(nn_V,dtype=np.float64)  
e_xy_n=np.zeros(nn_V,dtype=np.float64)  
e_xz_n=np.zeros(nn_V,dtype=np.float64)  
e_yz_n=np.zeros(nn_V,dtype=np.float64)  

r_V=np.array([-1, 1, 1,-1,-1, 1, 1,-1],np.float64)
s_V=np.array([-1,-1, 1, 1,-1,-1, 1, 1],np.float64)
t_V=np.array([ 0, 0, 0, 0, 1, 1, 1, 1],np.float64)

for iel in range(0,nel):
    for i in range(0,m_V):
        rq = r_V[i]
        sq = s_V[i]
        tq = t_V[i]

        exx=np.dot(dNdx_V,u[icon_V[:,iel]])
        eyy=np.dot(dNdy_V,v[icon_V[:,iel]])
        ezz=np.dot(dNdz_V,w[icon_V[:,iel]])

        exy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
           +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
        exz=np.dot(dNdx_V,w[icon_V[:,iel]])*0.5\
           +np.dot(dNdz_V,u[icon_V[:,iel]])*0.5
        eyz=np.dot(dNdy_V,w[icon_V[:,iel]])*0.5\
           +np.dot(dNdz_V,v[icon_V[:,iel]])*0.5

        e_xx_n[icon_V[i,iel]]+=exx
        e_yy_n[icon_V[i,iel]]+=eyy
        e_zz_n[icon_V[i,iel]]+=ezz
        e_xy_n[icon_V[i,iel]]+=exy
        e_xz_n[icon_V[i,iel]]+=exz
        e_yz_n[icon_V[i,iel]]+=eyz

        cc[icon_V[i,iel]]+=1.
    #end for
#end for

e_xx_n[:]/=cc[:]
e_yy_n[:]/=cc[:]
e_zz_n[:]/=cc[:]
e_xy_n[:]/=cc[:]
e_xz_n[:]/=cc[:]
e_yz_n[:]/=cc[:]

e_n[:]=np.sqrt(0.5*(e_xx_n[:]**2+e_yy_n[:]**2)+e_xy_n[:]**2) # CHANGE

print("     -> e_xx_n (m,M) %e %e " %(np.min(e_xx_n),np.max(e_xx_n)))
print("     -> e_yy_n (m,M) %e %e " %(np.min(e_yy_n),np.max(e_yy_n)))
print("     -> e_zz_n (m,M) %e %e " %(np.min(e_zz_n),np.max(e_zz_n)))
print("     -> e_xy_n (m,M) %e %e " %(np.min(e_xy_n),np.max(e_xy_n)))
print("     -> e_xz_n (m,M) %e %e " %(np.min(e_xz_n),np.max(e_xz_n)))
print("     -> e_yz_n (m,M) %e %e " %(np.min(e_yz_n),np.max(e_yz_n)))

print("compute nodal strain components: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve elemental strain tensor components 
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
e_e=np.zeros(nel,dtype=np.float64)  
e_xx_e=np.zeros(nel,dtype=np.float64)  
e_yy_e=np.zeros(nel,dtype=np.float64)  
e_zz_e=np.zeros(nel,dtype=np.float64)  
e_xy_e=np.zeros(nel,dtype=np.float64)  
e_xz_e=np.zeros(nel,dtype=np.float64)  
e_yz_e=np.zeros(nel,dtype=np.float64)  

sigma_e=np.zeros(nel,dtype=np.float64)  
sigma_xx_e=np.zeros(nel,dtype=np.float64)  
sigma_yy_e=np.zeros(nel,dtype=np.float64)  
sigma_zz_e=np.zeros(nel,dtype=np.float64)  
sigma_xy_e=np.zeros(nel,dtype=np.float64)  
sigma_xz_e=np.zeros(nel,dtype=np.float64)  
sigma_yz_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0
    sq=0
    tq=0
    N_V=basis_functions_V(rq,sq,tq)
    dNdr_V=basis_functions_V_dr(rq,sq,tq)
    dNds_V=basis_functions_V_ds(rq,sq,tq)
    dNdt_V=basis_functions_V_dt(rq,sq,tq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

    e_xx_e[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
    e_yy_e[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
    e_zz_e[iel]=np.dot(dNdz_V,w[icon_V[:,iel]])

    e_xy_e[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
               +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
    e_xz_e[iel]=np.dot(dNdx_V,w[icon_V[:,iel]])*0.5\
               +np.dot(dNdz_V,u[icon_V[:,iel]])*0.5
    e_yz_e[iel]=np.dot(dNdy_V,w[icon_V[:,iel]])*0.5\
               +np.dot(dNdz_V,v[icon_V[:,iel]])*0.5

#end for iel

p[:]=-(lambdaa[:]+2./3.*G[:])*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])

sigma_xx_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_xx_e[:] 
sigma_yy_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_yy_e[:] 
sigma_zz_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_zz_e[:] 
sigma_xy_e[:]=2*G[:]*e_xy_e[:]
sigma_xz_e[:]=2*G[:]*e_xz_e[:]
sigma_yz_e[:]=2*G[:]*e_yz_e[:]

print("compute elemental strain components: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve nodal stress tensor components 
###############################################################################
start=clock.time()

sigma_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xx_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_zz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xy_n=np.zeros(nn_V,dtype=np.float64)  
sigma_xz_n=np.zeros(nn_V,dtype=np.float64)  
sigma_yz_n=np.zeros(nn_V,dtype=np.float64)  

print("compute press & strain: %.3f s" % (clock.time()-start))

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
    vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],w[i]))
vtufile.write("</DataArray>\n")
#--
#vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
#for i in range(0,nn_V):
#    vtufile.write("%e  \n" %q[i])
#vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
e_xx_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
e_yy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_zz' Format='ascii'> \n")
e_zz_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
e_xy_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xz' Format='ascii'> \n")
e_xz_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yz' Format='ascii'> \n")
e_yz_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
e_n.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
#vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
#sigma_xx_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
#sigma_yy_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
#sigma_zz_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
#sigma_xy_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma_xz' Format='ascii'> \n")
#sigma_xz_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma_yz' Format='ascii'> \n")
#sigma_yz_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
#sigma_n.tofile(vtufile,sep=' ',format='%.4e')
#vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
rr.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
theta.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='E' Format='ascii'> \n")
E.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='G' Format='ascii'> \n")
G.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='nu' Format='ascii'> \n")
nu.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='lambdaa' Format='ascii'> \n")
lambdaa.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
p.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
e_xx_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
e_yy_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_zz' Format='ascii'> \n")
e_zz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
e_xy_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_xz' Format='ascii'> \n")
e_xz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e_yz' Format='ascii'> \n")
e_yz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
e_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='sigma_xx' Format='ascii'> \n")
sigma_xx_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_yy' Format='ascii'> \n")
sigma_yy_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_zz' Format='ascii'> \n")
sigma_zz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_xy' Format='ascii'> \n")
sigma_xy_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_xz' Format='ascii'> \n")
sigma_xz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma_yz' Format='ascii'> \n")
sigma_yz_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='sigma' Format='ascii'> \n")
sigma_e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--


vtufile.write("</CellData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],\
                                                icon_V[2,iel],icon_V[3,iel],\
                                                icon_V[4,iel],icon_V[5,iel],\
                                                icon_V[6,iel],icon_V[7,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %12)
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
