import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
import jatten as jatten 

#------------------------------------------------------------------------------

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq)
    dNdr_1=+0.25*(1.-sq)
    dNdr_2=+0.25*(1.+sq)
    dNdr_3=-0.25*(1.+sq)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

def paint(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=2
    else:
       val=1
    return val

#------------------------------------------------------------------------------

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)

    if (limitX != 0.0):
       axes[plotX][plotY].set_xlim(0,limitX)

    if (limitY != 0.0):
       axes[plotX][plotY].set_ylim(0,limitY)

    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

def scatterPlot(x,y,val, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].scatter(x,y,c=val,s=0.2)
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)
    axes[plotX][plotY].set_xlim(0,limitX)
    axes[plotX][plotY].set_ylim(0,limitY)
    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    axes[plotX][plotY].set_aspect('equal')
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

print("variable declaration")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 8):
   nelx           =int(sys.argv[1])
   nely           =int(sys.argv[2])
   visu           =int(sys.argv[3])
   avrg           =int(sys.argv[4])
   nmarker_per_dim=int(sys.argv[5])
   mdistribution  =int(sys.argv[6])
   proj           =int(sys.argv[7])
else:
   nelx = 24
   nely = 24
   visu = 1
   avrg = 3
   nmarker_per_dim=4
   mdistribution=2 # 1: random, 2: regular, 3: Poisson disc
   proj = 3
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

penalty=1.e7  # penalty coefficient value

Nfem=nnp*ndof  # Total number of degrees of freedom

gx=0.
gy=-10.

eps=1.e-10

sqrt3=np.sqrt(3.)

rho_mat = np.array([1.,2.],dtype=np.float64) 
eta_mat = np.array([1.,1.e3],dtype=np.float64) 

#################################################################
# grid point setup
#################################################################

print("grid point setup")

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

#################################################################
# connectivity
#################################################################

print("connectivity")

icon =np.zeros((m, nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

#################################################################
# marker setup
#################################################################
start = time.time()

nmarker_per_element=nmarker_per_dim*nmarker_per_dim

if mdistribution==1: # pure random
   nmarker=nel*nmarker_per_element
   swarm_x=np.empty(nmarker,dtype=np.float64)  
   swarm_y=np.empty(nmarker,dtype=np.float64)  
   swarm_mat=np.empty(nmarker,dtype=np.int32)  
   counter=0
   for iel in range(0,nel):
       x1=x[icon[0,iel]] ; y1=y[icon[0,iel]]
       x2=x[icon[1,iel]] ; y2=y[icon[1,iel]]
       x3=x[icon[2,iel]] ; y3=y[icon[2,iel]]
       x4=x[icon[3,iel]] ; y4=y[icon[3,iel]]
       for im in range(0,nmarker_per_element):
           # generate random numbers r,s between 0 and 1
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N1=0.25*(1-r)*(1-s)
           N2=0.25*(1+r)*(1-s)
           N3=0.25*(1+r)*(1+s)
           N4=0.25*(1-r)*(1+s)
           swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
           swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
           counter+=1

elif mdistribution==2: # regular
   nmarker=nel*nmarker_per_element
   swarm_x=np.empty(nmarker,dtype=np.float64)  
   swarm_y=np.empty(nmarker,dtype=np.float64)  
   swarm_mat=np.empty(nmarker,dtype=np.int32)  
   counter=0
   for iel in range(0,nel):
       x1=x[icon[0,iel]] ; y1=y[icon[0,iel]]
       x2=x[icon[1,iel]] ; y2=y[icon[1,iel]]
       x3=x[icon[2,iel]] ; y3=y[icon[2,iel]]
       x4=x[icon[3,iel]] ; y4=y[icon[3,iel]]
       for j in range(0,nmarker_per_dim):
           for i in range(0,nmarker_per_dim):
               r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
               s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
               N1=0.25*(1-r)*(1-s)
               N2=0.25*(1+r)*(1-s)
               N3=0.25*(1+r)*(1+s)
               N4=0.25*(1-r)*(1+s)
               swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
               swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
               counter+=1

else: # Poisson disc

   kpoisson=30
   nmarker_wish=nel*nmarker_per_element # target
   avrgdist=np.sqrt(Lx*Ly/nmarker_wish)/1.25
   #print (avrgdist)
   nmarker,swarm_x,swarm_y = jatten.PoissonDisc(kpoisson,avrgdist,Lx,Ly)
   swarm_mat=np.empty(nmarker,dtype=np.int32)  
   print ('nmarker_wish, nmarker, ratio: %d %d %e ' % (nmarker_wish,nmarker,nmarker/nmarker_wish) )

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

#################################################################
# material layout
#################################################################
start = time.time()

for im in range(0,nmarker):
    swarm_mat[im] = paint(swarm_x[im],swarm_y[im])

print("     -> swarm_mat (m,M) %.4f %.4f " %(np.min(swarm_mat),np.max(swarm_mat)))

np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')

print("material layout: %.3f s" % (time.time() - start))

#################################################################
# compute elemental averagings 
#################################################################
start = time.time()

rho_elemental=np.zeros(nel,dtype=np.float64) 
eta_elemental=np.zeros(nel,dtype=np.float64) 

for im in range(0,nmarker):
    ielx=int(swarm_x[im]/Lx*nelx)
    if ielx<0:
       print ('ielx<0',ielx)
    if ielx>nelx-1:
       print ('ielx>nelx-1')
    iely=int(swarm_y[im]/Ly*nely)
    if iely<0:
       print ('iely<0')
    if iely>nely-1:
       print ('iely>nely-1')
    iel=nelx*(iely)+ielx
    if iel<0:
       print ('iel<0')
    if iel>nel-1:
       print ('iel>nel-1')
    rho_elemental[iel]+=rho_mat[swarm_mat[im]-1]
    if avrg==1: # arithmetic
       eta_elemental[iel]+=eta_mat[swarm_mat[im]-1]
    if avrg==2: # geometric
       eta_elemental[iel]+=math.log(eta_mat[swarm_mat[im]-1],10)
    if avrg==3: # harmonic
       eta_elemental[iel]+=1./eta_mat[swarm_mat[im]-1]
       
for iel in range(0,nel):
    rho_elemental[iel]/=nmarker_per_element
    if avrg==1:
       eta_elemental[iel]/=nmarker_per_element
    if avrg==2:
       eta_elemental[iel]=10.**(eta_elemental[iel]/nmarker_per_element)
    if avrg==3:
       eta_elemental[iel]=nmarker_per_element/eta_elemental[iel]

print("     -> rho_elemental (m,M) %.4f %.4f " %(np.min(rho_elemental),np.max(rho_elemental)))
print("     -> eta_elemental (m,M) %.4f %.4f " %(np.min(eta_elemental),np.max(eta_elemental)))

print("projection elemental: %.3f s" % (time.time() - start))

#################################################################
# compute nodal averagings 1,2,3
#################################################################
start = time.time()

rho_nodal1=np.zeros(nnp,dtype=np.float64) 
rho_nodal2=np.zeros(nnp,dtype=np.float64) 
rho_nodal3=np.zeros(nnp,dtype=np.float64) 
eta_nodal1=np.zeros(nnp,dtype=np.float64) 
eta_nodal2=np.zeros(nnp,dtype=np.float64) 
eta_nodal3=np.zeros(nnp,dtype=np.float64) 
count_nodal1=np.zeros(nnp,dtype=np.float64) 
count_nodal2=np.zeros(nnp,dtype=np.float64) 
count_nodal3=np.zeros(nnp,dtype=np.float64) 

for im in range(0,nmarker):
    ielx=int(swarm_x[im]/Lx*nelx)
    iely=int(swarm_y[im]/Ly*nely)
    iel=nelx*(iely)+ielx
    xmin=x[icon[0,iel]]
    xmax=x[icon[2,iel]]
    ymin=y[icon[0,iel]]
    ymax=y[icon[2,iel]]
    r=((swarm_x[im]-xmin)/(xmax-xmin)-0.5)*2
    s=((swarm_y[im]-ymin)/(ymax-ymin)-0.5)*2

    # compute rho_nodal1,eta_nodal1
    # proj=2: use all four elements around node
    # nodal A

    for i in range(0,m):
        rho_nodal1[icon[i,iel]]+=rho_mat[swarm_mat[im]-1]
        count_nodal1[icon[i,iel]]+=1
        if avrg==1: # arithmetic
           eta_nodal1[icon[i,iel]]+=eta_mat[swarm_mat[im]-1]
        if avrg==2: # geometric
           eta_nodal1[icon[i,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)
        if avrg==3: # harmonic
           eta_nodal1[icon[i,iel]]+=1./eta_mat[swarm_mat[im]-1]

    # compute rho_nodal2,eta_nodal2
    # proj=3: use all four quadrants around node
    # nodal B

    # marker is in lower left quadrant
    if (r<=0 and s<=0):
       rho_nodal2[icon[0,iel]]+=rho_mat[swarm_mat[im]-1]
       count_nodal2[icon[0,iel]]+=1
       if avrg==1: # arithmetic
          eta_nodal2[icon[0,iel]]+=eta_mat[swarm_mat[im]-1]
       if avrg==2: # geometric
          eta_nodal2[icon[0,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)
       if avrg==3: # harmonic
          eta_nodal2[icon[0,iel]]+=1./eta_mat[swarm_mat[im]-1]

    # marker is in lower right quadrant 
    if (r>=0 and s<=0):
       rho_nodal2[icon[1,iel]]+=rho_mat[swarm_mat[im]-1]
       count_nodal2[icon[1,iel]]+=1
       if avrg==1: # arithmetic
          eta_nodal2[icon[1,iel]]+=eta_mat[swarm_mat[im]-1]
       if avrg==2: # geometric
          eta_nodal2[icon[1,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)
       if avrg==3: # harmonic
          eta_nodal2[icon[1,iel]]+=1./eta_mat[swarm_mat[im]-1]

    # marker is in upper right quadrant
    if (r>=0 and s>=0):
       rho_nodal2[icon[2,iel]]+=rho_mat[swarm_mat[im]-1]
       count_nodal2[icon[2,iel]]+=1
       if avrg==1: # arithmetic
          eta_nodal2[icon[2,iel]]+=eta_mat[swarm_mat[im]-1]
       if avrg==2: # geometric
          eta_nodal2[icon[2,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)
       if avrg==3: # harmonic
          eta_nodal2[icon[2,iel]]+=1./eta_mat[swarm_mat[im]-1]

    # marker is in upper left quadrant
    if (r<=0 and s>=0):
       rho_nodal2[icon[3,iel]]+=rho_mat[swarm_mat[im]-1]
       count_nodal2[icon[3,iel]]+=1
       if avrg==1: # arithmetic
          eta_nodal2[icon[3,iel]]+=eta_mat[swarm_mat[im]-1]
       if avrg==2: # geometric
          eta_nodal2[icon[3,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)
       if avrg==3: # harmonic
          eta_nodal2[icon[3,iel]]+=1./eta_mat[swarm_mat[im]-1]

    # compute rho_nodal3,eta_nodal3
    # proj=4: use all four elements around node w/ averaging
    # nodal C

    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)

    rho_nodal3[icon[0,iel]]+=rho_mat[swarm_mat[im]-1]*N0
    rho_nodal3[icon[1,iel]]+=rho_mat[swarm_mat[im]-1]*N1
    rho_nodal3[icon[2,iel]]+=rho_mat[swarm_mat[im]-1]*N2
    rho_nodal3[icon[3,iel]]+=rho_mat[swarm_mat[im]-1]*N3

    if avrg==1: # arithmetic
       eta_nodal3[icon[0,iel]]+=eta_mat[swarm_mat[im]-1]*N0
       eta_nodal3[icon[1,iel]]+=eta_mat[swarm_mat[im]-1]*N1
       eta_nodal3[icon[2,iel]]+=eta_mat[swarm_mat[im]-1]*N2
       eta_nodal3[icon[3,iel]]+=eta_mat[swarm_mat[im]-1]*N3
    if avrg==2: # geometric
       eta_nodal3[icon[0,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)*N0
       eta_nodal3[icon[1,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)*N1
       eta_nodal3[icon[2,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)*N2
       eta_nodal3[icon[3,iel]]+=math.log(eta_mat[swarm_mat[im]-1],10)*N3
    if avrg==3: # harmonic
       eta_nodal3[icon[0,iel]]+=1./eta_mat[swarm_mat[im]-1]*N0
       eta_nodal3[icon[1,iel]]+=1./eta_mat[swarm_mat[im]-1]*N1
       eta_nodal3[icon[2,iel]]+=1./eta_mat[swarm_mat[im]-1]*N2
       eta_nodal3[icon[3,iel]]+=1./eta_mat[swarm_mat[im]-1]*N3

    count_nodal3[icon[0,iel]]+=N0
    count_nodal3[icon[1,iel]]+=N1
    count_nodal3[icon[2,iel]]+=N2
    count_nodal3[icon[3,iel]]+=N3


if np.min(count_nodal2)==0:
   quit() 

if np.min(count_nodal3)==0:
   quit() 

for i in range(0,nnp):
    rho_nodal1[i]/=count_nodal1[i]
    rho_nodal2[i]/=count_nodal2[i]
    rho_nodal3[i]/=count_nodal3[i]
    if avrg==1: # arithmetic
       eta_nodal1[i]/=count_nodal1[i]
       eta_nodal2[i]/=count_nodal2[i]
       eta_nodal3[i]/=count_nodal3[i]
    if avrg==2: # geometric
       eta_nodal1[i]=10.**(eta_nodal1[i]/count_nodal1[i])
       eta_nodal2[i]=10.**(eta_nodal2[i]/count_nodal2[i])
       eta_nodal3[i]=10.**(eta_nodal3[i]/count_nodal3[i])
    if avrg==3: # harmonic
       eta_nodal1[i]=count_nodal1[i]/eta_nodal1[i]
       eta_nodal2[i]=count_nodal2[i]/eta_nodal2[i]
       eta_nodal3[i]=count_nodal3[i]/eta_nodal3[i]
 
print("     -> count_nodal1 (m,M) %.4f %.4f " %(np.min(count_nodal1),np.max(count_nodal1)))
print("     -> count_nodal2 (m,M) %.4f %.4f " %(np.min(count_nodal2),np.max(count_nodal2)))
print("     -> count_nodal3 (m,M) %.4f %.4f " %(np.min(count_nodal3),np.max(count_nodal3)))
print("     -> rho_nodal1 (m,M) %.4f %.4f " %(np.min(rho_nodal1),np.max(rho_nodal1)))
print("     -> rho_nodal2 (m,M) %.4f %.4f " %(np.min(rho_nodal2),np.max(rho_nodal2)))
print("     -> rho_nodal3 (m,M) %.4f %.4f " %(np.min(rho_nodal3),np.max(rho_nodal3)))
print("     -> eta_nodal1 (m,M) %.4f %.4f " %(np.min(eta_nodal1),np.max(eta_nodal1)))
print("     -> eta_nodal2 (m,M) %.4f %.4f " %(np.min(eta_nodal2),np.max(eta_nodal2)))
print("     -> eta_nodal3 (m,M) %.4f %.4f " %(np.min(eta_nodal3),np.max(eta_nodal3)))

np.savetxt('count_nodal1.ascii',np.array([x,y,count_nodal1]).T,header='# x,y,count')
np.savetxt('count_nodal2.ascii',np.array([x,y,count_nodal2]).T,header='# x,y,count')
np.savetxt('count_nodal3.ascii',np.array([x,y,count_nodal3]).T,header='# x,y,count')
np.savetxt('rho_nodal1.ascii',np.array([x,y,rho_nodal1]).T,header='# x,y,rho')
np.savetxt('rho_nodal2.ascii',np.array([x,y,rho_nodal2]).T,header='# x,y,rho')
np.savetxt('rho_nodal3.ascii',np.array([x,y,rho_nodal3]).T,header='# x,y,rho')
np.savetxt('eta_nodal1.ascii',np.array([x,y,eta_nodal1]).T,header='# x,y,eta')
np.savetxt('eta_nodal2.ascii',np.array([x,y,eta_nodal2]).T,header='# x,y,eta')
np.savetxt('eta_nodal3.ascii',np.array([x,y,eta_nodal3]).T,header='# x,y,eta')

print("projection nodal 1,2,3: %.3f s" % (time.time() - start))

#################################################################
# compute nodal averagings 4
#################################################################
# This method is not documented in the manual. It uses a Q1 projection
# by means of the Q1 mass matrix. 
# The problem is that it does not work well, even in the ideal case 
# of regularly spaced markers. The rhs is \int N(r,s) rho dV
# where dV is actually the average volume a marker takes.
# If the mass matrix is lumped, one logically recovers the results
# of the nodal3 algorithm.
# This mass matrix approach is bound to fail in the sense that 
# inside the element with 2 different materials the jump cannot
# be represented by Q1. 
#################################################################

start = time.time()

N=np.zeros(m,dtype=np.float64)
dNdr  = np.zeros(m,dtype=np.float64)
dNds  = np.zeros(m,dtype=np.float64)
A_mat=np.zeros((nnp,nnp),dtype=np.float64) # Q1 mass matrix
rhs=np.zeros(nnp,dtype=np.float64)      # rhs
Adiag=np.zeros(nnp,dtype=np.float64) # Q1 mass matrix
rho_nodal4=np.zeros(nnp,dtype=np.float64) 
markervolume=Lx*Ly/nmarker

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros(m,dtype=np.float64)
    M_el =np.zeros((m,m),dtype=np.float64)
    Mdiag_el=np.zeros(m,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N[0:m]=NNV(rq,sq)
            dNdr[0:m]=dNNVdr(rq,sq)
            dNds[0:m]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob = np.linalg.det(jcb)

            for i in range(0,m):
                for j in range(0,m):
                    M_el[i,j]+=N[i]*N[j]*wq*jcob
                    Mdiag_el[i]+=N[i]*N[j]*wq*jcob

    for im in range(0,nmarker):
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        ielm=nelx*(iely)+ielx
        if ielm==iel:
           xmin=x[icon[0,iel]]
           xmax=x[icon[2,iel]]
           ymin=y[icon[0,iel]]
           ymax=y[icon[2,iel]]
           r=((swarm_x[im]-xmin)/(xmax-xmin)-0.5)*2
           s=((swarm_y[im]-ymin)/(ymax-ymin)-0.5)*2
           N[0:m]=NNV(r,s)
           for i in range(0,m):
               f_el[i]+=N[i]*markervolume*rho_mat[swarm_mat[im]-1]

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        ik=icon[k1,iel]
        for k2 in range(0,m):
            jk=icon[k2,iel]
            A_mat[ik,jk]+=M_el[k1,k2]
        rhs[ik]+=f_el[k1]
        Adiag[ik]+=Mdiag_el[k1]


rho_nodal4=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
#rho_nodal4[:]=rhs[:]/Adiag[:]

np.savetxt('rho_nodal4.ascii',np.array([x,y,rho_nodal4]).T,header='# x,y,rho')

print("projection nodal 4: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if y[i]<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
k_mat = np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]],dtype=np.float64) 
c_mat = np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m * ndof)
    a_el = np.zeros((m * ndof, m * ndof), dtype=float)

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
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # assign quad point a density and viscosity

            if proj==1:
               rhoq=rho_elemental[iel]
               etaq=eta_elemental[iel]
            if proj==2:
               rhoq=0
               etaq=0
               for k in range(0, m):
                   rhoq+=N[k]*rho_nodal1[icon[k,iel]]
                   etaq+=N[k]*eta_nodal1[icon[k,iel]]
            if proj==3:
               rhoq=0
               etaq=0
               for k in range(0, m):
                   rhoq+=N[k]*rho_nodal2[icon[k,iel]]
                   etaq+=N[k]*eta_nodal2[icon[k,iel]]
            if proj==4:
               rhoq=0
               etaq=0
               for k in range(0, m):
                   rhoq+=N[k]*rho_nodal3[icon[k,iel]]
                   etaq+=N[k]*eta_nodal3[icon[k,iel]]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*etaq*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*wq*rhoq*gx
                b_el[2*i+1]+=N[i]*jcob*wq*rhoq*gy

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    wq=2.*2.

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob = np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi = np.linalg.inv(jcb)

    # compute dNdx and dNdy
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]

    # compute elemental matrix
    a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

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
            rhs[m1]+=b_el[ikk]

print("build FE matrix: %.3f s" % (time.time() - start))

#################################################################
# impose boundary conditions
#################################################################
start = time.time()

for i in range(0, Nfem):
    if bc_fix[i]:
       a_matref = a_mat[i,i]
       for j in range(0,Nfem):
           rhs[j]-= a_mat[i, j] * bc_val[i]
           a_mat[i,j]=0.
           a_mat[j,i]=0.
           a_mat[i,i] = a_matref
       rhs[i]=a_matref*bc_val[i]

#print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat),np.max(a_mat)))
#print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

print("imposing b.c.: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################

u,v=np.reshape(sol,(nnp,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

#####################################################################
# retrieve pressure
#####################################################################

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  

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

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0, m):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

#################################################################
# compute vrms 
#################################################################

vrms=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.
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
            jcob = np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0,m):
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            uq=0.
            vq=0.
            for k in range(0,m):
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
            vrms+=(uq**2+vq**2)*wq*jcob

vrms=np.sqrt(vrms/(Lx*Ly))

print("     -> nel= %.d ; nmarker= %d ;  vrms= %.6f" %(nel,nmarker,vrms))

#####################################################################
# plot of solution
#####################################################################

u_temp=np.reshape(u,(nny,nnx))
v_temp=np.reshape(v,(nny,nnx))
p_temp=np.reshape(p,(nely,nelx))
vel_temp=np.reshape((u**2+v**2),(nny,nnx))

fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,18))

uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

onePlot(u_temp,  0, 0, "$v_x$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(v_temp,  0, 1, "$v_y$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(p_temp,  0, 2, "$p$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
onePlot(vel_temp,1, 0, "$|v|$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
axes[1][0].quiver(x,y,u_temp,v_temp, alpha=.95)

scatterPlot(swarm_x,swarm_y,swarm_mat, 1, 1, "markers",                 "x", "y", uextent,  Lx,  Ly, 'Spectral_r')
#axes[1][1].scatter(swarm_x,swarm_y,s=0.1,c=swarm_mat)

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

   vtufile=open('markers.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))

   #vtufile.write("<PointData Scalars='scalars'>\n")
   #vtufile.write("</PointData>\n")

   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nmarker):
       vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")

   vtufile.write("<Cells>\n")

   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nmarker):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nmarker):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")

   vtufile.write("</Cells>\n")

   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
