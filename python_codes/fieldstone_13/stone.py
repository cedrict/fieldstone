import random
import numpy as np
import sys 
import math 
import time as clock
import scipy.sparse as sps
import matplotlib.pyplot as plt
import jatten as jatten 
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
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

def material_layout(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=2
    else:
       val=1
    return val

###############################################################################

def locate_particle(x,y):
    ielx=int(x/Lx*nelx)
    iely=int(y/Ly*nely)
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]]
    xmax=x_V[icon_V[2,iel]]
    ymin=y_V[icon_V[0,iel]]
    ymax=y_V[icon_V[2,iel]]
    r=((swarm_x[ip]-xmin)/(xmax-xmin)-0.5)*2
    s=((swarm_y[ip]-ymin)/(ymax-ymin)-0.5)*2
    return iel,r,s

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 013 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 8):
   nelx             =int(sys.argv[1])
   nely             =int(sys.argv[2])
   visu             =int(sys.argv[3])
   avrg             =int(sys.argv[4])
   nparticle_per_dim=int(sys.argv[5])
   pdistribution    =int(sys.argv[6])
   proj             =int(sys.argv[7])
else:
   nelx = 64
   nely = nelx
   visu = 1
   avrg = 3
   nparticle_per_dim=8
   pdistribution=2 # 1: random, 2: regular, 3: Poisson disc
   proj = 2 # 0: eltal, 1: nodal1, 2: nodal2, 3: nodal 3
    
nel=nelx*nely          # number of elements, total
nn_V=(nelx+1)*(nely+1) # number of nodes
Nfem=nn_V*ndof_V       # Total number of degrees of freedom

penalty=1.e7 # penalty coefficient value

gx=0.
gy=-10.

debug=False

rho_mat=np.array([1.,2.],dtype=np.float64) 
eta_mat=np.array([1.,1.e3],dtype=np.float64) 

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter = 0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter += 1

if debug: np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

print("grid points layout: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("connectivity array: %.3f s" % (clock.time()-start))

###############################################################################
# particle setup
###############################################################################
start=clock.time()

nparticle_per_element=nparticle_per_dim*nparticle_per_dim

if pdistribution==1: # pure random
   nparticle=nel*nparticle_per_element
   swarm_x=np.zeros(nparticle,dtype=np.float64)  
   swarm_y=np.zeros(nparticle,dtype=np.float64)  
   swarm_mat=np.zeros(nparticle,dtype=np.int32)  
   counter=0
   for iel in range(0,nel):
       for ip in range(0,nparticle_per_element):
           r=random.uniform(-1.,+1)
           s=random.uniform(-1.,+1)
           N=basis_functions_V(r,s)
           swarm_x[counter]=np.dot(N,x_V[icon_V[:,iel]])
           swarm_y[counter]=np.dot(N,y_V[icon_V[:,iel]])
           counter+=1

elif pdistribution==2: # regular
   nparticle=nel*nparticle_per_element
   swarm_x=np.zeros(nparticle,dtype=np.float64)  
   swarm_y=np.zeros(nparticle,dtype=np.float64)  
   swarm_mat=np.zeros(nparticle,dtype=np.int32)  
   counter=0
   for iel in range(0,nel):
       for j in range(0,nparticle_per_dim):
           for i in range(0,nparticle_per_dim):
               r=-1.+i*2./nparticle_per_dim+1./nparticle_per_dim
               s=-1.+j*2./nparticle_per_dim+1./nparticle_per_dim
               N=basis_functions_V(r,s)
               swarm_x[counter]=np.dot(N,x_V[icon_V[:,iel]])
               swarm_y[counter]=np.dot(N,y_V[icon_V[:,iel]])
               counter+=1

else: # Poisson disc
   kpoisson=30
   nparticle_wish=nel*nparticle_per_element # target
   avrgdist=np.sqrt(Lx*Ly/nparticle_wish)/1.25
   nparticle,swarm_x,swarm_y = jatten.PoissonDisc(kpoisson,avrgdist,Lx,Ly)
   swarm_mat=np.zeros(nparticle,dtype=np.int32)  
   print ('nparticle_wish, nparticle, ratio: %d %d %e ' % (nparticle_wish,nparticle,nparticle/nparticle_wish) )

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))

print("particle layout: %.3f s" % (clock.time()-start))

###############################################################################
# material layout
###############################################################################
start=clock.time()

for ip in range(0,nparticle):
    swarm_mat[ip]=material_layout(swarm_x[ip],swarm_y[ip])

print("     -> swarm_mat (m,M) %.4f %.4f " %(np.min(swarm_mat),np.max(swarm_mat)))

if debug: np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')

print("material layout: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental averagings 
###############################################################################
start=clock.time()

rho_elemental=np.zeros(nel,dtype=np.float64) 
eta_elemental=np.zeros(nel,dtype=np.float64) 

if proj==0:

   for ip in range(0,nparticle):
       iel,r,s,=locate_particle(swarm_x[ip],swarm_y[ip])
       rho_elemental[iel]+=rho_mat[swarm_mat[ip]-1]
       if avrg==1: eta_elemental[iel]+=eta_mat[swarm_mat[ip]-1]              # arithmetic
       if avrg==2: eta_elemental[iel]+=math.log(eta_mat[swarm_mat[ip]-1],10) # geometric
       if avrg==3: eta_elemental[iel]+=1./eta_mat[swarm_mat[ip]-1]           # harmonic
       
   for iel in range(0,nel):
       rho_elemental[iel]/=nparticle_per_element
       if avrg==1: eta_elemental[iel]/=nparticle_per_element
       if avrg==2: eta_elemental[iel]=10.**(eta_elemental[iel]/nparticle_per_element)
       if avrg==3: eta_elemental[iel]=nparticle_per_element/eta_elemental[iel]

   print("     -> rho_elemental (m,M) %.4f %.4f " %(np.min(rho_elemental),np.max(rho_elemental)))
   print("     -> eta_elemental (m,M) %.4f %.4f " %(np.min(eta_elemental),np.max(eta_elemental)))

print("projection elemental: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal averagings 1
# proj=1: use all four elements around node
###############################################################################
start=clock.time()

rho_nodal1=np.zeros(nn_V,dtype=np.float64) 
eta_nodal1=np.zeros(nn_V,dtype=np.float64) 
count_nodal1=np.zeros(nn_V,dtype=np.float64) 

if proj==1:

   for ip in range(0,nparticle):
       iel,r,s,=locate_particle(swarm_x[ip],swarm_y[ip])
       for i in range(0,m_V):
           rho_nodal1[icon_V[i,iel]]+=rho_mat[swarm_mat[ip]-1]
           count_nodal1[icon_V[i,iel]]+=1
           if avrg==1: eta_nodal1[icon_V[i,iel]]+=eta_mat[swarm_mat[ip]-1]
           if avrg==2: eta_nodal1[icon_V[i,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)
           if avrg==3: eta_nodal1[icon_V[i,iel]]+=1./eta_mat[swarm_mat[ip]-1]

   for i in range(0,nn_V):
       rho_nodal1[i]/=count_nodal1[i]
       if avrg==1: eta_nodal1[i]/=count_nodal1[i]
       if avrg==2: eta_nodal1[i]=10.**(eta_nodal1[i]/count_nodal1[i])
       if avrg==3: eta_nodal1[i]=count_nodal1[i]/eta_nodal1[i]

   print("     -> count_nodal1 (m,M) %.4f %.4f " %(np.min(count_nodal1),np.max(count_nodal1)))
   print("     -> rho_nodal1 (m,M) %.4f %.4f " %(np.min(rho_nodal1),np.max(rho_nodal1)))
   print("     -> eta_nodal1 (m,M) %.4f %.4f " %(np.min(eta_nodal1),np.max(eta_nodal1)))

   if debug:
      np.savetxt('count_nodal1.ascii',np.array([x,y,count_nodal1]).T,header='# x,y,count')
      np.savetxt('rho_nodal1.ascii',np.array([x,y,rho_nodal1]).T,header='# x,y,rho')
      np.savetxt('eta_nodal1.ascii',np.array([x,y,eta_nodal1]).T,header='# x,y,eta')

print("projection nodal 1: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal averagings 2 
# proj=2: use all four quadrants around node
###############################################################################
start=clock.time()

rho_nodal2=np.zeros(nn_V,dtype=np.float64) 
eta_nodal2=np.zeros(nn_V,dtype=np.float64) 
count_nodal2=np.zeros(nn_V,dtype=np.float64) 

if proj==2:

   for ip in range(0,nparticle):
       iel,r,s,=locate_particle(swarm_x[ip],swarm_y[ip])
       # particle is in lower left quadrant
       if (r<=0 and s<=0):
          rho_nodal2[icon_V[0,iel]]+=rho_mat[swarm_mat[ip]-1]
          count_nodal2[icon_V[0,iel]]+=1
          if avrg==1: eta_nodal2[icon_V[0,iel]]+=eta_mat[swarm_mat[ip]-1]
          if avrg==2: eta_nodal2[icon_V[0,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)
          if avrg==3: eta_nodal2[icon_V[0,iel]]+=1./eta_mat[swarm_mat[ip]-1]
       # particle is in lower right quadrant 
       if (r>=0 and s<=0):
          rho_nodal2[icon_V[1,iel]]+=rho_mat[swarm_mat[ip]-1]
          count_nodal2[icon_V[1,iel]]+=1
          if avrg==1: eta_nodal2[icon_V[1,iel]]+=eta_mat[swarm_mat[ip]-1]
          if avrg==2: eta_nodal2[icon_V[1,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)
          if avrg==3: eta_nodal2[icon_V[1,iel]]+=1./eta_mat[swarm_mat[ip]-1]
       # particle is in upper right quadrant
       if (r>=0 and s>=0):
          rho_nodal2[icon_V[2,iel]]+=rho_mat[swarm_mat[ip]-1]
          count_nodal2[icon_V[2,iel]]+=1
          if avrg==1: eta_nodal2[icon_V[2,iel]]+=eta_mat[swarm_mat[ip]-1]
          if avrg==2: eta_nodal2[icon_V[2,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)
          if avrg==3: eta_nodal2[icon_V[2,iel]]+=1./eta_mat[swarm_mat[ip]-1]
       # particle is in upper left quadrant
       if (r<=0 and s>=0):
          rho_nodal2[icon_V[3,iel]]+=rho_mat[swarm_mat[ip]-1]
          count_nodal2[icon_V[3,iel]]+=1
          if avrg==1: eta_nodal2[icon_V[3,iel]]+=eta_mat[swarm_mat[ip]-1]
          if avrg==2: eta_nodal2[icon_V[3,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)
          if avrg==3: eta_nodal2[icon_V[3,iel]]+=1./eta_mat[swarm_mat[ip]-1]

   if np.min(count_nodal2)==0:
      quit() 

   for i in range(0,nn_V):
       rho_nodal2[i]/=count_nodal2[i]
       if avrg==1: eta_nodal2[i]/=count_nodal2[i]
       if avrg==2: eta_nodal2[i]=10.**(eta_nodal2[i]/count_nodal2[i])
       if avrg==3: eta_nodal2[i]=count_nodal2[i]/eta_nodal2[i]

   print("     -> count_nodal2 (m,M) %.4f %.4f " %(np.min(count_nodal2),np.max(count_nodal2)))
   print("     -> rho_nodal2 (m,M) %.4f %.4f " %(np.min(rho_nodal2),np.max(rho_nodal2)))
   print("     -> eta_nodal2 (m,M) %.4f %.4f " %(np.min(eta_nodal2),np.max(eta_nodal2)))

   if debug:
      np.savetxt('count_nodal2.ascii',np.array([x,y,count_nodal2]).T,header='# x,y,count')
      np.savetxt('rho_nodal2.ascii',np.array([x,y,rho_nodal2]).T,header='# x,y,rho')
      np.savetxt('eta_nodal2.ascii',np.array([x,y,eta_nodal2]).T,header='# x,y,eta')

print("projection nodal 2: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal averagings 3 
# proj=3: use all four elements around node w/ weighting
###############################################################################
start=clock.time()

rho_nodal3=np.zeros(nn_V,dtype=np.float64) 
eta_nodal3=np.zeros(nn_V,dtype=np.float64) 
count_nodal3=np.zeros(nn_V,dtype=np.float64) 

if proj==3:

   for ip in range(0,nparticle):
       iel,r,s,=locate_particle(swarm_x[ip],swarm_y[ip])
       N0=0.25*(1-r)*(1-s)
       N1=0.25*(1+r)*(1-s)
       N2=0.25*(1+r)*(1+s)
       N3=0.25*(1-r)*(1+s)

       rho_nodal3[icon_V[0,iel]]+=rho_mat[swarm_mat[ip]-1]*N0
       rho_nodal3[icon_V[1,iel]]+=rho_mat[swarm_mat[ip]-1]*N1
       rho_nodal3[icon_V[2,iel]]+=rho_mat[swarm_mat[ip]-1]*N2
       rho_nodal3[icon_V[3,iel]]+=rho_mat[swarm_mat[ip]-1]*N3

       if avrg==1: # arithmetic
          eta_nodal3[icon_V[0,iel]]+=eta_mat[swarm_mat[ip]-1]*N0
          eta_nodal3[icon_V[1,iel]]+=eta_mat[swarm_mat[ip]-1]*N1
          eta_nodal3[icon_V[2,iel]]+=eta_mat[swarm_mat[ip]-1]*N2
          eta_nodal3[icon_V[3,iel]]+=eta_mat[swarm_mat[ip]-1]*N3
       if avrg==2: # geometric
          eta_nodal3[icon_V[0,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)*N0
          eta_nodal3[icon_V[1,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)*N1
          eta_nodal3[icon_V[2,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)*N2
          eta_nodal3[icon_V[3,iel]]+=math.log(eta_mat[swarm_mat[ip]-1],10)*N3
       if avrg==3: # harmonic
          eta_nodal3[icon_V[0,iel]]+=1./eta_mat[swarm_mat[ip]-1]*N0
          eta_nodal3[icon_V[1,iel]]+=1./eta_mat[swarm_mat[ip]-1]*N1
          eta_nodal3[icon_V[2,iel]]+=1./eta_mat[swarm_mat[ip]-1]*N2
          eta_nodal3[icon_V[3,iel]]+=1./eta_mat[swarm_mat[ip]-1]*N3

       count_nodal3[icon_V[0,iel]]+=N0
       count_nodal3[icon_V[1,iel]]+=N1
       count_nodal3[icon_V[2,iel]]+=N2
       count_nodal3[icon_V[3,iel]]+=N3

   if np.min(count_nodal3)==0:
      quit() 

   for i in range(0,nn_V):
       rho_nodal3[i]/=count_nodal3[i]
       if avrg==1: eta_nodal3[i]/=count_nodal3[i]
       if avrg==2: eta_nodal3[i]=10.**(eta_nodal3[i]/count_nodal3[i])
       if avrg==3: eta_nodal3[i]=count_nodal3[i]/eta_nodal3[i]
 
   print("     -> count_nodal3 (m,M) %.4f %.4f " %(np.min(count_nodal3),np.max(count_nodal3)))
   print("     -> rho_nodal3 (m,M) %.4f %.4f " %(np.min(rho_nodal3),np.max(rho_nodal3)))
   print("     -> eta_nodal3 (m,M) %.4f %.4f " %(np.min(eta_nodal3),np.max(eta_nodal3)))

   if debug:
      np.savetxt('count_nodal3.ascii',np.array([x,y,count_nodal3]).T,header='# x,y,count')
      np.savetxt('rho_nodal3.ascii',np.array([x,y,rho_nodal3]).T,header='# x,y,rho')
      np.savetxt('eta_nodal3.ascii',np.array([x,y,eta_nodal3]).T,header='# x,y,eta')

print("projection nodal 3: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal averagings 4
###############################################################################
# This method is not documented in the manual. It uses a Q1 projection
# by means of the Q1 mass matrix. 
# The problem is that it does not work well, even in the ideal case 
# of regularly spaced particles. The rhs is \int N(r,s) rho dV
# where dV is actually the average volume a particle takes.
# If the mass matrix is lumped, one logically recovers the results
# of the nodal3 algorithm.
# This mass matrix approach is bound to fail in the sense that 
# inside the element with 2 different materials the jump cannot
# be represented by Q1. 
###############################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
A_fem=np.zeros((nn_V,nn_V),dtype=np.float64) # Q1 mass matrix
rhs=np.zeros(nn_V,dtype=np.float64)      # rhs
Adiag=np.zeros(nn_V,dtype=np.float64) # Q1 mass matrix
rho_nodal4=np.zeros(nn_V,dtype=np.float64) 
particlevolume=Lx*Ly/nparticle

for iel in range(0,0): # nel):

    # set arrays to 0 every loop
    f_el =np.zeros(m_V,dtype=np.float64)
    M_el =np.zeros((m_V,m_V),dtype=np.float64)
    Mdiag_el=np.zeros(m_V,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
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

            for i in range(0,m_V):
                for j in range(0,m_V):
                    M_el[i,j]+=N_V[i]*N_V[j]*JxWq
                    Mdiag_el[i]+=N_V[i]*N_V[j]*JxWq

        #end for
    #end for

    for im in range(0,nparticle):
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        ielm=nelx*iely+ielx
        if ielm==iel:
           xmin=x_V[icon_V[0,iel]]
           xmax=x_V[icon_V[2,iel]]
           ymin=y_V[icon_V[0,iel]]
           ymax=y_V[icon_V[2,iel]]
           r=((swarm_x[im]-xmin)/(xmax-xmin)-0.5)*2
           s=((swarm_y[im]-ymin)/(ymax-ymin)-0.5)*2
           N=basis_functions_V(r,s)
           for i in range(0,m_V):
               f_el[i]+=N[i]*particlevolume*rho_mat[swarm_mat[im]-1]
    #end for

    # assemble matrix and right hand side
    for k1 in range(0,m_V):
        ik=icon_V[k1,iel]
        for k2 in range(0,m_V):
            jk=icon_V[k2,iel]
            A_fem[ik,jk]+=M_el[k1,k2]
        rhs[ik]+=f_el[k1]
        Adiag[ik]+=Mdiag_el[k1]
    #end for

#end for nel

#rho_nodal4=sps.linalg.spsolve(sps.csr_matrix(A_fem),rhs)
#rho_nodal4[:]=rhs[:]/Adiag[:]

#if debug: np.savetxt('rho_nodal4.ascii',np.array([x,y,rho_nodal4]).T,header='# x,y,rho')

#print("projection nodal 4: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions: no slip on all four sides
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool) # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
H=np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]],dtype=np.float64) 
C=np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],dtype=np.float64) 

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
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

            # assign quad point a density and viscosity

            if proj==0:
               rhoq=rho_elemental[iel]
               etaq=eta_elemental[iel]

            if proj==1:
               rhoq=np.dot(N_V,rho_nodal1[icon_V[:,iel]])
               etaq=np.dot(N_V,eta_nodal1[icon_V[:,iel]])

            if proj==2:
               rhoq=np.dot(N_V,rho_nodal2[icon_V[:,iel]])
               etaq=np.dot(N_V,eta_nodal2[icon_V[:,iel]])

            if proj==3:
               rhoq=np.dot(N_V,rho_nodal3[icon_V[:,iel]])
               etaq=np.dot(N_V,eta_nodal3[icon_V[:,iel]])

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            A_el+=B.T.dot(C.dot(B))*etaq*JxWq

            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*rhoq*gx*JxWq
                b_el[2*i+1]+=N_V[i]*rhoq*gy*JxWq

        #end for
    #end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    JxWq=np.linalg.det(jcb)*weightq
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    # compute elemental matrix
    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]: 
               fixt=bc_val_V[m1]
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

    # assemble matrix and right hand side
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

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.0
    sq=0.0
    weightq=2.0 * 2.0
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %e %e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %e %e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %e %e " %(np.min(exy),np.max(exy)))

if debug: 
   np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p and strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# compute vrms 
###############################################################################
start=clock.time()

vrms=0.
for iel in range (0,nel):
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxWq
        #end for
    #end for
#end for

vrms=np.sqrt(vrms/(Lx*Ly))

print("     -> nel= %.d ; nparticle= %d ; vrms= %e" %(nel,nparticle,vrms))

print("compute vrms: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################

if visu==1:

   filename = 'solution.vtu'
   vtufile=open(filename,"w")
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
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (p[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Density (1)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % rho_nodal1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Density (2)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % rho_nodal2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Density (3)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % rho_nodal3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Viscosity (1)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % eta_nodal1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Viscosity (2)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % eta_nodal2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Viscosity (3)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" % eta_nodal3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (eta_elemental[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],\
                                       icon_V[2,iel],icon_V[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*m_V))
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

   ####################################

   vtufile=open('particles.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nparticle,nparticle))
   #-----
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nparticle):
       vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #-----
   vtufile.write("<PointData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='swarm_mat' Format='ascii'> \n")
   for im in range(0,nparticle):
       vtufile.write("%e \n" % swarm_mat[im])
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #-----
   vtufile.write("<Cells>\n")
   #-
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nparticle):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nparticle):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nparticle):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")
   #-
   vtufile.write("</Cells>\n")
   #-----
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   print("export to vtu: %.3f s" % (clock.time() - start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
