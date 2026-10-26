import numpy as np
import time as clock
import scipy
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import numba
from compute_principal_stresses import *
from mesher_cube import * 
from mesher_borehole import * 
from basis_functions import *
from build_elemental_matrix import *
from compute_strain_tensor_components import *
from export_mesh_to_vtu import *

###############################################################################
# square root of second invariant of tensor 
###############################################################################

def effective(txx,tyy,tzz,txy,txz,tyz):
    return np.sqrt(0.5*(txx**2+tyy**2+tzz**2)+txy**2+txz**2+tyz**2)

###############################################################################
# see Eq 9 of Glerum et al, 2018 (https://doi.org/10.5194/se-9-267-2018)

def yield_criterion(c,phi,p):
    return 6*c*np.cos(phi)/sqrt3/(3-np.sin(phi))+\
           6*np.sin(phi)*p/sqrt3/(3-np.sin(phi))

###############################################################################

sqrt3=np.sqrt(3.)
eps=1e-8

print("*******************************")
print("********** stone 185 **********")
print("*******************************")

m_V=8     # number of nodes making up an element
ndof_V=3  # number of degrees of freedom per node

debug=True

#######################################
#experiment=0,1: full block
#experiment=2: 2d Kirsch (hole in a plate)
#experiment=3: 3d solution
#experiment=4: iowa setup
#######################################
# layout=1: one material
# layout=2: 3 slanted layers
#######################################

experiment=1

match experiment:
 case 0 | 1:
   meshtype=1 # full cuboid
   Lx=1
   Ly=1
   Lz=0.25
   nelx=32
   nely=nelx
   nelz=int(nelx/4)
   volume_analytical=Lx*Ly*Lz
   layout=1
   sigma_H=0
   sigma_h=0
   sigma_v=0
   nstep=1
   E1=2.5e6 
   G1=1e6
   radius=0
   if (experiment==0): 
      from block0 import *
   if (experiment==1): 
      from block1 import *

 case 2:
   meshtype=2 # hole
   Lx=1
   Ly=1
   Lz=0.125
   nelx=48
   nely=nelx
   nelz=4
   radius=0.02
   volume_analytical=Lx*Ly*Lz-(np.pi*radius**2*Lz)
   layout=1
   sigma_H1=1e3
   sigma_H=0
   sigma_h=0
   sigma_v=0
   nstep=1
   E1=2.5e6 
   G1=1e6
   from kirsch import *

 case 3:
   meshtype=2
   layout=1
   sigma_H=9e3
   sigma_h=0
   sigma_v=0
   nstep=1

 case 4:
   meshtype=2
   Lx=0.1
   Ly=0.1
   Lz=0.1
   radius=0.01
   volume_analytical=Lx*Ly*Lz-(np.pi*radius**2*Lz)
   nelx=10
   nely=nelx
   nelz=10  #must be even number
   layout=2
   sigma_H=9e3
   sigma_h=0
   sigma_v=0
   nstep=1

#######################################
#1: fast
#2: slow

bem=1 # build elementary matrix
ass=1 # assemble matrix

#######################################
#1: direct solver
#2: conjugate gradients

solver=2

#######################################

print('     -> meshtype=',meshtype) 
print('     -> nelx=',nelx)
print('     -> nely=',nely)
print('     -> nelz=',nelz)
print('     -> layout=',layout) 
print('     -> solver=',solver) 

#######################################
# layout
# 1: single material
# 2: one layer in matrix
#######################################

match layout:

 case 1:
  #E1=2.5e6 ; G1=1e6
  nu1=nu=(E1-2*G1)/(2*G1)
  lambdaa1=G1*(E1-2*G1)/(3*G1-E1) 
  print('     -> nu1=',nu1,'lambda1=',lambdaa1)

 case 2:
  E1=3e6   ; G1=1.15385e6
  E2=2.5e6 ; G2=1e6
  nu1=nu=(E1-2*G1)/(2*G1)
  nu2=nu=(E2-2*G2)/(2*G2)
  lambdaa1=G1*(E1-2*G1)/(3*G1-E1) 
  lambdaa2=G2*(E2-2*G2)/(3*G2-E2)
  print('     -> nu1=',nu1,'lambda1=',lambdaa1)
  print('     -> nu2=',nu2,'lambda2=',lambdaa2)

  #parameters controlling the tilt #z=z0+a*(x-Lx/2)+b*(y-Ly/2)

  aaa=0.1
  bbb=0.15

###############################################################################
# create mesh
###############################################################################
start=clock.time()

match meshtype:
 case 1:
   x_V,y_V,z_V,icon_V,nn_V,nel=mesher_cube(Lx,Ly,Lz,nelx,nely,nelz)
 case 2:
   x_V,y_V,z_V,icon_V,nn_V,nel=mesher_borehole(Lx,Ly,Lz,nelx,nely,nelz,radius)
 case 3:
   x_V,y_V,z_V,icon_V,nn_V,nel=mesher_quarter_borehole(radius)
 case _:
   exit('unknown mesh value')

#print('     -> nel=',nel)
#print('     -> nn_V=',nn_V)
print("*******************************")

Nfem=ndof_V*nn_V

wall_V=np.zeros(nn_V,dtype=bool)
theta_V=np.zeros(nn_V,dtype=np.float64)  
rad_V=np.zeros(nn_V,dtype=np.float64)  
for i in range(nn_V):
    rad_V[i]=np.sqrt((x_V[i]-Lx/2)**2+(y_V[i]-Ly/2)**2)
    theta_V[i]=np.arctan2((y_V[i]-Ly/2),(x_V[i]-Lx/2))
    if abs(rad_V[i]-radius)<1e-6*radius: wall_V[i]=True

#for iel in range (0,nel):
#    print ("iel=",iel,'icon:',icon_V[:,iel])

print("mesh domain: %.3f s" % (clock.time()-start))

###############################################################################
# precompute basis functions at quadrature points 
# quadrature rule based on 2x2x2 points per element
###############################################################################
start=clock.time()

nq_per_el=2*2*2  # nb of quadrature points per element
rq=np.zeros(nq_per_el,dtype=np.float64)
sq=np.zeros(nq_per_el,dtype=np.float64)
tq=np.zeros(nq_per_el,dtype=np.float64)
N_V=np.zeros((nq_per_el,m_V),dtype=np.float64)
dNdr_V=np.zeros((nq_per_el,m_V),dtype=np.float64)
dNds_V=np.zeros((nq_per_el,m_V),dtype=np.float64)
dNdt_V=np.zeros((nq_per_el,m_V),dtype=np.float64)

cq=0
for iq in [-1,1]:
    for jq in [-1,1]:
        for kq in [-1,1]:
            rq[cq]=iq/sqrt3
            sq[cq]=jq/sqrt3
            tq[cq]=kq/sqrt3
            weightq=1.*1.*1.
            N_V[cq,:]=basis_functions_V(rq[cq],sq[cq],tq[cq])
            dNdr_V[cq,:]=basis_functions_V_dr(rq[cq],sq[cq],tq[cq])
            dNds_V[cq,:]=basis_functions_V_ds(rq[cq],sq[cq],tq[cq])
            dNdt_V[cq,:]=basis_functions_V_dt(rq[cq],sq[cq],tq[cq])
            cq+=1
        #end for
    #end for
#end for

print("precompute basis functions: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental volume / sanity check
###############################################################################
start=clock.time()

jcb=np.zeros((3,3),dtype=np.float64)
volume=np.zeros(nel,dtype=np.float64) 

if True:
   for iel in range(0,nel):
       for k in range(0,nq_per_el):
           jcb[0,0]=np.dot(dNdr_V[k,:],x_V[icon_V[:,iel]])
           jcb[0,1]=np.dot(dNdr_V[k,:],y_V[icon_V[:,iel]])
           jcb[0,2]=np.dot(dNdr_V[k,:],z_V[icon_V[:,iel]])
           jcb[1,0]=np.dot(dNds_V[k,:],x_V[icon_V[:,iel]])
           jcb[1,1]=np.dot(dNds_V[k,:],y_V[icon_V[:,iel]])
           jcb[1,2]=np.dot(dNds_V[k,:],z_V[icon_V[:,iel]])
           jcb[2,0]=np.dot(dNdt_V[k,:],x_V[icon_V[:,iel]])
           jcb[2,1]=np.dot(dNdt_V[k,:],y_V[icon_V[:,iel]])
           jcb[2,2]=np.dot(dNdt_V[k,:],z_V[icon_V[:,iel]])
           JxWq=np.linalg.det(jcb)*weightq
           volume[iel]+=JxWq
       #end for
   #end for

   print("     -> volume (m,M) %.6e %.6e " %(np.min(volume),np.max(volume)))
   print("     -> total volume meas %.8e " %(volume.sum()))
   print("     -> total volume anal %.8e " %(volume_analytical))

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

match meshtype:
 case 1:
   iel=0
   for i in range(0,nelx):
       for j in range(0,nely):
           for k in range(0,nelz):
               if i==0: face1[iel]=True
               if i==nelx-1: face2[iel]=True
               if j==0: face3[iel]=True
               if j==nely-1: face4[iel]=True
               if k==0: face5[iel]=True
               if k==nelz-1: face6[iel]=True
               iel+=1

 case 2:
   for iel in range(0,nel):
       for k in range(m_V):
           #x=0 side
           if abs((x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])*0.5)<eps: face1[iel]=True
           #x=Lx side
           if abs((x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])*0.5-Lx)<eps: face2[iel]=True
           #y=0 side
           if abs((y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])*0.5)<eps: face3[iel]=True
           #y=Ly side
           if abs((y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])*0.5-Ly)<eps: face4[iel]=True
           #z=0 side
           if abs((z_V[icon_V[0,iel]]+z_V[icon_V[3,iel]])*0.5)<eps: face5[iel]=True
           #z=Lz side
           if abs((z_V[icon_V[4,iel]]+z_V[icon_V[7,iel]])*0.5-Lz)<eps: face6[iel]=True

 case _:
   exit('unknown meshtype')

print("flag elements: %.3f s" % (clock.time()-start))

###############################################################################
# export mesh to vtu
###############################################################################
start=clock.time()

if debug:
   export_mesh_to_vtu(nn_V,nel,x_V,y_V,z_V,icon_V,volume,\
                      face1,face2,face3,face4,face5,face6)

print("export 3d mesh to vtu: %.3f s" % (clock.time()-start))

###############################################################################
# compute Cartesian and polar coordinates of element centers
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
z_e=np.zeros(nel,dtype=np.float64)  
r_e=np.zeros(nel,dtype=np.float64)  
theta_e=np.zeros(nel,dtype=np.float64)  
    
for iel in range(0,nel):
    x_e[iel]=np.sum(x_V[icon_V[:,iel]])/m_V
    y_e[iel]=np.sum(y_V[icon_V[:,iel]])/m_V
    z_e[iel]=np.sum(z_V[icon_V[:,iel]])/m_V
    r_e[iel]=np.sqrt((x_e[iel]-Lx/2)**2+(y_e[iel]-Ly/2)**2)
    theta_e[iel]=np.arctan2((y_e[iel]-Ly/2),(x_e[iel]-Lx/2))

print("compute coords.: %.3f s" % (clock.time()-start))

###############################################################################
# assign elemental physical properties (E,G)
# compute elemental arrays lambdaa & nu from E,G 
###############################################################################
start=clock.time()

E=np.zeros(nel,dtype=np.float64)       # Young's modulus
G=np.zeros(nel,dtype=np.float64)       # shear modulus
lambdaa=np.zeros(nel,dtype=np.float64) # Lame's first parameter 

match layout:
 case 1:
  G[:]=G1
  E[:]=E1

 case 2:
  for iel in range(0,nel):
      if z_e[iel]<Lz/3 + aaa*(x_e[iel]-Lx/2) + bbb*(y_e[iel]-Ly/2):
         G[iel]=G1
         E[iel]=E1
      elif z_e[iel]<2*Lz/3+ aaa*(x_e[iel]-Lx/2) + bbb*(y_e[iel]-Ly/2):
         G[iel]=G2
         E[iel]=E2
      else:
         G[iel]=G1
         E[iel]=E1
    
lambdaa=G*(E-2*G)/(3*G-E) # Table 4.1 of Sadd's book, wiki

#nu=E/(2*G)-1     # https://en.wikipedia.org/wiki/Elastic_modulus, identical
nu=(E-2*G)/(2*G) # Table 4.1 of Sadd's book

print("     -> Young's modulus: E (m,M) %e %e " %(np.min(E),np.max(E)))
print("     -> Shear modulus: G (m,M) %e %e " %(np.min(G),np.max(G)))
print("     -> Poisson's ratio: nu (m,M) %e %e " %(np.min(nu),np.max(nu)))
print("     -> Lame's 1st parameter: lambdaa (m,M) %e %e " %(np.min(lambdaa),np.max(lambdaa)))

print("compute coords.: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
start=clock.time()

u_th=np.zeros(nn_V,dtype=np.float64)  
v_th=np.zeros(nn_V,dtype=np.float64)  
w_th=np.zeros(nn_V,dtype=np.float64)  
e_xx_th=np.zeros(nn_V,dtype=np.float64)  
e_yy_th=np.zeros(nn_V,dtype=np.float64)  
e_zz_th=np.zeros(nn_V,dtype=np.float64)  
e_xy_th=np.zeros(nn_V,dtype=np.float64)  
e_xz_th=np.zeros(nn_V,dtype=np.float64)  
e_yz_th=np.zeros(nn_V,dtype=np.float64)  
sigma_xx_th=np.zeros(nn_V,dtype=np.float64)  
sigma_yy_th=np.zeros(nn_V,dtype=np.float64)  
sigma_zz_th=np.zeros(nn_V,dtype=np.float64)  
sigma_xy_th=np.zeros(nn_V,dtype=np.float64)  
sigma_xz_th=np.zeros(nn_V,dtype=np.float64)  
sigma_yz_th=np.zeros(nn_V,dtype=np.float64)  

match experiment:
 case 0 | 1 :
   for i in range(0,nn_V):
       u_th[i],v_th[i],w_th[i]=\
       analytical_solution_displacement(x_V[i],y_V[i],z_V[i])
       e_xx_th[i],e_yy_th[i],e_zz_th[i],e_xy_th[i],e_xz_th[i],e_yz_th[i]=\
       analytical_solution_strain(x_V[i],y_V[i],z_V[i])
       sigma_xx_th[i],sigma_yy_th[i],sigma_zz_th[i],sigma_xy_th[i],sigma_xz_th[i],sigma_yz_th[i]=\
       analytical_solution_stress(x_V[i],y_V[i],z_V[i],lambdaa[0],G[0])
 case 2:
   for i in range(0,nn_V):
       u_th[i],v_th[i],w_th[i]=\
       analytical_solution_displacement(rad_V[i],theta_V[i],radius,sigma_H1,E1,nu1)
       e_xx_th[i],e_yy_th[i],e_zz_th[i],e_xy_th[i],e_xz_th[i],e_yz_th[i]=\
       analytical_solution_strain(rad_V[i],theta_V[i],radius,sigma_H1,E1,nu1)
       sigma_xx_th[i],sigma_yy_th[i],sigma_zz_th[i],sigma_xy_th[i],sigma_xz_th[i],sigma_yz_th[i]=\
       analytical_solution_stress(rad_V[i],theta_V[i],radius,sigma_H1)

print("compute analytical solution on nodes: %.3f s" % (clock.time()-start))

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

match experiment:
 case 10:
   # w=0 on all six faces
   for i in range(0,nn_V):
    # x=0 face 
    if abs(x_V[i]-   0)/Lx<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # x=Lx face
    if abs(x_V[i]-  Lx)/Lx<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # y=0 face 
    if abs(y_V[i]-   0)/Ly<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # y=Ly face
    if abs(y_V[i]-  Ly)/Ly<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # z=0 face 
    if abs(z_V[i]-   0)/Lz<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # z=Ly face
    if abs(z_V[i]-  Lz)/Lz<eps :
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

 case 0 | 1 | 2 :

    for i in range(0,nn_V):
        if abs(x_V[i])/Lx<eps or abs(x_V[i]-Lx)/Lx<eps or \
           abs(y_V[i])/Ly<eps or abs(y_V[i]-Ly)/Ly<eps or \
           abs(z_V[i])/Lz<eps or abs(z_V[i]-Lz)/Lz<eps:
           bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = u_th[i] 
           bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = v_th[i]
           bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = w_th[i]
        if wall_V[i]:
           bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = u_th[i] 
           bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = v_th[i]
           bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = w_th[i]

 case 3:
   2
 case 4:

   for i in range(0,nn_V):

    # point A
    if abs(x_V[i]-Lx/2)/Lx<eps and \
       abs(y_V[i]-   0)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = 0 # fix u
       #bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # point B
    if abs(x_V[i]-Lx/2)/Lx<eps and \
       abs(y_V[i]-  Ly)/Ly<eps and \
       abs(z_V[i]-Lz/2)/Lz<eps: 
       bc_fix[i*ndof_V+0] = True ; bc_val[i*ndof_V+0] = 0 # fix u
       #bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # point C
    #if abs(x_V[i]-   0)/Lx<eps and \
    #   abs(y_V[i]-Ly/2)/Ly<eps and \
    #   abs(z_V[i]-Lz/2)/Lz<eps: 
    #   bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
    #   bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w
    # point D
    #if abs(x_V[i]-  Lx)/Lx<eps and \
    #   abs(y_V[i]-Ly/2)/Ly<eps and \
    #   abs(z_V[i]-Lz/2)/Lz<eps: 
    #   bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
    #   bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # x=0 face 
    if abs(x_V[i]-   0)/Lx<eps :
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

    # x=Lx face
    if abs(x_V[i]-  Lx)/Lx<eps :
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0 # fix v
       bc_fix[i*ndof_V+2] = True ; bc_val[i*ndof_V+2] = 0 # fix w

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# fill II_matrix,JJ_matrix arrays for sparse storage of FEM matrix
###############################################################################
start=clock.time()

ndof_V_el=m_V*ndof_V

local_to_globalV=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1

bignb=nel*ndof_V_el**2

II_matrix=np.zeros(bignb,dtype=np.int32)    
JJ_matrix=np.zeros(bignb,dtype=np.int32)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndof_V_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndof_V_el):
            m2=local_to_globalV[jkk,iel]
            II_matrix[counter]=m1
            JJ_matrix[counter]=m2
            counter+=1

print("fill II_matrix,JJ_matrix arrays: %.3f s" % (clock.time()-start))

#==============================================================================
#==============================================================================
# time stepping
#==============================================================================
#==============================================================================

hx=Lx/nelx ; print('     -> hx=',hx)
hy=Ly/nely ; print('     -> hy=',hy)
hz=Lz/nelz ; print('     -> hz=',hz)
    
total_strain_xx=np.zeros(nn_V,dtype=np.float64)
total_strain_yy=np.zeros(nn_V,dtype=np.float64)
total_strain_zz=np.zeros(nn_V,dtype=np.float64)
total_strain_xy=np.zeros(nn_V,dtype=np.float64)
total_strain_xz=np.zeros(nn_V,dtype=np.float64)
total_strain_yz=np.zeros(nn_V,dtype=np.float64)

for istep in range(0,nstep):

    print("*******************************")
    print('istep=',istep)
    print("*******************************")

    ###############################################################################
    # build FE matrix | section 22.1 in fieldstone
    ###############################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    b_fem=np.zeros(Nfem,dtype=np.float64)
    B=np.zeros((6,ndof_V*m_V),dtype=np.float64)
    VV_matrix=np.zeros(bignb,dtype=np.float64) ; counter_K=0

    time_bcD=0
    time_bcN=0
    time_ass=0
    time_elmat=0

    for iel in range(0,nel):

        A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

        D=np.array([[2*G[iel]+lambdaa[iel],         lambdaa[iel],         lambdaa[iel],     0,     0,     0],
                    [         lambdaa[iel],2*G[iel]+lambdaa[iel],         lambdaa[iel],     0,     0,     0],
                    [         lambdaa[iel],         lambdaa[iel],2*G[iel]+lambdaa[iel],     0,     0,     0],
                    [                    0,                    0,                    0,G[iel],     0,     0],
                    [                    0,                    0,                    0,     0,G[iel],     0],
                    [                    0,                    0,                    0,     0,     0,G[iel]]],dtype=np.float64) 


        start1=clock.time()
        if bem==1:
           nodes=icon_V[:,iel]
           A_el=build_elemental_matrix(m_V,nq_per_el,ndof_V_el,D,dNdr_V,dNds_V,dNdt_V,nodes,x_V,y_V,z_V)
        else:
           for k in range(0,nq_per_el):
               jcb[0,0]=np.dot(dNdr_V[k,:],x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V[k,:],y_V[icon_V[:,iel]])
               jcb[0,2]=np.dot(dNdr_V[k,:],z_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V[k,:],x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V[k,:],y_V[icon_V[:,iel]])
               jcb[1,2]=np.dot(dNds_V[k,:],z_V[icon_V[:,iel]])
               jcb[2,0]=np.dot(dNdt_V[k,:],x_V[icon_V[:,iel]])
               jcb[2,1]=np.dot(dNdt_V[k,:],y_V[icon_V[:,iel]])
               jcb[2,2]=np.dot(dNdt_V[k,:],z_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)
               JxWq=np.linalg.det(jcb)*weightq
               dNdx_V=jcbi[0,0]*dNdr_V[k,:]+jcbi[0,1]*dNds_V[k,:]+jcbi[0,2]*dNdt_V[k,:]
               dNdy_V=jcbi[1,0]*dNdr_V[k,:]+jcbi[1,1]*dNds_V[k,:]+jcbi[1,2]*dNdt_V[k,:]
               dNdz_V=jcbi[2,0]*dNdr_V[k,:]+jcbi[2,1]*dNds_V[k,:]+jcbi[2,2]*dNdt_V[k,:]
               for i in range(0,m_V):
                   B[0,3*i  ]=dNdx_V[i]
                   B[1,3*i+1]=dNdy_V[i]
                   B[2,3*i+2]=dNdz_V[i]
                   B[3,3*i  ]=dNdy_V[i] ; B[3,3*i+1]=dNdx_V[i]
                   B[4,3*i  ]=dNdz_V[i] ; B[4,3*i+2]=dNdx_V[i]
                   B[5,3*i+1]=dNdz_V[i] ; B[5,3*i+2]=dNdy_V[i]
               # end for i
               A_el+=B.T.dot(D.dot(B))*JxWq
           # end for k
        time_elmat+=clock.time()-start1

        # impose stress boundary conditions
        start1=clock.time()

        match meshtype:
         case 1:
           if face1[iel]:
                 surf=hz*hy
                 b_el[ndof_V*0+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*3+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*4+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*7+0]+=sigma_H*surf*0.25
           if face2[iel]:
                 surf=hz*hy
                 b_el[ndof_V*1+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*2+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*5+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*6+0]-=sigma_H*surf*0.25
           if face3[iel]: 
                 surf=hx*hz
                 b_el[ndof_V*0+1]+=sigma_h*surf*0.25
                 b_el[ndof_V*1+1]+=sigma_h*surf*0.25
                 b_el[ndof_V*4+1]+=sigma_h*surf*0.25
                 b_el[ndof_V*5+1]+=sigma_h*surf*0.25
           if face4[iel]: 
                 surf=hx*hz
                 b_el[ndof_V*2+1]-=sigma_h*surf*0.25
                 b_el[ndof_V*3+1]-=sigma_h*surf*0.25
                 b_el[ndof_V*6+1]-=sigma_h*surf*0.25
                 b_el[ndof_V*7+1]-=sigma_h*surf*0.25
           if face5[iel]: 
                 surf=hx*hy
                 b_el[ndof_V*0+2]+=sigma_v*surf*0.25
                 b_el[ndof_V*1+2]+=sigma_v*surf*0.25
                 b_el[ndof_V*2+2]+=sigma_v*surf*0.25
                 b_el[ndof_V*3+2]+=sigma_v*surf*0.25
           if face6[iel]: 
                 surf=hx*hy
                 b_el[ndof_V*4+2]-=sigma_v*surf*0.25
                 b_el[ndof_V*5+2]-=sigma_v*surf*0.25
                 b_el[ndof_V*6+2]-=sigma_v*surf*0.25
                 b_el[ndof_V*7+2]-=sigma_v*surf*0.25

         case 2:
           if face1[iel]: # face is nodes 1-2-5-6
                 surf=hz*abs(y_V[icon_V[1,iel]]-y_V[icon_V[2,iel]])
                 b_el[ndof_V*1+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*2+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*5+0]+=sigma_H*surf*0.25
                 b_el[ndof_V*6+0]+=sigma_H*surf*0.25
           if face2[iel]: # face is nodes 1-2-5-6
                 surf=hz*abs(y_V[icon_V[1,iel]]-y_V[icon_V[2,iel]])
                 b_el[ndof_V*1+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*2+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*5+0]-=sigma_H*surf*0.25
                 b_el[ndof_V*6+0]-=sigma_H*surf*0.25
           if sigma_h>0: exit('sigma_h not implemented yet')
           if sigma_v>0: exit('sigma_v not implemented yet')
        time_bcN+=clock.time()-start1

        # apply boundary conditions # could be improved with local_to_globalV
        start1=clock.time()
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix[m1]: 
                   fixt=bc_val[m1]
                   ikk=ndof_V*k1+i1
                   Aref=A_el[ikk,ikk]
                   for jkk in range(0,m_V*ndof_V):
                       b_el[jkk]-=A_el[jkk,ikk]*fixt
                   #end for
                   A_el[ikk,:]=0.
                   A_el[:,ikk]=0.
                   A_el[ikk,ikk]=Aref
                   b_el[ikk]=Aref*fixt
                #end if
            #end for
        #end for
        time_bcD+=clock.time()-start1

        # assemble matrix A_fem and right hand side rhs
        start1=clock.time()
        if ass==2:
           for ikk in range(ndof_V_el):
              for jkk in range(ndof_V_el):
                  VV_matrix[counter_K]=A_el[ikk,jkk]
                  counter_K+=1
              m1=local_to_globalV[ikk,iel]
              b_fem[m1]+=b_el[ikk]
        else:
           VV_matrix[counter_K:counter_K+ndof_V_el**2]=A_el.flatten()
           counter_K+=ndof_V_el**2
           for ikk in range(ndof_V_el):
              m1=local_to_globalV[ikk,iel]
              b_fem[m1]+=b_el[ikk]

        time_ass+=clock.time()-start1

    #end for

    print('     -> time bound cond D:',time_bcD)
    print('     -> time bound cond N:',time_bcN)
    print('     -> time el mat:',time_elmat)
    print('     -> time assembly:',time_ass)

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ###############################################################################
    # solve system
    ###############################################################################
    start=clock.time()

    A_fem=sps.coo_matrix((VV_matrix,(II_matrix,JJ_matrix)),shape=(Nfem,Nfem)).tocsr()

    match solver:
     case 1:
      sol=sps.linalg.spsolve(A_fem,b_fem)
     case 2:
      sol,info=sps.linalg.cg(A_fem,b_fem,tol=1e-06)
      if info!=0 : print('pb with cg solver',info)
     case _:
      sol=np.zeros(Nfem,dtype=np.float64)  

    print("solve time: %.3f s" % (clock.time()-start))

    ###############################################################################
    # put solution into separate u,v,w arrays (nn_V long)
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
    # note that dNdr_V etc are stored on q points. I here therefore use dNdr__V
    # for some weird reason there seems to be an asymmetry/error in the strain
    # components (only nodal ones) but it's gone if I sample in the middle ...
    ###############################################################################
    start=clock.time()

    cc=np.zeros(nn_V,dtype=np.float64)  
    e_xx_n=np.zeros(nn_V,dtype=np.float64)  
    e_yy_n=np.zeros(nn_V,dtype=np.float64)  
    e_zz_n=np.zeros(nn_V,dtype=np.float64)  
    e_xy_n=np.zeros(nn_V,dtype=np.float64)  
    e_xz_n=np.zeros(nn_V,dtype=np.float64)  
    e_yz_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_xx_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_yy_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_zz_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_xy_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_xz_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_yz_n=np.zeros(nn_V,dtype=np.float64)  

    #for debugging
    #u=np.copy(x_V-Lx/2)/50
    #v=np.copy(y_V-Ly/2)/50
    #w=np.copy(z_V)

    r_V=np.array([-1, 1, 1,-1,-1, 1, 1,-1],np.float64)
    s_V=np.array([-1,-1, 1, 1,-1,-1, 1, 1],np.float64)
    t_V=np.array([ 0, 0, 0, 0, 1, 1, 1, 1],np.float64)

    for iel in range(0,nel):
        nodes=icon_V[:,iel]
        for i in range(0,m_V):
            inode=nodes[i]
            rq=r_V[i]
            sq=s_V[i]
            tq=t_V[i]

            exx,eyy,ezz,exy,exz,eyz=\
            compute_strain_tensor_components(rq,sq,tq,x_V,y_V,z_V,nodes,u,v,w)

            e_xx_n[inode]+=exx
            e_yy_n[inode]+=eyy
            e_zz_n[inode]+=ezz
            e_xy_n[inode]+=exy
            e_xz_n[inode]+=exz
            e_yz_n[inode]+=eyz

            #q[inode]+=-(lambdaa[iel]+2./3.*G[iel])*(exx+eyy+ezz)

            sxx=lambdaa[iel]*(exx+eyy+ezz)+2*G[iel]*exx
            syy=lambdaa[iel]*(exx+eyy+ezz)+2*G[iel]*eyy
            szz=lambdaa[iel]*(exx+eyy+ezz)+2*G[iel]*ezz
            sxy=2*G[iel]*exy
            sxz=2*G[iel]*exz
            syz=2*G[iel]*eyz

            sigma_xx_n[inode]+=sxx
            sigma_yy_n[inode]+=syy
            sigma_zz_n[inode]+=szz
            sigma_xy_n[inode]+=sxy
            sigma_xz_n[inode]+=sxz
            sigma_yz_n[inode]+=syz

            cc[inode]+=1.
        #end for
    #end for

    e_xx_n[:]/=cc[:]
    e_yy_n[:]/=cc[:]
    e_zz_n[:]/=cc[:]
    e_xy_n[:]/=cc[:]
    e_xz_n[:]/=cc[:]
    e_yz_n[:]/=cc[:]

    sigma_xx_n[:]/=cc[:]
    sigma_yy_n[:]/=cc[:]
    sigma_zz_n[:]/=cc[:]
    sigma_xy_n[:]/=cc[:]
    sigma_xz_n[:]/=cc[:]
    sigma_yz_n[:]/=cc[:]

    e_n=effective(e_xx_n,e_yy_n,e_zz_n,e_xy_n,e_xz_n,e_yz_n)

    sigma_n=effective(sigma_xx_n,sigma_yy_n,sigma_zz_n,sigma_xy_n,sigma_xz_n,sigma_yz_n)

    print("     -> e_xx_n (m,M) %e %e " %(np.min(e_xx_n),np.max(e_xx_n)))
    print("     -> e_yy_n (m,M) %e %e " %(np.min(e_yy_n),np.max(e_yy_n)))
    print("     -> e_zz_n (m,M) %e %e " %(np.min(e_zz_n),np.max(e_zz_n)))
    print("     -> e_xy_n (m,M) %e %e " %(np.min(e_xy_n),np.max(e_xy_n)))
    print("     -> e_xz_n (m,M) %e %e " %(np.min(e_xz_n),np.max(e_xz_n)))
    print("     -> e_yz_n (m,M) %e %e " %(np.min(e_yz_n),np.max(e_yz_n)))
    print("     -> sigma_xx_n (m,M) %e %e " %(np.min(sigma_xx_n),np.max(sigma_xx_n)))
    print("     -> sigma_yy_n (m,M) %e %e " %(np.min(sigma_yy_n),np.max(sigma_yy_n)))
    print("     -> sigma_zz_n (m,M) %e %e " %(np.min(sigma_zz_n),np.max(sigma_zz_n)))
    print("     -> sigma_xy_n (m,M) %e %e " %(np.min(sigma_xy_n),np.max(sigma_xy_n)))
    print("     -> sigma_xz_n (m,M) %e %e " %(np.min(sigma_xz_n),np.max(sigma_xz_n)))
    print("     -> sigma_yz_n (m,M) %e %e " %(np.min(sigma_yz_n),np.max(sigma_yz_n)))

    total_strain_xx+=e_xx_n
    total_strain_yy+=e_yy_n
    total_strain_zz+=e_zz_n
    total_strain_xy+=e_xy_n
    total_strain_xz+=e_xz_n
    total_strain_yz+=e_yz_n

    total_strain=effective(total_strain_xx,total_strain_yy,total_strain_zz,\
                           total_strain_xy,total_strain_xz,total_strain_yz)

    if debug:
       np.savetxt('strain_nodal.ascii',np.array([x_V,y_V,z_V,e_xx_n,e_yy_n,e_zz_n,e_xy_n,e_xz_n,e_yz_n]).T,\
                  header='# x,y,z,exx,eyy,ezz,exy,exz,eyz')

    print("compute nodal strain & stress components: %.3f s" % (clock.time()-start))

    ###############################################################################
    # compute principal stresses & directions on the nodes
    ###############################################################################
    start=clock.time()

    sigma_p1_n=np.zeros(nn_V,dtype=np.float64) # principal stress 1 
    sigma_v1x_n=np.zeros(nn_V,dtype=np.float64) # eigenvector of ps 1  
    sigma_v1y_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v1z_n=np.zeros(nn_V,dtype=np.float64)  

    sigma_p2_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v2x_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v2y_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v2z_n=np.zeros(nn_V,dtype=np.float64)  

    sigma_p3_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v3x_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v3y_n=np.zeros(nn_V,dtype=np.float64)  
    sigma_v3z_n=np.zeros(nn_V,dtype=np.float64)  

    for i in range(0,nn_V):
        sigma_p1_n[i],sigma_v1x_n[i],sigma_v1y_n[i],sigma_v1z_n[i],\
        sigma_p2_n[i],sigma_v2x_n[i],sigma_v2y_n[i],sigma_v2z_n[i],\
        sigma_p3_n[i],sigma_v3x_n[i],sigma_v3y_n[i],sigma_v3z_n[i]=\
        compute_principal_stresses(sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i],\
                                   sigma_xy_n[i],sigma_xz_n[i],sigma_yz_n[i])

    print("compute principal stresses: %.3f s" % (clock.time()-start))

    ###############################################################################
    # retrieve elemental strain tensor components 
    ###############################################################################
    start=clock.time()

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

    rq=0
    sq=0
    tq=0
    for iel in range(0,nel):
        nodes=icon_V[:,iel]
        e_xx_e[iel],e_yy_e[iel],e_zz_e[iel],e_xy_e[iel],e_xz_e[iel],e_yz_e[iel]=\
        compute_strain_tensor_components(rq,sq,tq,x_V,y_V,z_V,nodes,u,v,w)

    e_e=effective(e_xx_e,e_yy_e,e_zz_e,e_xy_e,e_xz_e,e_yz_e)

    print("     -> e_xx_e (m,M) %e %e " %(np.min(e_xx_e),np.max(e_xx_e)))
    print("     -> e_yy_e (m,M) %e %e " %(np.min(e_yy_e),np.max(e_yy_e)))
    print("     -> e_zz_e (m,M) %e %e " %(np.min(e_zz_e),np.max(e_zz_e)))
    print("     -> e_xy_e (m,M) %e %e " %(np.min(e_xy_e),np.max(e_xy_e)))
    print("     -> e_xz_e (m,M) %e %e " %(np.min(e_xz_e),np.max(e_xz_e)))
    print("     -> e_yz_e (m,M) %e %e " %(np.min(e_yz_e),np.max(e_yz_e)))

    if debug:
       np.savetxt('strain_elemental.ascii',\
                  np.array([x_e,y_e,z_e,e_xx_e,e_yy_e,e_zz_e,e_xy_e,e_xz_e,e_yz_e]).T,\
                  header='# x,y,z,exx,eyy,ezz,exy,exz,eyz')

    print("compute elemental strain components: %.3f s" % (clock.time()-start))

    ###############################################################################
    # compute elemental stress
    ###############################################################################
    start=clock.time()

    p=-(lambdaa[:]+2./3.*G[:])*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])

    sigma_xx_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_xx_e[:] 
    sigma_yy_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_yy_e[:] 
    sigma_zz_e[:]=lambdaa[:]*(e_xx_e[:]+e_yy_e[:]+e_zz_e[:])+2*G[:]*e_zz_e[:] 
    sigma_xy_e[:]=2*G[:]*e_xy_e[:]
    sigma_xz_e[:]=2*G[:]*e_xz_e[:]
    sigma_yz_e[:]=2*G[:]*e_yz_e[:]

    sigma_e=effective(sigma_xx_e,sigma_yy_e,sigma_zz_e,sigma_xy_e,sigma_xz_e,sigma_yz_e)

    print("     -> sigma_xx_e (m,M) %e %e " %(np.min(sigma_xx_e),np.max(sigma_xx_e)))
    print("     -> sigma_yy_e (m,M) %e %e " %(np.min(sigma_yy_e),np.max(sigma_yy_e)))
    print("     -> sigma_zz_e (m,M) %e %e " %(np.min(sigma_zz_e),np.max(sigma_zz_e)))
    print("     -> sigma_xy_e (m,M) %e %e " %(np.min(sigma_xy_e),np.max(sigma_xy_e)))
    print("     -> sigma_xz_e (m,M) %e %e " %(np.min(sigma_xz_e),np.max(sigma_xz_e)))
    print("     -> sigma_yz_e (m,M) %e %e " %(np.min(sigma_yz_e),np.max(sigma_yz_e)))

    print("compute elemental stress components: %.3f s" % (clock.time()-start))

    ###############################################################################
    # compute principal stresses & directions on the elements 
    ###############################################################################
    start=clock.time()

    sigma_p1_e=np.zeros(nel,dtype=np.float64)  
    sigma_p2_e=np.zeros(nel,dtype=np.float64)  
    sigma_p3_e=np.zeros(nel,dtype=np.float64)  
    sigma_v1x_e=np.zeros(nel,dtype=np.float64)  
    sigma_v2x_e=np.zeros(nel,dtype=np.float64)  
    sigma_v3x_e=np.zeros(nel,dtype=np.float64)  
    sigma_v1y_e=np.zeros(nel,dtype=np.float64)  
    sigma_v2y_e=np.zeros(nel,dtype=np.float64)  
    sigma_v3y_e=np.zeros(nel,dtype=np.float64)  
    sigma_v1z_e=np.zeros(nel,dtype=np.float64)  
    sigma_v2z_e=np.zeros(nel,dtype=np.float64)  
    sigma_v3z_e=np.zeros(nel,dtype=np.float64)  

    for i in range(0,nel):
        sigma_p1_e[i],sigma_v1x_e[i],sigma_v1y_e[i],sigma_v1z_e[i],\
        sigma_p2_e[i],sigma_v2x_e[i],sigma_v2y_e[i],sigma_v2z_e[i],\
        sigma_p3_e[i],sigma_v3x_e[i],sigma_v3y_e[i],sigma_v3z_e[i]=\
        compute_principal_stresses(sigma_xx_e[i],sigma_yy_e[i],sigma_zz_e[i],\
                                   sigma_xy_e[i],sigma_xz_e[i],sigma_yz_e[i])

    print("compute principal stresses: %.3f s" % (clock.time()-start))

    ###############################################################################
    # evolve mesh (very crude for now)
    ###############################################################################

    disp=np.sqrt(u**2+v**2+w**2)
    dd=0.01*min(hx,hz)/max(disp) # in %
    dd=0

    x_V+=u*dd
    y_V+=v*dd
    z_V+=w*dd

    for i in range(nn_V):
        rad_V[i]=np.sqrt((x_V[i]-Lx/2)**2+(y_V[i]-Ly/2)**2)
        theta_V[i]=np.arctan2((y_V[i]-Ly/2),(x_V[i]-Lx/2))

    ###############################################################################
    # plot of solution
    # note the specific ordering of tensor terms xx,yy,zz,xy,yz,xz
    ###############################################################################
    start=clock.time()
       
    vtufile=open('solution'+str(istep)+'.vtu',"w")
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
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(u_th[i],v_th[i],w_th[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='displacement (error)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(u[i]-u_th[i],v[i]-v_th[i],w[i]-w_th[i]))
    vtufile.write("</DataArray>\n")


    #--
    #vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    #q.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
    rad_V.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
    theta_V.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (effective)' Format='ascii'> \n")
    e_n.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strain' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e %e %e %e\n" %(e_xx_n[i],e_yy_n[i],e_zz_n[i],\
                                              e_xy_n[i],e_yz_n[i],e_xz_n[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strain (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e %e %e %e\n" %(e_xx_th[i],e_yy_th[i],e_zz_th[i],e_xy_th[i],e_yz_th[i],e_xz_th[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='total strain (effective)' Format='ascii'> \n")
    total_strain.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='total strain (tensor)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e %e %e %e\n" %(total_strain_xx[i],total_strain_yy[i],total_strain_zz[i],\
                                              total_strain_xy[i],total_strain_yz[i],total_strain_xz[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='stress (effective)' Format='ascii'> \n")
    sigma_n.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='stress' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e %e %e %e\n" %(sigma_xx_n[i],sigma_yy_n[i],sigma_zz_n[i],\
                                              sigma_xy_n[i],sigma_yz_n[i],sigma_xz_n[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='stress (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e %e %e %e\n" %(sigma_xx_th[i],sigma_yy_th[i],sigma_zz_th[i],sigma_xy_th[i],sigma_yz_th[i],sigma_xz_th[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='bc fix u' Format='ascii'> \n")
    for i in range(0,nn_V):
        val=int(bc_fix[3*i+0])
        vtufile.write("%d \n" % val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='bc fix v' Format='ascii'> \n")
    for i in range(0,nn_V):
        val=int(bc_fix[3*i+1])
        vtufile.write("%d \n" % val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='bc fix w' Format='ascii'> \n")
    for i in range(0,nn_V):
        val=int(bc_fix[3*i+2])
        vtufile.write("%d \n" % val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='wall' Format='ascii'> \n")
    for i in range(0,nn_V):
        val=int(wall_V[i]) 
        vtufile.write("%d \n" % val)
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='E' Format='ascii'> \n")
    E.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='G' Format='ascii'> \n")
    G.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='nu' Format='ascii'> \n")
    nu.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='lambdaa' Format='ascii'> \n")
    lambdaa.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    p.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='r' Format='ascii'> \n")
    r_e.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='theta' Format='ascii'> \n")
    theta_e.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (effective)' Format='ascii'> \n")
    e_e.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")

    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strain' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%e %e %e %e %e %e\n" %(e_xx_e[iel],e_yy_e[iel],e_zz_e[iel],\
                                              e_xy_e[iel],e_yz_e[iel],e_xz_e[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='stress (effective)' Format='ascii'> \n")
    sigma_e.tofile(vtufile,sep=' ',format='%.4e') ; vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='stress (tensor)' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%e %e %e %e %e %e\n" %(sigma_xx_e[iel],sigma_yy_e[iel],sigma_zz_e[iel],\
                                              sigma_xy_e[iel],sigma_yz_e[iel],sigma_xz_e[iel]))
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

    print("export to vtu: %.3f s" % (clock.time()-start))

    ###########################################################################
    # export wall nodes to ascii file
    ###########################################################################

    np.savetxt('wall_displacement'+str(istep)+'.ascii',\
               np.array([theta_V[wall_V],z_V[wall_V],\
                         rad_V[wall_V],disp[wall_V],e_n[wall_V],sigma_n[wall_V]]).T,header='# theta,z,r,disp',fmt='%.4e')

    #INSERT matplotlib commands to generate colormaps

    ###########################################################################
    # export analytical solution on line 
    ###########################################################################

    online=np.logical_and(  (abs(y_V-Ly/2)/Ly<eps) , (abs(z_V-Lz/2)/Lz<eps) )

    np.savetxt('line_displacement.ascii',np.array([x_V[online],y_V[online],z_V[online],\
                                                   u[online],v[online],w[online],\
                                                   e_xx_n[online],e_yy_n[online],e_xy_n[online],\
                                                   sigma_xx_n[online],sigma_yy_n[online],sigma_xy_n[online]  ]).T)

    np.savetxt('line_displacement_th.ascii',np.array([x_V[online],y_V[online],z_V[online],\
                                                      u_th[online],v_th[online],w_th[online],\
                                                      e_xx_th[online],e_yy_th[online],e_xy_th[online],\
                                                      sigma_xx_th[online],sigma_yy_th[online],sigma_xy_th[online]  ]).T)
#end of time stepping loop

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
