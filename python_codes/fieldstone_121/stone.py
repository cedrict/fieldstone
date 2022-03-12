import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix 
import time as time
from numpy import linalg as LA
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

Rgas=8.3145
Tkelvin=273
cm=0.01
year=3600*24*365.25
MPa=1e6

#------------------------------------------------------------------------------
# experiment=0 (F. Garel & C. Thoraval)

a0=4.4e8
b0=-5.26e4
a1=2.11e-2
b1=1.74e-4
a2=-41.8
b2=4.21e-2
c2=-1.14e-5

#------------------------------------------------------------------------------
# experiment=1 (project 'rheology')



#------------------------------------------------------------------------------
# experiment=2 (projet 'inclusion')
#
#    +-----------------------------------------------------+
#    |                                                     |
#    |                      /------\                       |
#    |                     /        \                      |
#    |          mat=1      | mat=2  |                      |
#    |           an        \  cpx   /                      |
#    |                      \------/                       |   
#    |                                                     |
#    +-----------------------------------------------------+

A_cpx=6.80220477
Q_cpx=534e3
n_cpx=5.52

A_ab=398.1
n_ab=3
Q_ab=356e3

cohesion=[20e6,20e6]
phi=[30/180*np.pi,30/180*np.pi]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def viscosity(x,y,ee,T,imat,iter):
    if experiment==0:
       if iter==0:
          xxxx
       else:
          A0=a0+b0*T
          A1=a1+b1*T
          A2=a2+b2*T+c2*T**2
          val = A0/2/ee*(1+np.tanh(A1*(np.log10(ee)-A2)))
          #val=1e21

    if experiment==1:
       val=1e21

    if experiment==2:
       sr=max(1e-20,ee) # minimum strain rate
       if imat==1:
          val=A_ab**(-1/n_ab)*sr**(1/n_ab-1)*np.exp(Q_ab/Rgas/T/n_ab)*MPa
       else:
          val=A_cpx**(-1./n_cpx)*sr**(1./n_cpx-1)*np.exp(Q_cpx/Rgas/T/n_cpx)*MPa

    return val

#------------------------------------------------------------------------------
#   Vspace=Q2     Pspace=Q1       
#
#  3----6----2   3---------2  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  7----8----5   |         |  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  0----4----1   0---------1  
#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------stone 121----------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

experiment=2 

if experiment==0:
   Lx=1
   Ly=1
   nelx=12
   nely=12
   v0=1e-15*Ly # bc velocity so that shear strain rate is 10^-15
   #time stepping
   nstep=1
   CFL_nb=0
   depth=30e3 # 30 to 60km
   background_pressure=3000*9.81*depth
   background_temperature=400+273 #between 300 and 500C
   nmarker_per_dim=5
   avrg=3
   #nonlinear iterations parameters
   niter=10
   tol=1e-2

if experiment==2:
   Lx=4e-2 # horizontal extent of the domain in m 
   Ly=1e-2 # vertical extent of the domain in m
   nelx = 80             #number of elements in horizontal direction
   nely = int(nelx*Ly/Lx) #number of elements in vertical direction
   v0=1e-15*Ly # bc velocity so that shear strain rate is 10^-15
   x_inclusion=Lx/2
   y_inclusion=Ly/2
   R_inclusion=Ly/5
   #time stepping
   nstep=100
   CFL_nb=0.25
   depth=30e3 # 30 to 60km
   background_pressure=3000*9.81*depth
   background_temperature=400+273 #between 300 and 500C
   pf=0.9*background_pressure
   nmarker_per_dim=5
   avrg=3
   #nonlinear iterations parameters
   niter=20
   tol=1e-2

#################################################################
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NP=(nelx+1)*(nely+1)

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

#quadrature points
nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx # size of element
hy=Ly/nely

gx=0 #gravity vector
gy=0

convfile=open('conv.ascii',"w")

nmarker_per_element=nmarker_per_dim**2
nmarker=nel*nmarker_per_element

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("------------------------------")

rVnodes=[-1,+1,+1,-1, 0,+1, 0,-1,0]
sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]

eta_ref=1e19 # purely numerical parameter - do not change

pnormalise=False 

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

iconV=np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

print("velocity grid points: %.3f s" % (time.time() - start))

#################################################################
# pressure connectivity array
#################################################################
start = time.time()

xP=np.zeros(NP,dtype=np.float64)     # x coordinates
yP=np.zeros(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

counter = 0
for j in range(0, nely+1):
    for i in range(0, nelx+1):
        xP[counter]=i*Lx/float(nelx)
        yP[counter]=j*Ly/float(nely)
        counter += 1

print("pressure connectivity & nodes: %.3f s" % (time.time() - start))

#################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
#################################################################
start = time.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV=np.zeros(mV,dtype=np.float64)       
dNNNVdr=np.zeros(mV,dtype=np.float64)   
dNNNVds= np.zeros(mV,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
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
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.8e " %(area.sum()))
print("     -> total area anal %.8e " %(Lx*Ly))

print("compute elements areas: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

eps=1.e-10

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    #left boundary 
    if xV[i]<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # vy
    #right boundary 
    if xV[i]/Lx>1-eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # vy
    #bottom boundary 
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV  ] = -v0 # vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   # vy
    #top boundary 
    if yV[i]/Ly>1-eps:
       bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV  ] = +v0 # vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   # vy
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# swarm (=all the particles) setup
#################################################################
start = time.time()

swarm_x=np.empty(nmarker,dtype=np.float64)        # x coordinates   
swarm_y=np.empty(nmarker,dtype=np.float64)        # y coordinates 
swarm_mat=np.empty(nmarker,dtype=np.int8)         # type of material 
swarm_paint=np.empty(nmarker,dtype=np.float64) 
swarm_r=np.empty(nmarker,dtype=np.float64)        # reduced coordinates r
swarm_s=np.empty(nmarker,dtype=np.float64)        # reduced coordinates s
swarm_u=np.empty(nmarker,dtype=np.float64)        # velocity x
swarm_v=np.empty(nmarker,dtype=np.float64)        # velocity y
swarm_exx=np.empty(nmarker,dtype=np.float64)      # strain rate xx
swarm_exy=np.empty(nmarker,dtype=np.float64)      # strain rate yy
swarm_eyy=np.empty(nmarker,dtype=np.float64)      # strain rate xy
swarm_ee=np.empty(nmarker,dtype=np.float64)       # effective strain rate
swarm_strainxx=np.empty(nmarker,dtype=np.float64) # strain xx
swarm_strainxy=np.empty(nmarker,dtype=np.float64) # strain yy
swarm_strainyy=np.empty(nmarker,dtype=np.float64) # strain xy
swarm_tauxx=np.empty(nmarker,dtype=np.float64)    # dev stress xx
swarm_tauxy=np.empty(nmarker,dtype=np.float64)    # dev stress yy
swarm_tauyy=np.empty(nmarker,dtype=np.float64)    # dev stress xy
swarm_iel=np.empty(nmarker,dtype=np.int32)        # element identity
swarm_eta=np.empty(nmarker,dtype=np.float64)      # viscosity
swarm_p=np.empty(nmarker,dtype=np.float64)        # pressure
swarm_yield=np.empty(nmarker,dtype=np.float64)    # yield value

counter=0
for iel in range(0,nel):
    x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
    x2=xV[iconV[1,iel]] ; y2=yV[iconV[1,iel]]
    x3=xV[iconV[2,iel]] ; y3=yV[iconV[2,iel]]
    x4=xV[iconV[3,iel]] ; y4=yV[iconV[3,iel]]
    for j in range(0,nmarker_per_dim):
        for i in range(0,nmarker_per_dim):
            r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
            s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
            swarm_r[counter]=r
            swarm_s[counter]=s
            N1=0.25*(1-r)*(1-s)
            N2=0.25*(1+r)*(1-s)
            N3=0.25*(1+r)*(1+s)
            N4=0.25*(1-r)*(1+s)
            swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
            swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
            counter+=1
        #end for 
    #end for 
#end for 

print("swarm setup: %.3f s" % (time.time() - start))

#################################################################
# assign material id to markers 
#################################################################
start = time.time()

if experiment==0:
   for im in range (0,nmarker):
       swarm_mat[im]=1

if experiment==1:
   swarm_mat[:]=1

if experiment==2:
   for im in range (0,nmarker):
       swarm_mat[im]=1
       if (swarm_x[im]-x_inclusion)**2+(swarm_y[im]-y_inclusion)**2<R_inclusion**2:
          swarm_mat[im]=2

print("assigning mat to swarm: %.3f s" % (time.time() - start))

#################################################################
# paint markers 
#################################################################
start = time.time()

for im in range (0,nmarker):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))**4
#end for 

print("paint swarm: %.3f s" % (time.time() - start))

###################################################################################################
###################################################################################################
# time stepping loop 
###################################################################################################
###################################################################################################

total_time=0
    
umem=np.zeros(NV,dtype=np.float64)      
vmem=np.zeros(NV,dtype=np.float64)    
u   =np.zeros(NV,dtype=np.float64)   
v   =np.zeros(NV,dtype=np.float64)    
exx =np.zeros(NV,dtype=np.float64)  
eyy =np.zeros(NV,dtype=np.float64)  
exy =np.zeros(NV,dtype=np.float64)  

for istep in range(0,nstep):

    print ('||******************************************************||')
    print ('||    istep= %i ' %istep)
    print ('||******************************************************||')

    #################################################################
    # localise markers 
    #################################################################
    start = time.time()

    nmarker_in_element=np.zeros(nel,dtype=np.int16)
    list_of_markers_in_element=np.zeros((2*nmarker_per_element,nel),dtype=np.int32)
    for im in range(0,nmarker):
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        iel=nelx*(iely)+ielx
        list_of_markers_in_element[nmarker_in_element[iel],iel]=im
        nmarker_in_element[iel]+=1
        swarm_iel[im]=iel
    #end for

    print("     localise markers: %.3f s" % (time.time() - start))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # non linear iterations
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for iter in range(0,niter):

        print('     ----------------------------------')
        print('     ----- iteration------',iter,'----------')
        print('     ----------------------------------')

        #################################################################
        # compute elemental averagings 
        #################################################################
        start = time.time()
        eta_elemental=np.zeros(nel,dtype=np.float64)
        eta_nodal=np.zeros(NP,dtype=np.float64)
        nodal_counter=np.zeros(NP,dtype=np.float64)
        for im in range(0,nmarker):
            iel=swarm_iel[im]
            N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
            N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
            N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
            N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
            nodal_counter[iconP[0,iel]]+=N1
            nodal_counter[iconP[1,iel]]+=N2
            nodal_counter[iconP[2,iel]]+=N3
            nodal_counter[iconP[3,iel]]+=N4
            #compute viscosity on marker
            NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
            swarm_exx[im]=sum(NNNV[0:mV]*exx[iconV[0:mV,iel]])
            swarm_eyy[im]=sum(NNNV[0:mV]*eyy[iconV[0:mV,iel]])
            swarm_exy[im]=sum(NNNV[0:mV]*exy[iconV[0:mV,iel]])
            swarm_ee[im]=np.sqrt(0.5*(swarm_exx[im]**2+swarm_eyy[im]**2+2*swarm_exy[im]**2) ) 
            swarm_eta[im]=viscosity(swarm_x[im],swarm_y[im],swarm_ee[im],background_temperature,swarm_mat[im],iter)

            if abs(avrg)==1 : # arithmetic
               eta_elemental[iel]     +=swarm_eta[im]
               eta_nodal[iconP[0,iel]]+=swarm_eta[im]*N1
               eta_nodal[iconP[1,iel]]+=swarm_eta[im]*N2
               eta_nodal[iconP[2,iel]]+=swarm_eta[im]*N3
               eta_nodal[iconP[3,iel]]+=swarm_eta[im]*N4
            if abs(avrg)==2: # geometric
               eta_elemental[iel]     +=np.log10(swarm_eta[im])
               eta_nodal[iconP[0,iel]]+=np.log10(swarm_eta[im])*N1
               eta_nodal[iconP[1,iel]]+=np.log10(swarm_eta[im])*N2
               eta_nodal[iconP[2,iel]]+=np.log10(swarm_eta[im])*N3
               eta_nodal[iconP[3,iel]]+=np.log10(swarm_eta[im])*N4
            if abs(avrg)==3: # harmonic
               eta_elemental[iel]     +=1/swarm_eta[im]
               eta_nodal[iconP[0,iel]]+=1/swarm_eta[im]*N1
               eta_nodal[iconP[1,iel]]+=1/swarm_eta[im]*N2
               eta_nodal[iconP[2,iel]]+=1/swarm_eta[im]*N3
               eta_nodal[iconP[3,iel]]+=1/swarm_eta[im]*N4
        #end for
        if abs(avrg)==1:
           eta_nodal/=nodal_counter
           eta_elemental[:]/=nmarker_in_element[:]
        if abs(avrg)==2:
           eta_nodal[:]=10.**(eta_nodal[:]/nodal_counter[:])
           eta_elemental[:]=10.**(eta_elemental[:]/nmarker_in_element[:])
        if abs(avrg)==3:
           eta_nodal[:]=nodal_counter[:]/eta_nodal[:]
           eta_elemental[:]=nmarker_in_element[:]/eta_elemental[:]

        print("          -> nmarker_in_elt(m,M) %.5e %.5e " %(np.min(nmarker_in_element),np.max(nmarker_in_element)))
        print("          -> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))
        print("          -> eta_nodal     (m,M) %.5e %.5e " %(np.min(eta_nodal),np.max(eta_nodal)))

        print("     markers onto grid: %.3f s" % (time.time() - start))

        #################################################################
        # build FE matrix
        # [ K G ][u]=[f]
        # [GT 0 ][p] [h]
        #################################################################
        start = time.time()

        if pnormalise:
           A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
        else:
           A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)

        f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
        h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
        constr  = np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
        b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
        N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
        NNNV    = np.zeros(mV,dtype=np.float64)            # shape functions V
        NNNP    = np.zeros(mP,dtype=np.float64)            # shape functions P
        dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
        dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
        dNNNVdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
        dNNNVds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
        c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

        for iel in range(0,nel):

            f_el =np.zeros((mV*ndofV),dtype=np.float64)
            K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
            G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
            h_el=np.zeros((mP*ndofP),dtype=np.float64)
            NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   

            for iq in range(0,nqperdim):
                for jq in range(0,nqperdim):

                    # position & weight of quad. point
                    rq=qcoords[iq]
                    sq=qcoords[jq]
                    weightq=qweights[iq]*qweights[jq]

                    NNNV[0:mV]=NNV(rq,sq)
                    dNNNVdr[0:mV]=dNNVdr(rq,sq)
                    dNNNVds[0:mV]=dNNVds(rq,sq)
                    NNNP[0:mP]=NNP(rq,sq)

                    # calculate jacobian matrix
                    jcb=np.zeros((2,2),dtype=np.float64)
                    for k in range(0,mV):
                        jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                        jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                        jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                        jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                    jcob = np.linalg.det(jcb)
                    jcbi = np.linalg.inv(jcb)

                    # compute dNdx & dNdy
                    #xq=0.0
                    #yq=0.0
                    for k in range(0,mV):
                        #xq+=NNNV[k]*xV[iconV[k,iel]]
                        #yq+=NNNV[k]*yV[iconV[k,iel]]
                        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

                    # construct b_mat matrix
                    for i in range(0,mV):
                        b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                                 [0.        ,dNNNVdy[i]],
                                                 [dNNNVdy[i],dNNNVdx[i]]]

                    etaq=eta_elemental[iel]

                    # compute elemental a_mat matrix
                    K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                    # compute elemental rhs vector
                    #for i in range(0,mV):
                    #    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*density(xq,yq)
                    #    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*density(xq,yq)

                    for i in range(0,mP):
                        N_mat[0,i]=NNNP[i]
                        N_mat[1,i]=NNNP[i]
                        N_mat[2,i]=0.

                    G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                    #NNNNP[:]+=NNNP[:]*jcob*weightq

                # end for jq
            # end for iq

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
                #end for
            #end for

            G_el*=eta_ref/Ly
            h_el*=eta_ref/Ly

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
                #constr[m2]+=NNNNP[k2]
                #if pnormalise:
                #   A_sparse[Nfem,NfemV+m2]+=constr[m2]
                #   A_sparse[NfemV+m2,Nfem]+=constr[m2]
            #end for
        #end for iel

        print("     build FE matrix: %.3f s" % (time.time() - start))

        ######################################################################
        # assemble K, G, GT, f, h into A and rhs
        ######################################################################
        start = time.time()

        if pnormalise:
           rhs=np.zeros(Nfem+1,dtype=np.float64) # right hand side of Ax=b
        else:
           rhs=np.zeros(Nfem,dtype=np.float64)   # right hand side of Ax=b

        sparse_matrix=A_sparse.tocsr()

        rhs[0:NfemV]=f_rhs
        rhs[NfemV:Nfem]=h_rhs

        print("     assemble blocks: %.3f s" % (time.time() - start))

        ######################################################################
        # solve system
        ######################################################################
        start = time.time()

        sol=sps.linalg.spsolve(sparse_matrix,rhs)

        print("     solve time: %.3f s" % (time.time() - start))

        ######################################################################
        # put solution into separate x,y velocity arrays
        ######################################################################
        start = time.time()

        u,v=np.reshape(sol[0:NfemV],(NV,2)).T
        p=sol[NfemV:Nfem]*(eta_ref/Ly)

        print("          -> u (m,M) %.4e %.4e (cm/year)" %(np.min(u)/cm*year,np.max(u)/cm*year))
        print("          -> v (m,M) %.4e %.4e (cm/year)" %(np.min(v)/cm*year,np.max(v)/cm*year))
        print("          -> p (m,M) %.4e %.4e (Pa)     " %(np.min(p),np.max(p)))

        #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
        #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

        print("     split vel into u,v: %.3f s" % (time.time() - start))

        ######################################################################
        #normalise pressure
        ######################################################################
        start = time.time()

        avrg_p=0
        for iel in range(0,nel):
            for iq in range(0,nqperdim):
                for jq in range(0,nqperdim):
                    rq=qcoords[iq]
                    sq=qcoords[jq]
                    weightq=qweights[iq]*qweights[jq]
                    NNNP[0:mP]=NNP(rq,sq)
                    pq=NNNP.dot(p[iconP[0:mP,iel]])
                    jcob=hx*hy/4
                    avrg_p+=pq*jcob*weightq

        p-=avrg_p

        print("          -> p (m,M) %.4e %.4e (Pa)     " %(np.min(p),np.max(p)))
            
        print("     pressure normalisation: %.3f s" % (time.time() - start))

        ######################################################################
        # compute nodal strainrate 
        ######################################################################
        start = time.time()
 
        exx = np.zeros(NV,dtype=np.float64)  
        eyy = np.zeros(NV,dtype=np.float64)  
        exy = np.zeros(NV,dtype=np.float64)  
        ee  = np.zeros(NV,dtype=np.float64)  
        ccc = np.zeros(NV,dtype=np.float64)  
 
        for iel in range(0,nel):
            for k in range(0,mV):
                rq = rVnodes[k]
                sq = sVnodes[k]
                inode=iconV[k,iel]
                NNNV[0:mV]=NNV(rq,sq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq)
                dNNNVds[0:mV]=dNNVds(rq,sq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                jcbi=np.linalg.inv(jcb)
                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                ccc[inode]+=1
                exx[inode]+=dNNNVdx.dot(u[iconV[:,iel]])
                eyy[inode]+=dNNNVdy.dot(v[iconV[:,iel]])
                exy[inode]+=dNNNVdx.dot(v[iconV[:,iel]])*0.5+dNNNVdy.dot(u[iconV[:,iel]])*0.5
            #end for
        #end for
        exx[:]/=ccc[:]
        eyy[:]/=ccc[:]
        exy[:]/=ccc[:]
        ee[:]=np.sqrt(0.5*(exx[:]*exx[:]+eyy[:]*eyy[:])+exy[:]*exy[:])
 
        print("          -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
        print("          -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
        print("          -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
 
        print("     compute strain rate: %.3f s" % (time.time() - start))

        ######################################################################
        # convergence criterion is based on difference between two consecutively
        # obtained velocity fields, normalised by the boundary condition velocity

        chi_u=LA.norm(u-umem,2)/v0 # vx convergence indicator
        chi_v=LA.norm(v-vmem,2)/v0 # vy convergence indicator

        print('          -> convergence u,v: %.3e %.3e | tol= %.2e' %(chi_u,chi_v,tol))

        convfile.write("%f %10e %10e %10e\n" %(istep+iter/200,chi_u,chi_v,tol))
        convfile.flush()

        umem[:]=u[:]
        vmem[:]=v[:]

        if chi_u<tol and chi_v<tol:
           print('     ***converged***')
           break

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #end for iter
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print('     ----------------------------------')
    print('     end of nl iteration ')
    print('     ----------------------------------')

    ######################################################################
    # compute timestep using CFL condition 
    ######################################################################

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    total_time+=dt

    print("     -> dt= %.3e yr" %(dt/year))
    print("     -> time= %.3e yr" %(total_time/year))

    ######################################################################
    # advect markers 
    # a simple one-step Euler method is used so timesteps should
    # be kept to rather low values, i.e. CFL_nb<=0.25
    # Periodic boundary conditions are implemented on the markers.
    ######################################################################
    start = time.time()

    if True:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*iely+ielx
           swarm_r[im]=-1.+2*(swarm_x[im]-xV[iconV[0,iel]])/hx
           swarm_s[im]=-1.+2*(swarm_y[im]-yV[iconV[0,iel]])/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           NNNP[0:mP]=NNP(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           exxm=sum(NNNV[0:mV]*exx[iconV[0:mV,iel]])
           eyym=sum(NNNV[0:mV]*eyy[iconV[0:mV,iel]])
           exym=sum(NNNV[0:mV]*exy[iconV[0:mV,iel]])
           #assign velocity to marker
           swarm_u[im]=um 
           swarm_v[im]=vm
           #update its position
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
           #update its strain tensor components
           swarm_strainxx[im]+=exxm*dt
           swarm_strainyy[im]+=eyym*dt
           swarm_strainxy[im]+=exym*dt
           #assign strain rate tensor components
           swarm_exx[im]=exxm
           swarm_eyy[im]=eyym
           swarm_exy[im]=exym
           #assign effective strain rate and viscosity
           swarm_ee[im]=np.sqrt(0.5*(swarm_exx[im]**2+swarm_eyy[im]**2+2*swarm_exy[im]**2) ) 
           swarm_eta[im]=viscosity(swarm_x[im],swarm_y[im],swarm_ee[im],background_temperature,swarm_mat[im],iter)
           #assign dev stress values
           swarm_tauxx[im]=2*exxm*swarm_eta[im]
           swarm_tauyy[im]=2*eyym*swarm_eta[im]
           swarm_tauxy[im]=2*exym*swarm_eta[im]
           #assign pressure
           swarm_p[im]=NNNP.dot(p[iconP[0:mP,iel]])
           swarm_yield[im]=swarm_p[im]*np.sin(phi[swarm_mat[im]-1])+\
                           cohesion[swarm_mat[im]-1]*np.cos(phi[swarm_mat[im]-1])
           if swarm_x[im]>Lx: swarm_x[im]-=Lx #periodic b.c. on right side
           if swarm_x[im]<0:  swarm_x[im]+=Lx #periodic b.c. on left side
       #end for
    #end for

    print("     advect markers: %.3f s" % (time.time() - start))

    #####################################################################
    # interpolate pressure onto velocity grid points
    #####################################################################
    start = time.time()

    q=np.zeros(NV,dtype=np.float64)
    counter=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        q[iconV[0,iel]]=p[iconP[0,iel]]
        q[iconV[1,iel]]=p[iconP[1,iel]]
        q[iconV[2,iel]]=p[iconP[2,iel]]
        q[iconV[3,iel]]=p[iconP[3,iel]]
        q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
        q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
        q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
        q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
        q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

    print("     project p on Q2: %.3f s" % (time.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    # the 9-node Q2 element does not exist in vtk, but the 8-node one 
    # does, i.e. type=23. 

    start = time.time()

    #np.savetxt('markers.ascii',np.array([swarm_x,swarm_y,swarm_mat,swarm_eta]).T)

    filename = 'solution_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
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
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % eta_elemental[iel]) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % exx[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % eyy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % exy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='ee' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % ee[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],\
                                                    iconV[5,iel],iconV[6,iel],iconV[7,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*8))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %23)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    filename = 'swarm_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_mat[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='iel' Format='ascii'>\n")
    #for i in range(0,nmarker):
    #    vtufile.write("%3e \n" %swarm_iel[i])
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_paint[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (xx)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_strainxx[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (yy)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_strainyy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (xy)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_strainxy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain (eff.)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" % np.sqrt(0.5*(swarm_strainxx[i]**2+swarm_strainyy[i]**2+2*swarm_strainxy[i]**2) ) )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (xx)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_exx[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (yy)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_eyy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (xy)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_exy[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate (eff.)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" % swarm_ee[i] )
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tauxx (MPa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %(swarm_tauxx[i]/MPa))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tauyy (MPa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %(swarm_tauyy[i]/MPa))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tauxy (MPa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %(swarm_tauxy[i]/MPa))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau (eff.) (MPa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" % (np.sqrt(0.5*(swarm_tauxx[i]**2+swarm_tauyy[i]**2+2*swarm_tauxy[i]**2))/MPa))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='viscosity (Pa.s)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%10e \n" %swarm_eta[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure (Pa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%10e \n" %swarm_p[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='yield value (Pa)' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%10e \n" %swarm_yield[i])
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='r,s,t' NumberOfComponents='3' Format='ascii'>\n")
    #for i in range(0,nmarker):
    #    vtufile.write("%5e %5e %5e \n" %(swarm_r[i],swarm_s[i],0.))
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='velocity (cm/year)' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%5e %5e %5e \n" %(swarm_u[i]/cm*year,swarm_v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
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

    print("     export to vtu: %.3f s" % (time.time() - start))

    #----------------------------------
    start = time.time()

    plt.figure(figsize=(20,5))
    plt.scatter(swarm_x,swarm_y,c=np.log10(swarm_eta),s=1)
    plt.title('effective viscosity')
    plt.xlim([0,Lx])
    plt.ylim([0,Ly])
    filename = 'swarm_eta_{:04d}.png'.format(istep)
    plt.savefig(filename,bbox_inches='tight')

    print("     export to png: %.3f s" % (time.time() - start))

###################################################################################################
# end for istep
###################################################################################################

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
