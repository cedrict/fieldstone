import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix 
import time as time
from numpy import linalg as LA

#------------------------------------------------------------------------------

cm=0.01
year=3600*24*365.25

a0=4.4e8
b0=-5.26e4
a1=2.11e-2
b1=1.74e-4
a2=-41.8
b2=4.21e-2
c2=-1.14e-5

def density(x,y):
    val=3300
    return val

def viscosity(x,y,e,T):
    #A0=a0+b0*T
    #A1=a1+b1*T
    #A2=a2+b2*T+c2*T**2
    #val = A0/2/e*(1+np.tanh(A1*(np.log10(e)-A2)))
    val=1e21
    return val

#------------------------------------------------------------------------------
#      Q2            Q1       
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
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=200e3 # horizontal extent of the domain 
Ly=100e3 # vertical extent of the domain 

nelx = 64
nely = int(nelx*Ly/Lx)
visu = 1
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NP=(nelx+1)*(nely+1)

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

pnormalise=False

gx=0
gy=0

eta_ref=1e21

v0=1*cm/year

reg=1e-20

niter=50
tol=1e-3
convfile=open('conv.ascii',"w")

experiment=2
x_inclusion=Lx/2
y_inclusion=Ly/2
R_inclusion=Ly/8
rho_mat = np.array([3000,3000],dtype=np.float64)
eta_mat = np.array([1e19,1e23],dtype=np.float64)

nmarker_per_dim=8
nmarker_per_element=nmarker_per_dim**2
nmarker=nel*nmarker_per_element
avrg=1
rk=1

nstep=1
CFL_nb=0.5
rk=1

every=1

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

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
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
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

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
# compute coordinates of element centers
#################################################################

NNNV=np.zeros(mV,dtype=np.float64)       
xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0
    sq=0
    NNNV[0:9]=NNV(rq,sq)
    xc[iel]=NNNV[:].dot(xV[iconV[:,iel]])
    yc[iel]=NNNV[:].dot(yV[iconV[:,iel]])
#end for

#################################################################
# temperature layout
#################################################################

T = np.zeros(NV,dtype=np.float64)  
    
T[:]=300+273

for i in range(0,NV):
    if abs(yV[i]-Ly/2)<Ly/1000:
       T[i]*=0.99

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    if xV[i]<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -v0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
    if yV[i]/Ly>1-eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +v0
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# swarm setup
#################################################################
start = time.time()

swarm_x=np.empty(nmarker,dtype=np.float64) 
swarm_y=np.empty(nmarker,dtype=np.float64) 
swarm_mat=np.empty(nmarker,dtype=np.int8)  
swarm_paint=np.empty(nmarker,dtype=np.float64) 
swarm_x0=np.empty(nmarker,dtype=np.float64) 
swarm_y0=np.empty(nmarker,dtype=np.float64) 
swarm_r=np.empty(nmarker,dtype=np.float64) 
swarm_s=np.empty(nmarker,dtype=np.float64)

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

swarm_x0[0:nmarker]=swarm_x[0:nmarker]
swarm_y0[0:nmarker]=swarm_y[0:nmarker]

print("setup: swarm: %.3f s" % (time.time() - start))

#################################################################
# assign material id to markers 
#################################################################
start = time.time()

if experiment==1:
   swarm_mat[:]=1

if experiment==2:
   for im in range (0,nmarker):
       swarm_mat[im]=1
       if (swarm_x[im]-x_inclusion)**2+(swarm_y[im]-y_inclusion)**2<R_inclusion**2:
          swarm_mat[im]=2


#################################################################
# paint markers 
#################################################################
start = time.time()

for im in range (0,nmarker):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*4)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))
#end for 

np.savetxt('markers.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')



###################################################################################################
###################################################################################################
# time stepping loop 
###################################################################################################
###################################################################################################
    
umem = np.zeros(NV,dtype=np.float64)      
vmem = np.zeros(NV,dtype=np.float64)    
u    = np.zeros(NV,dtype=np.float64)   
v    = np.zeros(NV,dtype=np.float64)    

for istep in range(0,nstep):

    print ('||-----------------------------------------------||')
    print ('||--------------------istep= %i ------------------||' %istep)
    print ('||-----------------------------------------------||')

    #################################################################
    # compute elemental averagings 
    #################################################################
    start = time.time()

    nmarker_in_element=np.zeros(nel,dtype=np.int16)
    list_of_markers_in_element=np.zeros((2*nmarker_per_element,nel),dtype=np.int32)
    eta_elemental=np.zeros(nel,dtype=np.float64)
    rho_nodal=np.zeros(NP,dtype=np.float64)
    eta_nodal=np.zeros(NP,dtype=np.float64)
    nodal_counter=np.zeros(NP,dtype=np.float64)

    for im in range(0,nmarker):
        #localise marker
        ielx=int(swarm_x[im]/Lx*nelx)
        iely=int(swarm_y[im]/Ly*nely)
        iel=nelx*(iely)+ielx
        list_of_markers_in_element[nmarker_in_element[iel],iel]=im
        nmarker_in_element[iel]+=1
        N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
        N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
        N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
        N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
        nodal_counter[iconP[0,iel]]+=N1
        nodal_counter[iconP[1,iel]]+=N2
        nodal_counter[iconP[2,iel]]+=N3
        nodal_counter[iconP[3,iel]]+=N4


        if abs(avrg)==1 : # arithmetic
           eta_elemental[iel]     +=eta_mat[swarm_mat[im]-1]
           eta_nodal[iconP[0,iel]]+=eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[iconP[1,iel]]+=eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[iconP[2,iel]]+=eta_mat[swarm_mat[im]-1]*N3
           eta_nodal[iconP[3,iel]]+=eta_mat[swarm_mat[im]-1]*N4
        if abs(avrg)==2: # geometric
           eta_elemental[iel]     +=np.log10(eta_mat[swarm_mat[im]-1])
           eta_nodal[iconP[0,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N1
           eta_nodal[iconP[1,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N2
           eta_nodal[iconP[2,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N3
           eta_nodal[iconP[3,iel]]+=np.log10(eta_mat[swarm_mat[im]-1])*N4
        if abs(avrg)==3: # harmonic
           eta_elemental[iel]     +=1/eta_mat[swarm_mat[im]-1]
           eta_nodal[iconP[0,iel]]+=1/eta_mat[swarm_mat[im]-1]*N1
           eta_nodal[iconP[1,iel]]+=1/eta_mat[swarm_mat[im]-1]*N2
           eta_nodal[iconP[2,iel]]+=1/eta_mat[swarm_mat[im]-1]*N3
           eta_nodal[iconP[3,iel]]+=1/eta_mat[swarm_mat[im]-1]*N4
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

    print("     -> nmarker_in_elt(m,M) %.5e %.5e " %(np.min(nmarker_in_element),np.max(nmarker_in_element)))
    print("     -> rho_nodal     (m,M) %.5e %.5e " %(np.min(rho_nodal),np.max(rho_nodal)))
    print("     -> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))

    print("     markers onto grid: %.3f s" % (time.time() - start))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # non linear iterations
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for iter in range(0,niter):

        print('     ----- iteration------',iter,'----------')


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
        dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
        dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
        dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
        dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
        c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

        for iel in range(0,nel):

            # set arrays to 0 every loop
            f_el =np.zeros((mV*ndofV),dtype=np.float64)
            K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
            G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
            h_el=np.zeros((mP*ndofP),dtype=np.float64)
            NNNNP= np.zeros(mP*ndofP,dtype=np.float64)   

            # integrate viscous term at 4 quadrature points
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
                    exxq=0.0
                    eyyq=0.0
                    exyq=0.0
                    xq=0.0
                    yq=0.0
                    Tq=0.0
                    for k in range(0,mV):
                        xq+=NNNV[k]*xV[iconV[k,iel]]
                        yq+=NNNV[k]*yV[iconV[k,iel]]
                        Tq+=NNNV[k]*T[iconV[k,iel]]
                        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                        exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                        eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                        exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

                    ee=np.sqrt(0.5*(exxq**2+eyyq**2+2*exyq**2) + reg**2)

                    # construct b_mat matrix
                    for i in range(0,mV):
                        b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                                 [0.        ,dNNNVdy[i]],
                                                 [dNNNVdy[i],dNNNVdx[i]]]

                    #viscosity(xq,yq,ee,Tq)
                    etaq=eta_elemental[iel]

                    # compute elemental a_mat matrix
                    K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                    # compute elemental rhs vector
                    for i in range(0,mV):
                        f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*density(xq,yq)
                        f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*density(xq,yq)

                    for i in range(0,mP):
                        N_mat[0,i]=NNNP[i]
                        N_mat[1,i]=NNNP[i]
                        N_mat[2,i]=0.

                    G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                    NNNNP[:]+=NNNP[:]*jcob*weightq

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
                    for k2 in range(0,mP):
                        jkk=k2
                        m2 =iconP[k2,iel]
                        A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]#*eta_ref/Ly
                        A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]#*eta_ref/Ly
                    f_rhs[m1]+=f_el[ikk]
            for k2 in range(0,mP):
                m2=iconP[k2,iel]
                h_rhs[m2]+=h_el[k2]
                constr[m2]+=NNNNP[k2]
                if pnormalise:
                   A_sparse[Nfem,NfemV+m2]+=constr[m2]
                   A_sparse[NfemV+m2,Nfem]+=constr[m2]


        print("     build FE matrix: %.3f s" % (time.time() - start))

        ######################################################################
        # assemble K, G, GT, f, h into A and rhs
        ######################################################################
        start = time.time()

        if pnormalise:
           rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
        else:
           rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

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

        print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
        print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
        print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

        np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
        np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

        print("     split vel into u,v: %.3f s" % (time.time() - start))

        ######################################################################

        xi_u=LA.norm(u-umem,2)/v0
        xi_v=LA.norm(v-vmem,2)/v0

        print('     -> convergence u,v',xi_u,xi_v,tol)

        convfile.write("%3d %10e %10e %10e\n" %(iter,xi_u,xi_v,tol))
        convfile.flush()

        umem[:]=u[:]
        vmem[:]=v[:]

        if xi_u<tol and xi_v<tol:
           print('     ***converged***')
           break

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #end for iter
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print('     ----- end nl iteration------------')

    ######################################################################
    # compute timestep 
    ######################################################################

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    print("     -> dt= %.3e yr" %(dt/year))

    ######################################################################
    # advect markers 
    ######################################################################
    start = time.time()

    if rk==1:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=xV[iconV[0,iel]]
           y0=yV[iconV[0,iel]]
           swarm_r[im]=-1.+2*(swarm_x[im]-x0)/hx
           swarm_s[im]=-1.+2*(swarm_y[im]-y0)/hy
           NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
           um=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
       #end for

    print("     advect markers: %.3f s" % (time.time() - start))

    ######################################################################
    # compute strainrate 
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

            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)

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

    print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

    print("     compute press & sr: %.3f s" % (time.time() - start))

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

    filename = 'solution.vtu'
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
    #vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % rho_el[iel]) 
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % eta_elemental[iel]) 
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %T[i])
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
    vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
    for i in range (0,NV):
        vtufile.write("%10e\n" % viscosity(xV[i],yV[i],ee[i],T[i]))
    vtufile.write("</DataArray>\n")

    #--
    #vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    #for i in range (0,NV):
    #    vtufile.write("%10e\n" % eta_nodal[i])
    #vtufile.write("</DataArray>\n")

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
    vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_paint[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='r,s,t' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%5e %5e %5e \n" %(swarm_r[i],swarm_s[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='displacement' NumberOfComponents='3' Format='ascii'>\n")
    #for i in range(0,nmarker):
    #    vtufile.write("%5e %5e %5e \n" %(swarm_x[i]-swarm_x0[i],swarm_y[i]-swarm_y0[i],0.))
    #vtufile.write("</DataArray>\n")
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


###################################################################################################
# end for istep
###################################################################################################

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
