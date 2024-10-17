import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix 
import time as time

#------------------------------------------------------------------------------
Ra=1e6 

def a(x):
    return x**2*(1-x)**2

def ap(x):
    return 2*x*(1-x)*(1-2*x)

def app(x):
    return 2*(1-6*x+6*x**2)

def appp(x):
    return 24*x-12

def bx(x, y): 
    if experiment==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==2:
       val=0
    if experiment==3 or experiment==4 or experiment==5:
       val=4*np.pi**2*np.cos(2*np.pi*y) 
    if experiment==6:
       dpdx=10*(3*(x-0.5)**2*y**2-3*(1-x)**2*(y-0.5)**3)
       val=dpdx-100*eta*(app(x)*ap(y)+a(x)*appp(y))
    return val 

def by(x, y):
    if experiment==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==2:
       val=Ra*(1-y+3*y**2)
    if experiment==3:
       val=2*np.pi*np.cos(2*np.pi*y) + 4*np.pi**2*np.sin(2*np.pi*x) 
    if experiment==4:
       val=8*np.pi*np.cos(8*np.pi*y) + 4*np.pi**2*np.sin(2*np.pi*x) 
    if experiment==5:
       val=1e4*2*np.pi*np.cos(2*np.pi*y) + 4*np.pi**2*np.sin(2*np.pi*x) 
    if experiment==6:
       dpdy=10*((x-0.5)**3*2*y+(1-x)**3*3*(y-0.5)**2)
       val=dpdy+100*eta*(ap(x)*app(y)+appp(x)*a(y))
    return val 

#------------------------------------------------------------------------------

def velocity(x,y):
    if experiment==1:
       vx=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
       vy=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==2:
       vx=0
       vy=0
    if experiment==3 or experiment==4 or experiment==5:
       vx=np.cos(2*np.pi*y)
       vy=np.sin(2*np.pi*x)
    if experiment==6:
       vx= 100*a(x)*ap(y) 
       vy=-100*ap(x)*a(y)
    return vx,vy

def pressure(x,y):
    if experiment==1:
       val=x*(1.-x)-1./6.
    if experiment==2:
       val=Ra*(y**3-y**2/2+y-7/12)
    if experiment==3:
       val=np.sin(2*np.pi*y)
    if experiment==4:
       val=np.sin(8*np.pi*y)
    if experiment==5:
       val=1e4*np.sin(2*np.pi*y)
    if experiment==6:
       val=10*((x-0.5)**3*y**2+(1-x)**3*(y-0.5)**3)
    return val

def dudx(x,y):
    if experiment==1:
       val=x**2*(2*x-2)*(4*y**3-6*y**2+2*y)+2*x*(-x+1)**2*(4*y**3-6*y**2+2*y)
    elif experiment==2:
       val=0
    if experiment==3 or experiment==4 or experiment==5:
       val=0
    elif experiment==6:
       val=100*ap(x)*ap(y)
    else:
       val=0
    return val 

def dudy(x,y):
    if experiment==1:
       val=2*(1-x)**2*x**2*(1-6*y+6*y**2)
    elif experiment==2:
       val=0
    if experiment==3 or experiment==4 or experiment==5:
       val=-2*np.pi*np.sin(2*np.pi*y)
    elif experiment==6:
       val=100*a(x)*app(y)
    else:
       val=0
    return val 

def dvdx(x,y):
    if experiment==1:
       val=-2*(1-6*x+6*x**2)*y**2*(1-y)**2
    elif experiment==2:
       val=0
    if experiment==3 or experiment==4 or experiment==5:
       val=2*np.pi*np.cos(2*np.pi*x)
    elif experiment==6:
       val=-100*app(x)*a(y)
    else:
       val=0
    return val 

def dvdy(x,y):
    if experiment==1:
       val=-(x**2*(2*x-2)*(4*y**3-6*y**2+2*y)+2*x*(-x+1)**2*(4*y**3-6*y**2+2*y))
    elif experiment==2:
       val=0
    if experiment==3 or experiment==4 or experiment==5:
       val=0
    elif experiment==6:
       val=-100*ap(x)*ap(y)
    else:
       val=0
    return val 

#------------------------------------------------------------------------------
#      Q2            Q1           P-1
#
#  3----6----2   3---------2  +----2----+
#  |    |    |   |         |  |    |    |
#  |    |    |   |         |  |    |    |
#  7----8----5   |         |  |    0----1
#  |    |    |   |         |  |         |
#  |    |    |   |         |  |         |
#  0----4----1   0---------1  +---------+
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
    if disc:
       NP_0=1-rq-sq
       NP_1=rq
       NP_2=sq
       return NP_0,NP_1,NP_2
    else:
       NP_0=0.25*(1-rq)*(1-sq)
       NP_1=0.25*(1+rq)*(1-sq)
       NP_2=0.25*(1+rq)*(1+sq)
       NP_3=0.25*(1-rq)*(1+sq)
       return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------

disc=True # Q2Q1 vs Q2P-1

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
if disc:
   mP=3     # number of pressure nodes making up an element
else:
   mP=4     # number of pressure nodes making up an element

Lx=1. # horizontal extent of the domain 
Ly=1. # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   gamma = float(sys.argv[4])
else:
   nelx = 16
   nely = 16
   visu = 1
   gamma= 0

#######################################
#exp1: donea & huerta
#exp2: example1.1 in jolm17
#exp3: jojl14
#exp4: jojl14
#exp5: jojl14
#exp6: jolm17 p 528

experiment=5

#######################################
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

if disc:
   NP=3*nel
else:
   NP=(nelx+1)*(nely+1)

NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eps=1.e-10

nqperdim=3

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

hx=Lx/nelx
hy=Ly/nely

eta = 1
eta_ref=eta

pnormalise=False

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("------------------------------")

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

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("velocity grid points: %.3f s" % (time.time() - start))

#################################################################
# pressure connectivity array
#################################################################

xP=np.zeros(NP,dtype=np.float64)     # x coordinates
yP=np.zeros(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

if disc:
   for iel in range(nel):
       iconP[0,iel]=3*iel
       iconP[1,iel]=3*iel+1
       iconP[2,iel]=3*iel+2

   NNNV = np.zeros(mV,dtype=np.float64)           # shape functions V
   counter=0
   for iel in range(nel):
       #pressure node 0
       rq=0.0
       sq=0.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1
       #pressure node 1
       rq=1.0
       sq=0.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1
       #pressure node 2
       rq=0.0
       sq=1.0
       NNNV[0:mV]=NNV(rq,sq)
       xq=NNNV[:].dot(xV[iconV[:,iel]])
       yq=NNNV[:].dot(yV[iconV[:,iel]])
       xP[counter]=xq
       yP[counter]=yq
       counter+=1

else:
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

   #for iel in range(0,nel):
   #    print('elt',iel,'composed of Pnodes',iconP[:,iel])

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

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
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, NV):
    vx,vy=velocity(xV[i],yV[i])
    if xV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vy 
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vy 
    if yV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vy 
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = vx
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vy 
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

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
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
#c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
c_mat   = np.array([[2*eta+gamma,gamma,0],[gamma,2*eta+gamma,0],[0,0,eta]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    T_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
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
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                #f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*rho
                #f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)

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

    # assemble matrix and rhs 
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


print("build FE matrix: %.3f s" % (time.time() - start))

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

print("assemble blocks: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

if pnormalise:
   print(sol[Nfem])

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

#u[:]=xV[:]
#v[:]=yV[:]

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    NNNV[0:9]=NNV(rq,sq)
    dNNNVdr[0:9]=dNNVdr(rq,sq)
    dNNNVds[0:9]=dNNVds(rq,sq)

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

    for k in range(0,mV):
        xc[iel] += NNNV[k]*xV[iconV[k,iel]]
        yc[iel] += NNNV[k]*yV[iconV[k,iel]]
        exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# normalise pressure
######################################################################
start = time.time()

avrg_p=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            xq=0.
            yq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
            #end for
            pq=0.0
            for k in range(0,mP):
                 pq+=NNNP[k]*p[iconP[k,iel]]

            #print (xq,yq,pq)
            avrg_p+=pq*jcob*weightq

        #end for jq
    #end for iq
#end for iel

avrg_p/=(Lx*Ly)

print('     -> avrg_p=',avrg_p)

p[:]-=avrg_p

print("normalise pressure: %.3f s" % (time.time() - start))

######################################################################
# compute vrms and errors 
######################################################################
start = time.time()

avrg_p=0.
vrms=0.
err_v=0
err_p=0
err_divv=0
err_gradvel=0
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            xq=0.
            yq=0.
            uq=0.
            vq=0.
            dudxq=0.
            dudyq=0.
            dvdxq=0.
            dvdyq=0.
            divvq=0.
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                dudxq += dNNNVdx[k]*u[iconV[k,iel]] 
                dudyq += dNNNVdy[k]*u[iconV[k,iel]] 
                dvdxq += dNNNVdx[k]*v[iconV[k,iel]] 
                dvdyq += dNNNVdy[k]*v[iconV[k,iel]] 
                divvq += dNNNVdx[k]*u[iconV[k,iel]] + dNNNVdy[k]*v[iconV[k,iel]]  
            #end for
            vrms+=(uq**2+vq**2)*weightq*jcob
            uth,vth=velocity(xq,yq)
            err_v+=((uq-uth)**2+(vq-vth)**2)*weightq*jcob
            err_divv+=divvq**2*weightq*jcob
            err_gradvel+=(dudxq-dudx(xq,yq))**2*weightq*jcob
            err_gradvel+=(dudyq-dudy(xq,yq))**2*weightq*jcob
            err_gradvel+=(dvdxq-dvdx(xq,yq))**2*weightq*jcob
            err_gradvel+=(dvdyq-dvdy(xq,yq))**2*weightq*jcob

            pq=0.0
            for k in range(0,mP):
                 pq+=NNNP[k]*p[iconP[k,iel]]

            #print (xq,yq,pq)
            avrg_p+=pq*jcob*weightq
            err_p+=(pq-pressure(xq,yq))**2*weightq*jcob

        #end for
    #end for
#end for

vrms=np.sqrt(vrms/(Lx*Ly))
avrg_p/=(Lx*Ly)
err_v=np.sqrt(err_v)
err_p=np.sqrt(err_p)
err_divv=np.sqrt(err_divv)
err_gradvel=np.sqrt(err_gradvel)

print('     -> avrg_p=',avrg_p)

print("     -> hx= %.6e  vrms= %.7e " % (hx,vrms))

print('benchmark ',hx,err_v,err_p,vrms,err_divv,err_gradvel,gamma)

print("compute vrms & errors: %.3f s" % (time.time() - start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################
start = time.time()

q=np.zeros(NV,dtype=np.float64)
counter=np.zeros(NV,dtype=np.float64)

if disc:
   rVnodes=[-1,1,1,-1,0,1,0,-1,0]
   sVnodes=[-1,-1,1,1,-1,0,1,0,0]
   for iel in range(0,nel):
       for c in range(0,9):
          rq=rVnodes[c]
          sq=sVnodes[c]
          NNNP[0:mP]=NNP(rq,sq)
          pq=0
          for k in range(0,mP):
              pq+=NNNP[k]*p[iconP[k,iel]]
          q[iconV[c,iel]]+=pq
          counter[iconV[c,iel]]+=1
       #end for
   #end for
   q[:]/=counter[:]

else:
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

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("compute q: %.3f s" % (time.time() - start))

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
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % eyy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy[iel])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
if disc:
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % p[3*iel]) 
   vtufile.write("</DataArray>\n")

vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (analytical)' Format='ascii'> \n")
for i in range(0,NV):
    vx,vy=velocity(xV[i],yV[i])
    vtufile.write("%10e %10e %10e \n" %(vx,vy,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (analytical)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % (pressure(xV[i],yV[i]))  )
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" % (q[i]-pressure(xV[i],yV[i]))  )
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
