import numpy as np
import time as timing
import sys as sys
import scipy.sparse as sps
import random
from scipy.sparse import csr_matrix, lil_matrix 

###############################################################################
# bx and by are the body force components
# and analytical solution

def a(x):
    return -2*x*x*(x-1)**2
def b(y):
    return y*(2*y-1)*(y-1)  
def c(x):
    return x*(2*x-1)*(x-1) 
def d(y):
    return 2*y*y*(y-1)**2

def ap(x): 
    return -4*x*(2*x**2-3*x+1)
def app(x):
    return -4*(6*x**2-6*x+1) 
def bp(y): 
    return 6*y**2-6*y+1 
def bpp(y):
    return 12*y-6 
def cp(x): 
    return 6*x**2-6*x+1 
def cpp(x):
    return 12*x-6  
def dp(y): 
    return 4*y*(2*y**2-3*y+1)  
def dpp(y):
    return 4*(6*y**2-6*y+1)  

def exx_th(x,y):
    return ap(x)*b(y)
def eyy_th(x,y):
    return c(x)*dp(y)
def exx_th(x,y):
    return 0.5*(a(x)*bp(y)+cp(x)*d(y))

def dpdx_th(x,y):
    return (1-2*x)*(1-2*y)
def dpdy_th(x,y):
    return -2*x*(1-x)

#------------------------------------------------------------------------------

def bx(x,y):
    if bench==1: return dpdx_th(x,y)-2*app(x)*b(y) -(a(x)*bpp(y)+cp(x)*dp(y))
    if bench==2: return 0. 
    if bench==3:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
           (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
           (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
           1.-4.*y+12.*y*y-8.*y*y*y)
       return val
    if bench==9: return 3*x**2*y**2-y-1

def by(x,y):
    if bench==1: return dpdy_th(x,y)-(ap(x)*bp(y)+cpp(x)*d(y)) -2*c(x)*dpp(y) 
    if bench==2:
       if abs(x-0.5)<0.0625 and abs(y-0.5)<0.0625:
          return -1.01+1
       else:
          return -1+1
    if bench==3:
       val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
       return val
    if bench==9: return 2*x**3*y+3*x-1

#------------------------------------------------------------------------------

def eta(x,y):
    if bench==1: return 1
    if bench==2:
       if abs(x-0.5)<0.0625 and abs(y-0.5)<0.0625:
          return 1000
       else:
          return 1
    if bench==3: return 1 
    if bench==9: return 1 

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if bench==1: return a(x)*b(y)
    if bench==2: return 0
    if bench==3: return x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==9: return x+x**2-2*x*y+x**3-3*x*y**2+x**2*y

def velocity_y(x,y):
    if bench==1: return c(x)*d(y)
    if bench==2: return 0
    if bench==3: return -y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==9: return -y-2*x*y+y**2-3*x**2*y+y**3-x*y**2

def pressure(x,y):
    if bench==1: return x*(1-x)*(1-2*y)
    if bench==2: return 0
    if bench==3: return x*(1.-x)-1./6.
    if bench==9: return x*y+x+y+x**3*y**2-4/3

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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,\
                     NV_5,NV_6,NV_7,NV_8],dtype=np.float64)

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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,\
                     dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,\
                     dNVds_5,dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(r,s):
    return np.array([1-r-s,r,s],dtype=np.float64)

#------------------------------------------------------------------------------
###############################################################################

eps=1e-8
cm=0.01
year=365.25*24*3600

print("-----------------------------")
print("---------- stone 76 ---------")
print("-----------------------------")

ndim=2
ndofV=2
ndofP=1
mV=9
mP=3

# bench=1 : mms #1 (lami17)
# bench=2 : sinking cube
# bench=3 : Donea & Huerta
# bench=9 : mms #2 (lami17)

bench=2

Lx=1
Ly=1

if int(len(sys.argv) == 6):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   nqperdim=int(sys.argv[4])
   meth=int(sys.argv[5])
else:
   nelx = 32
   nely = nelx
   visu = 1
   nqperdim=3
   meth = 1

# 1: square elements
# 2: randomised
# 3: wave deformation
mesh_type=1

if mesh_type==2: xi=0.1 # controls level of mesh randomness (between 0 and 0.5 max)

nnx=2*nelx+1
nny=2*nely+1
nel=nelx*nely
NV=(2*nelx+1)*(2*nely+1)
NP=3*nel
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP
hx=Lx/nelx
hy=Ly/nely

###############################################################################

print('bench=',bench)
print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('method=',meth)

###############################################################################

if nqperdim==2:
   qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweights=[1.,1.]

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

if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

eta_ref=1.
pnormalise=True
sparse=True

###############################################################################
# grid point setup
###############################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("build V grid points: %.3f s" % (timing.time() - start))

###############################################################################
# connectivity
###############################################################################
start = timing.time()

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

print("build V grid icon: %.3f s" % (timing.time() - start))

###############################################################################
# add random noise to node positions
###############################################################################
start = timing.time()

if mesh_type==2:
   for i in range(0,NV):
       if xV[i]>0 and xV[i]<Lx and yV[i]>0 and yV[i]<Ly:
          xV[i]+=random.uniform(-1.,+1)*hx*xi
          yV[i]+=random.uniform(-1.,+1)*hy*xi
       #end if
   #end for

print("add randomness: %.3f s" % (timing.time() - start))

###############################################################################
# deform mesh with wave a la van Keken 1997
###############################################################################
start=timing.time()

on_interface=np.zeros(NV,dtype=bool) 

if mesh_type==3:
   jtarget=2*int(nely/5)+1 -1 
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           yinterface=0.2+0.02*np.cos(np.pi*xV[counter]/Lx)
           if j==jtarget:
              yV[counter]=yinterface
              on_interface[counter]=True
           if j<jtarget:
              yV[counter]=yinterface*(j+1-1.)/(jtarget+1-1.)
           if j>jtarget:
              dy=(Ly-yinterface)/(nny-jtarget-1)
              yV[counter]=yinterface+dy*(j-jtarget)
           if j==nny-1:
              yV[counter]=1.
           if j==0:
              yV[counter]=0.
           counter += 1

print("deform with wave: %.3f s" % (timing.time() - start))

###############################################################################
# straighten the edges
###############################################################################
start=timing.time()

for iel in range(0,nel):
    xV[iconV[4,iel]]=0.5*(xV[iconV[0,iel]]+xV[iconV[1,iel]])
    yV[iconV[4,iel]]=0.5*(yV[iconV[0,iel]]+yV[iconV[1,iel]])
    xV[iconV[5,iel]]=0.5*(xV[iconV[1,iel]]+xV[iconV[2,iel]])
    yV[iconV[5,iel]]=0.5*(yV[iconV[1,iel]]+yV[iconV[2,iel]])
    xV[iconV[6,iel]]=0.5*(xV[iconV[2,iel]]+xV[iconV[3,iel]])
    yV[iconV[6,iel]]=0.5*(yV[iconV[2,iel]]+yV[iconV[3,iel]])
    xV[iconV[7,iel]]=0.5*(xV[iconV[3,iel]]+xV[iconV[0,iel]])
    yV[iconV[7,iel]]=0.5*(yV[iconV[3,iel]]+yV[iconV[0,iel]])
    xV[iconV[8,iel]]=0.25*(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]]+xV[iconV[3,iel]])
    yV[iconV[8,iel]]=0.25*(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]]+yV[iconV[3,iel]])

print("straighten edges: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure grid and iconP 
###############################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

for iel in range(nel):
    iconP[0,iel]=3*iel
    iconP[1,iel]=3*iel+1
    iconP[2,iel]=3*iel+2

counter=0
for iel in range(nel):
    xP[counter]=xV[iconV[8,iel]]
    yP[counter]=yV[iconV[8,iel]]
    counter+=1
    xP[counter]=xV[iconV[8,iel]]+hx/2
    yP[counter]=yV[iconV[8,iel]]
    counter+=1
    xP[counter]=xV[iconV[8,iel]]
    yP[counter]=yV[iconV[8,iel]]+hy/2
    counter+=1

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            jcob=np.linalg.det(jcb)
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

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)        # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if bench==1 or bench==9 or bench==3:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=velocity_y(xV[i],yV[i])
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=velocity_y(xV[i],yV[i])
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=velocity_y(xV[i],yV[i])
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=velocity_y(xV[i],yV[i])

else:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=0 
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ]=True ; bc_val[i*ndofV  ]=0
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1]=True ; bc_val[i*ndofV+1]=0

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = timing.time()

if sparse:
   if pnormalise:
      A_sparse = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   else:
      A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT

constr  = np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions 
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    if meth==2:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det


    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)
    NNNNP= np.zeros(mP*ndofP,dtype=np.float64)           # int of shape functions P

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            if meth==1:
               NNNP[0:mP]=NNP(rq,sq)
            else:
               NNNP[0]=(m11+m21*xq+m31*yq)
               NNNP[1]=(m12+m22*xq+m32*yq)
               NNNP[2]=(m13+m23*xq+m33*yq)
            #print(NNNP[0],NNNP[1],NNNP[2])

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
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
            #end if
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
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for
            f_rhs[m1]+=f_el[ikk]
        #end for
    #end for
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
        if sparse and pnormalise:
           A_sparse[Nfem,NfemV+m2]=constr[m2]
           A_sparse[NfemV+m2,Nfem]=constr[m2]
        #end if
    #end for

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (timing.time()-start, nel))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start = timing.time()

if not sparse:
   if pnormalise:
      a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
      a_mat[Nfem,NfemV:Nfem]=constr
      a_mat[NfemV:Nfem,Nfem]=constr
   else:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
      a_mat[0:NfemV,0:NfemV]=K_mat
      a_mat[0:NfemV,NfemV:Nfem]=G_mat
      a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   #end if
else:
   if pnormalise:
      rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   else:
      rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
#else:

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (timing.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
#np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (timing.time() - start))

###############################################################################
# compute vrms 
###############################################################################
start = timing.time()

vrms=0.
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            uq=NNNV.dot(u[iconV[:,iel]])
            vq=NNNV.dot(v[iconV[:,iel]])
            vrms+=(uq**2+vq**2)*weightq*jcob
        # end for jq
    # end for iq
# end for iel

vrms=np.sqrt(vrms/(Lx*Ly))

vel=np.sqrt(u**2+v**2)

print("     -> nel= %6d ; vrms= %.14f ; max(vel)= %.14f" %(nel,vrms,np.max(vel)))

print("compute v_rms : %.3f s" % (timing.time()-start))

###############################################################################
# compute error fields for visualisation
###############################################################################
start = timing.time()

error_u=np.empty(NV,dtype=np.float64)
error_v=np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i])
    error_v[i]=v[i]-velocity_y(xV[i],yV[i])

print("compute nodal error for visu: %.3f s" % (timing.time()-start))

###############################################################################
# compute error in L2 norm
###############################################################################
if bench==1 or bench==9 or bench==3:

   start = timing.time()

   errv=0.
   errp=0.
   for iel in range (0,nel):

       if meth==2:
          det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
             -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
             +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
          m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
          m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
          m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
          m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
          m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
          m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
          m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
          m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
          m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

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
                   jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
               #end for
               jcob=np.linalg.det(jcb)

               xq=NNNV.dot(xV[iconV[:,iel]])
               yq=NNNV.dot(yV[iconV[:,iel]])

               uq=NNNV.dot(u[iconV[:,iel]])
               vq=NNNV.dot(v[iconV[:,iel]])
               errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob

               if meth==1:
                  NNNP[0:mP]=NNP(rq,sq)
               else:
                  NNNP[0]=(m11+m21*xq+m31*yq)
                  NNNP[1]=(m12+m22*xq+m32*yq)
                  NNNP[2]=(m13+m23*xq+m33*yq)
               pq=NNNP.dot(p[iconP[:,iel]])
               errp+=(pq-pressure(xq,yq))**2*weightq*jcob
           #end for
       #end for
   #end for
   errv=np.sqrt(errv)
   errp=np.sqrt(errp)

   print("     -> nel= %6d ; errv= %.10f ; errp= %.10f" %(nel,errv,errp))

   print("compute errors: %.3f s" % (timing.time() - start))

###############################################################################
# export velocity and pressure on vertical profile at x=0.5
# since a node belongs to up to 4 elts and the pressure is discontinuous,
# we export the four pressure values at the node.
###############################################################################

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]
    
profile=open('profile.ascii',"w")

for iel in range(0,nel): 

    if meth==2:
       det=xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]]\
          -xP[iconP[0,iel]]*yP[iconP[2,iel]]+xP[iconP[2,iel]]*yP[iconP[0,iel]]\
          +xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]]
       m11=(xP[iconP[1,iel]]*yP[iconP[2,iel]]-xP[iconP[2,iel]]*yP[iconP[1,iel]])/det
       m12=(xP[iconP[2,iel]]*yP[iconP[0,iel]]-xP[iconP[0,iel]]*yP[iconP[2,iel]])/det
       m13=(xP[iconP[0,iel]]*yP[iconP[1,iel]]-xP[iconP[1,iel]]*yP[iconP[0,iel]])/det
       m21=(yP[iconP[1,iel]]-yP[iconP[2,iel]])/det
       m22=(yP[iconP[2,iel]]-yP[iconP[0,iel]])/det
       m23=(yP[iconP[0,iel]]-yP[iconP[1,iel]])/det
       m31=(xP[iconP[2,iel]]-xP[iconP[1,iel]])/det
       m32=(xP[iconP[0,iel]]-xP[iconP[2,iel]])/det
       m33=(xP[iconP[1,iel]]-xP[iconP[0,iel]])/det

    for k in range(0,mV):
       xq=xV[iconV[k,iel]]
       if abs(xq-0.5)<1e-6:
          yq=yV[iconV[k,iel]]
          uq=u[iconV[k,iel]]
          vq=v[iconV[k,iel]]
          rq=rVnodes[k]
          sq=sVnodes[k]
          if meth==1:
             NNNP=NNP(rq,sq)
          else:
             NNNP[0]=(m11+m21*xq+m31*yq)
             NNNP[1]=(m12+m22*xq+m32*yq)
             NNNP[2]=(m13+m23*xq+m33*yq)
          pq=NNNP.dot(p[iconP[:,iel]])
          profile.write("%10e %10e %10e %10e %10e\n" %(xq,yq,uq,vq,pq))





###############################################################################
# plot of solution
# using in fact only 4 Vnodes and leaving the bubble out. 
###############################################################################
start = timing.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
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
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%10e \n" %(area[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
    for iel in range(0,nel):
        vtufile.write("%10e \n" %(p[iconP[0,iel]]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='error vel' Format='ascii'> \n")
    for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(error_u[i],error_v[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel th' Format='ascii'> \n")
    for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(velocity_x(xV[i],yV[i]),velocity_y(xV[i],yV[i]),0))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Float32' Name='p th' Format='ascii'> \n")
    for i in range(0,NV):
           vtufile.write("%10e \n" %(pressure(xV[i],yV[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                       iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                       iconV[6,iel],iconV[7,iel],iconV[8,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*9))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %28)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("export to vtu: %.3f s" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
###############################################################################
###############################################################################
