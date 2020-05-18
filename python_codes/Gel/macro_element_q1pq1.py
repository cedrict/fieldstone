import sys as sys
import numpy as np
import time as timing
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import random
from numpy import linalg 
from scipy.linalg import null_space

#------------------------------------------------------------------------------
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
    if bench==1:
       return dpdx_th(x,y)-2*app(x)*b(y) -(a(x)*bpp(y)+cp(x)*dp(y))
    if bench==9:
       return 3*x**2*y**2-y-1

def by(x,y):
    if bench==1:
       return dpdy_th(x,y)-(ap(x)*bp(y)+cpp(x)*d(y)) -2*c(x)*dpp(y) 
    if bench==9:
       return 2*x**3*y+3*x-1

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if bench==1:
       return a(x)*b(y)
    if bench==9:
       return x+x**2-2*x*y+x**3-3*x*y**2+x**2*y

def velocity_y(x,y):
    if bench==1:
       return c(x)*d(y)
    if bench==9:
       return -y-2*x*y+y**2-3*x**2*y+y**3-x*y**2

def pressure(x,y):
    if bench==1:
       return x*(1-x)*(1-2*y)
    if bench==9:
       return x*y+x+y+x**3*y**2-4/3

#------------------------------------------------------------------------------

def B(r,s):
    if bubble==1:
       return (1-r**2)*(1-s**2)*(1-r)*(1-s)
    elif bubble==2:
       return (1-r**2)*(1-s**2)*(1+beta*(r+s))
    else:
       return (1-r**2)*(1-s**2)

def dBdr(r,s):
    if bubble==1:
       return (1-s**2)*(1-s)*(-1-2*r+3*r**2)
    elif bubble==2:
       return (s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
    else:
       return (-2*r)*(1-s**2)

def dBds(r,s):
    if bubble==1:
       return (1-r**2)*(1-r)*(-1-2*s+3*s**2) 
    elif bubble==2:
       return (r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
    else:
       return (1-r**2)*(-2*s)

#------------------------------------------------------------------------------

def NNV(r,s):
    NV_0= 0.25*(1-r)*(1-s) - 0.25*B(r,s)
    NV_1= 0.25*(1+r)*(1-s) - 0.25*B(r,s)
    NV_2= 0.25*(1+r)*(1+s) - 0.25*B(r,s)
    NV_3= 0.25*(1-r)*(1+s) - 0.25*B(r,s)
    NV_4= B(r,s)
    return NV_0,NV_1,NV_2,NV_3,NV_4

def dNNVdr(r,s):
    dNVdr_0=-0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_1=+0.25*(1.-s) -0.25*dBdr(r,s)
    dNVdr_2=+0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_3=-0.25*(1.+s) -0.25*dBdr(r,s)
    dNVdr_4=dBdr(r,s) 
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4

def dNNVds(r,s):
    dNVds_0=-0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_1=-0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_2=+0.25*(1.+r) -0.25*dBds(r,s)
    dNVds_3=+0.25*(1.-r) -0.25*dBds(r,s)
    dNVds_4=dBds(r,s) 
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4

def NNP(r,s):
    NP_0= 0.25*(1-r)*(1-s)
    NP_1= 0.25*(1+r)*(1-s)
    NP_2= 0.25*(1+r)*(1+s)
    NP_3= 0.25*(1-r)*(1+s)
    return NP_0,NP_1,NP_2,NP_3 

#------------------------------------------------------------------------------

def eta(x,y):
    if bench==1:
       val=1.
    if bench==2 or bench==3:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=eta2
       else:
          val=eta1
    if bench==4:
       if x<0.5:
          val=1.
       else:
          val=1.e6
    if bench==5:
       val= np.exp(13.8155*y) 
    if bench==6:
       if (np.sqrt(x*x+y*y) < 0.2):
          val=1e3
       else:
          val=1.
    if bench==7:
       if ((x-0.5)**2+(y-0.5)**2 < 0.123**2):
          val=100.
       else:
          val=1.
    if bench==8:
       if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
          val=eta1
       else:
          val=eta2
    if bench==9:
       val=1
    if bench==10:
       if y>600e3:
          val=1e23
       else:
          val=1e21
    return val

#------------------------------------------------------------------------------

def rho(x,y):
    if bench==2:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=rho2 
       else:
          val=rho1 
    if bench==3:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=rho2-rho1
       else:
          val=rho1-rho1
    if bench==4:
       val=np.sin(np.pi*y)*np.cos(np.pi*x)
    if bench==5:
       val=np.sin(2.*y)*np.cos(3.*np.pi*x)
    if bench==6:
       val=0.
    if bench==7:
       if ((x-0.5)**2+(y-0.5)**2 < 0.123**2):
          val=2.
       else:
          val=1.
    if bench==8:
       if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
          val=3300
       else:
          val=3000
    if bench==10:
       val=3300
    return val

#------------------------------------------------------------------------------

def vy_th(phi1,phi2,rho1,rho2):
    c11 = (eta1*2*phi1**2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) - (2*phi2**2)/(np.cosh(2*phi2)-1-2*phi2**2)
    d12 = (eta1*(np.sinh(2*phi1) -2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) + (np.sinh(2*phi2)-2*phi2)/(np.cosh(2*phi2)-1-2*phi2**2)
    i21 = (eta1*phi2*(np.sinh(2*phi1)+2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) + (phi2*(np.sinh(2*phi2)+2*phi2))/(np.cosh(2*phi2)-1-2*phi2**2) 
    j22 = (eta1*2*phi1**2*phi2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2))-(2*phi2**3)/(np.cosh(2*phi2)-1-2*phi2**2 )
    K=-d12/(c11*j22-d12*i21)
    val=K*(rho1-rho2)/2/eta2*(Ly/2.)*abs(gy)*amplitude
    return val

#------------------------------------------------------------------------------

cm=0.01
year=365.25*24*3600

ndim=2
ndofV=2
ndofP=1
mV=5
mP=4

# bench=1 : mms (lami17)
# bench=2 : block full density
# bench=3 : block reduced density
# bench=4 : solcx
# bench=5 : solkz
# bench=6 : solvi
# bench=7 : Stokes sphere
# bench=8 : RT-instability
# bench=9 : mms (lami17)
# bench=10: free surf. crsg12

bench=1

if bench==1 or bench==4 or bench==5 or bench==6 or bench==7 or bench==9:
   Lx=4
   Ly=4
if bench==2 or bench==3 or bench==8:
   Lx=512e3
   Ly=512e3
if bench==10:
   Lx=2800e3
   Ly=700e3

bubble=2

if int(len(sys.argv) == 9):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   drho=float(sys.argv[4])
   eta1=10.**(float(sys.argv[5]))
   eta2=10.**(float(sys.argv[6]))
   nqperdim=int(sys.argv[7])
   beta=float(sys.argv[8])
else:
   nelx = 2
   nely = 2
   visu = 1
   drho = 8
   eta1 = 1e21
   eta2 = 1e22
   nqperdim=2
   beta=0.

compute_eigenvalues=False

nel=nelx*nely
NV=(nelx+1)*(nely+1)+nel
NP=(nelx+1)*(nely+1)
NfemV=NV*ndofV
NfemP=NP*ndofP
Nfem=NfemV+NfemP
hx=Lx/nelx
hy=Ly/nely

print('bench=',bench)
print('nelx =',nelx)
print('nely =',nely)
print('nel  =',nel)
print('NV   =',NV)
print('NP   =',NP)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('bubble=',bubble)
print('beta=',beta)

nqperdim=10

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

if nqperdim==10:
   qcoords=[-0.973906528517172,\
            -0.865063366688985,\
            -0.679409568299024,\
            -0.433395394129247,\
            -0.148874338981631,\
             0.148874338981631,\
             0.433395394129247,\
             0.679409568299024,\
             0.865063366688985,\
             0.973906528517172]
   qweights=[0.066671344308688,\
             0.149451349150581,\
             0.219086362515982,\
             0.269266719309996,\
             0.295524224714753,\
             0.295524224714753,\
             0.269266719309996,\
             0.219086362515982,\
             0.149451349150581,\
             0.066671344308688]
eps=1e-8

if bench==1 or bench==9:
   eta_ref=1.
   pnormalise=True
if bench==2 or bench==3:
   gy=-10.
   rho1=3200.
   rho2=rho1+drho
   eta_ref=1e21      # scaling of G blocks
   xc_block=256e3
   yc_block=384e3
   d_block=64e3
   print('rho1=',rho1)
   print('rho2=',rho2)
   print('eta1=',eta1)
   print('eta2=',eta2)
   pnormalise=True
if bench==4 or bench==5 or bench==6:
   eta_ref=1.
   gy=1
   pnormalise=True
if bench==7:
   eta_ref=1.
   gy=-1
   pnormalise=True
if bench==8:
   llambda=256e3
   amplitude=2000
   eta_ref=1e21      # scaling of G blocks
   gy=-10
   phi1=2.*np.pi*(Ly/2.)/llambda
   phi2=2.*np.pi*(Ly/2.)/llambda
   pnormalise=True
if bench==10:
   gy=-10
   eta_ref=1e22
   amplitude=7e3
   pnormalise=False

sparse=False

rVnodes=[-1,1,1,-1,0]
sVnodes=[-1,-1,1,1,0]

xi=0.0 # controls level of mesh randomness (between 0 and 0.5 max)

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV = np.empty(NV,dtype=np.float64)  # x coordinates
yV = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        xV[counter]=i*hx
        yV[counter]=j*hy
        counter += 1

for j in range(0,nely):
    for i in range(0,nelx):
        xV[counter]=i*hx+1/2.*hx
        yV[counter]=j*hy+1/2.*hy
        counter += 1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        iconV[0, counter] = i + j * (nelx + 1)
        iconV[1, counter] = i + 1 + j * (nelx + 1)
        iconV[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        iconV[3, counter] = i + (j + 1) * (nelx + 1)
        iconV[4, counter] = (nelx+1)*(nely+1)+counter
        counter += 1

#################################################################
# add random noise to node positions
#################################################################

for i in range(0,NV):
    if xV[i]>0 and xV[i]<Lx and yV[i]>0 and yV[i]<Ly:
       xV[i]+=random.uniform(-1.,+1)*hx*xi
       yV[i]+=random.uniform(-1.,+1)*hy*xi
    #end if
#end for

for iel in range(0,nel):
    xV[iconV[4,iel]]=0.25*xV[iconV[0,iel]]+\
                    +0.25*xV[iconV[1,iel]]+\
                    +0.25*xV[iconV[2,iel]]+\
                    +0.25*xV[iconV[3,iel]]
    yV[iconV[4,iel]]=0.25*yV[iconV[0,iel]]+\
                    +0.25*yV[iconV[1,iel]]+\
                    +0.25*yV[iconV[2,iel]]+\
                    +0.25*yV[iconV[3,iel]]

#################################################################
# add sine perturbation for RT-instability
#################################################################

if bench==8: 
   for i in range(0,NV):
       if abs(yV[i]-Ly/2.)/Ly<eps:
          yV[i]+=amplitude*np.cos(2*np.pi*xV[i]/llambda)

   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           k=j*(nelx+1)+i
           ya=256e3+amplitude*np.cos(2*np.pi*xV[k]/llambda)
           if j<(nely+1)/2:
              dy=ya/(nely/2)
              yV[k]=j*dy
           else:
              dy=(Ly-ya)/(nely/2)
              yV[k]=ya+(j-nely/2)*dy

   for iel in range(0,nel):
       xV[iconV[4,iel]]=0.25*xV[iconV[0,iel]]+\
                       +0.25*xV[iconV[1,iel]]+\
                       +0.25*xV[iconV[2,iel]]+\
                       +0.25*xV[iconV[3,iel]]
       yV[iconV[4,iel]]=0.25*yV[iconV[0,iel]]+\
                       +0.25*yV[iconV[1,iel]]+\
                       +0.25*yV[iconV[2,iel]]+\
                       +0.25*yV[iconV[3,iel]]

#################################################################
# add sine perturbation for free surface benchmark 
#################################################################

if bench==10:
   for i in range(0,NV):
       if abs(yV[i]-Ly)/Ly<eps:
          yV[i]+=amplitude*np.cos(2*np.pi*xV[i]/Lx)

   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           k=j*(nelx+1)+i
           ymax=Ly+amplitude*np.cos(2*np.pi*xV[k]/Lx)-600e3
           dy=ymax/10
           if yV[k]>600e3:
              yV[k]=600e3+(j-60)*dy

   for iel in range(0,nel):
       xV[iconV[4,iel]]=0.25*xV[iconV[0,iel]]+\
                       +0.25*xV[iconV[1,iel]]+\
                       +0.25*xV[iconV[2,iel]]+\
                       +0.25*xV[iconV[3,iel]]
       yV[iconV[4,iel]]=0.25*yV[iconV[0,iel]]+\
                       +0.25*yV[iconV[1,iel]]+\
                       +0.25*yV[iconV[2,iel]]+\
                       +0.25*yV[iconV[3,iel]]

#################################################################
# build pressure grid and iconP 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates
iconP=np.zeros((mP,nel),dtype=np.int32)

xP[0:NP]=xV[0:NP]
yP[0:NP]=yV[0:NP]

iconP[0:mP,0:nel]=iconV[0:mP,0:nel]

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
#################################################################
start = timing.time()

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
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(NfemV, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(NfemV, dtype=np.float64)  # boundary condition, value

if bench==1:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

elif (bench==6 or bench==9):
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])

elif bench==8:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV  ] = 0 
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV  ] = 0
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

elif bench==10:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV]   = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

else: # free slip 
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0.
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
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
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

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

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
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

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            if bench==1 or bench==9:
               for i in range(0,mV):
                   f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)
            else:
               for i in range(0,mV):
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho(xq,yq)*gy

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNNP[:]+=NNNP[:]*jcob*weightq

        # end for jq
    # end for iq
    print('------------------')
    G_el*=18
    print(G_el)

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

    #G_el*=eta_ref/Ly
    #h_el*=eta_ref/Ly


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
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                if sparse:
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
    if sparse and pnormalise:
       A_sparse[Nfem,NfemV+m2]=constr[m2]
       A_sparse[NfemV+m2,Nfem]=constr[m2]

for i in range (NfemV):
    print ("%3i  & %3i & %3i & %3i & %3i &  %3i & %3i & %3i & %3i \\\\ " \
          %(int(round(G_mat[i,0])),int(round(G_mat[i,1])),int(round(G_mat[i,2])),int(round(G_mat[i,3])),int(round(G_mat[i,4])),int(round(G_mat[i,5])),int(round(G_mat[i,6])),int(round(G_mat[i,7])),int(round(G_mat[i,8]))))

#print (G_mat)
G2 = np.zeros((10,NfemP),dtype=np.float64) # matrix GT

print("----------------------------------------------")

G2[0,:]=G_mat[8,:]
G2[1,:]=G_mat[9,:]
G2[2,:]=G_mat[18,:]
G2[3,:]=G_mat[19,:]
G2[4,:]=G_mat[20,:]
G2[5,:]=G_mat[21,:]
G2[6,:]=G_mat[22,:]
G2[7,:]=G_mat[23,:]
G2[8,:]=G_mat[24,:]
G2[9,:]=G_mat[25,:]


#for i in range (10):
#    print ("%3f %3f %3f %3f %3f %3f %3f %3f %3f " %(G2[i,0],G2[i,1],G2[i,2],G2[i,3],G2[i,4],G2[i,5],G2[i,6],G2[i,7],G2[i,8]))

ns = null_space(G2)

print(ns)






