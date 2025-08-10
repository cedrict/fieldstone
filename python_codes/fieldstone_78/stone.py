import numpy as np
import sys as sys
import time as clock
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import mms_solkz as solkz
import mms_solcx as solcx
import mms_solvi as solvi
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numba import jit,float64
#from scikits.umfpack import spsolve, splu

eps=1.e-10

#exp=8
drho=8 # exp=8
rho0=3200

###############################################################################

@jit(nopython=True)
def bx(x,y):
    if experiment==2 or experiment==3 or experiment==4 or experiment==5 or\
       experiment==6 or experiment==7 or experiment==8 or experiment==10 or\
       experiment==11 or experiment==12 or experiment==13 or experiment==14:
       val=0.
    if experiment==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==9:
       val=(3*x**2*y**2-y-1)
    return val

@jit(nopython=True)
def by(x,y):
    if experiment==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==2:
       if np.abs(x-Lx/2)<0.125 and np.abs(y-Ly/2)<0.125:
          val=-1.02
       else:
          val= -1
    if experiment==3:
       if (x-Lx/2)**2+(y-Ly/2)**2<0.125**2:
          val=-1
       else:
          val=0
    if experiment==4:
       val=-1
    if experiment==5:
       val=np.sin(2*y)*np.cos(3*np.pi*x)
    if experiment==6:
       val=0.
    if experiment==7:
       val=-(8*x-2)
    if experiment==8:
       if abs(x-Lx/2)<64e3 and abs(y-384e3)<64e3:
          val=-10*(rho0+drho)
       else:
          val=-10*rho0
    if experiment==9:
       val=(2*x**3*y+3*x-1)
    if experiment==10:
       val=0.
    if experiment==11:
       val=0.
    if experiment==12:
       val=-10.
    if experiment==13:
       val=np.sin(np.pi*y)*np.cos(np.pi*x)
    if experiment==14:
       val=0
    return val


@jit(nopython=True)
def viscosity(x,y):
    if experiment==1 or experiment==2 or experiment==4 or \
       experiment==6 or experiment==7 or experiment==9 or \
       experiment==10 or experiment==11 or experiment==12:
       val=1
    if experiment==3:
       if (x-Lx/2)**2+(y-Ly/2)**2<0.125**2:
          val=1e4
       else:
          val=1
    if experiment==5:
       B=0.5*np.log(1e6)
       val=np.exp(2*B*y)
    if experiment==8:
       if abs(x-Lx/2)<64e3 and abs(y-384e3)<64e3:
          val=1e21*eta_star
       else:
          val=1e21
    if experiment==13:
       if x<0.5:
          val=1
       else:
          val=1e6
    if experiment==14:
       if (np.sqrt(x*x+y*y) < 0.2):
          val=1e3
       else:
          val=1.
    return val

###############################################################################

#@jit(nopython=True)
def velocity_x(x,y):
    if experiment==2 or experiment==3 or experiment==4 or\
       experiment==6 or experiment==8 or experiment==10 or\
       experiment==11 or experiment==12:
       val=0
    if experiment==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if experiment==5:
       val,xxx,xxxx=solkz.SolKzSolution(x,y)
    if experiment==7:
       val=(2*y-1)*x*(1-x)
    if experiment==9:
       val=x+x**2-2*x*y+x**3-3*x*y**2+x**2*y
    if experiment==13:
       val,xxx,xxx=solcx.SolCxSolution(x,y)
    if experiment==14:
       val,xxx,xxx=solvi.solution(x,y) 
    return val

#@jit(nopython=True)
def velocity_y(x,y):
    if experiment==2 or experiment==3 or experiment==4 or\
       experiment==6 or experiment==8 or experiment==10 or\
       experiment==11 or experiment==12:
       val=0
    if experiment==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==5:
       xxx,val,xxxx=solkz.SolKzSolution(x,y)
    if experiment==7:
       val=-(2*x-1)*y*(1-y)
    if experiment==9:
       val=-y-2*x*y+y**2-3*x**2*y+y**3-x*y**2
    if experiment==13:
       xxx,val,xxx=solcx.SolCxSolution(x,y)
    if experiment==14:
       xxx,val,xxx=solvi.solution(x,y) 
    return val

#@jit(nopython=True)
def pressure(x,y):
    if experiment==2 or experiment==3 or experiment==6 or\
       experiment==10 or experiment==11 or experiment==12:
       val=0
    if experiment==1:
       val=x*(1.-x)-1./6.
    if experiment==4:
       val=0.5-y
    if experiment==5:
       xxx,xxxx,val=solkz.SolKzSolution(x,y)
    if experiment==7:
       val=2*x*(1-2*y)
    if experiment==8:
       val=-32000*(y-Ly/2)
    if experiment==9:
       val=x*y+x+y+x**3*y**2-4./3.
    if experiment==13:
       xxx,xxx,val=solcx.SolCxSolution(x,y)
    if experiment==14:
       xxx,xxx,val=solvi.solution(x,y) 
    return val

###############################################################################

def NNV(r,s):
    NV_0= 0.25*(1-r)*(1-s) 
    NV_1= 0.25*(1+r)*(1-s) 
    NV_2= 0.25*(1+r)*(1+s) 
    NV_3= 0.25*(1-r)*(1+s) 
    return np.array([NV_0,NV_1,NV_2,NV_3],dtype=np.float64)

def dNNVdr(r,s):
    dNVdr_0=-0.25*(1.-s) 
    dNVdr_1=+0.25*(1.-s) 
    dNVdr_2=+0.25*(1.+s) 
    dNVdr_3=-0.25*(1.+s) 
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3],dtype=np.float64)

def dNNVds(r,s):
    dNVds_0=-0.25*(1.-r) 
    dNVds_1=-0.25*(1.+r) 
    dNVds_2=+0.25*(1.+r) 
    dNVds_3=+0.25*(1.-r) 
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3],dtype=np.float64)

###############################################################################

def area_triangle(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    ABx=x2-x1
    ABy=y2-y1
    ABz=z2-z1
    ACx=x3-x1
    ACy=y3-y1
    ACz=z3-z1
    # w1 = u2 v3 - u3 v2
    # w2 = u3 v1 - u1 v3
    # w3 = u1 v2 - u2 v1
    nx=ABy*ACz-ABz*ACy
    ny=ABz*ACx-ABx*ACz
    nz=ABx*ACy-ABy*ACx
    norm=np.sqrt(nx**2+ny**2+nz**2)
    return 0.5*norm


###############################################################################

print("-----------------------------")
print("---------- stone 78 ---------")
print("-----------------------------")

mV=4     # number of velocity nodes per element
mP=4     # number of pressure nodes per element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

# topo:
# 0: regular/structured (R)
# 1: Stenberg (S)
# 2: Le Tallec (LT)
# 3: Qin & Zhang (QZ1)
# 4: Qin & Zhang (QZ2)
# 5: Qin & Zhang (QZ3)
# 6: Thieulot (T1)
# 7: Thieulot (T2)
# 8: perturb R (Rp)
# 9: macro random (Rrp)
#10: fully random (FR)

# experiment:
# 1: mms donea huerta
# 2: block
# 3: sphere 
# 4: aquarium (retire)
# 5: mms solkz
# 6: regularised lid driven cavity
# 7: mms cavity (Elman et al)
# 8: sinking block
# 9: mms dohrmann bochev 
# 10: flow around square cylinder
# 11: flow over cavity
# 12: flow over obstacle
# 13: mms solcx
# 14: mms solvi

if int(len(sys.argv) == 7):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   topo = int(sys.argv[4])
   eta_star = int(sys.argv[5]) 
   experiment = int(sys.argv[6])
else:
   nelx = 16
   nely = 16
   visu = 1
   topo = 7
   eta_star=0
   experiment=5
   
if topo==0 or topo==8 or topo==9 or topo==10:
   nelx*=2
   nely*=2
   
eta_star=10**eta_star

nullspace=False
p_lagrange=False
matrix_snapshot=False
apply_RCM=False
correct_bcval=True

###############################################################################
# set specific values to some parameters for some experiments
###############################################################################

if nullspace: experiment=1

Lx=1. 
Ly=1.
eta_ref=1
vscaling=1.

if experiment==8:
   Lx=512e3
   Ly=512e3
   eta_ref=1e21
   vscaling=0.01/365.25/24/3600
 
if experiment==10 or experiment==11 or experiment==12:
   Lx=4
   nelx=4*nely

###############################################################################
###############################################################################
  
if topo==0: #regular
   import macro_R
   NV=(nelx+1)*(nely+1)
   nel=nelx*nely

if topo==1: #Stenberg  (S)
   import macro_S
   NV=nely*(5*nelx+2)+2*nelx+1
   nel=5*nelx*nely

if topo==2: #Le Tallec (LT)
   import macro_LT
   NV=(2*nelx+1)*(2*nely+1)+nely*nelx*8
   nel=12*nelx*nely

if topo==3: # qizh07 (QZ1)
   import macro_QZ1 
   nel=nelx*nely*12
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +9*nelx*nely

if topo==4: # qizh07 (QZ2)
   import macro_QZ2
   nel=nelx*nely*8
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +5*nelx*nely

if topo==5: # qizh07 (QZ3)
   import macro_QZ3
   nel=nelx*nely*6
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +3*nelx*nely

if topo==6: # mine (T1)
   import macro_T1
   nel=nelx*nely*7
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +4*nelx*nely

if topo==7: # mine (T2)
   import macro_T2
   nel=nelx*nely*5
   NV=(nelx+1)*(nely+1)+4*nelx*nely

if topo==8: # regular + perturbation
   import macro_Rp
   NV=(nelx+1)*(nely+1)
   nel=nelx*nely

if topo==9: # regular + rand perturbation
   import macro_Rrp
   NV=(nelx+1)*(nely+1)
   nel=nelx*nely

if topo==10: # fully random regular
   import macro_FR
   NV=(nelx+1)*(nely+1)
   nel=nelx*nely

###############################################################################

NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs
ndofV_el=mV*ndofV

print('nelx=',nelx)
print('nely=',nely)
print('NV=',NV)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('topo=',topo)
print('experiment=',experiment)
print('p_lagrange=',p_lagrange)
print('matrix_snapshot',matrix_snapshot)
print('apply_RCM=',apply_RCM)

print("-----------------------------")

###############################################################################
# quadrature points and weights
###############################################################################

nqperdim=2
qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
qweights=[1.,1.]
nqel=nqperdim**2

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

###############################################################################
# computing nodes coordinates and their connectivity
###############################################################################
start=clock.time()

if topo==0:  xV,yV,iconV=macro_R.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==1:  xV,yV,iconV=macro_S.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==2:  xV,yV,iconV=macro_LT.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==3:  xV,yV,iconV=macro_QZ1.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==4:  xV,yV,iconV=macro_QZ2.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==5:  xV,yV,iconV=macro_QZ3.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==6:  xV,yV,iconV=macro_T1.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==7:  xV,yV,iconV=macro_T2.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==8:  xV,yV,iconV=macro_Rp.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==9:  xV,yV,iconV=macro_Rrp.mesher(Lx,Ly,nelx,nely,nel,NV,mV)
if topo==10: xV,yV,iconV=macro_FR.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

print("build mesh: %.3f s" % (clock.time()-start))

###############################################################################
# compute coordinates of center of elements
###############################################################################
start=clock.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]=0.25*np.sum(xV[iconV[:,iel]])
    yc[iel]=0.25*np.sum(yV[iconV[:,iel]])

print("compute elt center coords: %.3f s" % (clock.time()-start))

###############################################################################
# compute area of elements
# This is a good test because it uses the quadrature points and 
# weights as well as the shape functions. If any area comes out
# negative or zero, or if the sum does not equal to the area of the 
# whole domain then there is a major problem which needs to 
# be addressed before FE are set into motion.
###############################################################################
start=clock.time()

area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

NNNV=np.zeros((nqel,mV),dtype=np.float64)
dNNNVdr=np.zeros((nqel,mV),dtype=np.float64)
dNNNVds=np.zeros((nqel,mV),dtype=np.float64)
JxWq=np.zeros((nqel,nel),dtype=np.float64)
xq=np.zeros((nqel,nel),dtype=np.float64)
yq=np.zeros((nqel,nel),dtype=np.float64)

for iel in range(0,nel):
    counterq=0
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[counterq,:]=NNV(rq,sq)
            dNNNVdr[counterq,:]=dNNVdr(rq,sq)
            dNNNVds[counterq,:]=dNNVds(rq,sq)
            jcb[0,0]=dNNNVdr[counterq,:].dot(xV[iconV[:,iel]])
            jcb[0,1]=dNNNVdr[counterq,:].dot(yV[iconV[:,iel]])
            jcb[1,0]=dNNNVds[counterq,:].dot(xV[iconV[:,iel]])
            jcb[1,1]=dNNNVds[counterq,:].dot(yV[iconV[:,iel]])
            jcob=np.linalg.det(jcb)
            JxWq[counterq,iel]=weightq*jcob
            xq[counterq,iel]=NNNV[counterq,:].dot(xV[iconV[:,iel]])
            yq[counterq,iel]=NNNV[counterq,:].dot(yV[iconV[:,iel]])
            area[iel]+=jcob*weightq
            counterq+=1
        #end for
    #end for

    if area[iel]<0: 
       for k in range(0,mV):
           print (xV[iconV[k,iel]],yV[iconV[k,iel]])
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area meas %.6f " %(area.sum()))
print("     -> total area anal %.6f " %(Lx*Ly))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if experiment==5 or experiment==8 or experiment==13:
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

elif experiment==6:
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 16.*xV[i]**2*(1-xV[i])**2
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

elif experiment==7 or experiment==9 or experiment==14:
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
       if yV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])
elif experiment==10:
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 1
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if abs(xV[i]-Lx/2-0.125)<0.0001 and abs(yV[i]-Ly/2)<0.1251:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-Lx/2+0.125)<0.0001 and abs(yV[i]-Ly/2)<0.1251:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-Lx/2)<0.1251 and abs(yV[i]-Ly/2-0.125)<0.0001:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-Lx/2)<0.1251 and abs(yV[i]-Ly/2+0.125)<0.0001:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

elif experiment==11:
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if yV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if yV[i]>0.4999*Ly and xV[i]<eps: 
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -(yV[i]-Ly/2)*(yV[i]-Ly)
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]<0.25001*Lx and abs(yV[i]-Ly/2)<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-0.25*Lx)<eps and yV[i]<0.5*Ly:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]>0.74999*Lx and abs(yV[i]-Ly/2)<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-0.75*Lx)<eps and yV[i]<0.5*Ly:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]>0.4999*Ly and xV[i]>Lx-eps: 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

elif experiment==12:
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 1
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if xV[i]>Lx-eps: 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if abs(xV[i]-0.5*Lx)<eps and yV[i]<0.25*Ly:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

else: # no slip on all sides
   for i in range(0,NV):
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if yV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute NV2 and allocate G2 for nullspace

if nullspace:

   N2=NfemV
   for i in range(0,NV):
       if bc_fix[i*ndofV  ]:  N2-=1 
       if bc_fix[i*ndofV+1]: N2-=1 

   G2=np.zeros((N2,nel),dtype=np.float64)

   print('NV=',NV)
   print('N2=',N2)
   print('G2=',np.shape(G2))

###############################################################################
# compute flux 
###############################################################################
start=clock.time()

flux_bottom=0
flux_top=0
flux_left=0
flux_right=0

for iel in range(0,nel):
    inode0=iconV[0,iel]
    inode1=iconV[1,iel]
    inode2=iconV[2,iel]
    inode3=iconV[3,iel]

    # bottom
    if abs(yV[inode0]-0)/Ly<eps and abs(yV[inode1]-0)/Ly<eps:
       flux_bottom+=abs(xV[inode0]-xV[inode1])*(bc_val[inode0*ndofV+1]+bc_val[inode1*ndofV+1])/2 *-1
    if abs(yV[inode1]-0)/Ly<eps and abs(yV[inode2]-0)/Ly<eps:
       flux_bottom+=abs(xV[inode1]-xV[inode2])*(bc_val[inode1*ndofV+1]+bc_val[inode2*ndofV+1])/2 *-1
    if abs(yV[inode2]-0)/Ly<eps and abs(yV[inode3]-0)/Ly<eps:
       flux_bottom+=abs(xV[inode2]-xV[inode3])*(bc_val[inode2*ndofV+1]+bc_val[inode3*ndofV+1])/2 *-1
    if abs(yV[inode3]-0)/Ly<eps and abs(yV[inode0]-0)/Ly<eps:
       flux_bottom+=abs(xV[inode3]-xV[inode0])*(bc_val[inode3*ndofV+1]+bc_val[inode0*ndofV+1])/2 *-1

    # top
    if abs(yV[inode0]-Ly)/Ly<eps and abs(yV[inode1]-Ly)/Ly<eps:
       flux_top+=abs(xV[inode0]-xV[inode1])*(bc_val[inode0*ndofV+1]+bc_val[inode1*ndofV+1])/2 * 1
    if abs(yV[inode1]-Ly)/Ly<eps and abs(yV[inode2]-Ly)/Ly<eps:
       flux_top+=abs(xV[inode1]-xV[inode2])*(bc_val[inode1*ndofV+1]+bc_val[inode2*ndofV+1])/2 * 1
    if abs(yV[inode2]-Ly)/Ly<eps and abs(yV[inode3]-Ly)/Ly<eps:
       flux_top+=abs(xV[inode2]-xV[inode3])*(bc_val[inode2*ndofV+1]+bc_val[inode3*ndofV+1])/2 * 1
    if abs(yV[inode3]-Ly)/Ly<eps and abs(yV[inode0]-Ly)/Ly<eps:
       flux_top+=abs(xV[inode3]-xV[inode0])*(bc_val[inode3*ndofV+1]+bc_val[inode0*ndofV+1])/2 * 1

    # right
    if abs(xV[inode0]-Lx)/Lx<eps and abs(xV[inode1]-Lx)/Lx<eps:
       flux_right+=abs(yV[inode0]-yV[inode1])*(bc_val[inode0*ndofV]+bc_val[inode1*ndofV])/2 * 1
    if abs(xV[inode1]-Lx)/Lx<eps and abs(xV[inode2]-Lx)/Lx<eps:
       flux_right+=abs(yV[inode1]-yV[inode2])*(bc_val[inode1*ndofV]+bc_val[inode2*ndofV])/2 * 1
    if abs(xV[inode2]-Lx)/Lx<eps and abs(xV[inode3]-Lx)/Lx<eps:
       flux_right+=abs(yV[inode2]-yV[inode3])*(bc_val[inode2*ndofV]+bc_val[inode3*ndofV])/2 * 1
    if abs(xV[inode3]-Lx)/Lx<eps and abs(xV[inode0]-Lx)/Lx<eps:
       flux_right+=abs(yV[inode3]-yV[inode0])*(bc_val[inode3*ndofV]+bc_val[inode0*ndofV])/2 * 1

    # left 
    if abs(xV[inode0]-0)/Lx<eps and abs(xV[inode1]-0)/Lx<eps:
       flux_left+=abs(yV[inode0]-yV[inode1])*(bc_val[inode0*ndofV]+bc_val[inode1*ndofV])/2 * -1
    if abs(xV[inode1]-0)/Lx<eps and abs(xV[inode2]-0)/Lx<eps:
       flux_left+=abs(yV[inode1]-yV[inode2])*(bc_val[inode1*ndofV]+bc_val[inode2*ndofV])/2 * -1
    if abs(xV[inode2]-0)/Lx<eps and abs(xV[inode3]-0)/Lx<eps:
       flux_left+=abs(yV[inode2]-yV[inode3])*(bc_val[inode2*ndofV]+bc_val[inode3*ndofV])/2 * -1
    if abs(xV[inode3]-0)/Lx<eps and abs(xV[inode0]-0)/Lx<eps:
       flux_left+=abs(yV[inode3]-yV[inode0])*(bc_val[inode3*ndofV]+bc_val[inode0*ndofV])/2 * -1

PHI=flux_bottom+flux_top+flux_left+flux_right

print('     -> flux b,t,l,r=',flux_bottom,flux_top,flux_left,flux_right)
print('     -> total_flux=',PHI)

print("compute bc val flux: %.3f s" % (clock.time()-start))

###############################################################################
# correct boundary conditions
###############################################################################
start=clock.time()

if correct_bcval:

   perim=2*Lx+2*Ly

   for i in range(0,NV):
       if yV[i]<eps: # bottom, only need to correct v 
          ny=-1
          bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])-PHI/perim*ny
       if yV[i]>(Ly-eps): #top, only need to correct v
          ny=+1
          bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i])-PHI/perim*ny
       if xV[i]<eps: # left, only need to correct u
          nx=-1
          bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])-PHI/perim*nx
       if xV[i]>(Lx-eps): #right, only need to correct u
          nx=+1
          bc_val[i*ndofV  ] = velocity_x(xV[i],yV[i])-PHI/perim*nx

   print("correct bc val: %.3f s" % (clock.time() - start))

###############################################################################
# compute flux again 
###############################################################################
start=clock.time()

if correct_bcval:

   flux_bottom=0
   flux_top=0
   flux_left=0
   flux_right=0

   for iel in range(0,nel):
       inode0=iconV[0,iel]
       inode1=iconV[1,iel]
       inode2=iconV[2,iel]
       inode3=iconV[3,iel]

       # bottom
       if abs(yV[inode0]-0)/Ly<eps and abs(yV[inode1]-0)/Ly<eps:
          flux_bottom+=abs(xV[inode0]-xV[inode1])*(bc_val[inode0*ndofV+1]+bc_val[inode1*ndofV+1])/2 *-1
       if abs(yV[inode1]-0)/Ly<eps and abs(yV[inode2]-0)/Ly<eps:
          flux_bottom+=abs(xV[inode1]-xV[inode2])*(bc_val[inode1*ndofV+1]+bc_val[inode2*ndofV+1])/2 *-1
       if abs(yV[inode2]-0)/Ly<eps and abs(yV[inode3]-0)/Ly<eps:
          flux_bottom+=abs(xV[inode2]-xV[inode3])*(bc_val[inode2*ndofV+1]+bc_val[inode3*ndofV+1])/2 *-1
       if abs(yV[inode3]-0)/Ly<eps and abs(yV[inode0]-0)/Ly<eps:
          flux_bottom+=abs(xV[inode3]-xV[inode0])*(bc_val[inode3*ndofV+1]+bc_val[inode0*ndofV+1])/2 *-1

       # top
       if abs(yV[inode0]-Ly)/Ly<eps and abs(yV[inode1]-Ly)/Ly<eps:
          flux_top+=abs(xV[inode0]-xV[inode1])*(bc_val[inode0*ndofV+1]+bc_val[inode1*ndofV+1])/2 * 1
       if abs(yV[inode1]-Ly)/Ly<eps and abs(yV[inode2]-Ly)/Ly<eps:
          flux_top+=abs(xV[inode1]-xV[inode2])*(bc_val[inode1*ndofV+1]+bc_val[inode2*ndofV+1])/2 * 1
       if abs(yV[inode2]-Ly)/Ly<eps and abs(yV[inode3]-Ly)/Ly<eps:
          flux_top+=abs(xV[inode2]-xV[inode3])*(bc_val[inode2*ndofV+1]+bc_val[inode3*ndofV+1])/2 * 1
       if abs(yV[inode3]-Ly)/Ly<eps and abs(yV[inode0]-Ly)/Ly<eps:
          flux_top+=abs(xV[inode3]-xV[inode0])*(bc_val[inode3*ndofV+1]+bc_val[inode0*ndofV+1])/2 * 1

       # right
       if abs(xV[inode0]-Lx)/Lx<eps and abs(xV[inode1]-Lx)/Lx<eps:
          flux_right+=abs(yV[inode0]-yV[inode1])*(bc_val[inode0*ndofV]+bc_val[inode1*ndofV])/2 * 1
       if abs(xV[inode1]-Lx)/Lx<eps and abs(xV[inode2]-Lx)/Lx<eps:
          flux_right+=abs(yV[inode1]-yV[inode2])*(bc_val[inode1*ndofV]+bc_val[inode2*ndofV])/2 * 1
       if abs(xV[inode2]-Lx)/Lx<eps and abs(xV[inode3]-Lx)/Lx<eps:
          flux_right+=abs(yV[inode2]-yV[inode3])*(bc_val[inode2*ndofV]+bc_val[inode3*ndofV])/2 * 1
       if abs(xV[inode3]-Lx)/Lx<eps and abs(xV[inode0]-Lx)/Lx<eps:
          flux_right+=abs(yV[inode3]-yV[inode0])*(bc_val[inode3*ndofV]+bc_val[inode0*ndofV])/2 * 1

       # left 
       if abs(xV[inode0]-0)/Lx<eps and abs(xV[inode1]-0)/Lx<eps:
          flux_left+=abs(yV[inode0]-yV[inode1])*(bc_val[inode0*ndofV]+bc_val[inode1*ndofV])/2 * -1
       if abs(xV[inode1]-0)/Lx<eps and abs(xV[inode2]-0)/Lx<eps:
          flux_left+=abs(yV[inode1]-yV[inode2])*(bc_val[inode1*ndofV]+bc_val[inode2*ndofV])/2 * -1
       if abs(xV[inode2]-0)/Lx<eps and abs(xV[inode3]-0)/Lx<eps:
          flux_left+=abs(yV[inode2]-yV[inode3])*(bc_val[inode2*ndofV]+bc_val[inode3*ndofV])/2 * -1
       if abs(xV[inode3]-0)/Lx<eps and abs(xV[inode0]-0)/Lx<eps:
          flux_left+=abs(yV[inode3]-yV[inode0])*(bc_val[inode3*ndofV]+bc_val[inode0*ndofV])/2 * -1

   PHI=flux_bottom+flux_top+flux_left+flux_right

   print('     -> flux b,t,l,r=',flux_bottom,flux_top,flux_left,flux_right)
   print('     -> total_flux=',PHI)

   print("compute bc val flux: %.3f s" % (clock.time()-start))

###############################################################################
# compute array for assembly
###############################################################################
start=clock.time()

local_to_globalV=np.zeros((ndofV_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
  
print("compute local_to_global: %.3f s" % (clock.time()-start))

###############################################################################
# fill I,J arrays
# bignb is nel*(nb of floats needed to store Kel and 2*Gel)
###############################################################################
start=clock.time()

bignb=nel*( (mV*ndofV)**2 + 2*(mV*ndofV*mP) )

I=np.zeros(bignb,dtype=np.int32)
J=np.zeros(bignb,dtype=np.int32)
V=np.zeros(bignb,dtype=np.float64)

counter=0
for iel in range(0,nel):
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndofV_el):
            m2=local_to_globalV[jkk,iel]
            I[counter]=m1
            J[counter]=m2
            counter+=1
        #for jkk in range(0,mP):
        #    m2 =iconP[jkk,iel]+NfemV
        m2=iel+NfemV
        I[counter]=m1
        J[counter]=m2
        counter+=1
        I[counter]=m2
        J[counter]=m1
        counter+=1

print("fill I,J arrays: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

if p_lagrange:
   A_mat = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)# matrix A 
   rhs   = np.zeros((Nfem+1),dtype=np.float64)         # right hand side 
else:
   A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side 

if (nullspace): 
   G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

tA=0 ; tB=0 ; tC=0

counter=0
for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    start=clock.time()
    for iq in range(0,nqel):

        # calculate jacobian matrix
        jcb[0,0]=dNNNVdr[iq,:].dot(xV[iconV[:,iel]])
        jcb[0,1]=dNNNVdr[iq,:].dot(yV[iconV[:,iel]])
        jcb[1,0]=dNNNVds[iq,:].dot(xV[iconV[:,iel]])
        jcb[1,1]=dNNNVds[iq,:].dot(yV[iconV[:,iel]])
        jcbi=np.linalg.inv(jcb)

        # compute dNdx & dNdy
        dNNNVdx[:]=jcbi[0,0]*dNNNVdr[iq,:]+jcbi[0,1]*dNNNVds[iq,:]
        dNNNVdy[:]=jcbi[1,0]*dNNNVdr[iq,:]+jcbi[1,1]*dNNNVds[iq,:]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0,2*i  ]=dNNNVdx[i]
            b_mat[1,2*i+1]=dNNNVdy[i]
            b_mat[2,2*i  ]=dNNNVdy[i]
            b_mat[2,2*i+1]=dNNNVdx[i]

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq[iq,iel],yq[iq,iel])*JxWq[iq,iel]

        # compute elemental rhs vector
        for i in range(0,mV):
            f_el[ndofV*i  ]+=NNNV[iq,i]*bx(xq[iq,iel],yq[iq,iel])*JxWq[iq,iel]
            f_el[ndofV*i+1]+=NNNV[iq,i]*by(xq[iq,iel],yq[iq,iel])*JxWq[iq,iel]
            G_el[ndofV*i  ,0]-=dNNNVdx[i]*JxWq[iq,iel]
            G_el[ndofV*i+1,0]-=dNNNVdy[i]*JxWq[iq,iel]

    #end for iq
    G_el*=eta_ref/Lx
    tA+=clock.time()-start

    # impose b.c. 
    start=clock.time()
    for ikk in range(0,ndofV_el):
            m1=local_to_globalV[ikk,iel]
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,ndofV_el):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0,0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0
    tB+=clock.time()-start

    # assemble matrix K_mat and right hand side rhs
    start=clock.time()
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        if (nullspace): G_mat[m1,iel]+=G_el[ikk,0]
        for jkk in range(ndofV_el):
            V[counter]=K_el[ikk,jkk]
            counter+=1
        V[counter]=G_el[ikk,0]
        counter+=1
        V[counter]=G_el[ikk,0]
        counter+=1
        rhs[m1]+=f_el[ikk]
    rhs[NfemV+iel]+=h_el[0,0]



#            m2=local_to_globalV[jkk,iel]
#            A_mat[m1,m2]+=K_el[ikk,jkk]
#        rhs[m1]+=f_el[ikk]
#        A_mat[m1,NfemV+iel]+=G_el[ikk,0]
#        A_mat[NfemV+iel,m1]+=G_el[ikk,0]
#    rhs[NfemV+iel]+=h_el[0,0]
    tC+=clock.time()-start



    if p_lagrange:
       A_mat[Nfem,NfemV+iel]=area[iel]
       A_mat[NfemV+iel,Nfem]=area[iel]
    #if p_lagrange and iel==nel-1:
    #   A_mat[Nfem,NfemV+iel]=1
    #   A_mat[NfemV+iel,Nfem]=1
    #if p_lagrange and iel<=nelx-1:
    #   A_mat[Nfem,NfemV+iel]=1
    #   A_mat[NfemV+iel,Nfem]=1

#end for iel

print('    -> make elt matrices: %.3f s' %tA)
print('    -> bound conds: %.3f s' %tB)
print('    -> assembly: %.3f s' %tC)

print("build FE matrix: %.3f s" % (tA+tB+tC))

###############################################################################
# compute null space of G matrix
###############################################################################

if nullspace:

   print("-----------------------------")
   #print('G matrix:')
   #for i in range (NfemV):
   #    print (i,'|',G_mat[i,:])

   counter=0
   for i in range(0,NV):
       for i1 in range(0,ndofV):
           m1=ndofV*i+i1
           if not bc_fix[m1]:
              #print (m1)
              G2[counter,:]=G_mat[m1,:]
              counter+=1 

   ns=null_space(G2)
   ns_size=np.shape(ns)[1]
   if topo==0 or topo==8 or topo==9 or topo==10: 
      coeff=2
   else:
      coeff=1
   print('topo=',topo,'null space size:',ns_size,'|',nelx/coeff,nel)
   for ins in range(ns_size):
       #print(np.average(ns[:,ins]))
       ns[:,ins]-=np.average(ns[:,ins])
       ns[:,ins]/=np.max(ns[:,ins])
   #print(ns)
   exit()

###############################################################################
# apply reverse Cuthill-McKee algorithm 
###############################################################################
start=clock.time()
   
A_csr=A_mat.tocsr()

#take snapshot of matrix before reordering
if matrix_snapshot:
   plt.spy(A_csr, markersize=0.2)
   plt.savefig('A_bef.png', bbox_inches='tight')
   plt.clf()
   print('     -> A_bef.png')

if apply_RCM:
   #compute reordering array
   perm = reverse_cuthill_mckee(A_csr,symmetric_mode=True)
   #build reverse perm array
   perm_inv=np.empty(len(perm),dtype=np.int32)
   for i in range(0,len(perm)):
       perm_inv[perm[i]]=i
   A_csr=A_csr[np.ix_(perm,perm)]
   rhs=rhs[np.ix_(perm)]

if matrix_snapshot:
   #take snapshot of matrix after reordering
   plt.spy(A_csr, markersize=0.2)
   plt.savefig('A_aft.png', bbox_inches='tight')
   plt.clf()
   print('     -> A_aft.png')

print("apply Reverse Cuthill-Mckee reordering: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

#NEW
sparse_matrix = sps.coo_matrix((V,(I,J)),shape=(Nfem,Nfem)).tocsr()
sol=sps.linalg.spsolve(sparse_matrix,rhs)
   
if apply_RCM:
   sol=sol[np.ix_(perm_inv)]

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*eta_ref/Lx

print("     -> u (m,M) %12.4e %12.4e nel= %d" %(np.min(u),np.max(u),nel))
print("     -> v (m,M) %12.4e %12.4e nel= %d" %(np.min(v),np.max(v),nel))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
start=clock.time()

avrg_p=np.sum(p[:]*area[:])/Lx/Ly

print('     -> avrg p=',avrg_p)

p[:]-=avrg_p

#np.savetxt('p.ascii',np.array([xc,yc,p]).T)

print("     -> p (m,M) %.6f %.6f nel= %d" %(np.min(p),np.max(p),nel))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute q1 &q2 nodal pressure (corner to node averaging)
###############################################################################
start=clock.time()

q1=np.zeros(NV,dtype=np.float64)  
q2=np.zeros(NV,dtype=np.float64)  
q3=np.zeros(NV,dtype=np.float64)  
count1=np.zeros(NV,dtype=np.int32) 
count2=np.zeros(NV,dtype=np.float64) 
count3=np.zeros(NV,dtype=np.float64) 

for iel in range(0,nel):
    for k in range(0,mV):
        inode=iconV[k,iel]
        q1[inode]+=p[iel]
        q2[inode]+=p[iel]*area[iel]
        count1[inode]+=1
        count2[inode]+=area[iel]
    #end for
    i0=iconV[0,iel]
    i1=iconV[1,iel]
    i2=iconV[2,iel]
    i3=iconV[3,iel]
    Atr0=area_triangle(xV[i3],yV[i3],0,xV[i0],yV[i0],0,xV[i1],yV[i1],0)
    Atr1=area_triangle(xV[i0],yV[i0],0,xV[i1],yV[i1],0,xV[i2],yV[i2],0)
    Atr2=area_triangle(xV[i1],yV[i1],0,xV[i2],yV[i2],0,xV[i3],yV[i3],0)
    Atr3=area_triangle(xV[i2],yV[i2],0,xV[i3],yV[i3],0,xV[i0],yV[i0],0)
    q3[i0]+=p[iel]*Atr0
    q3[i1]+=p[iel]*Atr1
    q3[i2]+=p[iel]*Atr2
    q3[i3]+=p[iel]*Atr3
    count3[i0]+=Atr0
    count3[i1]+=Atr1
    count3[i2]+=Atr2
    count3[i3]+=Atr3

q1/=count1
q2/=count2
q3/=count3

#np.savetxt('q1.ascii',np.array([xV,yV,q1]).T)
#np.savetxt('q2.ascii',np.array([xV,yV,q2]).T)
#np.savetxt('q3.ascii',np.array([xV,yV,q3]).T)

print("     -> q1 (m,M) %.6f %.6f nel= %d" %(np.min(q1),np.max(q1),nel))
print("     -> q2 (m,M) %.6f %.6f nel= %d" %(np.min(q2),np.max(q2),nel))
print("     -> q3 (m,M) %.6f %.6f nel= %d" %(np.min(q3),np.max(q3),nel))

print("compute nodal pressure q1,q2,q3: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

vrms=0.
errv=0.
errp=0.
errq1=0.
errq2=0.
errq3=0.
for iel in range (0,nel):
    for iq in range(0,nqel):
        uq=NNNV[iq,:].dot(u[iconV[:,iel]])
        vq=NNNV[iq,:].dot(v[iconV[:,iel]])
        q1q=NNNV[iq,:].dot(q1[iconV[:,iel]])
        q2q=NNNV[iq,:].dot(q2[iconV[:,iel]])
        q3q=NNNV[iq,:].dot(q3[iconV[:,iel]])
        errv+=((uq-velocity_x(xq[iq,iel],yq[iq,iel]))**2\
              +(vq-velocity_y(xq[iq,iel],yq[iq,iel]))**2)*JxWq[iq,iel]
        errp+=(p[iel]-pressure(xq[iq,iel],yq[iq,iel]))**2*JxWq[iq,iel]
        errq1+=(q1q-pressure(xq[iq,iel],yq[iq,iel]))**2*JxWq[iq,iel]
        errq2+=(q2q-pressure(xq[iq,iel],yq[iq,iel]))**2*JxWq[iq,iel]
        errq3+=(q3q-pressure(xq[iq,iel],yq[iq,iel]))**2*JxWq[iq,iel]
        vrms+=(uq**2+vq**2)*JxWq[iq,iel]
    #end for
#end for
errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq1=np.sqrt(errq1)
errq2=np.sqrt(errq2)
errq3=np.sqrt(errq3)
vrms=np.sqrt(vrms/(Lx*Ly))

print("     -> nel= %6d ; errv= %.11f ; errp= %.11f ; errq1= %.11f ; errq2= %.11f ; errq3= %.11f"\
          %(nel,errv,errp,errq1,errq2,errq3))
print("     -> nel= %6d ; vrms= %12.6e " %(nel,vrms))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# export profiles
# only export pressure if a complete edge is on boundary
###############################################################################
start=clock.time()

pfile=open('pressure_top.ascii',"w")
for iel in range(0,nel):
    if abs(yV[iconV[0,iel]]-Ly)<eps and abs(yV[iconV[1,iel]]-Ly)<eps:
       pfile.write("%e %e %e \n" %(xc[iel],p[iel],pressure(xc[iel],yc[iel])))
    if abs(yV[iconV[1,iel]]-Ly)<eps and abs(yV[iconV[2,iel]]-Ly)<eps:
       pfile.write("%e %e %e \n" %(xc[iel],p[iel],pressure(xc[iel],yc[iel])))
    if abs(yV[iconV[2,iel]]-Ly)<eps and abs(yV[iconV[3,iel]]-Ly)<eps:
       pfile.write("%e %e %e \n" %(xc[iel],p[iel],pressure(xc[iel],yc[iel])))
    if abs(yV[iconV[3,iel]]-Ly)<eps and abs(yV[iconV[0,iel]]-Ly)<eps:
       pfile.write("%e %e %e \n" %(xc[iel],p[iel],pressure(xc[iel],yc[iel])))
pfile.close()

vfile=open('vel_profile.ascii',"w")
qfile=open('q1_profile.ascii',"w")
for i in range(0,NV):
    if abs(xV[i]-Lx/2)<eps:
       vfile.write("%e %e %e %e %e\n" %(yV[i],u[i],v[i],\
                                        velocity_x(xV[i],yV[i]),\
                                        velocity_y(xV[i],yV[i])))
       qfile.write("%e %e\n" %(yV[i],q1[i]))
vfile.close()
qfile.close()

if experiment==8:
   xc_block=256e3
   yc_block=384e3
   for iel in range(0,nel):
       if abs(xV[iconV[0,iel]]-xc_block)/Lx<eps and abs(yV[iconV[0,iel]]-yc_block)/Ly<eps:
          print ('pblock:',eta_star,p[iel],q1[iconV[0,iel]],drho)
       if abs(xV[iconV[1,iel]]-xc_block)/Lx<eps and abs(yV[iconV[1,iel]]-yc_block)/Ly<eps:
          print ('pblock:',eta_star,p[iel],q1[iconV[1,iel]],drho)
       if abs(xV[iconV[2,iel]]-xc_block)/Lx<eps and abs(yV[iconV[2,iel]]-yc_block)/Ly<eps:
          print ('pblock:',eta_star,p[iel],q1[iconV[2,iel]],drho)
       if abs(xV[iconV[3,iel]]-xc_block)/Lx<eps and abs(yV[iconV[3,iel]]-yc_block)/Ly<eps:
          print ('pblock:',eta_star,p[iel],q1[iconV[3,iel]],drho)
   for i in range(0,NV):
       if abs(xV[i]-xc_block)/Lx<eps and abs(yV[i]-yc_block)/Ly<eps:
          print('vblock:',eta_star,u[i],v[i],drho)

print("export profiles: %.3f s" % (clock.time()-start))

###############################################################################
# compute error fields for visualisation
###############################################################################
start=clock.time()

error_u = np.zeros(NV,dtype=np.float64)
error_v = np.zeros(NV,dtype=np.float64)
error_q1 = np.zeros(NV,dtype=np.float64)
error_q2 = np.zeros(NV,dtype=np.float64)
error_q3 = np.zeros(NV,dtype=np.float64)
error_p = np.zeros(nel,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(xV[i],yV[i])
    error_v[i]=v[i]-velocity_y(xV[i],yV[i])
    error_q1[i]=q1[i]-pressure(xV[i],yV[i])
    error_q2[i]=q2[i]-pressure(xV[i],yV[i])
    error_q3[i]=q3[i]-pressure(xV[i],yV[i])

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i])

print("create error fields: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

if visu:
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
   #
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(p[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(error_p[iel]))
   vtufile.write("</DataArray>\n")
   if experiment==3 or experiment==5 or experiment==8 or experiment==13:
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for iel in range(0,nel):
           vtufile.write("%10e \n" %(viscosity(xc[iel],yc[iel])))
      vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='p analytical' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(pressure(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='bx' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(bx(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='by' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%e \n" %(by(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%e \n" %(area[iel]))
   vtufile.write("</DataArray>\n")
   #--
   if nullspace:
      vtufile.write("<DataArray type='Float32' Name='nullspace' Format='ascii'> \n")
      for iel in range(0,nel):
              vtufile.write("%10e \n" %(ns[iel,0]))
      vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i]/vscaling,v[i]/vscaling,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (analytical)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(velocity_x(xV[i],yV[i]),velocity_y(xV[i],yV[i]),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(error_u[i]/vscaling,error_v[i]/vscaling,0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(q1[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(q2[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(q3[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q1 (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(error_q1[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q2 (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(error_q2[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q3 (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(error_q3[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
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

   print("export to vtu: %.3f s" % (clock.time()-start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
