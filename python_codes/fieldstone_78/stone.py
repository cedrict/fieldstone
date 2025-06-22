import numpy as np
import sys as sys
import time as timing
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import solkz
import solcx
import solvi

eps=1.e-10

###############################################################################

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
          val=-10*3208#+32000
       else:
          val=-10*3200#+32000
       val=-32000
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
          val=1e24
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

print("-----------------------------")
print("---------- stone 78 ---------")
print("-----------------------------")

mV=4     # number of velocity nodes per element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

# topo:
# 0: regular/structured
# 1: Stenberg (S)
# 2: Le Tallec (LT)
# 3: Qin & Zhang (QZ1)
# 4: Qin & Zhang (QZ2)
# 5: Qin & Zhang (QZ3)
# 6: Thieulot (A)
# 7: Thieulot (B)

# experiment:
# 1: mms donea huerta
# 2: block
# 3: sphere 
# 4: aquarium (retire)
# 5: mms solkz
# 6: regularised lid driven cavity
# 7: mms cavity
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
   epsi = float(sys.argv[5]) # only for topo=2
   experiment = int(sys.argv[6])
else:
   nelx = 32
   nely = 32
   visu = 1
   topo = 1
   epsi = 0
   experiment=5

pnormalise=True 

###############################################################################
# set specific values to some parameters for some experiments
###############################################################################

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
   #pnormalise=False 
   Lx=4
   nelx=4*nely

###############################################################################
###############################################################################
  
if topo==0: #regular
   import regular
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

if topo==6: # mine (A)
   import macro_A
   nel=nelx*nely*7
   NV=(nelx+1)*(nely+1) +nelx*(nely+1) +nely*(nelx+1) +4*nelx*nely

if topo==7: # mine (B)
   import macro_B
   nel=nelx*nely*5
   NV=(nelx+1)*(nely+1)+4*nelx*nely

###############################################################################

NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

print('nelx=',nelx)
print('nely=',nely)
print('NV=',NV)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('topo=',topo)
print('experiment=',experiment)
print("-----------------------------")

###############################################################################
# quadrature points and weights
###############################################################################

nqperdim=2
qcoords=[-1./np.sqrt(3.),1./np.sqrt(3.)]
qweights=[1.,1.]

rVnodes=[-1,+1,+1,-1]
sVnodes=[-1,-1,+1,+1]

###############################################################################
# computing nodes coordinates and their connectivity
###############################################################################
start = timing.time()

if topo==0:
   xV,yV,iconV=regular.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==1:
   xV,yV,iconV=macro_S.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==2:
   xV,yV,iconV=macro_LT.mesher(Lx,Ly,nelx,nely,nel,NV,mV,epsi)

if topo==3:
   xV,yV,iconV=macro_QZ1.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==4:
   xV,yV,iconV=macro_QZ2.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==5:
   xV,yV,iconV=macro_QZ3.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==6:
   xV,yV,iconV=macro_A.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

if topo==7:
   xV,yV,iconV=macro_B.mesher(Lx,Ly,nelx,nely,nel,NV,mV)

print("build mesh: %.3f s" % (timing.time() - start))

###############################################################################
# compute coordinates of center of elements
###############################################################################

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]=0.25*np.sum(xV[iconV[:,iel]])
    yc[iel]=0.25*np.sum(yV[iconV[:,iel]])

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
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)
            jcb[0,0]=dNNNVdr[:].dot(xV[iconV[:,iel]])
            jcb[0,1]=dNNNVdr[:].dot(yV[iconV[:,iel]])
            jcb[1,0]=dNNNVds[:].dot(xV[iconV[:,iel]])
            jcb[1,1]=dNNNVds[:].dot(yV[iconV[:,iel]])
            jcob=np.linalg.det(jcb)
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

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

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

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = timing.time()

#if pnormalise:
#   A_mat = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)# matrix A 
#   rhs   = np.zeros((Nfem+1),dtype=np.float64)         # right hand side 
#else:
A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)
    jcb=np.zeros((2,2),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV=NNV(rq,sq)
            dNNNVdr=dNNVdr(rq,sq)
            dNNNVds=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=dNNNVdr[:].dot(xV[iconV[:,iel]])
            jcb[0,1]=dNNNVdr[:].dot(yV[iconV[:,iel]])
            jcb[1,0]=dNNNVds[:].dot(xV[iconV[:,iel]])
            jcb[1,1]=dNNNVds[:].dot(yV[iconV[:,iel]])
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute coordinates of quad points
            xq=NNNV[:].dot(xV[iconV[:,iel]])
            yq=NNNV[:].dot(yV[iconV[:,iel]])

            # compute dNdx & dNdy
            for k in range(0, mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0, mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.        ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xc[iel],yc[iel])*weightq*jcob
            #K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNNNVdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNNNVdy[i]*jcob*weightq

    G_el*=eta_ref/Lx

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
               h_el[0,0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    A_mat[m1,m2]+=K_el[ikk,jkk]
            rhs[m1]+=f_el[ikk]
            A_mat[m1,NfemV+iel]+=G_el[ikk,0]
            A_mat[NfemV+iel,m1]+=G_el[ikk,0]
    rhs[NfemV+iel]+=h_el[0,0]

    #if pnormalise:
    #   A_mat[Nfem,NfemV+iel]=area[iel]
    #   A_mat[NfemV+iel,Nfem]=area[iel]

#end for iel

print("build FE matrix: %.3f s" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

A_mat=A_mat.tocsr()
sol=sps.linalg.spsolve(A_mat,rhs)

print("solve time: %.3f s" % (timing.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*eta_ref/Lx

print("     -> u (m,M) %12.4e %12.4e nel= %d" %(np.min(u),np.max(u),nel))
print("     -> v (m,M) %12.4e %12.4e nel= %d" %(np.min(v),np.max(v),nel))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (timing.time() - start))

###############################################################################
###############################################################################
start = timing.time()

avrg_p=np.sum(p[:]*area[:])/Lx/Ly

print('     -> avrg p=',avrg_p)

p[:]-=avrg_p

#np.savetxt('p.ascii',np.array([xc,yc,p]).T)

print("     -> p (m,M) %.6f %.6f nel= %d" %(np.min(p),np.max(p),nel))

print("normalise pressure: %.3f s" % (timing.time() - start))

###############################################################################
# compute q1 &q2 nodal pressure (corner to node averaging)
###############################################################################
start = timing.time()

q1=np.zeros(NV,dtype=np.float64)  
q2=np.zeros(NV,dtype=np.float64)  
count1=np.zeros(NV,dtype=np.int32) 
count2=np.zeros(NV,dtype=np.float64) 

for iel in range(0,nel):
    for k in range(0,mV):
        inode=iconV[k,iel]
        q1[inode]+=p[iel]
        q2[inode]+=p[iel]*area[iel]
        count1[inode]+=1
        count2[inode]+=area[iel]

q1/=count1
q2/=count2

#np.savetxt('q1.ascii',np.array([xV,yV,q1]).T)
#np.savetxt('q2.ascii',np.array([xV,yV,q2]).T)

print("     -> q1 (m,M) %.6f %.6f nel= %d" %(np.min(q1),np.max(q1),nel))
print("     -> q2 (m,M) %.6f %.6f nel= %d" %(np.min(q2),np.max(q2),nel))

print("compute nodal pressure q1 & q2: %.3f s" % (timing.time() - start))

###############################################################################
# compute error
###############################################################################
start = timing.time()

error_u = np.zeros(NV,dtype=np.float64)
error_v = np.zeros(NV,dtype=np.float64)
error_q1 = np.zeros(NV,dtype=np.float64)
error_q2 = np.zeros(NV,dtype=np.float64)
error_p = np.zeros(nel,dtype=np.float64)
jcb=np.zeros((2,2),dtype=np.float64)

if True: 

   for i in range(0,NV): 
       error_u[i]=u[i]-velocity_x(xV[i],yV[i])
       error_v[i]=v[i]-velocity_y(xV[i],yV[i])
       error_q1[i]=q1[i]-pressure(xV[i],yV[i])
       error_q2[i]=q2[i]-pressure(xV[i],yV[i])

   for i in range(0,nel): 
       error_p[i]=p[i]-pressure(xc[i],yc[i])

   vrms=0.
   errv=0.
   errp=0.
   errq1=0.
   errq2=0.
   for iel in range (0,nel):
       for iq in range(0,nqperdim):
           for jq in range(0,nqperdim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV=NNV(rq,sq)
               dNNNVdr=dNNVdr(rq,sq)
               dNNNVds=dNNVds(rq,sq)
               jcb[0,0]=dNNNVdr[:].dot(xV[iconV[:,iel]])
               jcb[0,1]=dNNNVdr[:].dot(yV[iconV[:,iel]])
               jcb[1,0]=dNNNVds[:].dot(xV[iconV[:,iel]])
               jcb[1,1]=dNNNVds[:].dot(yV[iconV[:,iel]])
               jcob = np.linalg.det(jcb)
               xq=NNNV[:].dot(xV[iconV[:,iel]])
               yq=NNNV[:].dot(yV[iconV[:,iel]])
               uq=NNNV[:].dot(u[iconV[:,iel]])
               vq=NNNV[:].dot(v[iconV[:,iel]])
               q1q=NNNV[:].dot(q1[iconV[:,iel]])
               q2q=NNNV[:].dot(q2[iconV[:,iel]])
               errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
               errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
               errq1+=(q1q-pressure(xq,yq))**2*weightq*jcob
               errq2+=(q2q-pressure(xq,yq))**2*weightq*jcob
               vrms+=(uq**2+vq**2)*weightq*jcob
           #end for
       #end for
   #end for
   errv=np.sqrt(errv)
   errp=np.sqrt(errp)
   errq1=np.sqrt(errq1)
   errq2=np.sqrt(errq2)
   vrms=np.sqrt(vrms/(Lx*Ly))

   print("     -> nel= %6d ; errv= %.8f ; errp= %.8f ; errq1= %.8f ; errq2= %.8f" %(nel,errv,errp,errq1,errq2))
   print("     -> nel= %6d ; vrms= %12.4e " %(nel,vrms))

print("compute errors: %.3f s" % (timing.time() - start))

###############################################################################
# export profiles
###############################################################################
start = timing.time()

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
for i in range(0,NV):
    if abs(xV[i]-Lx/2)<eps:
       vfile.write("%e %e %e %e %e\n" %(yV[i],u[i],v[i],\
                                        velocity_x(xV[i],yV[i]),\
                                        velocity_y(xV[i],yV[i])))
vfile.close()

print("export profiles: %.3f s" % (timing.time() - start))

###############################################################################
# plot of solution
###############################################################################

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

   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(p[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10e \n" %(error_p[iel]))
   vtufile.write("</DataArray>\n")
   if experiment==3 or experiment==5 or experiment==8:
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
           vtufile.write("%10e \n" %(by(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
           vtufile.write("%10e \n" %(area[iel]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i]/vscaling,v[i]/vscaling,0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (analytical)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(velocity_x(xV[i],yV[i]),velocity_y(xV[i],yV[i]),0.))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
