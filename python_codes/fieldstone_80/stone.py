import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import solvi 
import solcx 

#------------------------------------------------------------------------------

def bx(x,y):
    if benchmark==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if benchmark==2:
       val=0
    if benchmark==3:
       val=0
    if benchmark==4:
       val= 3*x**2*y**2-y-1
    if benchmark==5:
       val=0
    if benchmark==6:
       val=0
    if benchmark==7:
       val=0
    if benchmark==8:
       val=0
    if benchmark==9:
       val=0
    return val

def by(x,y):
    if benchmark==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if benchmark==2:
       val=-1
    if benchmark==3:
       val=0
    if benchmark==4:
       val= 2*x**3*y+3*x-1
    if benchmark==5:
       if ((x-0.5)**2+(y-0.5)**2 < 0.123**2):
          val=-1.001+1
       else:
          val=-1.+1
    if benchmark==6:
       val=0
    if benchmark==7:
       val=np.sin(np.pi*y)*np.cos(np.pi*x)
    if benchmark==8:
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=rho2*gy 
       else:
          val=rho1*gy 
    if benchmark==9:
       val=0
    return val

def eta(x,y):
    if benchmark==5: #stokes sphere
       if ((x-0.5)**2+(y-0.5)**2 < 0.123**2):
          val=1#e6
       else:
          val=1
       return val
    elif benchmark==6: #solvi
       if (np.sqrt(x*x+y*y) < 0.2):
          val=1e3
       else:
          val=1.
       return val
    if benchmark==7: #solcx
       if x<0.5:
          val=1.
       else:
          val=1.e6
       return val
    if benchmark==8: #sinking block
       if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
          val=eta2
       else:
          val=eta1
       return val
    else: 
       return 1

#------------------------------------------------------------------------------
# analytical solution

def velocity_x(x,y):
    val=0
    if benchmark==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if benchmark==4:
       val=x+x**2-2*x*y+x**3-3*x*y**2+x**2*y
    if benchmark==5:
       val=0
    if benchmark==6:
       val,vi,pi=solvi.solution(x,y) 
    if benchmark==7:
       val,vi,pi=solcx.SolCxSolution(x,y) 
    if benchmark==9 and y>0.5:
       val=16*x**2*(1-x)**2
    return val

def velocity_y(x,y):
    val=0
    if benchmark==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if benchmark==4:
       val= -y-2*x*y+y**2-3*x**2*y+y**3-x*y**2
    if benchmark==5:
       val=0
    if benchmark==6:
       ui,val,pi=solvi.solution(x,y) 
    if benchmark==7:
       ui,val,pi=solcx.SolCxSolution(x,y) 
    if benchmark==9:
       val=0
    return val

def pressure(x,y):
    val=0
    if benchmark==1:
       val=x*(1.-x)-1./6.
    if benchmark==4:
       val= x*y+x+y+x**3*y**2-4/3
    if benchmark==5:
       val=0
    if benchmark==6:
       ui,vi,val=solvi.solution(x,y) 
    if benchmark==7:
       ui,vi,val=solcx.SolCxSolution(x,y) 
    return val

#------------------------------------------------------------------------------

def B4u(r,s):
    return 0.5*(1-r)*(1-s**2)
def dB4udr(r,s):
    return -0.5*(1-s**2)
def dB4uds(r,s):
    return -(1-r)*s

def B5u(r,s):
    return 0.5*(1+r)*(1-s**2)
def dB5udr(r,s):
    return 0.5*(1-s**2)
def dB5uds(r,s):
    return -(1+r)*s

#------------------------------------------------------------------------------

def B4v(r,s):
    return 0.5*(1-r**2)*(1-s)
def dB4vdr(r,s):
    return -r*(1-s)
def dB4vds(r,s):
    return -0.5*(1-r**2)

def B5v(r,s):
    return 0.5*(1-r**2)*(1+s)
def dB5vdr(r,s):
    return -r*(1+s)
def dB5vds(r,s):
    return 0.5*(1-r**2)

#------------------------------------------------------------------------------

def NNVu(r,s):
    N_0=0.25*(1-r)*(1-s) -0.5*B4u(r,s)
    N_1=0.25*(1+r)*(1-s) -0.5*B5u(r,s)
    N_2=0.25*(1+r)*(1+s) -0.5*B5u(r,s)
    N_3=0.25*(1-r)*(1+s) -0.5*B4u(r,s)
    N_4= B4u(r,s)
    N_5= B5u(r,s)
    return N_0,N_1,N_2,N_3,N_4,N_5

def dNNVudr(r,s):
    dNdr_0=-0.25*(1-s) -0.5*dB4udr(r,s) 
    dNdr_1=+0.25*(1-s) -0.5*dB5udr(r,s) 
    dNdr_2=+0.25*(1+s) -0.5*dB5udr(r,s) 
    dNdr_3=-0.25*(1+s) -0.5*dB4udr(r,s) 
    dNdr_4= dB4udr(r,s)
    dNdr_5= dB5udr(r,s)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5

def dNNVuds(r,s):
    dNds_0=-0.25*(1-r) -0.5*dB4uds(r,s) 
    dNds_1=-0.25*(1+r) -0.5*dB5uds(r,s) 
    dNds_2=+0.25*(1+r) -0.5*dB5uds(r,s) 
    dNds_3=+0.25*(1-r) -0.5*dB4uds(r,s) 
    dNds_4= dB4uds(r,s)
    dNds_5= dB5uds(r,s)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5

#------------------------------------------------------------------------------

def NNVv(r,s):
    N_0=0.25*(1-r)*(1-s) -0.5*B4v(r,s)
    N_1=0.25*(1+r)*(1-s) -0.5*B4v(r,s)
    N_2=0.25*(1+r)*(1+s) -0.5*B5v(r,s)
    N_3=0.25*(1-r)*(1+s) -0.5*B5v(r,s)
    N_4= B4v(r,s)
    N_5= B5v(r,s)
    return N_0,N_1,N_2,N_3,N_4,N_5

def dNNVvdr(r,s):
    dNdr_0=-0.25*(1-s) -0.5*dB4vdr(r,s) 
    dNdr_1=+0.25*(1-s) -0.5*dB4vdr(r,s) 
    dNdr_2=+0.25*(1+s) -0.5*dB5vdr(r,s) 
    dNdr_3=-0.25*(1+s) -0.5*dB5vdr(r,s) 
    dNdr_4= dB4vdr(r,s)
    dNdr_5= dB5vdr(r,s)
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5

def dNNVvds(r,s):
    dNds_0=-0.25*(1-r) -0.5*dB4vds(r,s) 
    dNds_1=-0.25*(1+r) -0.5*dB4vds(r,s) 
    dNds_2=+0.25*(1+r) -0.5*dB5vds(r,s) 
    dNds_3=+0.25*(1-r) -0.5*dB5vds(r,s) 
    dNds_4= dB4vds(r,s)
    dNds_5= dB5vds(r,s)
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5

def NNP(r,s):
    return 1

#------------------------------------------------------------------------------

print("-----------------------------")
print("--------fieldstone 80--------")
print("-----------------------------")

ndim=2
ndofV=2  # number of degrees of freedom per node
ndofP=1
mV=6
mP=1

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 


eta_ref=1.

# allowing for argument parsing through command line
if int(len(sys.argv) == 8):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   nqperdim=int(sys.argv[4])
   drho=float(sys.argv[5])
   eta1=10.**(float(sys.argv[6]))
   eta2=10.**(float(sys.argv[7]))
else:
   nelx = 4#96
   nely = 3#nelx
   visu = 1
   nqperdim = 2
   drho=8
   eta1=1e21
   eta2=1e22

#1: Donea & Huerta "dh"
#2: aquarium
#3: ldc
#4: Dohrmann & Bochev "db2d" 
#5: stokes sphere
#6: solvi
#7: solcx
#8: sinking block
#9: ldc regularised

benchmark=9

if benchmark==8:
   Lx=512e3
   Ly=512e3
   gy=-10
   xc_block=256e3
   yc_block=384e3
   d_block=64e3
   eta_ref=1e21      # scaling of G blocks
   year=365.25*24*3600
   cm=0.01
else:
   year=1.   
   cm=1

rho1=0#3200
rho2=rho1+drho
 
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nel=nelx*nely  # number of elements, total

NP=nel
NV=nnx*nny+nnx*nely+nny*nelx

NfemV=ndofV*nnx*nny +nnx*nely+nny*nelx
NfemP=NP*ndofP  # Total number of degrees of freedom
Nfem=NfemV+NfemP

hx=Lx/nelx
hy=Ly/nely

eps=1.e-10

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

pnormalise=True


#################################################################
#################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("NP=",NP)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("hx",hx)
print("hy",hy)
print("nqperdim",nqperdim)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

xV = np.zeros(NV,dtype=np.float64)  # x coordinates
yV = np.zeros(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx
        yV[counter]=j*hy
        counter += 1
    #end for
#end for

for j in range(0, nely):
    for i in range(0, nnx):
        xV[counter]=i*hx
        yV[counter]=(j+0.5)*hy
        counter += 1

for j in range(0, nny):
    for i in range(0, nelx):
        xV[counter]=(i+0.5)*hx
        yV[counter]=j*hy
        counter += 1

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')
   
print("mesh setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
# Each element is connected to 4+4=8 nodes, but in each direction 
# only to 4+2=6. We therefore make 3 icon arrays for all three 
# directions
#################################################################
start = time.time()

iconu =np.zeros((mV, nel),dtype=np.int32)
iconv =np.zeros((mV, nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        #----
        iconu[0, counter] = i + j * (nelx + 1)
        iconu[1, counter] = i + 1 + j * (nelx + 1)
        iconu[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        iconu[3, counter] = i + (j + 1) * (nelx + 1)
        iconu[4, counter] = nnx*nny + i + j*nnx 
        iconu[5, counter] = nnx*nny + i +1+ j*nnx 
        #----
        iconv[0, counter] = i + j * (nelx + 1)
        iconv[1, counter] = i + 1 + j * (nelx + 1)
        iconv[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        iconv[3, counter] = i + (j + 1) * (nelx + 1)
        iconv[4, counter] = nnx*nny + nnx*nely + i  + j*nelx
        iconv[5, counter] = nnx*nny + nnx*nely + nelx + i+ j*nelx
        #----
        counter += 1
    #end for
#end for

#for iel in range(0,nel):
    #print(iel,'|',iconu[:,iel])
    #print(iel,'|',iconv[:,iel])

print("connectivity setup: %.3f s" % (time.time() - start))

#################################################################
# compute xc,yc
#################################################################

xc = np.zeros(nel,dtype=np.float64)  # x coordinates
yc = np.zeros(nel,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xc[iel]=0.5*(xV[iconu[0,iel]]+xV[iconu[2,iel]])
    yc[iel]=0.5*(yV[iconu[0,iel]]+yV[iconu[2,iel]])

#np.savetxt('gridc.ascii',np.array([xc,yc]).T,header='# x,y')

#################################################################
# define boundary conditions
# no slip boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros((mV*ndofV,nel),dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros((mV*ndofV,nel),dtype=np.float64)  # boundary condition, value

if benchmark==5 or benchmark==8: #free slip
   for iel in range(0,nel):
       inode0=iconu[0,iel] 
       inode2=iconu[2,iel]
       if yV[inode0]<eps: #element is on face y=0
          bc_fix[0*ndofV+1,iel]=True ; bc_val[0*ndofV+1,iel]= velocity_y(xV[iconv[0,iel]],yV[iconv[0,iel]])
          bc_fix[1*ndofV+1,iel]=True ; bc_val[1*ndofV+1,iel]= velocity_y(xV[iconv[1,iel]],yV[iconv[1,iel]])
          bc_fix[        9,iel]=True ; bc_val[        9,iel]= velocity_y(xV[iconv[4,iel]],yV[iconv[4,iel]])
       if yV[inode2]>Ly-eps: #element is on face y=Ly
          bc_fix[2*ndofV+1,iel]=True ; bc_val[2*ndofV+1,iel]= velocity_y(xV[iconv[2,iel]],yV[iconv[2,iel]])
          bc_fix[3*ndofV+1,iel]=True ; bc_val[3*ndofV+1,iel]= velocity_y(xV[iconv[3,iel]],yV[iconv[3,iel]])
          bc_fix[       11,iel]=True ; bc_val[       11,iel]= velocity_y(xV[iconv[5,iel]],yV[iconv[5,iel]])
       if xV[inode0]<eps: #element is on face x=0
          bc_fix[0*ndofV+0,iel]=True ; bc_val[0*ndofV+0,iel]= velocity_x(xV[iconu[0,iel]],yV[iconu[0,iel]])
          bc_fix[3*ndofV+0,iel]=True ; bc_val[3*ndofV+0,iel]= velocity_x(xV[iconu[3,iel]],yV[iconu[3,iel]])
          bc_fix[        8,iel]=True ; bc_val[        8,iel]= velocity_x(xV[iconu[4,iel]],yV[iconu[4,iel]])
       if xV[inode2]>Lx-eps: #element is on face x=Lx
          bc_fix[1*ndofV+0,iel]=True ; bc_val[1*ndofV+0,iel]= velocity_x(xV[iconu[1,iel]],yV[iconu[1,iel]])
          bc_fix[2*ndofV+0,iel]=True ; bc_val[2*ndofV+0,iel]= velocity_x(xV[iconu[2,iel]],yV[iconu[2,iel]])
          bc_fix[       10,iel]=True ; bc_val[       10,iel]= velocity_x(xV[iconu[5,iel]],yV[iconu[5,iel]])
   #end for
else:
   for iel in range(0,nel):
       inode0=iconu[0,iel] 
       inode2=iconu[2,iel]
       if yV[inode0]<eps: #element is on face y=0
          bc_fix[0*ndofV+0,iel]=True ; bc_val[0*ndofV+0,iel]= velocity_x(xV[iconu[0,iel]],yV[iconu[0,iel]])
          bc_fix[0*ndofV+1,iel]=True ; bc_val[0*ndofV+1,iel]= velocity_y(xV[iconv[0,iel]],yV[iconv[0,iel]])
          bc_fix[1*ndofV+0,iel]=True ; bc_val[1*ndofV+0,iel]= velocity_x(xV[iconu[1,iel]],yV[iconu[1,iel]])
          bc_fix[1*ndofV+1,iel]=True ; bc_val[1*ndofV+1,iel]= velocity_y(xV[iconv[1,iel]],yV[iconv[1,iel]])
          bc_fix[        9,iel]=True ; bc_val[        9,iel]= velocity_y(xV[iconv[4,iel]],yV[iconv[4,iel]])
       if yV[inode2]>Ly-eps: #element is on face y=Ly
          bc_fix[2*ndofV+0,iel]=True ; bc_val[2*ndofV+0,iel]= velocity_x(xV[iconu[2,iel]],yV[iconu[2,iel]])
          bc_fix[2*ndofV+1,iel]=True ; bc_val[2*ndofV+1,iel]= velocity_y(xV[iconv[2,iel]],yV[iconv[2,iel]])
          bc_fix[3*ndofV+0,iel]=True ; bc_val[3*ndofV+0,iel]= velocity_x(xV[iconu[3,iel]],yV[iconu[3,iel]])
          bc_fix[3*ndofV+1,iel]=True ; bc_val[3*ndofV+1,iel]= velocity_y(xV[iconv[3,iel]],yV[iconv[3,iel]])
          bc_fix[       11,iel]=True ; bc_val[       11,iel]= velocity_y(xV[iconv[5,iel]],yV[iconv[5,iel]])
       if xV[inode0]<eps: #element is on face x=0
          bc_fix[0*ndofV+0,iel]=True ; bc_val[0*ndofV+0,iel]= velocity_x(xV[iconu[0,iel]],yV[iconu[0,iel]])
          bc_fix[0*ndofV+1,iel]=True ; bc_val[0*ndofV+1,iel]= velocity_y(xV[iconv[0,iel]],yV[iconv[0,iel]])
          bc_fix[3*ndofV+0,iel]=True ; bc_val[3*ndofV+0,iel]= velocity_x(xV[iconu[3,iel]],yV[iconu[3,iel]])
          bc_fix[3*ndofV+1,iel]=True ; bc_val[3*ndofV+1,iel]= velocity_y(xV[iconv[3,iel]],yV[iconv[3,iel]])
          bc_fix[        8,iel]=True ; bc_val[        8,iel]= velocity_x(xV[iconu[4,iel]],yV[iconu[4,iel]])
       if xV[inode2]>Lx-eps: #element is on face x=Lx
          bc_fix[1*ndofV+0,iel]=True ; bc_val[1*ndofV+0,iel]= velocity_x(xV[iconu[1,iel]],yV[iconu[1,iel]])
          bc_fix[1*ndofV+1,iel]=True ; bc_val[1*ndofV+1,iel]= velocity_y(xV[iconv[1,iel]],yV[iconv[1,iel]])
          bc_fix[2*ndofV+0,iel]=True ; bc_val[2*ndofV+0,iel]= velocity_x(xV[iconu[2,iel]],yV[iconu[2,iel]])
          bc_fix[2*ndofV+1,iel]=True ; bc_val[2*ndofV+1,iel]= velocity_y(xV[iconv[2,iel]],yV[iconv[2,iel]])
          bc_fix[       10,iel]=True ; bc_val[       10,iel]= velocity_x(xV[iconu[5,iel]],yV[iconu[5,iel]])
   #end for
#end if






print("define b.c.: %.3f s" % (time.time() - start))

#################################################################
# compute area of elements
#################################################################
start = time.time()

N        = np.zeros(4,dtype=np.float64)           # z-component velocity
area     = np.zeros(nel,dtype=np.float64) 
NNNVu    = np.zeros(mV,dtype=np.float64)           # shape functions u
NNNVv    = np.zeros(mV,dtype=np.float64)           # shape functions v
dNNNVudx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVuds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
jcbi     = np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                # compute xq,yq, using linear mapping
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                xq=0.0
                yq=0.0
                for k in range(0,4):
                    xq+=N[k]*xV[iconu[k,iel]]
                    yq+=N[k]*yV[iconu[k,iel]]
                #end for

                #print(xq,yq)

                NNNVu[0:mV]=NNVu(rq,sq)
                dNNNVudr[0:mV]=dNNVudr(rq,sq)
                dNNNVuds[0:mV]=dNNVuds(rq,sq)

                NNNVv[0:mV]=NNVv(rq,sq)
                dNNNVvdr[0:mV]=dNNVvdr(rq,sq)
                dNNNVvds[0:mV]=dNNVvds(rq,sq)

                #compute jacobian matrix and determinant
                jcob=hx*hy/4
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

                # compute dNdx, dNdy
                for k in range(0,mV):
                    dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                    dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
                    dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
                    dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
                #end for

                area[iel]+=jcob*weightq
            #end for
        #end for
    #end for
#end for

print("     -> vol  (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total vol meas %.6f " %(area.sum()))
print("     -> total vol anal %.6f " %(Lx*Ly))

print("compute elements area: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

K_mat    = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat    = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs    = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs    = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat    = np.zeros((3,mV*ndofV),dtype=np.float64)  # gradient matrix B 
rhs      = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
u        = np.zeros(NV,dtype=np.float64)            # x-component velocity
v        = np.zeros(NV,dtype=np.float64)            # y-component velocity
N        = np.zeros(4,dtype=np.float64)             # shape fct values vector
N_mat    = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
NNNP     = np.zeros(mP,dtype=np.float64)            # shape functions P
NNNVu    = np.zeros(mV,dtype=np.float64)            # shape functions u
NNNVv    = np.zeros(mV,dtype=np.float64)            # shape functions v
dNNNVudx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVudy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVudr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVuds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVvdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVvdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVvdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNVvds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
jcbi     = np.zeros((ndim,ndim),dtype=np.float64)
c_mat    = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

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

                # calculate shape functions
                NNNVu[0:mV]=NNVu(rq,sq)
                dNNNVudr[0:mV]=dNNVudr(rq,sq)
                dNNNVuds[0:mV]=dNNVuds(rq,sq)

                NNNVv[0:mV]=NNVv(rq,sq)
                dNNNVvdr[0:mV]=dNNVvdr(rq,sq)
                dNNNVvds[0:mV]=dNNVvds(rq,sq)

                NNNP[0:mP]=NNP(rq,sq)

                #compute jacobian matrix and determinant
                jcob=hx*hy/4
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

                # compute xq,yq
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                xq=0.0
                yq=0.0
                for k in range(0,4):
                    xq+=N[k]*xV[iconu[k,iel]]
                    yq+=N[k]*yV[iconu[k,iel]]
                #end for

                # compute dNdx, dNdy
                for k in range(0,mV):
                    dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                    dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
                    dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
                    dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0, mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVudx[i],0.         ],
                                             [0.         ,dNNNVvdy[i]],
                                             [dNNNVudy[i],dNNNVvdx[i]]]
                #end for

                K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i+0]+=NNNVu[i]*jcob*weightq*bx(xq,yq)
                    f_el[ndofV*i+1]+=NNNVv[i]*jcob*weightq*by(xq,yq)
                #end for
                #print(xq,yq,eta(xq,yq),by(xq,yq))

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                NNNNP[:]+=NNNP[:]*jcob*weightq

            #end for kq
        #end for jq
    #end for iq

    # impose b.c. 
    for ikk in range(0,mV*ndofV): # loop over lines
        if bc_fix[ikk,iel]: 
           K_ref=K_el[ikk,ikk] 
           for jkk in range(0,mV*ndofV): 
               f_el[jkk]-=K_el[jkk,ikk]*bc_val[ikk,iel]
               K_el[ikk,jkk]=0
               K_el[jkk,ikk]=0
           #end for
           K_el[ikk,ikk]=K_ref
           f_el[ikk]=K_ref*bc_val[ikk,iel]
           h_el[0]-=G_el[ikk,0]*bc_val[ikk,iel]
           G_el[ikk,0]=0
        #end if
    #end for

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

    # assemble matrix K_mat and right hand side rhs

    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1  # local 
            if k1<4: # Q1 nodes
               m1 =ndofV*iconu[k1,iel]+i1  # which iconu/v/w does nto matter
            else: # bubbles
               if i1==0:
                  m1 =iconu[k1,iel] -nnx*nny + ndofV*nnx*nny
               if i1==1:
                  m1 =iconv[k1,iel] -nnx*nny + ndofV*nnx*nny
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2 +i2  # local
                    if k2<4: # Q1 nodes
                       m2 =ndofV*iconu[k2,iel]+i2  # which iconu/v/w does nto matter
                    else: # bubbles
                       if i2==0:
                          m2 =iconu[k2,iel]  -nnx*nny + ndofV*nnx*nny
                       if i2==1:
                          m2 =iconv[k2,iel]  -nnx*nny + ndofV*nnx*nny
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0]
    
#end for iel

#plt.spy(K_mat)
#plt.savefig('K.pdf', bbox_inches='tight')

print("build FE system: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   a_mat[Nfem,NfemV:Nfem]=1
   a_mat[NfemV:Nfem,Nfem]=1
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
#end if

rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (time.time() - start))

#plt.spy(a_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

######################################################################
# solve system
######################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

for i in range(0,nnx*nny):
    u[i]=sol[i*ndofV+0]
    v[i]=sol[i*ndofV+1]

u[nnx*nny:nnx*nny+nnx*nely]=sol[nnx*nny*ndofV:nnx*nny*ndofV+nnx*nely]

v[nnx*nny+nnx*nely:nnx*nny+nnx*nely+nny*nelx]= sol[nnx*nny*ndofV+nnx*nely:nnx*nny*ndofV+nnx*nely+nny*nelx]

p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %6f %6f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %6f %6f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %6f %6f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

print("stats: ",nel,np.min(u),np.max(u),np.min(v),np.max(v),np.min(p),np.max(p))

#np.savetxt('velocity.ascii',np.array([xV[0:nnx*nny],yV[0:nnx*nny],u[0:nnx*nny],v[0:nnx*nny]]).T,header='# x,y,u,v')
#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# x,y,p')

#np.savetxt('pressure_top.ascii',np.array([xc[nel-nelx:nel],yc[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,y,p')

print("transfer solution: %.3f s" % (time.time() - start))

#####################################################################
# project p onto Q1 mesh 
#####################################################################

q = np.zeros(nnx*nny,dtype=np.float64)  # x coordinates
c = np.zeros(nnx*nny,dtype=np.float64)  # x coordinates

for iel in range(0,nel):
    q[iconu[0,iel]]+=p[iel]
    q[iconu[1,iel]]+=p[iel]
    q[iconu[2,iel]]+=p[iel]
    q[iconu[3,iel]]+=p[iel]
    c[iconu[0,iel]]+=1
    c[iconu[1,iel]]+=1
    c[iconu[2,iel]]+=1
    c[iconu[3,iel]]+=1

q[:]/=c[:]


#####################################################################
# measure vel at center of block
#####################################################################

if benchmark==8:
   for i in range(0,nnx*nny):
       if abs(xV[i]-xc_block)<1 and abs(yV[i]-yc_block)<1:
          print('vblock=',eta1/eta2,np.abs(v[i])*eta1/drho,u[i]*year,v[i]*year,nelx)
          print('pblock=',eta1/eta2,q[i]/drho/np.abs(gy)/128e3,q[i],nelx)

   pline_file=open('pline.ascii',"w")
   for i in range(0,nnx*nny):
       if abs(xV[i]-Lx/2)<Lx/10000:
          pline_file.write("%10e %10e \n" %(yV[i],q[i]))
   pline_file.close()

if benchmark==9:
   np.savetxt('p_top.ascii',np.array([xc[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,p')

#################################################################
# compute error in L2 norm
#################################################################
start = time.time()

errv=0.
errp=0.
vrms=0.
for iel in range (0,nel):

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNVu[0:mV]=NNVu(rq,sq)
                NNNVv[0:mV]=NNVv(rq,sq)

                # compute xq,yq
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                xq=0.0
                yq=0.0
                for k in range(0,4):
                    xq+=N[k]*xV[iconu[k,iel]]
                    yq+=N[k]*yV[iconu[k,iel]]

                uq=0.0
                vq=0.0
                for k in range(0,mV):
                    uq+=NNNVu[k]*u[iconu[k,iel]]
                    vq+=NNNVv[k]*v[iconv[k,iel]]
                #end for
                #print(xq,yq)

                #compute jacobian matrix and determinant
                jcob=hx*hy/4
                jcbi[0,0]=2/hx ; jcbi[0,1]=0    
                jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

                errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
                errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob

                vrms+=(uq**2+vq**2)*weightq*jcob

        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

vrms=np.sqrt(1/Lx/Ly*vrms)

print("     -> nel= %6d ; errv= %e ; errp= %e ; vrms= %e" %(nel,errv,errp,vrms))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# exporting u and v dofs into seperature vtu files
# not the most elegant way...
#####################################################################

if visu==1:
       filename = 'u_dofs.vtu' 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mV*nel,mV*nel))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<4: # Q1 nodes
                  m1 =ndofV*iconu[k1,iel]  
               else: # bubbles
                  m1 =iconu[k1,iel] -nnx*nny + ndofV*nnx*nny
               vtufile.write("%3e \n" %sol[m1])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<4: # Q1 nodes
                  m1 =ndofV*iconu[k1,iel]  
               else: # bubbles
                  m1 =iconu[k1,iel] -nnx*nny + ndofV*nnx*nny
               vtufile.write("%10e %10e %10e \n" %(xV[iconu[k1,iel]],yV[iconu[k1,iel]],0))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % i) 
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % 1) 
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       filename = 'v_dofs.vtu' 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(mV*nel,mV*nel))
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='v' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<4: # Q1 nodes
                  m1 =ndofV*iconv[k1,iel]  +1
               else: # bubbles
                  m1 =iconv[k1,iel] -nnx*nny + ndofV*nnx*nny
               vtufile.write("%3e \n" %sol[m1])
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for iel in range(0,nel):
           for k1 in range(0,mV):
               if k1<4: # Q1 nodes
                  m1 =ndofV*iconv[k1,iel]  +1
               else: # bubbles
                  m1 =iconv[k1,iel] -nnx*nny + ndofV*nnx*nny
               vtufile.write("%10e %10e %10e \n" %(xV[iconv[k1,iel]],yV[iconv[k1,iel]],0))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % i) 
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,mV*nel):
           vtufile.write("%d " % 1) 
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

#####################################################################
# plot of solution
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnx*nny,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],0))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%d\n" % iel)
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (pressure(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (p[iel]-pressure(xc[iel],yc[iel])))
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='body force' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(bx(xV[i],yV[i]),by(xV[i],yV[i]),0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e  \n" %(eta(xV[i],yV[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nnx*nny):
       vtufile.write("%e  \n" %(q[i]))
   vtufile.write("</DataArray>\n")


   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]/cm*year,v[i]/cm*year,0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(u[i]-velocity_x(xV[i],yV[i]),v[i]-velocity_y(xV[i],yV[i])   ,0))
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e %e %e \n" %(velocity_x(xV[i],yV[i]),velocity_y(xV[i],yV[i])   ,0))
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d \n" %(iconu[0,iel],iconu[1,iel],iconu[2,iel],iconu[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %5)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
