import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

#------------------------------------------------------------------------------
def density(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
    gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
    alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
    val=k*math.sin(k*theta)*alephr + rho0 
    return val

def Psi(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    val=-r*gr*math.cos(k*theta)
    return val

def velocity_x(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.cos(theta)-vtheta*math.sin(theta)
    return val

def velocity_y(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    fpr=A-B/r**2
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    vr=k *gr * math.sin (k * theta)
    vtheta = fr *math.cos(k* theta)
    val=vr*math.sin(theta)+vtheta*math.cos(theta)
    return val

def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    fr=A*r+B/r
    gr=A/2.*r + B/r*math.log(r) - 1./r
    hr=(2*gr-fr)/r
    val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    return val

def sr_xx(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.cos(theta))**2\
       +ett*(math.sin(theta))**2\
       -2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_yy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=err*(math.sin(theta))**2\
       +ett*(math.cos(theta))**2\
       +2*ert*math.sin(theta)*math.cos(theta)
    return val

def sr_xy(x,y,R1,R2,k):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
    gr=A/2.*r + B/r*math.log(r) - 1./r
    gpr=A/2 + B*((1-math.log(r)) / r**2 ) +1./r**2
    fr=A*r+B/r
    fpr=A-B/r**2
    err=gpr*k*math.sin(k*theta)
    ert=0.5*(k**2/r*gr+fpr-fr/r)*math.cos(k*theta)
    ett=(gr-fr)/r*k*math.sin(k*theta)
    val=ert*(math.cos(theta)**2-math.sin(theta)**2)\
       +(err-ett)*math.cos(theta)*math.sin(theta)
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

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

print("-----------------------------")
print("--------stone 74-------------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=5     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 4):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
   nqperdim = int(sys.argv[3])
else:
   nelr = 16
   visu = 1
   nqperdim=3

bubble=2
beta=0.01

R1=1.
R2=2.

dr=(R2-R1)/nelr
nelt=12*nelr 
nel=nelr*nelt  
dtheta=2*np.pi/nelt

rho0=0.
kk=4
g0=1.

viscosity=1.  # dynamic viscosity \eta

eps=1.e-10

sqrt3=np.sqrt(3.)

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




rVnodes=[-1,1,1,-1,0]
sVnodes=[-1,-1,1,1,0]

sparse=True
pnormalise=True

nnr=nelr+1
nnt=nelt
nnp=nnr*nnt+nel  # number of nodes

NfemV=nnp*ndofV    # Total number of degrees of V freedom 
NfemP=nnr*nnt*ndofP          # Total number of degrees of P freedom
Nfem=NfemV+NfemP         # total number of dofs

nq=nel*nqperdim**2

print('nelr=',nelr)
print('nelr=',nelt)
print('nel=',nel)
print('nnp=',nnp)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nfem=',Nfem)
print('nqperdim=',nqperdim)
print('nnr=',nnr)
print('nnt=',nnt)

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates
r=np.empty(nnp,dtype=np.float64)  
theta=np.empty(nnp,dtype=np.float64) 

Louter=2.*math.pi*R2
Lr=R2-R1
sx = Louter/float(nelt)
sz = Lr    /float(nelr)

counter=0
for j in range(0,nnr):
    for i in range(0,nelt):
        xV[counter]=i*sx
        yV[counter]=j*sz
        counter += 1
    #end for
#end for

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=xV[counter]
        yi=yV[counter]
        t=xi/Louter*2.*math.pi    
        xV[counter]=math.cos(t)*(R1+yi)
        yV[counter]=math.sin(t)*(R1+yi)
        r[counter]=R1+yi
        theta[counter]=math.atan2(yV[counter],xV[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*math.pi
        counter+=1
    #end for
#end for

print("building coordinate arrays (%.3fs)" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconV[0,counter] = icon2 
        iconV[1,counter] = icon1
        iconV[2,counter] = icon4
        iconV[3,counter] = icon3
        iconV[4,counter] = nnr*nnt+counter
        counter += 1
    #end for
#end for

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        icon1=counter
        icon2=counter+1
        icon3=i+(j+1)*nelt+1
        icon4=i+(j+1)*nelt
        if i==nelt-1:
           icon2-=nelt
           icon3-=nelt
        iconP[0,counter] = icon2 
        iconP[1,counter] = icon1
        iconP[2,counter] = icon4
        iconP[3,counter] = icon3
        counter += 1
    #end for
#end for

for iel in range(0,nel):
    #this yields very bad pressure!!
    #xV[iconV[4,iel]]=0.25*xV[iconV[0,iel]]+\
    #                +0.25*xV[iconV[1,iel]]+\
    #                +0.25*xV[iconV[2,iel]]+\
    #                +0.25*xV[iconV[3,iel]]
    #yV[iconV[4,iel]]=0.25*yV[iconV[0,iel]]+\
    #                +0.25*yV[iconV[1,iel]]+\
    #                +0.25*yV[iconV[2,iel]]+\
    #                +0.25*yV[iconV[3,iel]]

    r[iconV[4,iel]]=r[iconV[0,iel]]+dr/2
    theta[iconV[4,iel]]=theta[iconV[0,iel]]-dtheta/2
    xV[iconV[4,iel]]=r[iconV[4,iel]]*np.cos(theta[iconV[4,iel]])
    yV[iconV[4,iel]]=r[iconV[4,iel]]*np.sin(theta[iconV[4,iel]])

#end for

#np.savetxt('gridV.ascii',np.array([xV,yV,r,theta]).T)

#################################################################
#now that I have both connectivity arrays I can 
# easily build xP,yP

NP=NfemP
xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]]
    yP[iconP[0,iel]]=yV[iconV[0,iel]]
    yP[iconP[1,iel]]=yV[iconV[1,iel]]
    yP[iconP[2,iel]]=yV[iconV[2,iel]]
    yP[iconP[3,iel]]=yV[iconV[3,iel]]

#np.savetxt('gridP.ascii',np.array([xP,yP]).T)

print("building connectivity array (%.3fs)" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0,nnp):
    if r[i]<R1+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)
    if r[i]>(R2-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

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
            #end for
            jcob = np.linalg.det(jcb)
            area[iel]+=jcob*weightq
        #end for
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()))
print("     -> total area (anal) %.6f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
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

f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

xq   = np.zeros(nq,dtype=np.float64)     
yq   = np.zeros(nq,dtype=np.float64)      

iiq=0
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
            #end for 
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            for k in range(0,mV):
                xq[iiq]+=NNNV[k]*xV[iconV[k,iel]]
                yq[iiq]+=NNNV[k]*yV[iconV[k,iel]]
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for 

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                         [0.        ,dNNNVdy[i]],
                                         [dNNNVdy[i],dNNNVdx[i]]]
            #end for 

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq[iiq],yq[iiq],g0)*density(xq[iiq],yq[iiq],R1,R2,kk,rho0,g0)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq[iiq],yq[iiq],g0)*density(xq[iiq],yq[iiq],R1,R2,kk,rho0,g0)
            #end for 

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNNP[:]+=NNNP[:]*jcob*weightq

            iiq+=1

        #end for jq
    #end for iq

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
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

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
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]
    #end for 
    if sparse and pnormalise:
       A_sparse[Nfem,NfemV+m2]=constr[m2]
       A_sparse[NfemV+m2,Nfem]=constr[m2]

#end for iel

if not sparse:
   print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))

#np.savetxt('gridq.ascii',np.array([xq,yq]).T)

print("build FE matrixs & rhs (%.3fs)" % (timing.time() - start))

#################################################################
# solve system
#################################################################
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


rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (timing.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

print("reshape solution (%.3fs)" % (timing.time() - start))

#####################################################################
# compute strain rate - center to nodes - method 1
#####################################################################

count = np.zeros(nnp,dtype=np.int32)  
Lxx1 = np.zeros(nnp,dtype=np.float64)  
Lxy1 = np.zeros(nnp,dtype=np.float64)  
Lyx1 = np.zeros(nnp,dtype=np.float64)  
Lyy1 = np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.
    sq=0.
    NNNV[0:mV]=NNV(rq,sq)
    dNNNVdr[0:mV]=dNNVdr(rq,sq)
    dNNNVds[0:mV]=dNNVds(rq,sq)
    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
    #end for
    jcbi=np.linalg.inv(jcb)
    for k in range(0,mV):
        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    #end for
    L_xx=0.
    L_xy=0.
    L_yx=0.
    L_yy=0.
    for k in range(0,mV):
        L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
        L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
        L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
        L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
    #end for
    for i in range(0,mV):
        inode=iconV[i,iel]
        Lxx1[inode]+=L_xx
        Lxy1[inode]+=L_xy
        Lyx1[inode]+=L_yx
        Lyy1[inode]+=L_yy
        count[inode]+=1
    #end for
#end for
Lxx1/=count
Lxy1/=count
Lyx1/=count
Lyy1/=count

print("     -> Lxx1 (m,M) %.4f %.4f " %(np.min(Lxx1),np.max(Lxx1)))
print("     -> Lyy1 (m,M) %.4f %.4f " %(np.min(Lyy1),np.max(Lyy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lxy1),np.max(Lxy1)))
print("     -> Lxy1 (m,M) %.4f %.4f " %(np.min(Lyx1),np.max(Lyx1)))

print("compute vel gradient meth-1 (%.3fs)" % (timing.time() - start))

#################################################################
#################################################################

exx1 = np.zeros(nnp,dtype=np.float64)  
eyy1 = np.zeros(nnp,dtype=np.float64)  
exy1 = np.zeros(nnp,dtype=np.float64)  

exx1[:]=Lxx1[:]
eyy1[:]=Lyy1[:]
exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

#####################################################################
# compute strain rate - corners to nodes - method 2
#####################################################################
start = timing.time()

count = np.zeros(nnp,dtype=np.int32)  
Lxx2 = np.zeros(nnp,dtype=np.float64)  
Lxy2 = np.zeros(nnp,dtype=np.float64)  
Lyx2 = np.zeros(nnp,dtype=np.float64)  
Lyy2 = np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,mV):
        inode=iconV[i,iel]
        rq=rVnodes[i]
        sq=sVnodes[i]
        NNNV[0:mV]=NNV(rq,sq)
        dNNNVdr[0:mV]=dNNVdr(rq,sq)
        dNNNVds[0:mV]=dNNVds(rq,sq)
        NNNP[0:mP]=NNP(rq,sq)
        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        #end for
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        #end for
        L_xx=0.
        L_xy=0.
        L_yx=0.
        L_yy=0.
        for k in range(0,mV):
            L_xx+=dNNNVdx[k]*u[iconV[k,iel]]
            L_xy+=dNNNVdx[k]*v[iconV[k,iel]]
            L_yx+=dNNNVdy[k]*u[iconV[k,iel]]
            L_yy+=dNNNVdy[k]*v[iconV[k,iel]]
        #end for
        Lxx2[inode]+=L_xx
        Lxy2[inode]+=L_xy
        Lyx2[inode]+=L_yx
        Lyy2[inode]+=L_yy
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count

print("     -> Lxx2 (m,M) %.4f %.4f " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4f %.4f " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lyx2),np.max(Lyx2)))

#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

print("compute vel gradient meth-2 (%.3fs)" % (timing.time() - start))

#################################################################
#################################################################

exx2 = np.zeros(nnp,dtype=np.float64)  
eyy2 = np.zeros(nnp,dtype=np.float64)  
exy2 = np.zeros(nnp,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

#################################################################
#################################################################
start = timing.time()

M_mat= np.zeros((nnp,nnp),dtype=np.float64)
rhsLxx=np.zeros(nnp,dtype=np.float64)
rhsLyy=np.zeros(nnp,dtype=np.float64)
rhsLxy=np.zeros(nnp,dtype=np.float64)
rhsLyx=np.zeros(nnp,dtype=np.float64)

for iel in range(0,nel):

    M_el =np.zeros((mV,mV),dtype=np.float64)
    fLxx_el=np.zeros(mV,dtype=np.float64)
    fLyy_el=np.zeros(mV,dtype=np.float64)
    fLxy_el=np.zeros(mV,dtype=np.float64)
    fLyx_el=np.zeros(mV,dtype=np.float64)
    NNNV =np.zeros((mV,1),dtype=np.float64) 

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV,0]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for 
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            Lxxq=0.
            Lyyq=0.
            Lxyq=0.
            Lyxq=0.
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                Lxxq+=dNNNVdx[k]*u[iconV[k,iel]]
                Lyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                Lxyq+=dNNNVdx[k]*v[iconV[k,iel]]
                Lyxq+=dNNNVdy[k]*u[iconV[k,iel]]
            #end for 

            M_el +=NNNV.dot(NNNV.T)*weightq*jcob

            fLxx_el[:]+=NNNV[:,0]*Lxxq*jcob*weightq
            fLyy_el[:]+=NNNV[:,0]*Lyyq*jcob*weightq
            fLxy_el[:]+=NNNV[:,0]*Lxyq*jcob*weightq
            fLyx_el[:]+=NNNV[:,0]*Lyxq*jcob*weightq

        #end for
    #end for

    for k1 in range(0,mV):
        m1=iconV[k1,iel]
        for k2 in range(0,mV):
            m2=iconV[k2,iel]
            M_mat[m1,m2]+=M_el[k1,k2]
        #end for
        rhsLxx[m1]+=fLxx_el[k1]
        rhsLyy[m1]+=fLyy_el[k1]
        rhsLxy[m1]+=fLxy_el[k1]
        rhsLyx[m1]+=fLyx_el[k1]
    #end for

#end for

Lxx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxx)
Lyy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyy)
Lxy3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLxy)
Lyx3 = sps.linalg.spsolve(sps.csr_matrix(M_mat),rhsLyx)

print("     -> Lxx3 (m,M) %.4f %.4f " %(np.min(Lxx3),np.max(Lxx3)))
print("     -> Lyy3 (m,M) %.4f %.4f " %(np.min(Lyy3),np.max(Lyy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lxy3),np.max(Lxy3)))
print("     -> Lxy3 (m,M) %.4f %.4f " %(np.min(Lyx3),np.max(Lyx3)))

print("compute vel gradient meth-3 (%.3fs)" % (timing.time() - start))

#################################################################
#################################################################

exx3 = np.zeros(nnp,dtype=np.float64)  
eyy3 = np.zeros(nnp,dtype=np.float64)  
exy3 = np.zeros(nnp,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

#################################################################
# export pressure at both surfaces
#################################################################
#start = timing.time()

#np.savetxt('q_R1.ascii',np.array([xV[0:2*nelt],yV[0:2*nelt],q[0:2*nelt],theta[0:2*nelt]]).T)
#np.savetxt('q_R2.ascii',np.array([xV[nnp-2*nelt:nnp],\
#                                  yV[nnp-2*nelt:nnp],\
#                                   q[nnp-2*nelt:nnp],\
#                               theta[nnp-2*nelt:nnp]]).T)

#np.savetxt('p_R1.ascii',np.array([xP[0:nelt],yP[0:nelt],p[0:nelt]]).T)
#np.savetxt('p_R2.ascii',np.array([xP[NP-nelt:NP],yP[NP-nelt:NP],p[NP-nelt:NP]]).T)

#print("export p&q on R1,R2 (%.3fs)" % (timing.time() - start))

#################################################################
# compute error
#################################################################
start = timing.time()

NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

iiq=0
errv=0.
errp=0.
errexx1=0.
erreyy1=0.
errexy1=0.
errexx2=0.
erreyy2=0.
errexy2=0.
errexx3=0.
erreyy3=0.
errexy3=0.
vrms=0.
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

            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)

            uq=0.
            vq=0.
            exx1q=0.
            eyy1q=0.
            exy1q=0.
            exx2q=0.
            eyy2q=0.
            exy2q=0.
            exx3q=0.
            eyy3q=0.
            exy3q=0.
            for k in range(0,mV):
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                exx1q+=NNNV[k]*exx1[iconV[k,iel]]
                eyy1q+=NNNV[k]*eyy1[iconV[k,iel]]
                exy1q+=NNNV[k]*exy1[iconV[k,iel]]
                exx2q+=NNNV[k]*exx2[iconV[k,iel]]
                eyy2q+=NNNV[k]*eyy2[iconV[k,iel]]
                exy2q+=NNNV[k]*exy2[iconV[k,iel]]
                exx3q+=NNNV[k]*exx3[iconV[k,iel]]
                eyy3q+=NNNV[k]*eyy3[iconV[k,iel]]
                exy3q+=NNNV[k]*exy3[iconV[k,iel]]
            errv+=((uq-velocity_x(xq[iiq],yq[iiq],R1,R2,kk,rho0,g0))**2+\
                   (vq-velocity_y(xq[iiq],yq[iiq],R1,R2,kk,rho0,g0))**2)*weightq*jcob

            errexx1+=(exx1q-sr_xx(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            erreyy1+=(eyy1q-sr_yy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            errexy1+=(exy1q-sr_xy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            errexx2+=(exx2q-sr_xx(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            erreyy2+=(eyy2q-sr_yy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            errexy2+=(exy2q-sr_xy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            errexx3+=(exx3q-sr_xx(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            erreyy3+=(eyy3q-sr_yy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob
            errexy3+=(exy3q-sr_xy(xq[iiq],yq[iiq],R1,R2,kk))**2*weightq*jcob

            vrms+=(uq**2+vq**2)*weightq*jcob

            xxq=0.
            yyq=0.
            pq=0.
            for k in range(0,mP):
                xxq+=NNNP[k]*xP[iconP[k,iel]]
                yyq+=NNNP[k]*yP[iconP[k,iel]]
                pq+=NNNP[k]*p[iconP[k,iel]]
            errp+=(pq-pressure(xxq,yyq,R1,R2,kk,rho0,g0))**2*weightq*jcob

            iiq+=1 
        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errexx1=np.sqrt(errexx1)
erreyy1=np.sqrt(erreyy1)
errexy1=np.sqrt(errexy1)
errexx2=np.sqrt(errexx2)
erreyy2=np.sqrt(erreyy2)
errexy2=np.sqrt(errexy2)
errexx3=np.sqrt(errexx3)
erreyy3=np.sqrt(erreyy3)
errexy3=np.sqrt(errexy3)

vrms=np.sqrt(vrms/np.pi/(R2**2-R1**2))

print('     -> nelr=',nelr,' vrms=',vrms)
print("     -> nelr= %6d ; errv= %.8e ; errp= %.8e " %(nelr,errv,errp))
print("     -> nelr= %6d ; errexx1= %.8e ; erreyy1= %.8e ; errexy1= %.8e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %6d ; errexx2= %.8e ; erreyy2= %.8e ; errexy2= %.8e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %6d ; errexx3= %.8e ; erreyy3= %.8e ; errexy3= %.8e" %(nelr,errexx3,erreyy3,errexy3))

print("compute errors (%.3fs)" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = timing.time()

nnp2=nnr*nnt

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp2,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='area' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%10f \n" %area[iel])
   vtufile.write("</DataArray>\n")

   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f %10f %10f \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%13e %13e %13e \n" %(velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                           velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f %10f %10f \n" %(u[i]-velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                           v[i]-velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %density(xV[i],yV[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='Psi' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %Psi(xV[i],yV[i],R1,R2,kk))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxx (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxx2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyy (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyy2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxy (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxy2[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyx (NEW)' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyx2[i])
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %(sr_xx(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %(sr_yy(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %(sr_xy(xV[i],yV[i],R1,R2,kk)))
   vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyy' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyy[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lxy' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lxy[i])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='Lyx' Format='ascii'> \n")
   #for i in range(0,nnp):
   #    vtufile.write("%10f \n" %Lyx[i])
   #vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exx1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %eyy1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exy1[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %exy3[i])
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for i in range(0,nnp2):
       vtufile.write("%10f \n" %p[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for i in range (0,nnp2):
       vtufile.write("%f\n" % pressure(xV[i],yV[i],R1,R2,kk,rho0,g0))
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
   print("export to vtu file (%.3fs)" % (timing.time() - start))


   vtufile=open('qpts.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nq):
       vtufile.write("%10e %10e %10e \n" %(xq[i],yq[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nq):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nq):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nq):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   vtufile=open('gridV.vtu',"w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nnp))
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   vtufile.write("<Cells>\n")
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%d " % i)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%d " % (i+1))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for i in range(0,nnp):
       vtufile.write("%d " % 1)
   vtufile.write("</DataArray>\n")
   vtufile.write("</Cells>\n")
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()






print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
