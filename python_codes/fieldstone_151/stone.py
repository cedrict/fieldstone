import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg import *
import time as timing
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

###############################################################################

def velocity_x(x,y,r,theta,test):
    if test==1 or test==2 or test==5 or test==6:
       return 0
    if test==3:
       return -np.sin(theta)*vbc*(R2-r)/(R2-R1)
    if test==4:
       return -np.sin(theta)*vbc*r

def velocity_y(x,y,r,theta,test):
    if test==1 or test==2 or test==5 or test==6:
       return 0
    if test==3:
       return np.cos(theta)*vbc*(R2-r)/(R2-R1)
       return 0
    if test==4:
       return np.cos(theta)*vbc*r

def pressure(x,y,r,theta,test):
    if test==1 or test==2 or test==3 or test==4:
       return rho0*g0*(R2-r)
    if test==5:
       return 0
    if test==6:
       return -2+rho0*g0*(R2-y)

###############################################################################

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    if test==6: val=0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    if test==6: val=-g0
    return val

###############################################################################

def NNV(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def dNNVdr(rq,sq):
    dNdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr_8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,\
                     dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def dNNVds(rq,sq):
    dNds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds_8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,\
                     dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

def NNP(rq,sq):
    N_0=0.25*(1-rq)*(1-sq)
    N_1=0.25*(1+rq)*(1-sq)
    N_2=0.25*(1+rq)*(1+sq)
    N_3=0.25*(1-rq)*(1+sq)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("-------- stone 151 ----------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=9     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 5):
   nelr      = int(sys.argv[1])
   test      = int(sys.argv[2])
   fs_method = int(sys.argv[3])
   trapezes  = int(sys.argv[4])
else:
   nelr      = 16
   test      = 6 
   fs_method = 1
   trapezes  = 0

normal_type=1

R1=1. # inner radius
R2=2. # outer radius

dr=(R2-R1)/nelr
nelt=12*nelr 
nel=nelr*nelt  

viscosity=1.

#surface_bc
# 0: no slip
# 1: free slip 

if test==1:
   rho0=1.
   g0=1.
   vbc=0
   surface_bc=0

if test==2:
   rho0=1.
   g0=1.
   vbc=0
   surface_bc=1

if test==3:
   rho0=1.
   g0=1.
   vbc=1
   surface_bc=0

if test==4:
   rho0=1.
   g0=1.
   vbc=1
   surface_bc=1

if test==5:
   rho0=1.
   g0=1.
   vbc=0
   surface_bc=1

if test==6:
   rho0=1
   g0=1
   vbc=0
   surface_bc=1   

eps=1.e-3

###############################################################################

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

if surface_bc==0: fs_method=0

###############################################################################
# grid point setup
###############################################################################
start = timing.time()

nnr=nelr+1
nnt=nelt
NV=nnr*nnt  # number of nodes

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates
rV=np.empty(NV,dtype=np.float64)  
theta=np.empty(NV,dtype=np.float64) 

Louter=2.*np.pi*R2
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
        t=xi/Louter*2.*np.pi    
        xV[counter]=np.cos(t)*(R1+yi)
        yV[counter]=np.sin(t)*(R1+yi)
        rV[counter]=R1+yi
        theta[counter]=math.atan2(yV[counter],xV[counter])
        if theta[counter]<0.:
           theta[counter]+=2.*np.pi
        counter+=1
    #end for
#end for

print("building coordinate arrays (%.3fs)" % (timing.time() - start))

###############################################################################
# build iconQ1 array needed for vtu file
###############################################################################

iconQ1 =np.zeros((4,nel),dtype=np.int32)
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
        iconQ1[0,counter] = icon2 
        iconQ1[1,counter] = icon1
        iconQ1[2,counter] = icon4
        iconQ1[3,counter] = icon3
        counter += 1
    #end for

###############################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
# Nlm is the number of additional lines/columns to the matrix
###############################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4

if fs_method==3:
   Nlm=2*nelt
else:
   Nlm=0

NP=nelt*(nelr+1)

NfemV=NV*ndofV       # Total number of degrees of V freedom 
NfemP=NP*ndofP       # Total number of degrees of P freedom
Nfem=NfemV+NfemP+Nlm # total number of dofs

print('nelr=',nelr)
print('nelt=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nlm=',Nlm)
print('Nfem=',Nfem)
print('surface_bc=',surface_bc)
print('fs_method=',fs_method)
print('test=',test)
print('trapezes=',trapezes)
print("-----------------------------")

###############################################################################
# connectivity
###############################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        iconV[0,counter]=2*counter+2 +2*j*nelt
        iconV[1,counter]=2*counter   +2*j*nelt
        iconV[2,counter]=iconV[1,counter]+4*nelt
        iconV[3,counter]=iconV[1,counter]+4*nelt+2
        iconV[4,counter]=iconV[0,counter]-1
        iconV[5,counter]=iconV[1,counter]+2*nelt
        iconV[6,counter]=iconV[2,counter]+1
        iconV[7,counter]=iconV[5,counter]+2
        iconV[8,counter]=iconV[5,counter]+1
        if i==nelt-1:
           iconV[0,counter]-=2*nelt
           iconV[7,counter]-=2*nelt
           iconV[3,counter]-=2*nelt
        #print(j,i,counter,'|',iconV[0:mV,counter])
        counter += 1


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
        iconP[0,counter] = icon2 
        iconP[1,counter] = icon1
        iconP[2,counter] = icon4
        iconP[3,counter] = icon3
        counter += 1
    #end for


#for iel in range(0,nel):
#    print(iel,'|',iconP[:,iel])

###############################################################################

if trapezes==1:
   for iel in range(0,nel):
       i0=iconV[0,iel]
       i1=iconV[1,iel]
       i2=iconV[2,iel]
       i3=iconV[3,iel]
       xV[iconV[4,iel]]=0.5*(xV[i0]+xV[i1])
       yV[iconV[4,iel]]=0.5*(yV[i0]+yV[i1])
       xV[iconV[5,iel]]=0.5*(xV[i1]+xV[i2])
       yV[iconV[5,iel]]=0.5*(yV[i1]+yV[i2])
       xV[iconV[6,iel]]=0.5*(xV[i2]+xV[i3])
       yV[iconV[6,iel]]=0.5*(yV[i2]+yV[i3])
       xV[iconV[7,iel]]=0.5*(xV[i3]+xV[i0])
       yV[iconV[7,iel]]=0.5*(yV[i3]+yV[i0])
       xV[iconV[8,iel]]=0.25*(xV[i0]+xV[i1]+xV[i2]+xV[i3])
       yV[iconV[8,iel]]=0.25*(yV[i0]+yV[i1]+yV[i2]+yV[i3])

rV[:]=np.sqrt(xV[:]**2+yV[:]**2)

###############################################################################

#now that I have both connectivity arrays I can 
# easily build xP,yP

xP=np.empty(NP,dtype=np.float64)  # x coordinates
yP=np.empty(NP,dtype=np.float64)  # y coordinates
rP=np.empty(NP,dtype=np.float64)  # r coordinates

for iel in range(0,nel):
    xP[iconP[0,iel]]=xV[iconV[0,iel]]
    xP[iconP[1,iel]]=xV[iconV[1,iel]]
    xP[iconP[2,iel]]=xV[iconV[2,iel]]
    xP[iconP[3,iel]]=xV[iconV[3,iel]]
    yP[iconP[0,iel]]=yV[iconV[0,iel]]
    yP[iconP[1,iel]]=yV[iconV[1,iel]]
    yP[iconP[2,iel]]=yV[iconV[2,iel]]
    yP[iconP[3,iel]]=yV[iconV[3,iel]]

rP[:]=np.sqrt(xP[:]**2+yP[:]**2)

print("building connectivity array (%.3fs)" % (timing.time() - start))

###############################################################################
# compute normal vectors
###############################################################################
start = timing.time()

nx1=np.zeros(NV,dtype=np.float64) 
ny1=np.zeros(NV,dtype=np.float64) 
surfaceV=np.zeros(NV,dtype=bool) 

#compute normal 1 type
for i in range(0,NV):
    if rV[i]/R2>1-eps:
       surfaceV[i]=True
       nx1[i]=np.cos(theta[i])
       ny1[i]=np.sin(theta[i])

#compute normal 2 type
nx2=np.zeros(NV,dtype=np.float64) 
ny2=np.zeros(NV,dtype=np.float64) 
dNNNVdx=np.zeros(mV,dtype=np.float64) 
dNNNVdy=np.zeros(mV,dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):
    if surfaceV[iconV[2,iel]]: # element is at top
       for iq in [0,1,2]:
           for jq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV=NNV(rq,sq)
               dNNNVdr=dNNVdr(rq,sq)
               dNNNVds=dNNVds(rq,sq)
               jcb[0,0]=np.dot(dNNNVdr[:],xV[iconV[:,iel]])
               jcb[0,1]=np.dot(dNNNVdr[:],yV[iconV[:,iel]])
               jcb[1,0]=np.dot(dNNNVds[:],xV[iconV[:,iel]])
               jcb[1,1]=np.dot(dNNNVds[:],yV[iconV[:,iel]])
               jcob = np.linalg.det(jcb)
               jcbi=np.linalg.inv(jcb)
               for k in range(0,mV):
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
               #end for 
               nx2[iconV[2,iel]]+=dNNNVdx[2]*jcob*weightq
               ny2[iconV[2,iel]]+=dNNNVdy[2]*jcob*weightq
               nx2[iconV[3,iel]]+=dNNNVdx[3]*jcob*weightq
               ny2[iconV[3,iel]]+=dNNNVdy[3]*jcob*weightq
               nx2[iconV[6,iel]]+=dNNNVdx[6]*jcob*weightq
               ny2[iconV[6,iel]]+=dNNNVdy[6]*jcob*weightq

           #end for
       #end for
    #end if
#end for

for i in range(0,NV):
    if surfaceV[i]:
       norm=np.sqrt(nx2[i]**2+ny2[i]**2)
       nx2[i]/=norm
       ny2[i]/=norm

nx=np.zeros(NV,dtype=np.float64) 
ny=np.zeros(NV,dtype=np.float64) 
if normal_type==1:
   nx[:]=nx1[:]
   ny[:]=ny1[:]
else:
   nx[:]=nx2[:]
   ny[:]=ny2[:]

print("compute surface normals (%.3fs)" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()


bc_fix=np.zeros(NfemV,dtype=bool)  
bc_val=np.zeros(NfemV,dtype=np.float64) 

for i in range(0,NV):
    #bottom boundary
    if rV[i]/R1<1+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = -np.sin(theta[i])*vbc
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] =  np.cos(theta[i])*vbc
    #surface boundary
    if rV[i]/R2>1-eps:
       if surface_bc==0:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

###############################################################################
start = timing.time()

surfaceV=np.zeros(NV,dtype=bool)  
cmbV=np.zeros(NV,dtype=bool)  

for i in range(0,NV):
    if rV[i]/R1<1+eps:
       cmbV[i]=True
    if rV[i]/R2>1-eps:
       surfaceV[i]=True

surfaceP=np.zeros(NP,dtype=bool)  
cmbP=np.zeros(NP,dtype=bool)  

for i in range(0,NP):
    if rP[i]/R1<1+eps:
       cmbP[i]=True
    if rP[i]/R2>1-eps:
       surfaceP[i]=True

print('     -> surfaceV counts',np.sum(surfaceV))
print('     -> surfaceP counts',np.sum(surfaceP))

print("flagging nodes on cmb and surface (%.3fs)" % (timing.time() - start))

###############################################################################
# flag all elements with a node touching the surface r=R_outer
# or r=R_inner, used later for free slip b.c.
###############################################################################
start = timing.time()

flag_top=np.zeros(nel,dtype=np.float64)  
flag_bot=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    if surfaceV[iconV[2,iel]]:
       flag_top[iel]=1
    if cmbV[iconV[0,iel]]:
       flag_bot[iel]=1

print("flag elts on boundaries: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV=np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr=np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds=np.zeros(mV,dtype=np.float64)          # shape functions derivatives

for iel in range(0,nel):
    for iq in [0,1,2]:
        for jq in [0,1,2]:
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
print("     -> total area (meas) %.9f | nel= %d" %(area.sum(),nel))
print("     -> total area (anal) %.9f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# assign elemental density
###############################################################################
start = timing.time()

rho=np.zeros(nel,dtype=np.float64)
rhoQ1=np.zeros(4*nel,dtype=np.float64)

if test==1 or test==2 or test==3 or test==4 or test==6:
   rho[:]=rho0
   rhoQ1[:]=rho0
elif test==5:
   for iel in range(0,nel):
       xc=np.sum(xV[iconV[0:4,iel]])/4
       yc=np.sum(yV[iconV[0:4,iel]])/4
       if np.sqrt(xc**2+(yc-1.5)**2)<0.25:
          rho[iel]=1.01*rho0
       else:
          rho[iel]=rho0
       #end if
   #end for
   for iel in range(0,4*nel):
       xc=np.sum(xV[iconQ1[0:4,iel]])/4
       yc=np.sum(yV[iconQ1[0:4,iel]])/4
       if np.sqrt(xc**2+(yc-1.5)**2)<0.25:
          rhoQ1[iel]=1.01*rho0
       else:
          rhoQ1[iel]=rho0
       #end if
   #end for
else:
   exit('test value unknown')

print("assign density: %.3f s" % (timing.time() - start))

###############################################################################
# build FE matrix
###############################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    G_el1=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    G_el2=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

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
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
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
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq,yq,g0)*rho[iel]
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,g0)*rho[iel]
            #end for 

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        #end for jq
    #end for iq

    if fs_method==1 and flag_top[iel]==1:
       for k in range(0,mV):
           inode=iconV[k,iel]
           if surfaceV[inode]:
              RotMat=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
              for i in range(0,mV*ndofV):
                  RotMat[i,i]=1.
              angle=theta[inode]
              RotMat[2*k  ,2*k]= np.cos(angle) ; RotMat[2*k  ,2*k+1]=np.sin(angle)
              RotMat[2*k+1,2*k]=-np.sin(angle) ; RotMat[2*k+1,2*k+1]=np.cos(angle)
              # apply counter rotation
              K_el=RotMat.dot(K_el.dot(RotMat.T))
              f_el=RotMat.dot(f_el)
              G_el=RotMat.dot(G_el)
              # apply boundary conditions
              # x-component set to 0
              ikk=ndofV*k
              K_ref=K_el[ikk,ikk]
              for jkk in range(0,mV*ndofV):
                  K_el[ikk,jkk]=0
                  K_el[jkk,ikk]=0
              K_el[ikk,ikk]=K_ref
              f_el[ikk]=0#K_ref*bc_val[m1]
              #h_el[:]-=G_el[ikk,:]*bc_val[m1]
              G_el[ikk,:]=0
              # rotate back
              K_el=RotMat.T.dot(K_el.dot(RotMat))
              f_el=RotMat.T.dot(f_el)
              G_el=RotMat.T.dot(G_el)
           #end if
       #end for

    G_el1[:,:]=G_el[:,:]
    G_el2[:,:]=G_el[:,:]

    if fs_method==2:
       for k in range(0,mV):
           inode=iconV[k,iel]
           if surfaceV[inode]:
              #print(xV[inode],zV[inode])
              if abs(nx[inode])>=abs(ny[inode]):
                 ikk=ndofV*k
                 K_ref=K_el[ikk,ikk]
                 K_el[ikk,:]=0
                 K_el[ikk,ikk]=K_ref
                 K_el[ikk,ikk+1]=K_ref*ny[inode]/nx[inode]
                 G_el1[ikk,:]=0
                 f_el[ikk]=0
              else:
                 ikk=ndofV*k+1
                 K_ref=K_el[ikk,ikk]
                 K_el[ikk,:]=0
                 K_el[ikk,ikk-1]=K_ref*nx[inode]/ny[inode]
                 K_el[ikk,ikk  ]=K_ref
                 G_el1[ikk,:]=0
                 f_el[ikk]=0
              #end if
           #end if
       #end for
    #end if

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
               h_el[:]-=G_el2[ikk,:]*bc_val[m1]
               G_el1[ikk,:]=0
               G_el2[ikk,:]=0
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
                    A_sparse[m1,m2] += K_el[ikk,jkk]
            for k2 in range(0,mP):
                jkk=k2
                m2 =iconP[k2,iel]
                A_sparse[m1,NfemV+m2]+=G_el1[ikk,jkk]
                A_sparse[NfemV+m2,m1]+=G_el2[ikk,jkk]
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

#end for iel

print("build FE matrixs & rhs (%.3fs)" % (timing.time() - start))

###############################################################################
# Lagrange multipliers business
###############################################################################

if fs_method==3:

   start = timing.time()

   counter=NfemV+NfemP
   for i in range(0,NV):
       if surfaceV[i]:
          # we need nx[i]*u[i]+ny[i]*v[i]=0
          A_sparse[counter,2*i  ]=nx[i]
          A_sparse[counter,2*i+1]=ny[i]
          A_sparse[2*i  ,counter]=nx[i]
          A_sparse[2*i+1,counter]=ny[i]
          counter+=1

   print("build L block (%.3fs)" % (timing.time() - start))

###############################################################################
# solve system
###############################################################################
start = timing.time()

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:NfemV+NfemP]=h_rhs
    
sparse_matrix=A_sparse.tocsr()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (timing.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:NfemV+NfemP]

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))

if fs_method==3: 
   l=sol[NfemV+NfemP:Nfem]
   print("     -> l (m,M) %.4f %.4f " %(np.min(l),np.max(l)))

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %e %e " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %e %e " %(np.min(vt),np.max(vt)))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v,vr,vt,rV]).T,header='# x,y,u,v,vr,vt,r')
#np.savetxt('pressure.ascii',np.array([xP,yP,p,rP]).T,header='# x,y,p,r')

print("reshape solution (%.3fs)" % (timing.time() - start))

###############################################################################
# compute strain rate - center to nodes - method 1
###############################################################################

count = np.zeros(NV,dtype=np.int32)  
Lxx1 = np.zeros(NV,dtype=np.float64)  
Lxy1 = np.zeros(NV,dtype=np.float64)  
Lyx1 = np.zeros(NV,dtype=np.float64)  
Lyy1 = np.zeros(NV,dtype=np.float64)  

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
    L_xx=np.dot(dNNNVdx[:],u[iconV[:,iel]])
    L_xy=np.dot(dNNNVdx[:],v[iconV[:,iel]])
    L_yx=np.dot(dNNNVdy[:],u[iconV[:,iel]])
    L_yy=np.dot(dNNNVdy[:],v[iconV[:,iel]])
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

###############################################################################

exx1 = np.zeros(NV,dtype=np.float64)  
eyy1 = np.zeros(NV,dtype=np.float64)  
exy1 = np.zeros(NV,dtype=np.float64)  

exx1[:]=Lxx1[:]
eyy1[:]=Lyy1[:]
exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

###############################################################################
# compute strain rate - corners to nodes - method 2
###############################################################################
start = timing.time()

count=np.zeros(NV,dtype=np.int32)  
q=np.zeros(NV,dtype=np.float64)
Lxx2=np.zeros(NV,dtype=np.float64)  
Lxy2=np.zeros(NV,dtype=np.float64)  
Lyx2=np.zeros(NV,dtype=np.float64)  
Lyy2=np.zeros(NV,dtype=np.float64)  
exx2=np.zeros(NV,dtype=np.float64)  
eyy2=np.zeros(NV,dtype=np.float64)  
exy2=np.zeros(NV,dtype=np.float64)  

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
        Lxx2[inode]+=np.dot(dNNNVdx[:],u[iconV[:,iel]])
        Lxy2[inode]+=np.dot(dNNNVdx[:],v[iconV[:,iel]])
        Lyx2[inode]+=np.dot(dNNNVdy[:],u[iconV[:,iel]])
        Lyy2[inode]+=np.dot(dNNNVdy[:],v[iconV[:,iel]])
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

print("     -> Lxx2 (m,M) %.4f %.4f " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4f %.4f " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lyx2),np.max(Lyx2)))

print("compute vel gradient meth-2 (%.3fs)" % (timing.time() - start))

###############################################################################
start = timing.time()

M_mat= np.zeros((NV,NV),dtype=np.float64)
rhsLxx=np.zeros(NV,dtype=np.float64)
rhsLyy=np.zeros(NV,dtype=np.float64)
rhsLxy=np.zeros(NV,dtype=np.float64)
rhsLyx=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):

    M_el =np.zeros((mV,mV),dtype=np.float64)
    fLxx_el=np.zeros(mV,dtype=np.float64)
    fLyy_el=np.zeros(mV,dtype=np.float64)
    fLxy_el=np.zeros(mV,dtype=np.float64)
    fLyx_el=np.zeros(mV,dtype=np.float64)
    NNNV =np.zeros((mV,1),dtype=np.float64) 

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

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

###############################################################################

exx3 = np.zeros(NV,dtype=np.float64)  
eyy3 = np.zeros(NV,dtype=np.float64)  
exy3 = np.zeros(NV,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

###############################################################################
# normalise pressure
# all nodes at the surface are equidistant
###############################################################################
start = timing.time()

poffset=np.sum(p[surfaceP])/np.count_nonzero(surfaceP)

q-=poffset
p-=poffset

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

#np.savetxt('pressure_normalised.ascii',np.array([xP,yP,p,rP]).T,header='# x,y,p,r')

print("normalise pressure (%.3fs)" % (timing.time() - start))

###############################################################################
start = timing.time()
 
e_rr=exx2*np.sin(theta)**2+2*exy2*np.sin(theta)*np.cos(theta)+eyy2*np.cos(theta)**2
e_tt=exx2*np.cos(theta)**2-2*exy2*np.sin(theta)*np.cos(theta)+eyy2*np.sin(theta)**2
e_rt=(exx2-eyy2)*np.sin(theta)*np.cos(theta)+exy2*(-np.sin(theta)**2+np.cos(theta)**2)

print("     -> e_rr (m,M) %e %e | nel= %d" %(np.min(e_rr),np.max(e_rr),nel))
print("     -> e_tt (m,M) %e %e | nel= %d" %(np.min(e_tt),np.max(e_tt),nel))
print("     -> e_rt (m,M) %e %e | nel= %d" %(np.min(e_rt),np.max(e_rt),nel))

print("strain rate in polar coords (%.3fs)" % (timing.time()-start))

###############################################################################
# export pressure at both surfaces
###############################################################################
start = timing.time()

if fs_method==3: np.savetxt('lambda.ascii',np.array([theta[surfaceV],l]).T)
np.savetxt('q_R1.ascii',np.array([xV[cmbV],yV[cmbV],q[cmbV],theta[cmbV]]).T)
np.savetxt('q_R2.ascii',np.array([xV[surfaceV],yV[surfaceV],q[surfaceV],theta[surfaceV]]).T)
np.savetxt('e_rr_R2.ascii',np.array([xV[surfaceV],yV[surfaceV],e_rr[surfaceV],theta[surfaceV]]).T)
np.savetxt('e_tt_R2.ascii',np.array([xV[surfaceV],yV[surfaceV],e_tt[surfaceV],theta[surfaceV]]).T)
np.savetxt('e_rt_R2.ascii',np.array([xV[surfaceV],yV[surfaceV],e_rt[surfaceV],theta[surfaceV]]).T)
np.savetxt('p_R1.ascii',np.array([xP[cmbP],yP[cmbP],p[cmbP]]).T)
np.savetxt('p_R2.ascii',np.array([xP[surfaceP],yP[surfaceP],p[surfaceP]]).T)

print("export p&q on R1,R2 (%.3fs)" % (timing.time() - start))

###############################################################################
# compute error
###############################################################################
start = timing.time()

NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

errv=0.
errp=0.
errq=0.
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

    for iq in [0,1,2]:
        for jq in [0,1,2]:
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

            xq=np.dot(NNNV[:],xV[iconV[:,iel]])
            yq=np.dot(NNNV[:],yV[iconV[:,iel]])
            uq=np.dot(NNNV[:],u[iconV[:,iel]])
            vq=np.dot(NNNV[:],v[iconV[:,iel]])
            qq=np.dot(NNNV[:],q[iconV[:,iel]])
            exx1q=np.dot(NNNV[:],exx1[iconV[:,iel]])
            eyy1q=np.dot(NNNV[:],eyy1[iconV[:,iel]])
            exy1q=np.dot(NNNV[:],exy1[iconV[:,iel]])
            exx2q=np.dot(NNNV[:],exx2[iconV[:,iel]])
            eyy2q=np.dot(NNNV[:],eyy2[iconV[:,iel]])
            exy2q=np.dot(NNNV[:],exy2[iconV[:,iel]])
            exx3q=np.dot(NNNV[:],exx3[iconV[:,iel]])
            eyy3q=np.dot(NNNV[:],eyy3[iconV[:,iel]])
            exy3q=np.dot(NNNV[:],exy3[iconV[:,iel]])

            radiusq=np.sqrt(xq**2+yq**2)
            thetaq=math.atan2(yq,xq)
  
            errv+=((uq-velocity_x(xq,yq,radiusq,thetaq,test))**2+\
                   (vq-velocity_y(xq,yq,radiusq,thetaq,test))**2)*weightq*jcob
            errq+=(qq-pressure(xq,yq,radiusq,thetaq,test))**2*weightq*jcob

            vrms+=(uq**2+vq**2)*weightq*jcob

            xq=np.dot(NNNP[:],xP[iconP[:,iel]])
            yq=np.dot(NNNP[:],yP[iconP[:,iel]])
            pq=np.dot(NNNP[:],p[iconP[:,iel]])
            radiusq=np.sqrt(xq**2+yq**2)
            thetaq=math.atan2(yq,xq)
            errp+=(pq-pressure(xq,yq,radiusq,thetaq,test))**2*weightq*jcob

        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)
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
print("     -> nelr= %6d ; errv= %e ; errp= %e ; errq= %e" %(nelr,errv,errp,errq))
print("     -> nelr= %6d ; errexx1= %e ; erreyy1= %e ; errexy1= %e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %6d ; errexx2= %e ; erreyy2= %e ; errexy2= %e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %6d ; errexx3= %e ; erreyy3= %e ; errexy3= %e" %(nelr,errexx3,erreyy3,errexy3))

print("compute errors (%.3fs)" % (timing.time() - start))

###############################################################################
# plot of solution
###############################################################################
start = timing.time()

if True:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.8e %.8e %.8e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.8e %.8e %.8e \n" %(velocity_x(xV[i],yV[i],rV[i],theta[i],test),\
                                           velocity_y(xV[i],yV[i],rV[i],theta[i],test),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i]-velocity_x(xV[i],yV[i],rV[i],theta[i],test),\
                                           v[i]-velocity_y(xV[i],yV[i],rV[i],theta[i],test),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %rV[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exx1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %eyy1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exy1[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %exy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e_rr' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rr[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e_tt' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_tt[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e_rt' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%e \n" %e_rt[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.13e %.13e %.13e \n" %(nx1[i],ny1[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.13e %.13e %.13e \n" %(nx2[i],ny2[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal diff' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%.13e %.13e %.13e \n" %(nx1[i]-nx2[i],ny1[i]-ny2[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='flag_surface' Format='ascii'> \n")
   for i in range(0,NV):
       if surfaceV[i]:
          vtufile.write("%d \n" %1)
       else:
          vtufile.write("%d \n" %0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='flag_cmb' Format='ascii'> \n")
   for i in range(0,NV):
       if cmbV[i]:
          vtufile.write("%d \n" %1)
       else:
          vtufile.write("%d \n" %0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%f\n" % pressure(xV[i],yV[i],rV[i],theta[i],test))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (error)' Format='ascii'> \n")
   for i in range (0,NV):
       vtufile.write("%e \n" % (q[i]-pressure(xV[i],yV[i],rV[i],theta[i],test)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%e\n" % (rhoQ1[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d %d %d %d\n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,4*nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,4*nel):
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
