import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
import time as timing
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix,lil_matrix

benchmark=3

###############################################################################

def density(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    if benchmark==1:
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       gpr=A/2.+B/r**2*(1.-math.log(r))+1./r**2
       gppr=-B/r**3*(3.-2.*math.log(r))-2./r**3
       alephr=gppr - gpr/r -gr/r**2*(k**2-1.) +fr/r**2  +fpr/r
       val=k*math.sin(k*theta)*alephr + rho0 
    if benchmark==2:
       val=1e6
    if benchmark==3:
       val=4000
    return val

def velocity_x(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    if benchmark==1:
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       vr=k *gr * math.sin (k * theta)
       vtheta = fr *math.cos(k* theta)
       val=vr*math.cos(theta)-vtheta*math.sin(theta)
    if benchmark==2:
       val=0
    if benchmark==3:
       val=0
    return val

def velocity_y(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    if benchmark==1:
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       fpr=A-B/r**2
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       vr=k *gr * math.sin (k * theta)
       vtheta = fr *math.cos(k* theta)
       val=vr*math.sin(theta)+vtheta*math.cos(theta)
    if benchmark==2:
       val=0
    if benchmark==3:
       val=0
    return val

def pressure(x,y,R1,R2,k,rho0,g0):
    r=np.sqrt(x*x+y*y)
    theta=math.atan2(y,x)
    if benchmark==1:
       A=2.*(math.log(R1)-math.log(R2))/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       B=(R2**2-R1**2)/(R2**2*math.log(R1)-R1**2*math.log(R2) )
       fr=A*r+B/r
       gr=A/2.*r + B/r*math.log(r) - 1./r
       hr=(2*gr-fr)/r
       val=k*hr*math.sin(k*theta) + rho0*g0*(r-R2)
    if benchmark==2:
       val=(R2-r)*1e6*g0
    if benchmark==3:
       val=(R2-r)*4000*g0
    return val

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
    return val

###############################################################################

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
    return np.array([NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,\
                     NV_6,NV_7,NV_8],dtype=np.float64)

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
    return np.array([dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,\
                     dNVdr_6,dNVdr_7,dNVdr_8],dtype=np.float64)

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
    return np.array([dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,\
                     dNVds_6,dNVds_7,dNVds_8],dtype=np.float64)

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return np.array([NP_0,NP_1,NP_2,NP_3],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("--------stone 133------------")
print("-----------------------------")

ndim=2   # number of dimensions
mV=9     # number of nodes making up an element
mP=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 4):
   nelr = int(sys.argv[1])
   visu = int(sys.argv[2])
   nqperdim = int(sys.argv[3])
else:
   nelr = 128
   visu = 1
   nqperdim=3

if benchmark==1 or benchmark==2:
   R1=1.
   R2=2.
   viscosity=1.
   g0=1.
   velunit=1
   vunit=' '
if benchmark==3:
   R1=2890e3
   R2=6370e3
   viscosity=1e21
   g0=10
   velunit=1e-2/365.25/3600/24
   vunit="cm/year"

dr=(R2-R1)/nelr
nelt=12*nelr 
#nelt=4*nelr 
nel=nelr*nelt  

rho0=0.
kk=4

DJ=False

###########################################
# quadrature parameters

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


rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

eta_ref=viscosity

#################################################################
# grid point setup
#################################################################
start = timing.time()

nnr=nelr+1
nnt=nelt
nnp=nnr*nnt  # number of nodes

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates
uprho=np.empty(nnp,dtype=np.float64)  
uptheta=np.empty(nnp,dtype=np.float64) 

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

counter=0
for j in range(0,nnr):
    for i in range(0,nnt):
        xi=xV[counter]
        yi=yV[counter]
        t=xi/Louter*2.*math.pi    
        xV[counter]=math.cos(t)*(R1+yi)
        yV[counter]=math.sin(t)*(R1+yi)
        uprho[counter]=R1+yi
        uptheta[counter]=math.atan2(yV[counter],xV[counter])
        if uptheta[counter]<0.:
           uptheta[counter]+=2.*math.pi
        counter+=1

#for i in range(0,nnp):
#    print(i,'|',xV[i],yV[i],'|',uprho[i],uptheta[i]/np.pi*180)

print("building coordinate arrays (%.3fs)" % (timing.time() - start))

#################################################################
# build iconQ1 array needed for vtu file
# each Q2 element will be plotted as 4x Q1 elts
#################################################################

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
        iconQ1[0,counter] = icon1
        iconQ1[1,counter] = icon4
        iconQ1[2,counter] = icon3
        iconQ1[3,counter] = icon2
        counter += 1
    #end for
#end for

###############################################################################
# now that the grid has been built as if it was a Q1 grid, 
# we can simply use these same points to arrive at a Q2 
# connectivity array with 4 times less elements.
###############################################################################

nelr=nelr//2
nelt=nelt//2
nel=nel//4

NfemV=nnp*ndofV           # Total number of degrees of V freedom 
NfemP=nelt*(nelr+1)*ndofP # Total number of degrees of P freedom
Nfem=NfemV+NfemP          # total number of dofs

print('nelr=',nelr)
print('nelt=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV =np.zeros((mV,nel),dtype=np.int32)
iconP =np.zeros((mP,nel),dtype=np.int32)

counter = 0
for j in range(0, nelr):
    for i in range(0, nelt):
        iconV[0,counter]=2*counter +2*j*nelt
        iconV[1,counter]=2*counter +2*j*nelt +4*nelt
        iconV[2,counter]=2*counter +2*j*nelt +4*nelt+2
        iconV[3,counter]=2*counter +2*j*nelt +2
        iconV[4,counter]=2*counter +2*j*nelt +2*nelt
        iconV[5,counter]=2*counter +2*j*nelt +4*nelt+1
        iconV[6,counter]=2*counter +2*j*nelt +2*nelt+2
        iconV[7,counter]=2*counter +2*j*nelt +1
        iconV[8,counter]=2*counter +2*j*nelt +2*nelt+1
        if i==nelt-1:
           iconV[2,counter]-=2*nelt
           iconV[3,counter]-=2*nelt
           iconV[6,counter]-=2*nelt
        #print(counter,'|',iconV[0:mV,counter])
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
        iconP[0,counter]=icon1
        iconP[1,counter]=icon4
        iconP[2,counter]=icon3
        iconP[3,counter]=icon2
        counter += 1
    #end for

#for iel in range(0,nel):
#    print(iel,'|',iconP[:,iel])

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

print("building connectivity array (%.3fs)" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

eps=1.e-10

bc_fix = np.zeros(Nfem, dtype=bool)  
bc_val = np.zeros(Nfem, dtype=np.float64) 

for i in range(0,nnp):
    if uprho[i]/R1<1+eps:
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)
    if uprho[i]/R2>(1-eps):
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0)
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0)

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64) 
dNNNVds = np.zeros(mV,dtype=np.float64) 

for iel in range(0,nel):
    for jq in range(0,nqperdim):
        for iq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((2,2),dtype=np.float64)
            if DJ:
               uprho_tilde=0.5*(R2-R1)/nelr
               uptheta_bar=0.5*(2*np.pi/nelt)
               uprho_q=NNNV.dot(uprho[iconV[:,iel]])
               jcob=uprho_q*uprho_tilde*uptheta_bar
            else:
               #this is J_CL
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
    #print('area=',area[iel])
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.13f | %d" %(area.sum(),nelr))
print("     -> total area (anal) %.13f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (timing.time() - start))

#################################################################
# build FE matrix
#################################################################
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
f_rhs    = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
h_rhs    = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
b_mat    = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
N_mat    = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
NNNV     = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP     = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
u        = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v        = np.zeros(nnp,dtype=np.float64)          # y-component velocity
c_mat    = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for jq in range(0,nqperdim):
        for iq in range(0,nqperdim):

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
            jcbi=np.zeros((2,2),dtype=np.float64)

            if DJ:
               uprho_tilde=0.5*(R2-R1)/nelr
               uptheta_bar=0.5*(2*np.pi/nelt)
               uprho_q=NNNV.dot(uprho[iconV[:,iel]])
               uptheta_q=uptheta[iconV[8,iel]]+sq/2*(2*np.pi/nelt)
               #print(iq,jq,uprho_q,uptheta_q/np.pi*180,uprho_tilde,uptheta_bar/np.pi*180)
               #this is J_CL
               jcb[0,0] = np.cos(uptheta_q)*uprho_tilde
               jcb[0,1] = np.sin(uptheta_q)*uprho_tilde
               jcb[1,0] =-np.sin(uptheta_q)*uprho_q*uptheta_bar
               jcb[1,1] = np.cos(uptheta_q)*uprho_q*uptheta_bar
               #this is J_LC
               jcbi[0,0] = np.cos(uptheta_q)/uprho_tilde
               jcbi[0,1] =-np.sin(uptheta_q)/uptheta_bar/uprho_q
               jcbi[1,0] = np.sin(uptheta_q)/uprho_tilde
               jcbi[1,1] = np.cos(uptheta_q)/uptheta_bar/uprho_q
               #end for
               jcob=uprho_q*uprho_tilde*uptheta_bar
               xq=uprho_q*np.cos(uptheta_q) 
               yq=uprho_q*np.sin(uptheta_q) 
            else:
               #this is J_CL
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               #end for 
               jcob = np.linalg.det(jcb)
               jcbi = np.linalg.inv(jcb) # J_CL
               xq=NNNV.dot(xV[iconV[:,iel]])
               yq=NNNV.dot(yV[iconV[:,iel]])
            #end if

            # compute dNdx & dNdy
            for k in range(0,mV):
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
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,g0)*density(xq,yq,R1,R2,kk,rho0,g0)
            #end for 

            for i in range(0,mP):
                N_mat[0,i]=NNNP[i]
                N_mat[1,i]=NNNP[i]
                N_mat[2,i]=0.
            #end for 

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

        #end for jq
    #end for iq

    G_el*=eta_ref/R2

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
                    A_sparse[m1,m2] += K_el[ikk,jkk]
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
    #end for 

#end for iel

print("build FE matrixs & rhs (%.3fs)" % (timing.time() - start))

#################################################################
# solve system
#################################################################
start = timing.time()

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs
    
sparse_matrix=A_sparse.tocsr()

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solving system (%.3fs)" % (timing.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:Nfem]*eta_ref/R2

print("     -> u (m,M) %.8f %.8f %a" %(np.min(u)/velunit,np.max(u)/velunit,vunit))
print("     -> v (m,M) %.8f %.8f %a" %(np.min(v)/velunit,np.max(v)/velunit,vunit))

#np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

vr= np.cos(uptheta)*u+np.sin(uptheta)*v
vt=-np.sin(uptheta)*u+np.cos(uptheta)*v
    
print("     -> vr (m,M) %.8f %.8f %r" %(np.min(vr)/velunit,np.max(vr)/velunit,vunit))
print("     -> vt (m,M) %.8f %.8f %r" %(np.min(vt)/velunit,np.max(vt)/velunit,vunit))

print("reshape solution (%.3fs)" % (timing.time() - start))

#################################################################
# normalise pressure
#CAREFUL: pressure normalisation is not 100% clean !!!
#################################################################
start = timing.time()

if benchmark==1:
   poffset=np.sum(p[0:nelt])/(nelt) 
   p-=poffset
if benchmark==2 or benchmark==3:
   poffset=np.sum(p[NP-nelt:NP])/(nelt) 
   p-=poffset

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("normalise pressure (%.3fs)" % (timing.time() - start))

#################################################################
# export pressure at both surfaces
#################################################################
start = timing.time()

np.savetxt('p_R1.ascii',np.array([xP[0:nelt],yP[0:nelt],p[0:nelt]]).T)
np.savetxt('p_R2.ascii',np.array([xP[NP-nelt:NP],yP[NP-nelt:NP],p[NP-nelt:NP]]).T)

#compute analytical pressure solution on nodes
pth=np.zeros(NP,dtype=np.float64)
for i in range(NP):
    pth[i]=pressure(xP[i],yP[i],R1,R2,k,rho0,g0)

np.savetxt('pressure.ascii',np.array([xP,yP,p,pth]).T)

print("export p on R1,R2 (%.3fs)" % (timing.time() - start))

#################################################################
# scale velocities for error calculations and vtu output
#################################################################

u/=velunit
v/=velunit
vr/=velunit
vt/=velunit

#################################################################
# compute error
#################################################################
start = timing.time()

NNNV    = np.zeros(mV,dtype=np.float64) # shape functions V
dNNNVdr = np.zeros(mV,dtype=np.float64) # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64) # shape functions derivatives

errv=0.
errp=0.
vrms=0.
for iel in range (0,nel):

    for jq in range(0,nqperdim):
        for iq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)

            if DJ:
               uprho_tilde=0.5*(R2-R1)/nelr
               uptheta_bar=0.5*(2*np.pi/nelt)
               uprho_q=NNNV.dot(uprho[iconV[:,iel]])
               uptheta_q=uptheta[iconV[8,iel]]+sq/2*(2*np.pi/nelt)
               jcob=uprho_q*uprho_tilde*uptheta_bar
               xq=uprho_q*np.cos(uptheta_q) 
               yq=uprho_q*np.sin(uptheta_q) 
            else:
               jcb=np.zeros((2,2),dtype=np.float64)
               for k in range(0,mV):
                   jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                   jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                   jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                   jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               #end for 
               jcob = np.linalg.det(jcb)
               xq=NNNV.dot(xV[iconV[:,iel]])
               yq=NNNV.dot(yV[iconV[:,iel]])
            #end if

            uq=0.
            vq=0.
            for k in range(0,mV):
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
            errv+=((uq-velocity_x(xq,yq,R1,R2,kk,rho0,g0))**2+\
                   (vq-velocity_y(xq,yq,R1,R2,kk,rho0,g0))**2)*weightq*jcob

            vrms+=(uq**2+vq**2)*weightq*jcob

            pq=0.
            for k in range(0,mP):
                pq+=NNNP[k]*p[iconP[k,iel]]
            errp+=(pq-pressure(xq,yq,R1,R2,kk,rho0,g0))**2*weightq*jcob

        # end for jq
    # end for iq
# end for iel

areath=np.pi*(R2**2-R1**2)

errv=np.sqrt(errv/areath)
errp=np.sqrt(errp/areath)

vrms=np.sqrt(vrms/np.pi/(R2**2-R1**2))

print('     -> nelr=',nelr,' vrms=',vrms)
print("     -> nelr= %6d ; errv= %.8e ; errp= %.8e" %(nelr,errv,errp))

print("compute errors (%.3fs)" % (timing.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = timing.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,4*nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(xV[i],yV[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%e %e %e \n" %(velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                     velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%e %e %e \n" %(u[i]-velocity_x(xV[i],yV[i],R1,R2,kk,rho0,g0),\
                                     v[i]-velocity_y(xV[i],yV[i],R1,R2,kk,rho0,g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %uprho[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='uptheta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %uptheta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %density(xV[i],yV[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
   for i in range (0,nnp):
       vtufile.write("%f\n" % pressure(xV[i],yV[i],R1,R2,kk,rho0,g0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
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
