import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg import *
import time as timing
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

###############################################################################

def velocity_x(x,y,r,theta,test):
    if test==1 or test==2:
       return 0
    if test==3:
       return -np.sin(theta)*vbc*(R2-r)/(R2-R1)
    if test==4:
       return -np.sin(theta)*vbc*r

def velocity_y(x,y,r,theta,test):
    if test==1 or test==2:
       return 0
    if test==3:
       return np.cos(theta)*vbc*(R2-r)/(R2-R1)
       return 0
    if test==4:
       return np.cos(theta)*vbc*r

def pressure(x,y,r,theta,test):
    if test==1 or test==2 or test==3 or test==4:
       return rho0*g0*(R2-r)

###############################################################################

def gx(x,y,g0):
    val=-x/np.sqrt(x*x+y*y)*g0
    return val

def gy(x,y,g0):
    val=-y/np.sqrt(x*x+y*y)*g0
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

if int(len(sys.argv) == 4):
   nelr = int(sys.argv[1])
   test = int(sys.argv[2])
   fs_method = int(sys.argv[3])
else:
   nelr = 20
   test = 4 
   fs_method=3

R1=1.
R2=2.

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
nnp=nnr*nnt  # number of nodes

xV=np.empty(nnp,dtype=np.float64)  # x coordinates
yV=np.empty(nnp,dtype=np.float64)  # y coordinates
r=np.empty(nnp,dtype=np.float64)  
theta=np.empty(nnp,dtype=np.float64) 

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
        r[counter]=R1+yi
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

NfemV=nnp*ndofV           # Total number of degrees of V freedom 
NfemP=nelt*(nelr+1)*ndofP # Total number of degrees of P freedom
Nfem=NfemV+NfemP+Nlm          # total number of dofs

print('nelr=',nelr)
print('nelt=',nelt)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nlm=',Nlm)
print('Nfem=',Nfem)
print('surface_bc=',surface_bc)
print('fs_method=',fs_method)
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

#now that I have both connectivity arrays I can 
# easily build xP,yP

NP=NfemP
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
# define boundary conditions
###############################################################################
start = timing.time()

eps=1.e-6

bc_fix=np.zeros(NfemV,dtype=bool)  
bc_val=np.zeros(NfemV,dtype=np.float64) 
surface=np.zeros(nnp,dtype=bool)  
cmb=np.zeros(nnp,dtype=bool)  
nx=np.empty(nnp,dtype=np.float64) 
ny=np.empty(nnp,dtype=np.float64) 

for i in range(0,nnp):
    #bottom boundary
    if r[i]/R1<1+eps:
       cmb[i]=True
       nx[i]=np.cos(theta[i])
       ny[i]=np.sin(theta[i])
       bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = -np.sin(theta[i])*vbc
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] =  np.cos(theta[i])*vbc
    #surface boundary
    if r[i]/R2>1-eps:
       surface[i]=True
       nx[i]=np.cos(theta[i])
       ny[i]=np.sin(theta[i])
       if surface_bc==0:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = 0
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0

print("defining boundary conditions (%.3fs)" % (timing.time() - start))

###############################################################################

surfaceP=np.zeros(NP,dtype=bool)  
cmbP=np.zeros(NP,dtype=bool)  

for i in range(0,NP):
    if rP[i]/R1<1+eps:
       cmbP[i]=True
    if rP[i]/R2>1-eps:
       surfaceP[i]=True


###############################################################################
# flag all elements with a node touching the surface r=R_outer
# or r=R_inner, used later for free slip b.c.
###############################################################################
start = timing.time()

flag_top=np.zeros(nel,dtype=np.float64)  
flag_bot=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    if surface[iconV[2,iel]]:
       flag_top[iel]=1
    if cmb[iconV[0,iel]]:
       flag_bot[iel]=1

print("flag elts on boundaries: %.3f s" % (timing.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

area=np.zeros(nel,dtype=np.float64) 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
dNNNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
dNNNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

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
print("     -> total area (meas) %.9f " %(area.sum()))
print("     -> total area (anal) %.9f " %(np.pi*(R2**2-R1**2)))

print("compute elements areas: %.3f s" % (timing.time() - start))

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
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
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
                f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx(xq,yq,g0)*rho0
                f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq,g0)*rho0
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
           if surface[inode]:
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
           if surface[inode]:
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
   for i in range(0,nnp):
       if surface[i]:
          # we need nx[i]*u[i]+ny[i]*v[i]=0
          A_sparse[counter, 2*i  ]=nx[i]
          A_sparse[counter, 2*i+1]=ny[i]
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

u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
p=sol[NfemV:NfemV+NfemP]
l=sol[NfemV+NfemP:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> l (m,M) %.4f %.4f " %(np.min(l),np.max(l)))

vr= np.cos(theta)*u+np.sin(theta)*v
vt=-np.sin(theta)*u+np.cos(theta)*v
    
print("     -> vr (m,M) %.4f %.4f " %(np.min(vr),np.max(vr)))
print("     -> vt (m,M) %.4f %.4f " %(np.min(vt),np.max(vt)))

np.savetxt('velocity.ascii',np.array([xV,yV,u,v,vr,vt,r]).T,header='# x,y,u,v,vr,vt,r')
np.savetxt('pressure.ascii',np.array([xP,yP,p,rP]).T,header='# x,y,p,r')

print("reshape solution (%.3fs)" % (timing.time() - start))

###############################################################################
# compute strain rate - center to nodes - method 1
###############################################################################

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

###############################################################################

exx1 = np.zeros(nnp,dtype=np.float64)  
eyy1 = np.zeros(nnp,dtype=np.float64)  
exy1 = np.zeros(nnp,dtype=np.float64)  

exx1[:]=Lxx1[:]
eyy1[:]=Lyy1[:]
exy1[:]=0.5*(Lxy1[:]+Lyx1[:])

###############################################################################
# compute strain rate - corners to nodes - method 2
###############################################################################
start = timing.time()

count = np.zeros(nnp,dtype=np.int32)  
q=np.zeros(nnp,dtype=np.float64)
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
        q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
        count[inode]+=1
    #end for
#end for
Lxx2/=count
Lxy2/=count
Lyx2/=count
Lyy2/=count
q/=count

print("     -> Lxx2 (m,M) %.4f %.4f " %(np.min(Lxx2),np.max(Lxx2)))
print("     -> Lyy2 (m,M) %.4f %.4f " %(np.min(Lyy2),np.max(Lyy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lxy2),np.max(Lxy2)))
print("     -> Lxy2 (m,M) %.4f %.4f " %(np.min(Lyx2),np.max(Lyx2)))

#np.savetxt('pressure.ascii',np.array([xV,yV,q]).T)
#np.savetxt('strainrate.ascii',np.array([xV,yV,Lxx,Lyy,Lxy,Lyx]).T)

print("compute vel gradient meth-2 (%.3fs)" % (timing.time() - start))

###############################################################################

exx2 = np.zeros(nnp,dtype=np.float64)  
eyy2 = np.zeros(nnp,dtype=np.float64)  
exy2 = np.zeros(nnp,dtype=np.float64)  

exx2[:]=Lxx2[:]
eyy2[:]=Lyy2[:]
exy2[:]=0.5*(Lxy2[:]+Lyx2[:])

###############################################################################
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

exx3 = np.zeros(nnp,dtype=np.float64)  
eyy3 = np.zeros(nnp,dtype=np.float64)  
exy3 = np.zeros(nnp,dtype=np.float64)  

exx3[:]=Lxx3[:]
eyy3[:]=Lyy3[:]
exy3[:]=0.5*(Lxy3[:]+Lyx3[:])

###############################################################################
# normalise pressure
###############################################################################
start = timing.time()

#print(np.sum(q[0:2*nelt])/(2*nelt))
#print(np.sum(q[nnp-2*nelt:nnp])/(2*nelt))
#print(np.sum(p[0:nelt])/(nelt))

#poffset=np.sum(q[0:2*nelt])/(2*nelt)
poffset=np.sum(q[surface])/(2*nelt)

q-=poffset
p-=poffset

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

np.savetxt('pressure_normalised.ascii',np.array([xP,yP,p,rP]).T,header='# x,y,p,r')

print("normalise pressure (%.3fs)" % (timing.time() - start))

###############################################################################
# export pressure at both surfaces
###############################################################################
start = timing.time()

np.savetxt('lambda.ascii',np.array([theta[surface],l]).T)
np.savetxt('q_R1.ascii',np.array([xV[cmb],yV[cmb],q[cmb],theta[cmb]]).T)
np.savetxt('q_R2.ascii',np.array([xV[surface],yV[surface],q[surface],theta[surface]]).T)
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

            xq=0.
            yq=0.
            uq=0.
            vq=0.
            qq=0.
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
                xq+=NNNV[k]*xV[iconV[k,iel]]
                yq+=NNNV[k]*yV[iconV[k,iel]]
                uq+=NNNV[k]*u[iconV[k,iel]]
                vq+=NNNV[k]*v[iconV[k,iel]]
                qq+=NNNV[k]*q[iconV[k,iel]]
                exx1q+=NNNV[k]*exx1[iconV[k,iel]]
                eyy1q+=NNNV[k]*eyy1[iconV[k,iel]]
                exy1q+=NNNV[k]*exy1[iconV[k,iel]]
                exx2q+=NNNV[k]*exx2[iconV[k,iel]]
                eyy2q+=NNNV[k]*eyy2[iconV[k,iel]]
                exy2q+=NNNV[k]*exy2[iconV[k,iel]]
                exx3q+=NNNV[k]*exx3[iconV[k,iel]]
                eyy3q+=NNNV[k]*eyy3[iconV[k,iel]]
                exy3q+=NNNV[k]*exy3[iconV[k,iel]]

            rq=np.sqrt(xq**2+yq**2)
            thetaq=math.atan2(yq,xq)
  
            errv+=((uq-velocity_x(xq,yq,rq,thetaq,test))**2+(vq-velocity_y(xq,yq,rq,thetaq,test))**2)*weightq*jcob
            errq+=(qq-pressure(xq,yq,rq,thetaq,test))**2*weightq*jcob

            vrms+=(uq**2+vq**2)*weightq*jcob

            xq=0.
            yq=0.
            pq=0.
            for k in range(0,mP):
                xq+=NNNP[k]*xP[iconP[k,iel]]
                yq+=NNNP[k]*yP[iconP[k,iel]]
                pq+=NNNP[k]*p[iconP[k,iel]]
            rq=np.sqrt(xq**2+yq**2)
            thetaq=math.atan2(yq,xq)
            errp+=(pq-pressure(xq,yq,rq,thetaq,test))**2*weightq*jcob

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
print("     -> nelr= %6d ; errv= %.8e ; errp= %.8e ; errq= %.8e" %(nelr,errv,errp,errq))
print("     -> nelr= %6d ; errexx1= %.8e ; erreyy1= %.8e ; errexy1= %.8e" %(nelr,errexx1,erreyy1,errexy1))
print("     -> nelr= %6d ; errexx2= %.8e ; erreyy2= %.8e ; errexy2= %.8e" %(nelr,errexx2,erreyy2,errexy2))
print("     -> nelr= %6d ; errexx3= %.8e ; erreyy3= %.8e ; errexy3= %.8e" %(nelr,errexx3,erreyy3,errexy3))

print("compute errors (%.3fs)" % (timing.time() - start))

###############################################################################
# plot of solution
###############################################################################
start = timing.time()

if True:
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
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(gx(xV[i],yV[i],g0),gy(xV[i],yV[i],g0),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(x,y)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(th)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%13e %13e %13e \n" %(velocity_x(xV[i],yV[i],r[i],theta[i],test),\
                                           velocity_y(xV[i],yV[i],r[i],theta[i],test),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(error)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(u[i]-velocity_x(xV[i],yV[i],r[i],theta[i],test),\
                                           v[i]-velocity_y(xV[i],yV[i],r[i],theta[i],test),0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity(r,theta)' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f %10f %10f \n" %(vr[i],vt[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %r[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='theta' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %theta[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy1' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy1[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy2' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy2[i])
   vtufile.write("</DataArray>\n")

   vtufile.write("<DataArray type='Float32' Name='exx3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exx3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %eyy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy3' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %exy3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='normal' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10e %10e %10e \n" %(nx[i],ny[i],0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='flag_surface' Format='ascii'> \n")
   for i in range(0,nnp):
       if surface[i]:
          vtufile.write("%d \n" %1)
       else:
          vtufile.write("%d \n" %0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='flag_cmb' Format='ascii'> \n")
   for i in range(0,nnp):
       if cmb[i]:
          vtufile.write("%d \n" %1)
       else:
          vtufile.write("%d \n" %0)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nnp):
       vtufile.write("%10f \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (th)' Format='ascii'> \n")
   for i in range (0,nnp):
       vtufile.write("%f\n" % pressure(xV[i],yV[i],r[i],theta[i],test))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (error)' Format='ascii'> \n")
   for i in range (0,nnp):
       vtufile.write("%e \n" % (q[i]-pressure(xV[i],yV[i],r[i],theta[i],test)))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   vtufile.write("<DataArray type='Int32' Name='flag_top' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % (flag_top[iel]))
       vtufile.write("%d\n" % (flag_top[iel]))
       vtufile.write("%d\n" % (flag_top[iel]))
       vtufile.write("%d\n" % (flag_top[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='flag_bot' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d\n" % (flag_bot[iel]))
       vtufile.write("%d\n" % (flag_bot[iel]))
       vtufile.write("%d\n" % (flag_bot[iel]))
       vtufile.write("%d\n" % (flag_bot[iel]))
   vtufile.write("</DataArray>\n")
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
