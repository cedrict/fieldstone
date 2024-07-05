import numpy as np
import time as time
import numba
from scipy.sparse import lil_matrix
import sys as sys
import scipy.sparse as sps

bench=1 # donea-huerta
bench=2 # zhan09,huzh11

###############################################################################

def f(x):
    return (x-x**2)**2

def fp(x):
    return 2*(1-2*x)*(x-x**2)

def fpp(x):
    return 2*(1-6*x+6*x**2)

def fppp(x):
    return 24*x-12

###############################################################################
# bx and by are the body force components 

def bx(x, y):
    match bench:
       case 1:
          val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
              (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
              (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
              1.-4.*y+12.*y*y-8.*y*y*y)
       case 2:
          val=-fpp(x)*fp(y)-f(x)*fppp(y)-fppp(x)*f(y)
          val*=256
    return val

def by(x, y):
    match bench:
       case 1:
          val=((8.-48.*y+48.*y*y)*x*x*x+
              (-12.+72.*y-72.*y*y)*x*x+
              (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
              12.*y*y+24.*y*y*y-12.*y**4)
       case 2:
          val=fppp(x)*f(y)+fp(x)*fpp(y)-fpp(x)*fp(y)
          val*=256
    return val

###############################################################################
# analytical solution

def velocity_x(x,y):
    match bench:
       case 1:
          val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
       case 2:
          val=f(x)*fp(y)
          val*=256
    return val

def velocity_y(x,y):
    match bench:
       case 1:
          val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
       case 2:
          val=-fp(x)*f(y)
          val*=256
    return val

def pressure(x,y):
    match bench:
       case 1:
          val=x*(1.-x)-1./6.
       case 2:
          val=-fpp(x)*f(y)
          val*=256
    return val

###############################################################################
#given the placement of the nodes/internal numbering the order of the 
#basis functions differs from the section in the manual
###############################################################################
#1-3-6-4-2-5

@numba.njit
def NNVu(r,s):
    N_0=0.5*r*(r-1) * 0.5*(1-s)
    N_1=0.5*r*(r+1) * 0.5*(1-s)
    N_2=0.5*r*(r+1) * 0.5*(1+s)
    N_3=0.5*r*(r-1) * 0.5*(1+s)
    N_4=(1-r*r)*0.5*(1-s)
    N_5=(1-r*r)*0.5*(1+s)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5],dtype=np.float64)

@numba.njit
def dNNVudr(r,s):
    dNdr_0=0.5*(2*r-1)* 0.5*(1-s)
    dNdr_1=0.5*(2*r+1)* 0.5*(1-s)
    dNdr_2=0.5*(2*r+1)* 0.5*(1+s)
    dNdr_3=0.5*(2*r-1)* 0.5*(1+s)
    dNdr_4=-2*r*0.5*(1-s)
    dNdr_5=-2*r*0.5*(1+s)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5],dtype=np.float64)

@numba.njit
def dNNVuds(r,s):
    dNds_0=0.5*r*(r-1)*0.5*(-1) 
    dNds_1=0.5*r*(r+1)*0.5*(-1) 
    dNds_2=0.5*r*(r+1)*0.5*(+1) 
    dNds_3=0.5*r*(r-1)*0.5*(+1) 
    dNds_4=(1-r*r)*0.5*(-1) 
    dNds_5=(1-r*r)*0.5*(+1) 
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5],dtype=np.float64)

###############################################################################
# 1-2-6-5-3-4 

@numba.njit
def NNVv(r,s):
    N_0=0.5*(1-r) * 0.5*s*(s-1)
    N_1=0.5*(1+r) * 0.5*s*(s-1)
    N_2=0.5*(1+r) * 0.5*s*(s+1)
    N_3=0.5*(1-r) * 0.5*s*(s+1)
    N_4=0.5*(1-r) * (1-s**2)
    N_5=0.5*(1+r) * (1-s**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5],dtype=np.float64)

@numba.njit
def dNNVvdr(r,s):
    dNdr_0=0.5*(-1) * 0.5*s*(s-1)
    dNdr_1=0.5*(+1) * 0.5*s*(s-1)
    dNdr_2=0.5*(+1) * 0.5*s*(s+1)
    dNdr_3=0.5*(-1) * 0.5*s*(s+1)
    dNdr_4=0.5*(-1) * (1-s**2)
    dNdr_5=0.5*(+1) * (1-s**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5],dtype=np.float64)

@numba.njit
def dNNVvds(r,s):
    dNds_0=0.5*(1-r) * 0.5*(2*s-1)
    dNds_1=0.5*(1+r) * 0.5*(2*s-1)
    dNds_2=0.5*(1+r) * 0.5*(2*s+1)
    dNds_3=0.5*(1-r) * 0.5*(2*s+1)
    dNds_4=0.5*(1-r) * (-2*s)
    dNds_5=0.5*(1+r) * (-2*s)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("-------fieldstone 162--------")
print("-----------------------------")

ndim=2
ndofV=2 
ndofP=1
mV=6
mP=4

Lx=1. 
Ly=1. 

eta=1.

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   laambda=int(sys.argv[4])
   laambda=10**laambda
else:
   nelx = 40
   nely = 40
   visu = 1
   laambda=1e3 # penalty parameter, alpha in zhan09

nnx=nelx+1 
nny=nely+1 

nel=nelx*nely  # number of elements, total

NP=mP*nel
NV=nnx*nny+nnx*nely+nny*nelx
Nu=nnx*nny+nelx*nny
Nv=nnx*nny+nnx*nely
NfemV=Nu+Nv #ndofV*nnx*nny +nnx*nely+nny*nelx

hx=Lx/nelx
hy=Ly/nely

tol=1e-8
niter=20

###############################################################################

nqperdim=3

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

nqel=nqperdim**2
nq=nqel*nel

###############################################################################
# check basis functions
###############################################################################

rVnodes=[-1,+1,+1,-1,0,0]
sVnodes=[-1,-1,+1,+1,-1,+1]

#for i in range(0,mV):
#    print(NNVu(rVnodes[i],sVnodes[i]))

#print('----------------------')

rVnodes=[-1,+1,+1,-1,-1,+1]
sVnodes=[-1,-1,+1,+1,0,0]

#for i in range(0,mV):
#    print(NNVv(rVnodes[i],sVnodes[i]))

###############################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("Nu=",Nu)
print("Nv=",Nv)
print("NP=",NP)
print("NfemV=",NfemV)
print("hx",hx)
print("hy",hy)
print("nqperdim",nqperdim)
print("nqel",nqel)
print("nq",nq)
print("laambda",laambda)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

xu = np.zeros(Nu,dtype=np.float64)  # x coordinates
yu = np.zeros(Nu,dtype=np.float64)  # y coordinates
xv = np.zeros(Nv,dtype=np.float64)  # x coordinates
yv = np.zeros(Nv,dtype=np.float64)  # y coordinates

counter = 0
counteru = 0
counterv = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xu[counteru]=i*hx
        yu[counteru]=j*hy
        counteru += 1
        xv[counterv]=i*hx
        yv[counterv]=j*hy
        counterv += 1
    #end for
#end for

for j in range(0,nely):
    for i in range(0,nnx):
        xv[counterv]=i*hx
        yv[counterv]=(j+0.5)*hy
        counterv += 1

for j in range(0,nny):
    for i in range(0,nelx):
        xu[counteru]=(i+0.5)*hx
        yu[counteru]=j*hy
        counteru += 1

np.savetxt('gridu.ascii',np.array([xu,yu]).T,header='# x,y')
np.savetxt('gridv.ascii',np.array([xv,yv]).T,header='# x,y')
   
###############################################################################
# connectivity arrays
###############################################################################
start = time.time()

iconu=np.zeros((mV,nel),dtype=np.int32)
iconv=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        #----
        iconu[0,counter] = i + j * (nelx + 1)
        iconu[1,counter] = i + 1 + j * (nelx + 1)
        iconu[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        iconu[3,counter] = i + (j + 1) * (nelx + 1)
        iconu[4,counter] = nnx*nny +  i  + j*nelx
        iconu[5,counter] = nnx*nny +  nelx + i+ j*nelx
        #----
        iconv[0,counter] = i + j * (nelx + 1)
        iconv[1,counter] = i + 1 + j * (nelx + 1)
        iconv[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        iconv[3,counter] = i + (j + 1) * (nelx + 1)
        iconv[4,counter] = nnx*nny + i + j*nnx 
        iconv[5,counter] = nnx*nny + i +1+ j*nnx 
        #----
        counter += 1
    #end for
#end for

#for iel in range(0,nel):
#    print(iel,'|',iconu[:,iel])
#for iel in range(0,nel):
#    print(iel,'|',iconv[:,iel])

print("connectivity setup: %.3f s" % (time.time() - start))

###############################################################################
# compute xc,yc
###############################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  # x coordinates
yc = np.zeros(nel,dtype=np.float64)  # y coordinates

for iel in range(0,nel):
    xc[iel]=0.5*(xu[iconu[0,iel]]+xu[iconu[2,iel]])
    yc[iel]=0.5*(yu[iconu[0,iel]]+yu[iconu[2,iel]])

np.savetxt('gridc.ascii',np.array([xc,yc]).T,header='# x,y')

print("center coordinates: %.3f s" % (time.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = time.time()

xq       = np.zeros(nq,dtype=np.float64) 
yq       = np.zeros(nq,dtype=np.float64) 
area     = np.zeros(nel,dtype=np.float64) 
dNNNVudx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVudr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVuds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVvds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
jcbi     = np.zeros((ndim,ndim),dtype=np.float64)

counterq=0
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNVu=NNVu(rq,sq)
                dNNNVudr=dNNVudr(rq,sq)
                dNNNVuds=dNNVuds(rq,sq)

                NNNVv=NNVv(rq,sq)
                dNNNVvdr=dNNVvdr(rq,sq)
                dNNNVvds=dNNVvds(rq,sq)

                xq[counterq]=NNNVu.dot(xu[iconu[0:mV,iel]])
                yq[counterq]=NNNVu.dot(yu[iconu[0:mV,iel]])

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

                #oneq=dNNNVudx.dot(xu[iconu[:,iel]])
                #print(oneq)
                #oneq=dNNNVudy.dot(yu[iconu[:,iel]])
                #print(oneq)

                area[iel]+=jcob*weightq
                counterq+=1

            #end for
        #end for
    #end for
#end for

np.savetxt('gridq.ascii',np.array([xq,yq]).T,header='# x,y')
                
print("     -> vol  (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total vol meas %.6f " %(area.sum()))
print("     -> total vol anal %.6f " %(Lx*Ly))

print("compute elements area: %.3f s" % (time.time() - start))

###############################################################################
# define boundary conditions
# 3--5--2     3-----2
# |     |     |     |
# |  u  |     4  v  5
# |     |     |     |
# 0--4--1     0-----1
###############################################################################
start = time.time()

bc_fix=np.zeros((mV*ndofV,nel),dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros((mV*ndofV,nel),dtype=np.float64)  # boundary condition, value

counter = 0
for iely in range(0, nely):
    for ielx in range(0, nelx):
        iel=iely*nelx+ielx
        if ielx==0: #left boundary element
           bc_fix[0,iel]=True  ; bc_val[0,iel]=0  # u0=0
           bc_fix[6,iel]=True  ; bc_val[6,iel]=0  # u3=0
           bc_fix[1,iel]=True  ; bc_val[1,iel]=0  # v0=0
           bc_fix[7,iel]=True  ; bc_val[7,iel]=0  # v3=0
           bc_fix[9,iel]=True  ; bc_val[9,iel]=0  # v4=0
        if ielx==nelx-1: #right boundary element
           bc_fix[2,iel]=True  ; bc_val[2,iel]=0  # u1=0
           bc_fix[4,iel]=True  ; bc_val[4,iel]=0  # u2=0
           bc_fix[3,iel]=True  ; bc_val[3,iel]=0  # v1=0
           bc_fix[5,iel]=True  ; bc_val[5,iel]=0  # v2=0
           bc_fix[11,iel]=True ; bc_val[11,iel]=0 # v5=0
        if iely==0: #bottom boundary element
           bc_fix[0,iel]=True  ; bc_val[0,iel]=0  # u0=0
           bc_fix[2,iel]=True  ; bc_val[2,iel]=0  # u1=0
           bc_fix[8,iel]=True  ; bc_val[8,iel]=0  # u4=0
           bc_fix[1,iel]=True  ; bc_val[1,iel]=0  # v0=0
           bc_fix[3,iel]=True  ; bc_val[3,iel]=0  # v1=0
        if iely==nely-1: #top boundary element
           bc_fix[4,iel]=True  ; bc_val[4,iel]=0  # u2=0
           bc_fix[6,iel]=True  ; bc_val[6,iel]=0  # u3=0
           bc_fix[10,iel]=True ; bc_val[10,iel]=0 # u5=0
           bc_fix[5,iel]=True  ; bc_val[5,iel]=0  # v2=0
           bc_fix[7,iel]=True  ; bc_val[7,iel]=0  # v3=0

print("boundary conditions setup: %.3f s" % (time.time() - start))

#******************************************************************************
#******************************************************************************
# carry out iterations
#******************************************************************************
#******************************************************************************
   
iterfile=open("conv.ascii","w")

c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
d_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
sum_sol=np.zeros(NfemV,dtype=np.float64) 
vel_prev=np.zeros(ndofV*mV,dtype=np.float64) 
u = np.zeros(Nu,dtype=np.float64)            # x-component velocity
v = np.zeros(Nv,dtype=np.float64)            # y-component velocity
u_prev = np.zeros(Nu,dtype=np.float64)            # x-component velocity
v_prev = np.zeros(Nv,dtype=np.float64)            # y-component velocity

for iter in range(0,niter):

    print('******************************')
    print('********* iter=',iter,'************')
    print('******************************')

    ###############################################################################
    # build FE matrix
    ###############################################################################
    start = time.time()

    A_sparse = lil_matrix((NfemV,NfemV),dtype=np.float64)
    rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    b_mat = np.zeros((3,mV*ndofV),dtype=np.float64)  # gradient matrix B 

    counterq=0
    for iel in range(0, nel):

        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        K_div =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)

        for k in range(0,mV):
            vel_prev[2*k  ]=sum_sol[iconu[k,iel]]
            vel_prev[2*k+1]=sum_sol[Nu+iconv[k,iel]]

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                # calculate shape functions
                NNNVu=NNVu(rq,sq)
                dNNNVudr=dNNVudr(rq,sq)
                dNNNVuds=dNNVuds(rq,sq)

                NNNVv=NNVv(rq,sq)
                dNNNVvdr=dNNVvdr(rq,sq)
                dNNNVvds=dNNVvds(rq,sq)

                #compute jacobian matrix and determinant
                #jcob=hx*hy/4
                #jcbi[0,0]=2/hx ; jcbi[0,1]=0    
                #jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

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

                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

                for i in range(0,mV):
                    f_el[ndofV*i+0]+=NNNVu[i]*jcob*weightq*bx(xq[counterq],yq[counterq])
                    f_el[ndofV*i+1]+=NNNVv[i]*jcob*weightq*by(xq[counterq],yq[counterq])
                #end for

                K_div+=b_mat.T.dot(d_mat.dot(b_mat))*weightq*jcob

                counterq+=1

            #end for jq
        #end for iq

        K_el += laambda*K_div
        f_el -= laambda*K_div.dot(vel_prev) 

        #apply boundary conditions
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
            #end if
        #end for

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1+i1
                if i1==0:               # u dof
                   m1=iconu[k1,iel]
                else:
                   m1=Nu+iconv[k1,iel]  # v dof
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        if i2==0:
                           m2=iconu[k2,iel]
                        else:
                           m2=Nu+iconv[k2,iel]
                        A_sparse[m1,m2] += K_el[ikk,jkk]
                    #end for
                #end for
                rhs[m1]+=f_el[ikk]
            #end for
        #end for

    #end for iel

    print("build FE matrix: %.3f s" % (time.time() - start))

    ###############################################################################
    # solve linear system 
    ###############################################################################
    start = time.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

    print("solve linear system: %.3f s" % (time.time() - start))

    ###############################################################################
    # separate fields 
    ###############################################################################
    start = time.time()

    u[:]=sol[0:Nu]
    v[:]=sol[0+Nu:Nv+Nu]

    print("     -> u (m,M) %.5f %.5f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5f %.5f " %(np.min(v),np.max(v)))

    #np.savetxt('solution_u.ascii',np.array([xu,yu,u]).T,header='# x,y')
    #np.savetxt('solution_v.ascii',np.array([xv,yv,v]).T,header='# x,y')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ###############################################################################
    # assess convergence
    ###############################################################################

    xi_u=np.max(abs(u_prev-u))/np.max(abs(u))
    xi_v=np.max(abs(v_prev-v))/np.max(abs(v))

    print('     -> xi_u,xi_v,conv= %e %e %s' %(xi_u,xi_v,xi_u<tol and xi_v<tol))
           
    iterfile.write("%i %e %e %e\n" %(iter,xi_u,xi_v,tol))

    if xi_u<tol and xi_v<tol:
       break

    u_prev[:]=u[:]
    v_prev[:]=v[:]
    sum_sol[:]+=sol[:]

#end for iter
    
print('******************************')

###############################################################################
# compute pressure 
###############################################################################
start = time.time()

p = np.zeros(NP,dtype=np.float64)
divv = np.zeros(NP,dtype=np.float64)
sum_u=np.zeros(Nu,dtype=np.float64)
sum_v=np.zeros(Nv,dtype=np.float64)

sum_u[:]=sum_sol[0:Nu]
sum_v[:]=sum_sol[0+Nu:Nv+Nu]

counter=0
for iel in range(0,nel):

    for k in range(0,4):

        rq=rVnodes[k]
        sq=sVnodes[k]

        dNNNVudr=dNNVudr(rq,sq)
        #dNNNVuds=dNNVuds(rq,sq)

        #dNNNVvdr=dNNVvdr(rq,sq)
        dNNNVvds=dNNVvds(rq,sq)

        #compute jacobian matrix and determinant
        #jcob=hx*hy/4
        #jcbi[0,0]=2/hx ; jcbi[0,1]=0    
        #jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

        # compute dNdx, dNdy
        for k in range(0,mV):
            dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
            #dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
            #dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
            dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
        #end for
        
        p[counter]=-laambda*(dNNNVudx.dot(sum_u[iconu[:,iel]])+\
                             dNNNVvdy.dot(sum_v[iconv[:,iel]]))

        divv[counter]=dNNNVudx.dot(u[iconu[:,iel]])+\
                      dNNNVvdy.dot(v[iconv[:,iel]])

        counter+=1

    #end for
#end for

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("compute pressure: %.3f s" % (time.time() - start))


pressfile=open("pressure.ascii","w")
for iel in range(0,nel):
    for k in range(0,4):
        pressfile.write("%10f %10f %10f \n" %(xu[iconu[k,iel]],yu[iconu[k,iel]],p[4*iel+k]))


###############################################################################
# normalise pressure
###############################################################################
start = time.time()

int_p=0
counterq=0
for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNVudr=dNNVudr(rq,sq)
            #dNNNVuds=dNNVuds(rq,sq)
            #dNNNVvdr=dNNVvdr(rq,sq)
            dNNNVvds=dNNVvds(rq,sq)
            for k in range(0,mV):
                dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                #dNNNVudy[k]=jcbi[1,1]*dNNNVuds[k]
                #dNNNVvdx[k]=jcbi[0,0]*dNNNVvdr[k]
                dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
            #end for
            pq=-laambda*(dNNNVudx.dot(sum_u[iconu[:,iel]])+\
                         dNNNVvdy.dot(sum_v[iconv[:,iel]]))
            int_p+=pq*weightq*jcob
            counterq+=1
        #end for
    #end for
#end for

print('int p dV=',int_p)


print("compute avrg pressure: %.3f s" % (time.time() - start))

###############################################################################
# compute velocity divergence on swarm of points
###############################################################################

nmarker_per_dim=5
nmarker=nmarker_per_dim**2*nel

xm=np.zeros(nmarker,dtype=np.float64)
ym=np.zeros(nmarker,dtype=np.float64)
divvm=np.zeros(nmarker,dtype=np.float64)

counter=0
for iel in range(0,nel):

    for i in range(0,nmarker_per_dim):
        for j in range(0,nmarker_per_dim):

            rm=-1+(i+0.5)*2/nmarker_per_dim
            sm=-1+(j+0.5)*2/nmarker_per_dim

            NNNVu=NNVu(rm,sm)
            NNNVv=NNVv(rm,sm)

            xm[counter]=NNNVu.dot(xu[iconu[0:mV,iel]])
            ym[counter]=NNNVu.dot(yu[iconu[0:mV,iel]])

            dNNNVudr=dNNVudr(rm,sm)
            dNNNVvds=dNNVvds(rm,sm)

            for k in range(0,mV):
                dNNNVudx[k]=jcbi[0,0]*dNNNVudr[k]
                dNNNVvdy[k]=jcbi[1,1]*dNNNVvds[k]
            #end for

            exx=dNNNVudx.dot(u[iconu[:,iel]])
            eyy=dNNNVvdy.dot(v[iconv[:,iel]])
        
            divvm[counter]=exx+eyy

            counter+=1

        #end for j
    #end for i
#end for iel

np.savetxt('divv.ascii',np.array([xm,ym,divvm]).T,header='# x,y')

###############################################################################
# compute error in L2 norm
# here again assuming jcob does not change
# for pressure I use Q1 basis functions
###############################################################################
start = time.time()

N=np.zeros(4,dtype=np.float64)

errv=0.
errp=0.
counterq=0
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            NNNVu=NNVu(rq,sq)
            NNNVv=NNVv(rq,sq)
            uq=NNNVu.dot(u[iconu[0:mV,iel]])
            vq=NNNVv.dot(v[iconv[0:mV,iel]])

            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            pq=N.dot(p[4*iel:4*iel+4])

            errv+=((uq-velocity_x(xq[counterq],yq[counterq]))**2+(vq-velocity_y(xq[counterq],yq[counterq]))**2)*weightq*jcob
            errp+=(pq-pressure(xq[counterq],yq[counterq]))**2*weightq*jcob
            counterq+=1
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

###############################################################################
# plot of solution (given the node layout, the easiest vtu export is Q1)
###############################################################################
start = time.time()



if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(xu[iconu[k,iel]],yu[iconu[k,iel]],0))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           vtufile.write("%e  \n" %(p[4*iel+k]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='divv' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           vtufile.write("%e  \n" %(divv[4*iel+k]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           pth=pressure(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vtufile.write("%e  \n" %(p[4*iel+k]-pth))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           pth=pressure(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vtufile.write("%e  \n" %(pth))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           vtufile.write("%e %e %e \n" %(u[iconu[k,iel]],v[iconv[k,iel]],0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           uth=velocity_x(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vth=velocity_y(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vtufile.write("%e %e %e \n" %(u[iconu[k,iel]]-uth,v[iconv[k,iel]]-vth,0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (th)' Format='ascii'> \n")
   for iel in range(0,nel):
       for k in range(0,4):
           uth=velocity_x(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vth=velocity_y(xu[iconu[k,iel]],yu[iconu[k,iel]])
           vtufile.write("%e %e %e \n" %(uth,vth,0))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d \n" %(4*iel,4*iel+1,4*iel+2,4*iel+3))
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
