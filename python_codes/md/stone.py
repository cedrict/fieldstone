import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import *
import time as timing
from scipy import sparse
import numba

###############################################################################
# density function
###############################################################################

@numba.njit
def rho(rho0,alphaT,T,T0):
    val=rho0*(1.-alphaT*(T-T0)) 
    return val

@numba.njit
def eta(T,x,y,eta0):
    return eta0 #*np.exp(T)

###############################################################################
# velocity shape functions
###############################################################################

@numba.njit
def NNV(r,s):
    N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N_1=    (1.-r**2) * 0.5*s*(s-1.)
    N_2= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N_3= 0.5*r*(r-1.) *    (1.-s**2)
    N_4=    (1.-r**2) *    (1.-s**2)
    N_5= 0.5*r*(r+1.) *    (1.-s**2)
    N_6= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N_7=    (1.-r**2) * 0.5*s*(s+1.)
    N_8= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

@numba.njit
def dNNVdr(r,s):
    dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr_1=       (-2.*r) * 0.5*s*(s-1)
    dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr_3= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr_4=       (-2.*r) *   (1.-s**2)
    dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr_6= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr_7=       (-2.*r) * 0.5*s*(s+1)
    dNdr_8= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,\
                     dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

@numba.njit
def dNNVds(r,s):
    dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds_1=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds_3= 0.5*r*(r-1.) *       (-2.*s)
    dNds_4=    (1.-r**2) *       (-2.*s)
    dNds_5= 0.5*r*(r+1.) *       (-2.*s)
    dNds_6= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds_7=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds_8= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,\
                     dNds_6,dNds_7,dNds_8],dtype=np.float64)

###############################################################################
# pressure shape functions 
###############################################################################

@numba.njit
def NNP(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1-r)*(1+s)
    N_3=0.25*(1+r)*(1+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

###############################################################################
# constants

eps=1e-9

###############################################################################

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2   # number of dimensions
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom per node
ndofT=1  # number of temperature degrees of freedom per node

Lx=1.
Ly=1.

if int(len(sys.argv) == 4):
   nelx  = int(sys.argv[1])
   Ra_nb = float(sys.argv[2])
   nstep = int(sys.argv[3])
else:
   nelx = 24
   Ra_nb= 1e4
   nstep= 100

tol_ss=1e-7   # tolerance for steady state 

top_bc_noslip=False
bot_bc_noslip=False

nely=nelx

nel=nelx*nely # total number of elements
nnx=2*nelx+1  # number of V nodes, x direction
nny=2*nely+1  # number of V nodes, y direction
NV=nnx*nny    # number of V nodes
NP=(nelx+1)*(nely+1) # number of P nodes

mV=9     # number of velocity nodes per element
mP=4     # number of pressure nodes per element
mT=9     # number of temperature nodes per element

rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
rPnodes=[-1,+1,-1,+1]
sPnodes=[-1,-1,+1,+1]

ndofV_el=mV*ndofV

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs
NfemT=NV*ndofT       # nb of temperature dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

EBA=False

###############################################################################

t01=0 ; t02=0 ; t03=0 ; t04=0 ; t05=0 ; t06=0
t07=0 ; t08=0 ; t09=0 ; t10=0 ; t11=0 ; t12=0

###############################################################################
# definition: Ra_nb=alphaT*abs(gy)*Ly**3*rho0**2*hcapa/hcond/eta
# following parameters are somewhat arbitrary

alphaT=2.5e-3   # thermal expansion coefficient
hcond=1.      # thermal conductivity
hcapa=1e-2      # heat capacity
rho0=20        # reference density
T0=0          # reference temperature
relax=0.75    # relaxation coefficient (0,1)
gy=-1 #Ra/alphaT # vertical component of gravity vector

eta0 = alphaT*abs(gy)*Ly**3*rho0**2*hcapa/hcond/Ra_nb

Di_nb=alphaT*abs(gy)*Ly/hcapa

###############################################################################
# compute reference quantities
###############################################################################

L_ref=Ly
T_ref=1
eta_ref=eta0
kappa_ref=hcond/hcapa/rho0
vel_ref=kappa_ref/L_ref

###############################################################################
# quadrature rule points and weights
###############################################################################

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nqel=nqperdim**ndim

###############################################################################
# open output files & write headers
###############################################################################

Nu_vrms_file=open('Nu_vrms.ascii',"w")
Nu_vrms_file.write("#istep,Nusselt,vrms,qy bottom, qy top\n")
Tavrg_file=open('Tavrg.ascii',"w")
Tavrg_file.write("#istep,Tavrg\n")
conv_file=open('conv.ascii',"w")
conv_file.write("#istep,T_diff,Nu_diff,tol_ss\n")
pstats_file=open('pressure_stats.ascii',"w")
pstats_file.write("#istep,min p, max p\n")
vstats_file=open('velocity_stats.ascii',"w")
vstats_file.write("#istep,min(u),max(u),min(v),max(v)\n")

###############################################################################

print ('Ra       =',Ra_nb)
print ('Di       =',Di_nb)
print ('eta0     =',eta0)
print ('nnx      =',nnx)
print ('nny      =',nny)
print ('NV       =',NV)
print ('NP       =',NP)
print ('nel      =',nel)
print ('NfemV    =',NfemV)
print ('NfemP    =',NfemP)
print ('Nfem     =',Nfem)
print ('nqperdim =',nqperdim)
print("-----------------------------")

###############################################################################
# checking that all velocity shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i]))

###############################################################################
# checking that all pressure shape functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i]))

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2
        yV[counter]=j*hy/2
        counter+=1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("build V grid: %.3f s" % (timing.time() - start))

###############################################################################
# connectivity
###############################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,3):
            for l in range(0,3):
                iconV[counter2,counter]=i*2+l+j*2*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

# creating a dedicated connectivity array to plot the solution on Q1 space
# different icon array but same velocity nodes.

nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int32)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

print("build iconV: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure grid 
###############################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        xP[counter]=i*hx
        yP[counter]=j*hy
        counter+=1
    #end for
 #end for

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,2):
            for l in range(0,2):
                iconP[counter2,counter]=i+l+j*(nelx+1)+(nelx+1)*k 
                counter2+=1
            #end for
        #end for
        counter+=1
    #end for
#end for

print("build iconP: %.3f s" % (timing.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if bot_bc_noslip:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if yV[i]>(Ly-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if top_bc_noslip:
          bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.

print("velocity b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]<eps:
       bc_fixT[i]=True ; bc_valT[i]=1.
    if yV[i]>(Ly-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
#end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

###############################################################################
# initial temperature
###############################################################################

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]=1.-yV[i]-0.01*np.cos(np.pi*xV[i]/Lx)*np.sin(np.pi*yV[i]/Ly)
#end for

T_prev[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

###############################################################################
# compute area of elements
###############################################################################
start = timing.time()

area    = np.zeros(nel,dtype=np.float64) 
dNNNVdr = np.zeros(mV,dtype=np.float64)  # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)  # shape functions derivatives

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcob = np.linalg.det(jcb)
            #jcbi = np.linalg.inv(jcb)
            #print (jcob,hx*hy/4)
            #print(jcbi,2/hx,2/hy)
            area[iel]+=jcob*weightq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

###############################################################################
# precompute basis functions values at q points
###############################################################################
start = timing.time()

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcob=hx*hy/4

NNNV=np.zeros((nqel,mV),dtype=np.float64) 
NNNP=np.zeros((nqel,mP),dtype=np.float64) 
dNNNVdr=np.zeros((nqel,mV),dtype=np.float64) 
dNNNVds=np.zeros((nqel,mV),dtype=np.float64) 
dNNNVdx=np.zeros((nqel,mV),dtype=np.float64) 
dNNNVdy=np.zeros((nqel,mV),dtype=np.float64) 
rq=np.zeros(nqel,dtype=np.float64) 
sq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
   
counterq=0 
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        rq[counterq]=qcoords[iq]
        sq[counterq]=qcoords[jq]
        weightq[counterq]=qweights[iq]*qweights[jq]
        NNNV[counterq,0:mV]=NNV(rq[counterq],sq[counterq])
        dNNNVdr[counterq,0:mV]=dNNVdr(rq[counterq],sq[counterq])
        dNNNVds[counterq,0:mV]=dNNVds(rq[counterq],sq[counterq])
        NNNP[counterq,0:mP]=NNP(rq[counterq],sq[counterq])
        dNNNVdx[counterq,0:mV]=jcbi[0,0]*dNNNVdr[counterq,0:mV]
        dNNNVdy[counterq,0:mV]=jcbi[1,1]*dNNNVds[counterq,0:mV]
        counterq+=1

print("compute N & grad(N) at q pts: %.3f s" % (timing.time() - start))

###############################################################################
# precompute basis functions values at V nodes
###############################################################################
start = timing.time()

NNNV_n=np.zeros((mV,mV),dtype=np.float64) 
NNNP_n=np.zeros((mV,mP),dtype=np.float64) 
dNNNVdr_n=np.zeros((mV,mV),dtype=np.float64) 
dNNNVds_n=np.zeros((mV,mV),dtype=np.float64) 
dNNNVdx_n=np.zeros((mV,mV),dtype=np.float64) 
dNNNVdy_n=np.zeros((mV,mV),dtype=np.float64) 
   
for i in range(0,mV):
    rq=rVnodes[i]
    sq=sVnodes[i]
    NNNV_n[i,0:mV]=NNV(rq,sq)
    dNNNVdr_n[i,0:mV]=dNNVdr(rq,sq)
    dNNNVds_n[i,0:mV]=dNNVds(rq,sq)
    NNNP_n[i,0:mP]=NNP(rq,sq)
    dNNNVdx_n[i,0:mV]=jcbi[0,0]*dNNNVdr_n[i,0:mV]
    dNNNVdy_n[i,0:mV]=jcbi[1,1]*dNNNVds_n[i,0:mV]

print("compute N & grad(N) at V nodes: %.3f s" % (timing.time() - start))

###############################################################################
# compute array for assembly
###############################################################################
start = timing.time()

local_to_globalV=np.zeros((ndofV_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1+i1
            m1 =ndofV*iconV[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
                 
print("compute local_to_globalV: %.3f s" % (timing.time() - start))

###############################################################################
# fill I,J arrays
###############################################################################
start = timing.time()

bignb=nel*( (mV*ndofV)**2 + 2*(mV*ndofV*mP) )

II_V=np.zeros(bignb,dtype=np.int32)    
JJ_V=np.zeros(bignb,dtype=np.int32)    
VV_V=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndofV_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndofV_el):
            m2=local_to_globalV[jkk,iel]
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
        for jkk in range(0,mP):
            m2 =iconP[jkk,iel]+NfemV
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
            II_V[counter]=m2
            JJ_V[counter]=m1
            counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (timing.time()-start))

###############################################################################
# fill I,J arrays
###############################################################################
start = timing.time()

bignb=nel*mT**2 

II_T=np.zeros(bignb,dtype=np.int32)    
JJ_T=np.zeros(bignb,dtype=np.int32)    
VV_T=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(mT):
        m1=iconV[ikk,iel]
        for jkk in range(mT):
            m2=iconV[jkk,iel]
            II_T[counter]=m1
            JJ_T[counter]=m2
            counter+=1

print("fill II_T,JJ_T arrays: %.3f s" % (timing.time()-start))

###############################################################################
###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
###############################################################################
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

topstart = timing.time()

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start = timing.time()

    f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 
    b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  

    counter=0
    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((ndofV_el),dtype=np.float64)
        K_el =np.zeros((ndofV_el,ndofV_el),dtype=np.float64)
        G_el=np.zeros((ndofV_el,mP),dtype=np.float64)
        h_el=np.zeros((mP),dtype=np.float64)

        for iq in range(0,nqel):

            JxW=jcob*weightq[iq]

            xq=np.dot(NNNV[iq,:],xV[iconV[:,iel]])
            yq=np.dot(NNNV[iq,:],yV[iconV[:,iel]])
            Tq=np.dot(NNNV[iq,:],T[iconV[:,iel]])

            for i in range(0,mV):
                dNdx=dNNNVdx[iq,i] 
                dNdy=dNNNVdy[iq,i] 
                b_mat[0,2*i  ]=dNdx
                b_mat[1,2*i+1]=dNdy
                b_mat[2,2*i  ]=dNdy
                b_mat[2,2*i+1]=dNdx

            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(Tq,xq,yq,eta0)*JxW

            for i in range(0,mV):
                f_el[ndofV*i+1]+=NNNV[iq,i]*JxW*rho(rho0,alphaT,Tq,T0)*gy

            N_mat[0,0:mP]=NNNP[iq,0:mP]
            N_mat[1,0:mP]=NNNP[iq,0:mP]

            G_el-=b_mat.T.dot(N_mat)*JxW

        # end for iq

        G_el*=eta_ref/Lx

        # impose b.c. 
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
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(ndofV_el):
            m1=local_to_globalV[ikk,iel]
            for jkk in range(ndofV_el):
                VV_V[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,mP):
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
            f_rhs[m1]+=f_el[ikk]
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]

    print("build FE matrix: %.3fs" % (timing.time()-start))

    t01+=timing.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    sparse_matrix=sparse.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    t02+=timing.time()-start

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Lx)

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    vstats_file.write("%10e %10e %10e %10e %10e\n" % (istep,np.min(u),np.max(u),\
                                                            np.min(u),np.max(u)))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    ###########################################################################
    # normalise pressure
    ###########################################################################
    start = timing.time()

    pressure_avrg=0
    for iel in range(0,nel):
        for iq in range(0,nqel):
            pressure_avrg+=np.dot(NNNP[iq,0:mP],p[iconP[0:mP,iel]])*jcob*weightq[iq]
        #end for iq
    #end for iel
    p-=pressure_avrg

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    pstats_file.write("%10e %10e %10e\n" % (istep,np.min(p),np.max(p)))
        
    print("normalise pressure: %.3f s" % (timing.time() - start))

    t12+=timing.time()-start

    ###########################################################################
    # relaxation step
    ###########################################################################

    if istep>0:
       u[:]=relax*u[:]+(1-relax)*u_prev[:]
       v[:]=relax*v[:]+(1-relax)*v_prev[:]

    ###########################################################################
    # compute nodal pressure 
    ###########################################################################
    start = timing.time()
    
    count = np.zeros(NV,dtype=np.int32)  
    q=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
            inode=iconV[i,iel]
            q[inode]+=np.dot(NNNP_n[i,:],p[iconP[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    q/=count

    print("     -> q     (m,M) %.6e %.6e " %(np.min(q    ),np.max(q    )))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

    print("compute nodal press: %.3f s" % (timing.time() - start))

    t03+=timing.time()-start

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start = timing.time()

    Tvect=np.zeros(mT,dtype=np.float64)   
    rhs=np.zeros(NfemT,dtype=np.float64)    # FE rhs 
    B_mat=np.zeros((2,mT),dtype=np.float64)     # gradient matrix B 
    N_mat=np.zeros((mT,1),dtype=np.float64)   # shape functions

    counter=0
    for iel in range (0,nel):

        b_el=np.zeros(mT,dtype=np.float64)
        a_el=np.zeros((mT,mT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mT,mT),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        Tvect[0:mT]=T[iconV[0:mT,iel]]

        for iq in range(0,nqel):

            JxW=jcob*weightq[iq]

            N_mat[0:mV,0]=NNNV[iq,:]

            vel[0,0]=np.dot(N_mat[:,0],u[iconV[:,iel]])
            vel[0,1]=np.dot(N_mat[:,0],v[iconV[:,iel]])

            B_mat[0,0:mT]=dNNNVdx[iq,:]
            B_mat[1,0:mT]=dNNNVdy[iq,:]

            # compute mass matrix
            #MM+=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

            # compute diffusion matrix
            Kd+=B_mat.T.dot(B_mat)*hcond*JxW

            # compute advection matrix
            Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*JxW

            if EBA:
               xq=np.dot(NNNV[iq,:],xV[iconV[:,iel]])
               yq=np.dot(NNNV[iq,:],yV[iconV[:,iel]])
               Tq=np.dot(NNNV[iq,:],T[iconV[:,iel]])
               exxq=np.dot(dNNNVdx[iq,:],u[iconV[:,iel]])
               eyyq=np.dot(dNNNVdy[iq,:],v[iconV[:,iel]])
               exyq=np.dot(dNNNVdy[iq,:],u[iconV[:,iel]])*0.5\
                   +np.dot(dNNNVdx[iq,:],v[iconV[:,iel]])*0.5
               dpdxq=np.dot(dNNNVdx[iq,:],q[iconV[:,iel]])
               dpdyq=np.dot(dNNNVdy[iq,:],q[iconV[:,iel]])
               #viscous dissipation
               b_el[:]+=N_mat[:,0]*JxW*2*eta(Tq,xq,yq,eta0)*(exxq**2+eyyq**2+2*exyq**2) 
               #adiabatic heating
               b_el[:]+=N_mat[:,0]*JxW*alphaT*Tq*(vel[0,0]*dpdxq+vel[0,1]*dpdyq)  

        #end for

        a_el=Ka+Kd

        # apply boundary conditions
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mV):
                   m2=iconV[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(mT):
            m1=iconV[ikk,iel]
            for jkk in range(mT):
                VV_T[counter]=a_el[ikk,jkk]
                counter+=1
            rhs[m1]+=b_el[ikk]
        #end for

    #end for iel

    print("build FE matrix : %.3f s" % (timing.time() - start))

    t04+=timing.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start = timing.time()

    sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(NfemT,NfemT)).tocsr()

    T = sps.linalg.spsolve(sparse_matrix,rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (timing.time() - start))

    t05+=timing.time()-start

    ###########################################################################
    # relax
    ###########################################################################

    if istep>0:
       T=relax*T+(1-relax)*T_prev

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start = timing.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nqel):
            JxW=jcob*weightq[iq]
            uq=np.dot(NNNV[iq,:],u[iconV[:,iel]])
            vq=np.dot(NNNV[iq,:],v[iconV[:,iel]])
            Tq=np.dot(NNNV[iq,:],T[iconV[:,iel]])
            vrms+=(uq**2+vq**2)*JxW
            Tavrg+=Tq*JxW
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly)) / vel_ref
    Tavrg/=(Lx*Ly)             / T_ref

    Tavrg_file.write("%10e %10e\n" % (istep,Tavrg))
    Tavrg_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (timing.time() - start))

    t06+=timing.time()-start

    ###########################################################################
    # compute nodal heat flux 
    ###########################################################################
    start = timing.time()
    
    count = np.zeros(NV,dtype=np.int32)  
    qx_n = np.zeros(NV,dtype=np.float64)  
    qy_n = np.zeros(NV,dtype=np.float64)  
    dpdx_n = np.zeros(NV,dtype=np.float64)  
    dpdy_n = np.zeros(NV,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,mV):
            inode=iconV[i,iel]
            qx_n[inode]+=-np.dot(hcond*dNNNVdx_n[i,:],T[iconV[:,iel]])
            qy_n[inode]+=-np.dot(hcond*dNNNVdy_n[i,:],T[iconV[:,iel]])
            dpdx_n[inode]+=np.dot(dNNNVdx_n[i,:],q[iconV[:,iel]])
            dpdy_n[inode]+=np.dot(dNNNVdy_n[i,:],q[iconV[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count
    dpdx_n/=count
    dpdy_n/=count

    print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))

    print("compute nodal heat flux: %.3f s" % (timing.time() - start))

    t07+=timing.time()-start

    ###########################################################################
    # compute Nusselt number at top
    ###########################################################################
    start = timing.time()

    qy_top=0
    qy_bot=0
    Nusselt=0
    for iel in range(0,nel):
        if yV[iconV[mV-1,iel]]>1-eps: 
           sq=+1
           for iq in range(0,nqperdim):
               rq=qcoords[iq]
               NNNNV=NNV(rq,sq)
               q_y=np.dot(NNNNV[:],qy_n[iconV[:,iel]])
               Nusselt+=q_y*(hx/2)*qweights[iq]
               qy_top+=q_y*(hx/2)*qweights[iq]
           #end for
        #end if
        if yV[iconV[0,iel]]<eps: 
           sq=-1
           for iq in range(0,nqperdim):
               rq=qcoords[iq]
               NNNNV=NNV(rq,sq)
               q_y=np.dot(NNNNV[:],qy_n[iconV[:,iel]])
               qy_bot+=q_y*(hx/2)*qweights[iq]
        #end if
    #end for

    Nusselt=np.abs(Nusselt)/Lx

    Nu_vrms_file.write("%10e %.10f %.10f %.10f %.10f \n" % (istep,Nusselt,vrms,qy_bot,qy_top))
    Nu_vrms_file.flush()

    print("     istep= %d ; Nusselt= %e ; Ra= %e " %(istep,Nusselt,Ra_nb))

    print("compute Nu: %.3f s" % (timing.time() - start))

    t08+=timing.time()-start

    ###########################################################################
    # compute temperature profile
    ###########################################################################
    start = timing.time()

    T_profile = np.zeros(nny,dtype=np.float64)  
    y_profile = np.zeros(nny,dtype=np.float64)  

    counter=0    
    for j in range(0,nny):
        for i in range(0,nnx):
            T_profile[j]+=T[counter]/nnx
            y_profile[j]=yV[counter]
            counter+=1
        #end for
    #end for

    np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='#y,T')

    print("compute T profile: %.3f s" % (timing.time() - start))

    t09+=timing.time()-start

    ###########################################################################
    # assess convergence of iterations
    ###########################################################################

    if istep==0: Nusselt_prev=1

    T_diff=np.sum(abs(T-T_prev))/NV
    Nu_diff=np.abs(Nusselt-Nusselt_prev)/Nusselt

    print("T conv, T_diff, Nu_diff: " , T_diff<tol_ss,T_diff,Nu_diff)

    conv_file.write("%10e %10e %10e %10e\n" % (istep,T_diff,Nu_diff,tol_ss))
    conv_file.flush()

    converged=(T_diff<tol_ss and Nu_diff<tol_ss)

    if converged:
       print("***convergence reached***")

    u_prev=u.copy()
    v_prev=v.copy()
    T_prev=T.copy()
    Nusselt_prev=Nusselt

    ###########################################################################
    # compute nodal strainrate
    ###########################################################################
    start = timing.time()
    
    if converged: 

       exx_n = np.zeros(NV,dtype=np.float64)  
       eyy_n = np.zeros(NV,dtype=np.float64)  
       exy_n = np.zeros(NV,dtype=np.float64)  
       count = np.zeros(NV,dtype=np.int32)  

       for iel in range(0,nel):
           for i in range(0,mV):
               inode=iconV[i,iel]
               exx_n[inode]+=np.dot(dNNNVdx_n[i,:],u[iconV[:,iel]])
               eyy_n[inode]+=np.dot(dNNNVdy_n[i,:],v[iconV[:,iel]])
               exy_n[inode]+=0.5*np.dot(dNNNVdx_n[i,:],v[iconV[:,iel]])+\
                             0.5*np.dot(dNNNVdy_n[i,:],u[iconV[:,iel]])
               count[inode]+=1
           #end for
       #end for
    
       exx_n/=count
       eyy_n/=count
       exy_n/=count

       print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
       print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
       print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))

       #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T)

       print("compute nodal sr: %.3f s" % (timing.time() - start))

       t11+=timing.time()-start

    ###########################################################################
    # plot of solution
    ###########################################################################
    start = timing.time()

    if converged: 
       #filename = 'solution_{:04d}.vtu'.format(istep)
       filename = 'solution.vtu'
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
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
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='press' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='shear heating (2*eta*e)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.15f \n" % (2*eta(T[i],xV[i],yV[i],eta0)*np.sqrt(exx_n[i]**2+eyy_n[i]**2+exy_n[i]**2)))
       vtufile.write("</DataArray>\n")
       #--
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (linearised)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*rho0*v[i]*gy))
       #vtufile.write("</DataArray>\n")
       #
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (true)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i]))) 
       #vtufile.write("</DataArray>\n")
       #
       #vtufile.write("<DataArray type='Float32' Name='adiab heating (diff)' Format='ascii'> \n")
       #for i in range(0,NV):
       #    vtufile.write("%.15f \n" % (alphaT*T[i]*(u[i]*dpdx_n[i]+v[i]*dpdy_n[i])-\
       #                                alphaT*T[i]*rho0*v[i]*gy))
       #vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='heat flux' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f %10f %10f \n" %(qx_n[i],qy_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='pressure gradient' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f %10f %10f \n" %(dpdx_n[i],dpdy_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel2):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu file: %.3f s" % (timing.time() - start))

       t10+=timing.time()-start

    ###########################################################################

    if converged:
       break

#end for istep

print("     script ; Nusselt= %e ; Ra= %e " %(Nusselt,Ra_nb))

###############################################################################
# close files
###############################################################################
       
vstats_file.close()
pstats_file.close()
conv_file.close()
Tavrg_file.close()
Nu_vrms_file.close()

###############################################################################

duration=timing.time()-topstart

print("total compute time: %.3f s" % (duration))

print("-----------------------------------------------")
print("build FE matrix V: %.3f s       | %3d percent" % (t01,int(t01/duration*100))) 
print("solve system V: %.3f s          | %3d percent" % (t02,int(t02/duration*100))) 
print("compute nodal p: %.3f s         | %3d percent" % (t03,int(t03/duration*100))) 
print("build matrix T: %.3f s          | %3d percent" % (t04,int(t04/duration*100))) 
print("solve system T: %.3f s          | %3d percent" % (t05,int(t05/duration*100))) 
print("compute vrms: %.3f s            | %3d percent" % (t06,int(t06/duration*100))) 
print("compute nodal heat flux: %.3f s | %3d percent" % (t07,int(t07/duration*100))) 
print("compute Nusself nb: %.3f s      | %3d percent" % (t08,int(t08/duration*100))) 
print("compute T profile: %.3f s       | %3d percent" % (t09,int(t09/duration*100))) 
print("export to vtu: %.3f s           | %3d percent" % (t10,int(t10/duration*100))) 
print("compute nodal sr: %.3f s        | %3d percent" % (t11,int(t11/duration*100))) 
print("normalise pressure: %.3f s      | %3d percent" % (t12,int(t12/duration*100))) 
print("-----------------------------------------------")
    
###############################################################################
