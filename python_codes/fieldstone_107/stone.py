import numpy as np
import sys as sys
import scipy
import math as math
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import lil_matrix
import time as timing
import random
from scipy import special
import matplotlib.pyplot as plt
import random

#------------------------------------------------------------------------------
# constants

eps=1e-9
year=365.25*3600*24

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2   # number of dimensions
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom per node
ndofT=1  # number of temperature degrees of freedom per node
order= 2

########################################################################
# input parameters to play with
########################################################################

Lx      = 4000.
Ly      = 1000.
nelx    = 40
nely    = 10
nstep   = 200
g0      = 9.81
Kxx     = 1e-12
Kyy     = 1e-12
Ttop    = 100
Tbottom = 110
rho_f   = 997       # https://en.wikipedia.org/wiki/Density#Water
eta_f   = 1e-3      # https://en.wikipedia.org/wiki/Viscosity#Water
hcond_f = 0.6     # https://en.wikipedia.org/wiki/List_of_thermal_conductivities
hcapa_f = 4184    # https://en.wikipedia.org/wiki/Specific_heat_capacity 
alpha_f = 2.1e-4  # https://en.wikipedia.org/wiki/Thermal_expansion

plot_nliter = 1

########################################################################

Ra= Kxx*rho_f*g0*alpha_f*(Tbottom-Ttop)*Ly/ (hcond_f/rho_f/hcapa_f)/eta_f

CFL_nb=0.05
apply_filter=False
supg_type=0

every=1
T0=Ttop

use_relax=True
relax=0.25
tol=1e-4

tfinal=1000*year

nel=nelx*nely
nnx=order*nelx+1  # number of elements, x direction
nny=order*nely+1  # number of elements, y direction
NV=nnx*nny

gx=0
gy=-g0

NP=(nelx+1)*(nely+1)
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
rVnodes=[-1,0,+1,-1,0,+1,-1,0,+1]
sVnodes=[-1,-1,-1,0,0,0,+1,+1,+1]
rPnodes=[-1,+1,-1,+1]
sPnodes=[-1,-1,+1,+1]

NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total nb of dofs
NfemT=NV*ndofT       # nb of temperature dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

sparse=True # storage of FEM matrix 

#################################################################

sqrt2=np.sqrt(2)

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

#################################################################
# open output files

vrms_file=open('vrms.ascii',"w")
dt_file=open('dt.ascii',"w")
Tavrg_file=open('Tavrg.ascii',"w")
conv_file=open('conv.ascii',"w")
Tstats_file=open('stats_T.ascii',"w")
ustats_file=open('stats_u.ascii',"w")
vstats_file=open('stats_v.ascii',"w")

#################################################################

print ('Ra       =',Ra)
print ('nnx      =',nnx)
print ('nny      =',nny)
print ('NV       =',NV)
print ('NP       =',NP)
print ('nel      =',nel)
print ('NfemV    =',NfemV)
print ('NfemP    =',NfemP)
print ('Nfem     =',Nfem)
print ('nqperdim =',nqperdim)
print ('relax    =',relax)
print("-----------------------------")

#------------------------------------------------------------------------------
# Q2 velocity shape functions
#------------------------------------------------------------------------------
# 6---7---8  
# |       |  
# 3   4   5  
# |       |  
# 0---1---2  

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
    return N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8

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
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,dNdr_5,dNdr_6,dNdr_7,dNdr_8

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
    return dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,dNds_5,dNds_6,dNds_7,dNds_8

#------------------------------------------------------------------------------
# Q1 pressure shape functions 
#------------------------------------------------------------------------------

def NNP(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1-r)*(1+s)
    N_3=0.25*(1+r)*(1+s)
    return N_0,N_1,N_2,N_3

def dNNPdr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=-0.25*(1.+s) 
    dNdr_3=+0.25*(1.+s) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNPds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.-r)
    dNds_3=+0.25*(1.+r)
    return dNds_0,dNds_1,dNds_2,dNds_3

#################################################################
# build velocity nodes coordinates 
#################################################################
start = timing.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/order
        yV[counter]=j*hy/order
        counter+=1
    #end for
#end for

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')

print("build V grid: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order+1):
            for l in range(0,order+1):
                iconV[counter2,counter]=i*order+l+j*order*nnx+nnx*k
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

#################################################################
# build pressure grid 
#################################################################
start = timing.time()

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

counter=0    
for j in range(0,(order-1)*nely+1):
    for i in range(0,(order-1)*nelx+1):
        xP[counter]=i*hx/(order-1)
        yP[counter]=j*hy/(order-1)
        counter+=1
    #end for
#end for

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (timing.time() - start))

#################################################################
# build pressure connectivity array 
#################################################################
start = timing.time()

iconP=np.zeros((mP,nel),dtype=np.int32)

om1=order-1
counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,order):
            for l in range(0,order):
                iconP[counter2,counter]=i*om1+l+j*om1*(om1*nelx+1)+(om1*nelx+1)*k 
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

print("build iconP: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV] = True ; bc_val[i*ndofV]   = 0.
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("velocity b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]/Ly<eps:
       bc_fixT[i]=True ; bc_valT[i]=Tbottom
    if yV[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=Ttop
#end for

print("temperature b.c.: %.3f s" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)
T_prev = np.zeros(NV,dtype=np.float64)
T_init = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T[i]= Tbottom-yV[i]/Ly*(Tbottom-Ttop) # conductive profile
    T[i] += random.uniform(-2.,+2)
#end for

T_prev[:]=T[:]
T_init[:]=T[:]

print("     -> Tinit (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

#np.savetxt('temperature_init.ascii',np.array([xV,yV,T]).T,header='# x,y,T')

print("temperature init: %.3f s" % (timing.time() - start))

#################################################################
# compute area of elements
#################################################################
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
            area[iel]+=jcob*weightq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (timing.time() - start))

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================
u       = np.zeros(NV,dtype=np.float64)           # x-component velocity
v       = np.zeros(NV,dtype=np.float64)           # y-component velocity
u_prev  = np.zeros(NV,dtype=np.float64)           # x-component velocity
v_prev  = np.zeros(NV,dtype=np.float64)           # y-component velocity
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNPdx = np.zeros(mP,dtype=np.float64)           # shape functions derivatives
dNNNPdy = np.zeros(mP,dtype=np.float64)           # shape functions derivatives
dNNNPdr = np.zeros(mP,dtype=np.float64)           # shape functions derivatives
dNNNPds = np.zeros(mP,dtype=np.float64)           # shape functions derivatives

time=0

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    #A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    A_sparse =  np.zeros((Nfem,Nfem),dtype=np.float64)
    f_rhs   = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs   = np.zeros(NfemP,dtype=np.float64)        # right hand side h 

    for iel in range(0,nel):

        # set arrays to 0 every loop
        fx_el =np.zeros((mV),dtype=np.float64)
        fy_el =np.zeros((mV),dtype=np.float64)
        Nxx_el =np.zeros((mV,mV),dtype=np.float64)
        Nyy_el =np.zeros((mV,mV),dtype=np.float64)
        Nxy_el =np.zeros((mV,mV),dtype=np.float64)
        Nyx_el =np.zeros((mV,mV),dtype=np.float64)
        Gx_el=np.zeros((mV,mP),dtype=np.float64)
        Gy_el=np.zeros((mV,mP),dtype=np.float64)
        Hx_el=np.zeros((mP,mV),dtype=np.float64)
        Hy_el=np.zeros((mP,mV),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:mV]=NNV(rq,sq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq)
                dNNNVds[0:mV]=dNNVds(rq,sq)

                NNNP[0:mP]=NNP(rq,sq)
                dNNNPdr[0:mP]=dNNPdr(rq,sq)
                dNNNPds[0:mP]=dNNPds(rq,sq)

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
                xq=0.
                yq=0.
                Tq=0.
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for

                for k in range(0,mP):
                    dNNNPdx[k]=jcbi[0,0]*dNNNPdr[k]+jcbi[0,1]*dNNNPds[k]
                    dNNNPdy[k]=jcbi[1,0]*dNNNPdr[k]+jcbi[1,1]*dNNNPds[k]
                #end for

                Nxx_el-=eta_f/Kxx*np.outer(NNNV,NNNV)*weightq*jcob
                Nyy_el-=eta_f/Kyy*np.outer(NNNV,NNNV)*weightq*jcob

                #Gx_el-=np.outer(NNNV,dNNNPdx)*weightq*jcob
                #Gy_el-=np.outer(NNNV,dNNNPdy)*weightq*jcob
                Gx_el+=np.outer(dNNNVdx,NNNP)*weightq*jcob
                Gy_el+=np.outer(dNNNVdy,NNNP)*weightq*jcob

                Hx_el-=np.outer(NNNP,dNNNVdx)*weightq*jcob
                Hy_el-=np.outer(NNNP,dNNNVdy)*weightq*jcob 

                # compute elemental rhs vector
                for i in range(0,mV):
                    fx_el[i]-=NNNV[i]*jcob*weightq*(rho_f*(1-alpha_f*(Tq-T0)))*gx
                    fy_el[i]-=NNNV[i]*jcob*weightq*(rho_f*(1-alpha_f*(Tq-T0)))*gy
                #end for

            # end for jq
        # end for iq

        #print(np.min(Nxx_el),np.max(Nxx_el))
        #print(np.min(Gx_el),np.max(Gx_el))
        #print(np.min(Gy_el),np.max(Gy_el))
        #print(np.min(Hx_el),np.max(Hx_el))
        #print(np.min(Hy_el),np.max(Hy_el))

        # impose b.c.: only works for free slip !!! 
        for k1 in range(0,mV):
            #horizontal component
            m1 =ndofV*iconV[k1,iel]+0
            if bc_fix[m1]:
               N_ref=Nxx_el[k1,k1] 
               for k2 in range(0,mV):
                   fx_el[k2]-=Nxx_el[k2,k1]*bc_val[m1]
                   fy_el[k2]-=Nyx_el[k2,k1]*bc_val[m1]
                   Nxx_el[k1,k2]=0
                   Nxx_el[k2,k1]=0
               #end for
               Nxx_el[k1,k1]=N_ref 
               fx_el[k1]=0
               Gx_el[k1,:]=0
               Nxy_el[k1,:]=0
               Nyx_el[:,k1]=0
               Hx_el[:,k1]=0
            #vertical component
            m1 =ndofV*iconV[k1,iel]+1
            if bc_fix[m1]:
               N_ref=Nyy_el[k1,k1] 
               for k2 in range(0,mV):
                   fx_el[k2]-=Nxy_el[k2,k1]*bc_val[m1]
                   fy_el[k2]-=Nyy_el[k2,k1]*bc_val[m1]
                   Nyy_el[k1,k2]=0
                   Nyy_el[k2,k1]=0
               #end for
               Nyy_el[k1,k1]=N_ref 
               fy_el[k1]=0
               Gy_el[k1,:]=0
               Nyx_el[k1,:]=0
               Nxy_el[:,k1]=0
               Hy_el[:,k1]=0

        #assemble ( no h vector ! )
        for k1 in range(0,mV):
            m1 =iconV[k1,iel]
            for k2 in range(0,mV):
                m2 =iconV[k2,iel]
                A_sparse[m1   ,m2   ] += Nxx_el[k1,k2]
                A_sparse[m1   ,m2+NV] += Nxy_el[k1,k2]
                A_sparse[m1+NV,m2   ] += Nyx_el[k1,k2]
                A_sparse[m1+NV,m2+NV] += Nyy_el[k1,k2]
            f_rhs[m1   ]+=fx_el[k1]
            f_rhs[m1+NV]+=fy_el[k1]
            for k2 in range(0,mP):
                m2 =iconP[k2,iel]
                A_sparse[m1   ,m2+NfemV] += Gx_el[k1,k2]
                A_sparse[m1+NV,m2+NfemV] += Gy_el[k1,k2]
                A_sparse[m2+NfemV,m1   ] += Hx_el[k2,k1]
                A_sparse[m2+NfemV,m1+NV] += Hy_el[k2,k1]

    #end for iel

    print("build FE matrix: %.3fs" % (timing.time()-start))

    #plt.spy(A_sparse,markersize=.5)
    #plt.savefig('matrix.pdf', bbox_inches='tight')

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble rhs %.3f s" % (timing.time() - start))

    ######################################################################
    # assign extra pressure b.c. to remove null space
    ######################################################################

    A_sparse[Nfem-1,:]=0
    A_sparse[:,Nfem-1]=0
    A_sparse[Nfem-1,Nfem-1]=1
    rhs[Nfem-1]=0

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u=sol[0:NV]
    v=sol[NV:NfemV]
    p=sol[NfemV:Nfem]

    print("     -> u (m,M) %.6e %.6e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.6e %.6e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.6e %.6e " %(np.min(p),np.max(p)))

    ustats_file.write("%6e %6e %6e\n" % (time,np.min(u),np.max(u)))
    vstats_file.write("%6e %6e %6e\n" % (time,np.min(v),np.max(v)))
    ustats_file.flush()
    vstats_file.flush()

    np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    #################################################################
    # relaxation step
    #################################################################

    if use_relax and istep>0:
        u=relax*u+(1-relax)*u_prev
        v=relax*v+(1-relax)*v_prev
    
    #################################################################
    # compute timestep value
    #################################################################
    start = timing.time()

    if use_relax:
       dt1=0
       dt2=0
       dt=1
       time+=dt
    else:
       dt1=CFL_nb*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))
       dt2=CFL_nb*(Lx/nelx)**2 /(hcond_f/hcapa_f/rho_f)
       dt=np.min([dt1,dt2])
       print('     -> dt1 = %.6f (year)' %dt1/year)
       print('     -> dt2 = %.6f (year)' %dt2/year)
       print('     -> dt  = %.6f (year)' %dt/year)
       time+=dt
       print('     -> time= %.6f; tfinal= %.6f (year)' %(time/year,tfinal/year))

    dt_file.write("%10e %10e %10e %10e\n" % (time,dt1,dt2,dt))
    dt_file.flush()

    print("compute time step: %.3f s" % (timing.time() - start))

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions
    N_mat_supg = np.zeros((mV,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*nqperdim**ndim,dtype=np.float64)
    Tvect   = np.zeros(mV,dtype=np.float64)

    counterq=0   
    for iel in range (0,nel):

        b_el=np.zeros(mV*ndofT,dtype=np.float64)
        a_el=np.zeros((mV*ndofT,mV*ndofT),dtype=np.float64)
        Ka=np.zeros((mV,mV),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mV,mV),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mV,mV),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,mV):
            Tvect[k]=T[iconV[k,iel]]
        #end for

        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                N_mat[0:mV,0]=NNV(rq,sq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq)
                dNNNVds[0:mV]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,mV):
                    vel[0,0]+=N_mat[k,0]*u[iconV[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    B_mat[0,k]=dNNNVdx[k]
                    B_mat[1,k]=dNNNVdy[k]
                #end for

                if supg_type==0:
                   tau_supg[counterq]=0.
                elif supg_type==1:
                      tau_supg[counterq]=(hx*sqrt2)/2/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)
                elif supg_type==2:
                      tau_supg[counterq]=(hx*sqrt2)/order/np.sqrt(vel[0,0]**2+vel[0,1]**2)/sqrt15
                else:
                   exit("supg_type: wrong value")
    
                N_mat_supg=N_mat+tau_supg[counterq]*np.transpose(vel.dot(B_mat))

                # compute mass matrix
                MM+=N_mat_supg.dot(N_mat.T)*weightq*jcob*rho_f*hcapa_f

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*weightq*jcob*hcond_f

                # compute advection matrix
                Ka+=N_mat_supg.dot(vel.dot(B_mat))*weightq*jcob*rho_f*hcapa_f

                counterq+=1

            #end for
        #end for

        if use_relax:
           a_el=Ka+Kd
        else:
           a_el=MM+0.5*(Ka+Kd)*dt
           b_el=(MM-0.5*(Ka+Kd)*dt).dot(Tvect)

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

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mV):
            m1=iconV[k1,iel]
            for k2 in range(0,mV):
                m2=iconV[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    #end for iel

    #print("     -> tau_supg (m,M) %e %e " %(np.min(tau_supg),np.max(tau_supg)))

    print("build FE matrix : %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    Traw = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))

    Tstats_file.write("%6e %6e %6e\n" % (time,np.min(Traw),np.max(Traw)))
    Tstats_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #################################################################
    # apply Lenardic & Kaula filter (1993)
    #################################################################
    start = timing.time()

    if apply_filter:
       # step 1: compute the initial sum 'sum0' of all values of T
       sum0=np.sum(Traw)
       # step 2: find the minimum value Tmin of T
       minT=np.min(Traw)
       # step 3: find the maximum value Tmax of T
       maxT=np.max(Traw)
       # step 4: set T=0 if T<=|Tmin|
       for i in range(0,NV):
           if Traw[i]<=abs(minT):
              Traw[i]=0
       # step 5: set T=1 if T>=2-Tmax
       for i in range(0,NV):
           if Traw[i]>=2-maxT:
              Traw[i]=1
       # step 6: compute the sum sum1 of all values of T
       sum1=np.sum(Traw)
       # step 7: compute the number num of 0<T<1
       num=0
       for i in range(0,NV):
           if Traw[i]>0 and Traw[i]<1:
              num+=1
       # step 8: add (sum1-sum0)/num to all 0<T<1
       for i in range(0,NV):
           if Traw[i]>0 and Traw[i]<1:
              Traw[i]+=(sum1-sum0)/num 
       print("     T (m,M) %.4f %.4f " %(np.min(Traw),np.max(Traw)))
    #end if

    T[:]=Traw[:]

    print("apply L&K filter: %.3f s" % (timing.time() - start))

    #################################################################
    # relaxation step
    #################################################################

    if use_relax and istep>0:
        T=relax*T+(1-relax)*T_prev

    #################################################################
    # compute vrms 
    #################################################################
    start = timing.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV[0:mV]=NNV(rq,sq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq)
                dNNNVds[0:mV]=dNNVds(rq,sq)
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                uq=0.
                vq=0.
                Tq=0.
                for k in range(0,mV):
                    uq+=NNNV[k]*u[iconV[k,iel]]
                    vq+=NNNV[k]*v[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                #end for
                vrms+=(uq**2+vq**2)*weightq*jcob
                Tavrg+=Tq*weightq*jcob
            #end for jq
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly))
    Tavrg/=(Lx*Ly)

    Tavrg_file.write("%10e %10e\n" % (time,Tavrg))
    Tavrg_file.flush()

    vrms_file.write("%10e %.10e\n" % (time,vrms))
    vrms_file.flush()

    print("     istep= %.6d ; vrms   = %.5e (m/year)" %(istep,vrms*year))

    print("compute vrms: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = timing.time()
    
    exx_n = np.zeros(NV,dtype=np.float64)  
    eyy_n = np.zeros(NV,dtype=np.float64)  
    exy_n = np.zeros(NV,dtype=np.float64)  
    dTdx_n= np.zeros(NV,dtype=np.float64)  
    dTdy_n= np.zeros(NV,dtype=np.float64)  
    sr_n  = np.zeros(NV,dtype=np.float64)  
    count = np.zeros(NV,dtype=np.int32)  
    q=np.zeros(NV,dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
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
            e_xx=0.
            e_yy=0.
            e_xy=0.
            dTdx=0.
            dTdy=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
                dTdx += dNNNVdx[k]*T[iconV[k,iel]]
                dTdy += dNNNVdy[k]*T[iconV[k,iel]]
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            dTdx_n[inode]+=dTdx
            dTdy_n[inode]+=dTdy

            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    
    dTdx_n/=count
    dTdy_n/=count
    exx_n/=count
    eyy_n/=count
    exy_n/=count
    q/=count

    sr_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

    print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
    print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
    print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
    print("     -> sr_n (m,M) %.6e %.6e " %(np.min(sr_n),np.max(sr_n)))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute Nusselt number
    #####################################################################

    #TODO

    #####################################################################
    # matplotlib output 
    #####################################################################

    if plot_nliter==1 and istep%every==0:
        plot_x = np.flipud(np.reshape(xV,(nny,nnx)))
        plot_y = np.reshape(yV,(nny,nnx))
     
        plot_var = np.flipud(np.reshape(T,(nny,nnx)))
        plt.figure(1)
        plt.set_cmap('magma')
        bla = plt.pcolormesh(plot_x,-plot_y,plot_var,shading='gouraud')
        plt.colorbar(bla,label='Temperature (C°)')
        plt.xlabel('Distance along profile (m)')
        plt.ylabel('Depth (m)')
        plt.gca().set_aspect('equal')
        filename = 'temp_{:04d}.png'.format(istep)
        plt.savefig(filename,dpi=300)
        plt.close(1)
 
        plt.figure(2)
        maxvar=np.max(np.abs(v))
        plot_var = np.flipud(np.reshape(v,(nny,nnx)))
        plt.set_cmap('RdBu_r')
        bla = plt.pcolormesh(plot_x,-plot_y,plot_var, vmin=-maxvar, vmax=maxvar,shading='gouraud')
        plt.colorbar(bla,label='v (m/s)')
        plt.xlabel('Distance along profile (m)')
        plt.ylabel('Depth (m)')
        plt.gca().set_aspect('equal')
        filename = 'v_{:04d}.png'.format(istep)
        plt.savefig(filename,dpi=300)
        plt.close(2)
     
        plt.figure(3)
        maxvar=np.max(np.abs(u))
        plot_var = np.flipud(np.reshape(u,(nny,nnx)))
        plt.set_cmap('RdBu_r')
        bla = plt.pcolormesh(plot_x,-plot_y,plot_var, vmin=-maxvar, vmax=maxvar,shading='gouraud')
        plt.colorbar(bla,label='v (m/s)')
        plt.xlabel('Distance along profile (m)')
        plt.ylabel('Depth (m)')
        plt.gca().set_aspect('equal')
        filename = 'u_{:04d}.png'.format(istep)
        plt.savefig(filename,dpi=300)
        plt.close(3)

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    if istep%every==0:

       filename = 'solution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
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
       ####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %dTdx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %dTdy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (init)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T_init[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='press' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(iconV[0,iel],iconV[2,iel],iconV[8,iel],iconV[6,iel]))
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


    if False:
       filename = 's__olution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
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
       ####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[2,iel],iconV[8,iel],iconV[6,iel],\
                                                       iconV[1,iel],iconV[5,iel],iconV[7,iel],iconV[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")


    if False:
       filename = 's_olution_{:04d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='press' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %T[i])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %sr_n[i])
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

    ######################################################################
    # checking for convergence
    ######################################################################

    if use_relax:

       xi_u=np.linalg.norm(u-u_prev,2)/np.linalg.norm(u+u_prev,2)
       xi_v=np.linalg.norm(v-v_prev,2)/np.linalg.norm(v+v_prev,2)
       xi_T=np.linalg.norm(T-T_prev,2)/np.linalg.norm(T+T_prev,2)

       print("conv: u,v,T: %.6f %.6f %.6f | tol= %.6f" %(xi_u,xi_v,xi_T,tol))

       conv_file.write("%3d %10e %10e %10e %10e \n" %(istep,xi_u,xi_v,xi_T,tol)) 
       conv_file.flush()

       if xi_u<tol and xi_v<tol and xi_T<tol:
          print('*****convergence reached*****')
          break
       else:
          u_prev[:]=u[:]
          v_prev[:]=v[:]
          T_prev[:]=T[:]

    ######################################################################

    if time>tfinal:
       print("*****tfinal reached*****")
       break

#end for istep

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
