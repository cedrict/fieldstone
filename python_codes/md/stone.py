import numpy as np
import sys as sys
from scipy import sparse
import scipy.sparse as sps
from scipy.sparse import lil_matrix
import time as clock
import numba

###############################################################################
# density & viscosity functions
###############################################################################

@numba.njit
def rho(rho0,alphaT,T,T0):
    val=rho0*(1.-alphaT*(T-T0)) 
    return val

@numba.njit
def eta(T,x,y,eta0):
    return eta0 #*np.exp(T)

###############################################################################
# velocity basis functions
###############################################################################

@numba.njit
def basis_functions_V(r,s):
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
def basis_functions_V_dr(r,s):
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
def basis_functions_V_ds(r,s):
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
# pressure basis functions 
###############################################################################

@numba.njit
def basis_functions_P(r,s):
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
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.
Ly=1.

if int(len(sys.argv) == 4):
   nelx  = int(sys.argv[1])
   Ra_nb = float(sys.argv[2])
   nstep = int(sys.argv[3])
else:
   nelx = 32
   Ra_nb= 1e4
   nstep= 100

tol_ss=1e-7   # tolerance for steady state 

top_bc_noslip=False  
bot_bc_noslip=False

nely=nelx

nel=nelx*nely # total number of elements
nn_V=(2*nelx+1)*(2*nely+1)  # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes

m_V=9 # number of velocity nodes per element
m_P=4 # number of pressure nodes per element
m_T=9 # number of temperature nodes per element

r_V=[-1,0,+1,-1,0,+1,-1,0,+1]
s_V=[-1,-1,-1,0,0,0,+1,+1,+1]
r_P=[-1,+1,-1,+1]
s_P=[-1,-1,+1,+1]

ndof_V_el=m_V*ndof_V

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem_T=nn_V        # number of temperature dofs
Nfem=Nfem_V+Nfem_P # total nb of dofs

hx=Lx/nelx # element size in x direction
hy=Ly/nely # element size in y direction

EBA=False

debug=False

###############################################################################

t01=0 ; t02=0 ; t03=0 ; t04=0 ; t05=0 ; t06=0 ; t14=0
t07=0 ; t08=0 ; t09=0 ; t10=0 ; t11=0 ; t12=0 ; t13=0

###############################################################################
# definition: Ra_nb=alphaT*abs(gy)*Ly**3*rho0**2*hcapa/hcond/eta
# following parameters are somewhat arbitrary

alphaT=2.5e-3 # thermal expansion coefficient
hcond=1.      # thermal conductivity
hcapa=1e-2    # heat capacity
rho0=20       # reference density
T0=0          # reference temperature
relax=0.75    # relaxation coefficient (0,1)
gy=-1         #Ra/alphaT # vertical component of gravity vector

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
print ('NV       =',nn_V)
print ('nn_P     =',nn_P)
print ('nel      =',nel)
print ('Nfem_V   =',Nfem_V)
print ('Nfem_P   =',Nfem_P)
print ('Nfem     =',Nfem)
print ('nqperdim =',nqperdim)
print("-----------------------------")

###############################################################################
# checking that all velocity basis functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mV):
#   print ('node',i,':',NNV(rVnodes[i],sVnodes[i]))

###############################################################################
# checking that all pressure basis functions are 1 on their node 
# and  zero elsewhere
#for i in range(0,mP):
#   print ('node',i,':',NNP(rPnodes[i],sPnodes[i]))

###############################################################################
# build velocity nodes coordinates 
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
        counter+=1
    #end for
#end for

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("build V grid: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

nnx=2*nelx+1 
nny=2*nely+1 

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,3):
            for l in range(0,3):
                icon_V[counter2,counter]=i*2+l+j*2*nnx+nnx*k
                counter2+=1
            #end for
        #end for
        counter += 1
    #end for
#end for

print("build icon_V: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid 
###############################################################################
start = clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)     # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)     # y coordinates

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1
    #end for
 #end for

#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
# build pressure connectivity array 
###############################################################################
start = clock.time()

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        counter2=0
        for k in range(0,2):
            for l in range(0,2):
                icon_P[counter2,counter]=i+l+j*(nelx+1)+(nelx+1)*k 
                counter2+=1
            #end for
        #end for
        counter+=1
    #end for
#end for

print("build icon_P: %.3f s" % (clock.time() - start))

###############################################################################
# define velocity boundary conditions
###############################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if bot_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if top_bc_noslip:
          bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V]   = 0.

print("velocity b.c.: %.3f s" % (clock.time() - start))

###############################################################################
# define temperature boundary conditions
###############################################################################
start = clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool)  
bc_val_T=np.zeros(Nfem_T,dtype=np.float64) 

for i in range(0,nn_V):
    if y_V[i]<eps:
       bc_fix_T[i]=True ; bc_val_T[i]=1.
    if y_V[i]>(Ly-eps):
       bc_fix_T[i]=True ; bc_val_T[i]=0.
#end for

print("temperature b.c.: %.3f s" % (clock.time() - start))

###############################################################################
# initial temperature
###############################################################################

T=np.zeros(nn_V,dtype=np.float64)
#T_mem=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    T[i]=1.-y_V[i]-0.01*np.cos(np.pi*x_V[i]/Lx)*np.sin(np.pi*y_V[i]/Ly)
#end for

T_mem=T.copy()

if debug: np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

###############################################################################
# compute area of elements
###############################################################################
start = clock.time()

area=np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
       #end for
   #end for
#end for

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time() - start))

###############################################################################
# precompute basis functions values at q points
###############################################################################
start = clock.time()

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy
jcob=hx*hy/4

N_V=np.zeros((nqel,m_V),dtype=np.float64) 
N_P=np.zeros((nqel,m_P),dtype=np.float64) 
dNdr_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNds_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdx_V=np.zeros((nqel,m_V),dtype=np.float64) 
dNdy_V=np.zeros((nqel,m_V),dtype=np.float64) 

rq=np.zeros(nqel,dtype=np.float64) 
sq=np.zeros(nqel,dtype=np.float64) 
weightq=np.zeros(nqel,dtype=np.float64) 
   
counterq=0 
for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):

        rq[counterq]=qcoords[iq]
        sq[counterq]=qcoords[jq]
        weightq[counterq]=qweights[iq]*qweights[jq]

        N_V[counterq,0:m_V]=basis_functions_V(rq[counterq],sq[counterq])
        N_P[counterq,0:m_P]=basis_functions_P(rq[counterq],sq[counterq])
        dNdr_V[counterq,0:m_V]=basis_functions_V_dr(rq[counterq],sq[counterq])
        dNds_V[counterq,0:m_V]=basis_functions_V_ds(rq[counterq],sq[counterq])
        dNdx_V[counterq,0:m_V]=jcbi[0,0]*dNdr_V[counterq,0:m_V]
        dNdy_V[counterq,0:m_V]=jcbi[1,1]*dNds_V[counterq,0:m_V]
        counterq+=1

print("compute N & grad(N) at q pts: %.3f s" % (clock.time() - start))

###############################################################################
# precompute basis functions values at V nodes
###############################################################################
start = clock.time()

N_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
N_P_n=np.zeros((m_V,m_P),dtype=np.float64) 
dNdr_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNds_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdx_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
dNdy_V_n=np.zeros((m_V,m_V),dtype=np.float64) 
   
for i in range(0,m_V):
    rq=r_V[i]
    sq=s_V[i]
    N_V_n[i,0:m_V]=basis_functions_V(rq,sq)
    N_P_n[i,0:m_P]=basis_functions_P(rq,sq)
    dNdr_V_n[i,0:m_V]=basis_functions_V_dr(rq,sq)
    dNds_V_n[i,0:m_V]=basis_functions_V_ds(rq,sq)
    dNdx_V_n[i,0:m_V]=jcbi[0,0]*dNdr_V_n[i,0:m_V]
    dNdy_V_n[i,0:m_V]=jcbi[1,1]*dNds_V_n[i,0:m_V]

print("compute N & grad(N) at V nodes: %.3f s" % (clock.time() - start))

###############################################################################
# compute array for assembly
###############################################################################
start = clock.time()

local_to_globalV=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            local_to_globalV[ikk,iel]=m1
                 
print("compute local_to_globalV: %.3f s" % (clock.time() - start))

###############################################################################
# fill I,J arrays
###############################################################################
start = clock.time()

bignb=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )

II_V=np.zeros(bignb,dtype=np.int32)    
JJ_V=np.zeros(bignb,dtype=np.int32)    
VV_V=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndof_V_el):
        m1=local_to_globalV[ikk,iel]
        for jkk in range(ndof_V_el):
            m2=local_to_globalV[jkk,iel]
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
        for jkk in range(0,m_P):
            m2 =icon_P[jkk,iel]+Nfem_V
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
            II_V[counter]=m2
            JJ_V[counter]=m1
            counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (clock.time()-start))

###############################################################################
# fill I,J arrays
###############################################################################
start = clock.time()

bignb=nel*m_T**2 

II_T=np.zeros(bignb,dtype=np.int32)    
JJ_T=np.zeros(bignb,dtype=np.int32)    
VV_T=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(m_T):
        m1=icon_V[ikk,iel]
        for jkk in range(m_T):
            m2=icon_V[jkk,iel]
            II_T[counter]=m1
            JJ_T[counter]=m2
            counter+=1

print("fill II_T,JJ_T arrays: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
###############################################################################
# time stepping loop
###############################################################################
###############################################################################
###############################################################################
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

topstart = clock.time()

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    ###########################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    ###########################################################################
    start = clock.time()

    f_rhs=np.zeros(Nfem_V,dtype=np.float64) # right hand side f 
    h_rhs=np.zeros(Nfem_P,dtype=np.float64) # right hand side h 
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  

    counter=0
    for iel in range(0,nel):

        f_el=np.zeros((ndof_V_el),dtype=np.float64)
        K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
        G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nqel):

            JxW=jcob*weightq[iq]

            xq=np.dot(N_V[iq,:],x_V[icon_V[:,iel]])
            yq=np.dot(N_V[iq,:],y_V[icon_V[:,iel]])
            Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])

            for i in range(0,m_V):
                dNdx=dNdx_V[iq,i] 
                dNdy=dNdy_V[iq,i] 
                B[0,2*i  ]=dNdx
                B[1,2*i+1]=dNdy
                B[2,2*i  ]=dNdy
                B[2,2*i+1]=dNdx

            K_el+=B.T.dot(C.dot(B))*eta(Tq,xq,yq,eta0)*JxW

            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[iq,i]*JxW*rho(rho0,alphaT,Tq,T0)*gy

            N_mat[0,0:m_P]=N_P[iq,0:m_P]
            N_mat[1,0:m_P]=N_P[iq,0:m_P]

            G_el-=B.T.dot(N_mat)*JxW

        # end for iq

        G_el*=eta_ref/Lx

        # impose b.c. 
        for ikk in range(0,ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,ndof_V_el):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(ndof_V_el):
            m1=local_to_globalV[ikk,iel]
            for jkk in range(ndof_V_el):
                VV_V[counter]=K_el[ikk,jkk]
                counter+=1
            for jkk in range(0,m_P):
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
                VV_V[counter]=G_el[ikk,jkk]
                counter+=1
            f_rhs[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            h_rhs[m2]+=h_el[k2]

    print("build FE matrix: %.3fs" % (clock.time()-start))

    t01+=clock.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start = clock.time()

    rhs = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    rhs[0:Nfem_V]=f_rhs
    rhs[Nfem_V:Nfem]=h_rhs

    sparse_matrix=sparse.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (clock.time() - start))

    t02+=clock.time()-start

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/Lx)

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    vstats_file.write("%10e %10e %10e %10e %10e\n" % (istep,np.min(u),np.max(u),\
                                                            np.min(u),np.max(u)))

    if debug:
       np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
       np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (clock.time() - start))

    t14+=clock.time()-start

    ###########################################################################
    # normalise pressure
    ###########################################################################
    start = clock.time()

    pressure_avrg=0
    for iel in range(0,nel):
        for iq in range(0,nqel):
            pressure_avrg+=np.dot(N_P[iq,:],p[icon_P[:,iel]])*jcob*weightq[iq]
        #end for iq
    #end for iel
    p-=pressure_avrg

    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    pstats_file.write("%10e %10e %10e\n" % (istep,np.min(p),np.max(p)))
        
    print("normalise pressure: %.3f s" % (clock.time() - start))

    t12+=clock.time()-start

    ###########################################################################
    # relaxation step
    ###########################################################################

    if istep>0:
       u[:]=relax*u[:]+(1-relax)*u_mem[:]
       v[:]=relax*v[:]+(1-relax)*v_mem[:]

    ###########################################################################
    # project Q1 pressure onto Q2 (vel,T) mesh
    ###########################################################################
    start=clock.time()
    
    count=np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)

    for iel,nodes in enumerate(icon_V.T):
        for k in range(0,m_V):
            q[nodes[k]]+=np.dot(N_P_n[k,:],p[icon_P[:,iel]])
            count[nodes[k]]+=1
        #end for
    #end for
    
    q/=count

    print("     -> q     (m,M) %.6e %.6e " %(np.min(q),np.max(q)))

    if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

    print("compute nodal press: %.3f s" % (clock.time() - start))

    t03+=clock.time()-start

    ###########################################################################
    # build temperature matrix
    ###########################################################################
    start = clock.time()

    Tvect=np.zeros(m_T,dtype=np.float64)   
    rhs=np.zeros(Nfem_T,dtype=np.float64)    # FE rhs 
    B=np.zeros((2,m_T),dtype=np.float64)     # gradient matrix B 
    N_mat=np.zeros((m_T,1),dtype=np.float64)   # shape functions

    counter=0
    for iel in range (0,nel):

        b_el=np.zeros(m_T,dtype=np.float64)
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
        #MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
        velq=np.zeros((1,ndim),dtype=np.float64)

        Tvect[0:m_T]=T[icon_V[0:m_T,iel]]

        for iq in range(0,nqel):

            JxW=jcob*weightq[iq]

            N=N_V[iq,:]

            velq[0,0]=np.dot(N,u[icon_V[:,iel]])
            velq[0,1]=np.dot(N,v[icon_V[:,iel]])

            B[0,:]=dNdx_V[iq,:]
            B[1,:]=dNdy_V[iq,:]

            # compute mass matrix
            #MM+=np.outer(N,N)*rho0*hcapa*weightq*jcob

            # compute diffusion matrix
            Kd+=B.T.dot(B)*hcond*JxW

            # compute advection matrix
            Ka+=np.outer(N,velq.dot(B))*rho0*hcapa*JxW

            if EBA:
               xq=np.dot(N_V[iq,:],x_V[icon_V[:,iel]])
               yq=np.dot(N_V[iq,:],y_V[icon_V[:,iel]])
               Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])
               exxq=np.dot(dNdx_V[iq,:],u[icon_V[:,iel]])
               eyyq=np.dot(dNdy_V[iq,:],v[icon_V[:,iel]])
               exyq=np.dot(dNdy_V[iq,:],u[icon_V[:,iel]])*0.5\
                   +np.dot(dNdx_V[iq,:],v[icon_V[:,iel]])*0.5
               dpdxq=np.dot(dNdx_V[iq,:],q[icon_V[:,iel]])
               dpdyq=np.dot(dNdy_V[iq,:],q[icon_V[:,iel]])
               #viscous dissipation
               b_el[:]+=N[:]*JxW*2*eta(Tq,xq,yq,eta0)*(exxq**2+eyyq**2+2*exyq**2) 
               #adiabatic heating
               b_el[:]+=N[:]*JxW*alphaT*Tq*(velq[0,0]*dpdxq+velq[0,1]*dpdyq)  

        #end for

        A_el=Ka+Kd

        # apply boundary conditions
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_V):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(m_T):
            m1=icon_V[ikk,iel]
            for jkk in range(m_T):
                VV_T[counter]=A_el[ikk,jkk]
                counter+=1
            rhs[m1]+=b_el[ikk]
        #end for

    #end for iel

    print("build FE matrix : %.3f s" % (clock.time() - start))

    t04+=clock.time()-start

    ###########################################################################
    # solve system
    ###########################################################################
    start = clock.time()

    sparse_matrix=sparse.coo_matrix((VV_T,(II_T,JJ_T)),shape=(Nfem_T,Nfem_T)).tocsr()

    T = sps.linalg.spsolve(sparse_matrix,rhs)

    print("     T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("solve T time: %.3f s" % (clock.time() - start))

    t05+=clock.time()-start

    ###########################################################################
    # relax
    ###########################################################################

    if istep>0:
       T=relax*T+(1-relax)*T_mem

    ###########################################################################
    # compute vrms 
    ###########################################################################
    start = clock.time()

    vrms=0.
    Tavrg=0.
    for iel in range (0,nel):
        for iq in range(0,nqel):
            JxW=jcob*weightq[iq]
            uq=np.dot(N_V[iq,:],u[icon_V[:,iel]])
            vq=np.dot(N_V[iq,:],v[icon_V[:,iel]])
            Tq=np.dot(N_V[iq,:],T[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxW
            Tavrg+=Tq*JxW
        #end for iq
    #end for iel

    vrms=np.sqrt(vrms/(Lx*Ly)) / vel_ref
    Tavrg/=(Lx*Ly)             / T_ref

    Tavrg_file.write("%10e %10e\n" % (istep,Tavrg))
    Tavrg_file.flush()

    print("     istep= %.6d ; vrms   = %.6f" %(istep,vrms))

    print("compute vrms: %.3f s" % (clock.time() - start))

    t06+=clock.time()-start

    ###########################################################################
    # compute nodal heat flux 
    ###########################################################################
    start = clock.time()
    
    count=np.zeros(nn_V,dtype=np.int32)  
    qx_n=np.zeros(nn_V,dtype=np.float64)  
    qy_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            qx_n[inode]-=np.dot(hcond*dNdx_V_n[i,:],T[icon_V[:,iel]])
            qy_n[inode]-=np.dot(hcond*dNdy_V_n[i,:],T[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count

    print("     -> qx_n (m,M) %.6e %.6e " %(np.min(qx_n),np.max(qx_n)))
    print("     -> qy_n (m,M) %.6e %.6e " %(np.min(qy_n),np.max(qy_n)))

    print("compute nodal heat flux: %.3f s" % (clock.time() - start))

    t07+=clock.time()-start

    ###########################################################################
    # compute Nusselt number at top
    ###########################################################################
    start = clock.time()

    qy_top=0
    qy_bot=0
    Nusselt=0
    for iel in range(0,nel):
        if y_V[icon_V[m_V-1,iel]]/Ly>1-eps: # top row of nodes 
           sq=+1
           for iq in range(0,nqperdim):
               rq=qcoords[iq]
               N=basis_functions_V(rq,sq)
               q_y=np.dot(N,qy_n[icon_V[:,iel]])
               Nusselt+=q_y*(hx/2)*qweights[iq]
               qy_top+=q_y*(hx/2)*qweights[iq]
           #end for
        #end if
        if y_V[icon_V[0,iel]]/Ly<eps: # bottom row of nodes
           sq=-1
           for iq in range(0,nqperdim):
               rq=qcoords[iq]
               N=basis_functions_V(rq,sq)
               q_y=np.dot(N,qy_n[icon_V[:,iel]])
               qy_bot+=q_y*(hx/2)*qweights[iq]
        #end if
    #end for

    Nusselt=np.abs(Nusselt)/Lx

    Nu_vrms_file.write("%10e %.10f %.10f %.10f %.10f \n" % (istep,Nusselt,vrms,qy_bot,qy_top))
    Nu_vrms_file.flush()

    print("     istep= %d ; Nusselt= %e ; Ra= %e " %(istep,Nusselt,Ra_nb))

    print("compute Nu: %.3f s" % (clock.time() - start))

    t08+=clock.time()-start

    ###########################################################################
    # compute temperature profile
    ###########################################################################
    start = clock.time()

    T_profile=np.zeros(nny,dtype=np.float64)  
    y_profile=np.zeros(nny,dtype=np.float64)  

    counter=0    
    for j in range(0,nny):
        for i in range(0,nnx):
            T_profile[j]+=T[counter]/nnx
            y_profile[j]=y_V[counter]
            counter+=1
        #end for
    #end for

    np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='#y,T')

    print("compute T profile: %.3f s" % (clock.time() - start))

    t09+=clock.time()-start

    ###########################################################################
    # assess convergence of iterations
    ###########################################################################
    start=clock.time()

    if istep==0: Nusselt_mem=1

    T_diff=np.sum(abs(T-T_mem))/nn_V
    Nu_diff=np.abs(Nusselt-Nusselt_mem)/Nusselt

    print("T conv, T_diff, Nu_diff: " , T_diff<tol_ss,T_diff,Nu_diff)

    conv_file.write("%10e %10e %10e %10e\n" % (istep,T_diff,Nu_diff,tol_ss))
    conv_file.flush()

    converged=(T_diff<tol_ss and Nu_diff<tol_ss)

    if converged:
       print("***convergence reached***")

    u_mem=u.copy()
    v_mem=v.copy()
    T_mem=T.copy()
    Nusselt_mem=Nusselt

    print("assess convergence: %.3f s" % (clock.time()-start))

    t13+=clock.time()-start

    ###########################################################################
    # compute nodal strainrate
    ###########################################################################
    start=clock.time()
    
    if converged: 

       exx_n=np.zeros(nn_V,dtype=np.float64)  
       eyy_n=np.zeros(nn_V,dtype=np.float64)  
       exy_n=np.zeros(nn_V,dtype=np.float64)  
       count=np.zeros(nn_V,dtype=np.int32)  

       for iel in range(0,nel):
           for i in range(0,m_V):
               inode=icon_V[i,iel]
               exx_n[inode]+=np.dot(dNdx_V_n[i,:],u[icon_V[:,iel]])
               eyy_n[inode]+=np.dot(dNdy_V_n[i,:],v[icon_V[:,iel]])
               exy_n[inode]+=0.5*np.dot(dNdx_V_n[i,:],v[icon_V[:,iel]])+\
                             0.5*np.dot(dNdy_V_n[i,:],u[icon_V[:,iel]])
               count[inode]+=1
           #end for
       #end for
    
       exx_n/=count
       eyy_n/=count
       exy_n/=count

       print("     -> exx_n (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
       print("     -> eyy_n (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
       print("     -> exy_n (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))

       if debug: np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T)

       print("compute nodal sr: %.3f s" % (clock.time()-start))

       t11+=clock.time()-start

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if converged: 
       #filename = 'solution_{:04d}.vtu'.format(istep)
       filename = 'solution.vtu'
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f %10f %10f \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Pressure' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Temperature' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" %T[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" %exx_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" %eyy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" %exy_n[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Shear heating (2*eta*e)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.15f \n" % (2*eta(T[i],x_V[i],y_V[i],eta0)*np.sqrt(exx_n[i]**2+eyy_n[i]**2+exy_n[i]**2)))
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
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Heat flux' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f %10f %10f \n" %(qx_n[i],qy_n[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[2,iel],icon_V[8,iel],\
                                                          icon_V[6,iel],icon_V[1,iel],icon_V[5,iel],\
                                                          icon_V[7,iel],icon_V[3,iel],icon_V[4,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m_V))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu file: %.3f s" % (clock.time()-start))

       t10+=clock.time()-start

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

duration=clock.time()-topstart

print("total compute time: %.3f s" % (duration))

print("-----------------------------------------------")
print("build FE matrix V: %.3f s       | %.2f percent" % (t01,(t01/duration*100))) 
print("solve system V: %.3f s          | %.2f percent" % (t02,(t02/duration*100))) 
print("build matrix T: %.3f s          | %.2f percent" % (t04,(t04/duration*100))) 
print("solve system T: %.3f s          | %.2f percent" % (t05,(t05/duration*100))) 
print("compute vrms: %.3f s            | %.2f percent" % (t06,(t06/duration*100))) 
print("compute nodal p: %.3f s         | %.2f percent" % (t03,(t03/duration*100))) 
print("compute nodal heat flux: %.3f s | %.2f percent" % (t07,(t07/duration*100))) 
print("compute Nusself nb: %.3f s      | %.2f percent" % (t08,(t08/duration*100))) 
print("compute T profile: %.3f s       | %.2f percent" % (t09,(t09/duration*100))) 
print("export to vtu: %.3f s           | %.2f percent" % (t10,(t10/duration*100))) 
print("compute nodal sr: %.3f s        | %.2f percent" % (t11,(t11/duration*100))) 
print("normalise pressure: %.3f s      | %.2f percent" % (t12,(t12/duration*100))) 
print("assess convergence: %.3f s      | %.2f percent" % (t13,(t13/duration*100))) 
print("split solution: %.3f s          | %.2f percent" % (t14,(t14/duration*100))) 
print("-----------------------------------------------")

print(t01+t02+t03+t04+t05+t06+t07+t08+t09+t10+t11+t12+t13+t14,duration)
    
###############################################################################
