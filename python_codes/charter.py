import numpy as np
import sys as sys
import time as clock
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve

###############################################################################

def bx(x,y):

def by(x,y):

def bz(x,y):

###############################################################################

def basis_functions_V(r,s,t):
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_V_dt(r,s,t):
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

###############################################################################

def u_analytical(x,y,z):
    return

def v_analytical(x,y,z):
    return

def w_analytical(x,y,z):
    return

def p_analytical(x,y,z):
    return

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone xyz **********")
print("*******************************")

print("*******************************")
print("********** the end ************")
print("*******************************")

start=clock.time()

print("...: %.3f s" %(clock.time()-start))

ndof_V=
nel=
nn_V=
nn_P=
nn_T=
m_V=
m_P=
m_T=
Lx=
Ly=
ndof=
hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
nqel=


######NAME??!?! m_V*ndof_V

#avoid nnx,nny,nnz

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('Nfem=',Nfem)
print("-----------------------------")

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
z_V=np.zeros(nn_V,dtype=np.float64)

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)
icon_T=np.zeros((m_T,nel),dtype=np.int32)

bc_fix_V=np.zeros(Nfem_V,dtype=bool)
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)

bc_fix_T=np.zeros(Nfem_T,dtype=bool)
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)

A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
b_el=np.zeros(m_V*ndof_V,dtype=np.float64)


###############################################################################

#2D
N_V=basis_functions_V(rq,sq)
N_P=basis_functions_P(rq,sq)
dNdr_V=basis_functions_V_dr(rq,sq)
dNds_V=basis_functions_V_ds(rq,sq)

#3D
N_V=basis_functions_V(rq,sq,tq)
N_P=basis_functions_P(rq,sq,tq)
dNdr_V=basis_functions_V_dr(rq,sq,tq)
dNds_V=basis_functions_V_ds(rq,sq,tq)
dNdt_V=basis_functions_V_dt(rq,sq,tq)

N_T=basis_functions_T(rq,sq,tq)
dNdr_T=basis_functions_T_dr(rq,sq,tq)
dNds_T=basis_functions_T_ds(rq,sq,tq)
dNdt_T=basis_functions_t_dt(rq,sq,tq)

#or
 
N_V=np.zeros((nqel,m_V),dtype=np.float64)
N_P=np.zeros((nqel,m_P),dtype=np.float64)
dNdr_V=np.zeros((nqel,m_V),dtype=np.float64)
dNds_V=np.zeros((nqel,m_V),dtype=np.float64)
dNdt_V=np.zeros((nqel,m_V),dtype=np.float64)


###############################################################################

nq_per_dim=3
nqel=nq_per_dim**ndim

for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        for kq in range(0,nqperdim):

###############################################################################

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
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



for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                wq=np.dot(N_V,w[icon_V[:,iel]])
                pq=np.dot(N_P,p[icon_P[:,iel]])
                area[iel]+=JxWq
            #end for
        #end for
    #end for
#end for


#--- OR ---

cq=0
for iq in range(0,nq_per_dim):
    for jq in range(0,nq_per_dim):
        rq[cq]=qcoords[iq]
        sq[cq]=qcoords[jq]
        weightq[cq]=qweights[iq]*qweights[jq]
        N_V[cq,:]=basis_functions_V(rq[cq],sq[cq])
        dNdr_V[cq,:]=basis_functions_V_dr(rq[cq],sq[cq])
        dNds_V[cq,:]=basis_functions_V_ds(rq[cq],sq[cq])
        jcb[0,0]=np.dot(dNdr_V[cq,:],x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V[cq,:],y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V[cq,:],x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V[cq,:],y_V[icon_V[:,iel]])
        JxWq=np.linalg.det(jcb)*weightq
        area[iel]+=JxWq
        cq+=1
    #end for
#end for

###############################################################################

dofs_V=np.zeros(ndof_V*m_V,dtype=np.int32) 

for iel in range(0,nel):
    for k in range(0,m_V):
        dofs_V[k*ndof_V  ]=icon_V[k,iel]*ndof_V
        dofs_V[k*ndof_V+1]=icon_V[k,iel]*ndof_V+1

    for i_local,idof in enumerate(dofs):
        for j_local,jdof in enumerate(dofs):




###############################################################################

rq,sq,tq,weightq # either scalars
or
rq=np.zeros(nqel,dtype=np.float64)
sq=np.zeros(nqel,dtype=np.float64)
tq=np.zeros(nqel,dtype=np.float64)
weightq=np.zeros(nqel,dtype=np.float64)


jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])

jcbi=np.linalg.inv(jcb)

JxWq=np.linalg.det(jcb)*weightq # avoid jcob

r_V=np.array([-1, 1, 1,-1,-1, 1, 1 ,-1],np.float64)
s_V=() ...
t_V=() ...

xq=np.dot(N_V,x_V[icon_V[:,iel]])
yq=np.dot(N_V,y_V[icon_V[:,iel]])
zq=np.dot(N_V,z_V[icon_V[:,iel]])
Tq=np.dot(N_T,T[icon_T[:,iel]])
uq=np.dot(N_V,u[icon_V[:,iel]])
vq=np.dot(N_V,v[icon_V[:,iel]])
wq=np.dot(N_V,w[icon_V[:,iel]])
pq=np.dot(N_P,p[icon_P[:,iel]])
qq=np.dot(N_V,q[icon_V[:,iel]])

xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
zc[iel]=np.dot(N_V,z_V[icon_V[:,iel]])

dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    
dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix 
for i in range(0,m_V):
    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                      [0.       ,dNdy_V[i]],
                      [dNdy_V[i],dNdx_V[i]]]


B=np.zeros((6,ndofV*m_V),dtype=np.float64)  # gradient matrix B 
for i in range(0,m_V):
    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                        [0.       ,dNdy_V[i],0.       ],
                        [0.       ,0.       ,dNdz_V[i]],
                        [dNdy_V[i],dNdx_V[i],0.       ],
                        [dNdz_V[i],0.       ,dNdx_V[i]],
                        [0.       ,dNdz_V[i],dNdy_V[i]]]


Lxx=np.dot(dNdx_V,u[icon_V[:,iel]])
Lyy=np.dot(dNdy_V,v[icon_V[:,iel]])
Lxy=np.dot(dNdx_V,v[icon_V[:,iel]])
Lyx=np.dot(dNdy_V,u[icon_V[:,iel]])


exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
        +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    



exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
ezz[iel]=np.dot(dNdz_V[:],w[icon_V[:,iel]])
exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
        +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
exz[iel]=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5\
        +np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
eyz[iel]=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5\
        +np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5
    
e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)

###############################################################################

A_fem=csr_matrix(A_fem)
sol=sps.linalg.spsolve(A_fem,b_fem)

or

sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

sol=spsolve(A_fem,b_fem)

###############################################################################

u,v,w=np.reshape(sol,(NV,3)).T

errv # velocity discretization error
errp # pressure discretization error
vrms # root mean square velocity

H=np.zeros((6,6),dtype=np.float64)   # bad name ?! 
H[0,0]=1. ; H[0,1]=1. ; H[0,2]=1. 
H[1,0]=1. ; H[1,1]=1. ; H[1,2]=1. 
H[2,0]=1. ; H[2,1]=1. ; H[2,2]=1. 

#--- 2d ---
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

#--- 3d ---
C=np.zeros((6,6),dtype=np.float64) 
C[0,0]=2. ; C[1,1]=2. ; C[2,2]=2.
C[3,3]=1. ; C[4,4]=1. ; C[5,5]=1.


##############################################################################r


##################################################
# check list
##################################################

header

footer 

80 ### 

time as clock

rename basis functions

rename matrix and rhs

RESULTS folder

no empty

A_el, b_el

ndof_V
m_V
m_P
m_T
Nfem_V
Nfem_P
x_V, y_V, z_V
x_P, y_P, z_P
icon_V, icon_P
rad,theta
u_mem, v_mem, w_mem, p_mem, T_mem

debug

cleandata

best structure for icon ? [mvXnel or nelxmV) ]

REWRITE B ????
