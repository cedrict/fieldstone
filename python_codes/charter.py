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
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

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

print("-------------------------------")
print("---------- stone xyz ----------")
print("-------------------------------")

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
hx=
hy=
hz=
nqel=

#avoid nnx,nny,nnz

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nfem=',Nfem)
print("-----------------------------")

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)
z_V=np.zeros(nn_V,dtype=np.float64)

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

bc_fix=np.zeros(Nfem,dtype=bool)
bc_val=np.zeros(Nfem,dtype=np.float64)

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)

A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

for iq in range(0,nqperdim):
    for jq in range(0,nqperdim):
        for kq in range(0,nqperdim):


N_V=basis_functions_V(rq,sq,tq)
dNdr_V=basis_functions_V_dr(rq,sq,tq)
dNds_V=basis_functions_V_ds(rq,sq,tq)
dNdt_V=basis_functions_V_dt(rq,sq,tq)
N_P=basis_functions_P(rq,sq,tq)

jcb=np.zeros((ndim,ndim),dtype=np.float64) 

rq,sq,tq,weightq

jcb:  # jacobian matrix 
jcob: # determinant of jacobian
jcbi: # inverse of jacobian 

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

xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
zc[iel]=np.dot(N_V,z_V[icon_V[:,iel]])

uq=np.dot(N_V,u[icon_V[:,iel]])
vq=np.dot(N_V,v[icon_V[:,iel]])
wq=np.dot(N_V,w[icon_V[:,iel]])
pq=np.dot(N_P,p[icon_P[:,iel]])

dNdx_V=jcbi[0,0]*dNdr_V[:]+jcbi[0,1]*dNds_V[:]
dNdy_V=jcbi[1,0]*dNdr_V[:]+jcbi[1,1]*dNds_V[:]

dNdx_V=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]+jcbi[0,2]*dNdt[:]
dNdy_V=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]+jcbi[1,2]*dNdt[:]
dNdz_V=jcbi[2,0]*dNdr[:]+jcbi[2,1]*dNds[:]+jcbi[2,2]*dNdt[:]

exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
ezz[iel]=np.dot(dNdz_V[:],w[icon_V[:,iel]])
exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
        +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
exz[iel]=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5\
        +np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
eyz[iel]=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5\
        +np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5

A_fem=csr_matrix(A_fem)

sol=sps.linalg.spsolve(A_fem,b_fem)

sol=spsolve(A_fem,b_fem)

u,v,w=np.reshape(sol,(NV,3)).T

errv # velocity discretization error
errp # pressure discretization error
vrms # root mean square velocity

K=np.zeros((6,6),dtype=np.float64) 
K[0,0]=1. ; K[0,1]=1. ; K[0,2]=1. 
K[1,0]=1. ; K[1,1]=1. ; K[1,2]=1. 
K[2,0]=1. ; K[2,1]=1. ; K[2,2]=1. 

C=np.zeros((6,6),dtype=np.float64) 
C[0,0]=2. ; C[1,1]=2. ; C[2,2]=2.
C[3,3]=1. ; C[4,4]=1. ; C[5,5]=1.

##############################################################################r

