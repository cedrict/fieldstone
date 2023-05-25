import numpy as np
import triangle as tr
from tools import *
from basis_functions_numba import *
from scipy.sparse import lil_matrix
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from mms_solvi import *

#------------------------------------------------------------------------------
# this function returns true if the point passed as argument
# is inside the inclusion
# note that these rotations are backwards, by -phi!
#------------------------------------------------------------------------------

def is_in_inclusion(case,x,y,a_incl,b_incl,phi_incl):

    if case==0: # circle
       return (x-0.5)**2+(y-0.5)**2<a_incl**2

    if case==1: # rectangle
       return  abs( (x-.5)*np.cos(phi_incl)+(y-.5)*np.sin(phi_incl))<a_incl and\
               abs(-(x-.5)*np.sin(phi_incl)+(y-.5)*np.cos(phi_incl))<b_incl

    if case==2: # ellipse
       return  (( (x-.5)*np.cos(phi_incl)+(y-.5)*np.sin(phi_incl))**2/a_incl**2 + \
                (-(x-.5)*np.sin(phi_incl)+(y-.5)*np.cos(phi_incl))**2/b_incl**2 < 1)

#------------------------------------------------------------------------------

def viscosity(case,x,y,a_incl,b_incl,phi_incl,eta_B):

    is_in=is_in_inclusion(case,x,y,a_incl,b_incl,phi_incl)

    n_exp=3
    sr_ref= 1.5              

    if case==0: #SolVi
       if is_in:
          eta_eff=1000
       else:
          eta_eff=1

    if case==1: #rectangle
       if is_in:
          eta_eff=1000
       else:
          eta_eff=1

    if case==2: #elliptical
       if is_in:
          eta_eff=1000
       else:
          eta_eff=1

    #eta_eff=1/(1/eta_L+1/eta_PL)
    #eta_eff=1

    return eta_eff

#------------------------------------------------------------------------------

ndim=2
km=1e3
eps=1e-6

print("-----------------------------")
print("--------- stone 142 ---------")
print("-----------------------------")

#------------------------------------------------------------------------------
# Crouzeix-Raviart elements not implememented!

CR=True

if CR: 
   mV=7     # number of velocity nodes making up an element
else:
   mV=6

mP=3     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

eta_ref=1e0 # numerical parameter for FEM

Lx=1
Ly=1

tol=1e-6

#------------------------------------------------------------------------------
# case 0: SolVi benchmark (fig. 5)
# case 1: rectangle (fig. 6)
# case 2: elliptical inclusion (fig. 8)

case=2

v_bc=1

a_incl = 0.2
b_incl = 0.1

if case==0:
   phi_incl=0
if case==1:
   phi_incl = 60/180*np.pi
if case==2:
   phi_incl = 30/180*np.pi

#bc_type='simpleshear'
bc_type='pureshear'

eta_B=1
D_B=1

resolution='pqa0.0001'

if case==0:
   bc_type='solvi'

#------------------------------------------------------------------------------
# 6 point integration coeffs and weights 
#------------------------------------------------------------------------------

nqel=6

nb1=0.816847572980459
nb2=0.091576213509771
nb3=0.108103018168070
nb4=0.445948490915965
nb5=0.109951743655322/2.
nb6=0.223381589678011/2.

qcoords_r=np.array([nb1,nb2,nb2,nb4,nb3,nb4],dtype=np.float64)
qcoords_s=np.array([nb2,nb1,nb2,nb3,nb4,nb4],dtype=np.float64)
qweights =np.array([nb5,nb5,nb5,nb6,nb6,nb6],dtype=np.float64)

#------------------------------------------------------------------------------

print('CR=',CR)
print('nqel=',nqel)
print('case=',case)
print('bc_type=',bc_type)
print("-----------------------------")

#------------------------------------------------------------------------------
# make mesh
#------------------------------------------------------------------------------
start = timing.time()

square_vertices = np.array([[0,0],[0,Ly],[Lx,Ly],[Lx,0]])
square_edges = compute_segs(square_vertices)

if case==0:

   nnt=150
   theta = np.linspace(-np.pi, np.pi,nnt, endpoint=False)      
   pts_ib = np.stack([np.cos(theta)*a_incl+.5, np.sin(theta)*a_incl+0.5], axis=1) 
   seg_ib = np.stack([np.arange(nnt), np.arange(nnt) + 1], axis=1) 

elif case==1:

   na=100
   nb=50
   pts_ib = np.zeros((2*na+2*nb,2),dtype=np.float64)  
   ha=2*a_incl/na
   hb=2*b_incl/nb

   counter=0
   for i in range(0,na): #south
       xx=-a_incl+i*ha
       yy=-b_incl
       pts_ib[counter,0]=xx*np.cos(phi_incl)-yy*np.sin(phi_incl)+0.5
       pts_ib[counter,1]=xx*np.sin(phi_incl)+yy*np.cos(phi_incl)+0.5
       counter+=1

   for i in range(0,nb): #east
       xx=a_incl
       yy=-b_incl+i*hb
       pts_ib[counter,0]=xx*np.cos(phi_incl)-yy*np.sin(phi_incl)+0.5
       pts_ib[counter,1]=xx*np.sin(phi_incl)+yy*np.cos(phi_incl)+0.5
       counter+=1

   for i in range(0,na): #north
       xx=a_incl-i*ha
       yy=b_incl
       pts_ib[counter,0]=xx*np.cos(phi_incl)-yy*np.sin(phi_incl)+0.5
       pts_ib[counter,1]=xx*np.sin(phi_incl)+yy*np.cos(phi_incl)+0.5
       counter+=1

   for i in range(0,nb): #west
       xx=-a_incl
       yy=b_incl-i*hb
       pts_ib[counter,0]=xx*np.cos(phi_incl)-yy*np.sin(phi_incl)+0.5
       pts_ib[counter,1]=xx*np.sin(phi_incl)+yy*np.cos(phi_incl)+0.5
       counter+=1

elif case==2:

   nnt=200
   theta = np.linspace(-np.pi, np.pi,nnt, endpoint=False)      
   pts_ib = np.stack([np.cos(theta)*a_incl+.5, np.sin(theta)*b_incl+0.5], axis=1) 

   for i in range(0,nnt):
       xx=pts_ib[i,0]
       yy=pts_ib[i,1]
       pts_ib[i,0]=(xx-.5)*np.cos(phi_incl)+(yy-.5)*np.sin(phi_incl)+0.5
       pts_ib[i,1]=(xx-.5)*np.sin(phi_incl)-(yy-.5)*np.cos(phi_incl)+0.5

   seg_ib = np.stack([np.arange(nnt), np.arange(nnt) + 1], axis=1) 

else:

   exit('case unknown')


points = np.vstack([square_vertices,pts_ib])

SEGS = np.vstack([square_edges])

O1 = {'vertices' : points, 'segments' : SEGS}
T1 = tr.triangulate(O1, resolution)

area=compute_triangles_area(T1['vertices'], T1['triangles'])
iconP1=T1['triangles'] ; iconP1=iconP1.T
xP1=T1['vertices'][:,0]
yP1=T1['vertices'][:,1]
NP1=np.size(xP1)

np.savetxt('meshP1.ascii',np.array([xP1,yP1]).T)

print('     -> number of nodes P1 mesh=',NP1)

export_elements_to_vtu(xP1,yP1,iconP1,'meshP1.vtu',area)

mP,nel=np.shape(iconP1)
print('     -> nel=',nel)

NP2,xP2,yP2,iconP2=mesh_P1_to_P2(xP1,yP1,iconP1)

np.savetxt('meshP2.ascii',np.array([xP2,yP2]).T) 

print('     -> number of nodes P2 mesh=',NP2)

export_elements_to_vtuP2(xP2,yP2,iconP2,'meshP2.vtu')

print("use Delaunay mesher: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# build coordinates and connectivity arrays
#------------------------------------------------------------------------------
start = timing.time()

if CR:
   NV=NP2+nel
   NP=3*nel*ndofP
else:
   NV=NP2
   NP=NP1

xV = np.zeros(NV,dtype=np.float64)  
yV = np.zeros(NV,dtype=np.float64)
xP = np.zeros(NP,dtype=np.float64)
yP = np.zeros(NP,dtype=np.float64)
iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

if CR:

   iconV[0,:]=iconP2[0,:]
   iconV[1,:]=iconP2[1,:]
   iconV[2,:]=iconP2[2,:]
   iconV[3,:]=iconP2[3,:]
   iconV[4,:]=iconP2[4,:]
   iconV[5,:]=iconP2[5,:]
   for iel in range (0,nel):
       iconV[6,iel]=NP2+iel
   #-----
   xV[0:NP2]=xP2[0:NP2]
   yV[0:NP2]=yP2[0:NP2]
   for iel in range (0,nel):
       xV[NP2+iel]=(xV[iconV[0,iel]]+xV[iconV[1,iel]]+xV[iconV[2,iel]])/3.
       yV[NP2+iel]=(yV[iconV[0,iel]]+yV[iconV[1,iel]]+yV[iconV[2,iel]])/3.
   #-----
   counter=0
   for iel in range(0,nel):
       xP[counter]=xP1[iconP1[0,iel]]
       yP[counter]=yP1[iconP1[0,iel]]
       iconP[0,iel]=counter
       counter+=1
       xP[counter]=xP1[iconP1[1,iel]]
       yP[counter]=yP1[iconP1[1,iel]]
       iconP[1,iel]=counter
       counter+=1
       xP[counter]=xP1[iconP1[2,iel]]
       yP[counter]=yP1[iconP1[2,iel]]
       iconP[2,iel]=counter
       counter+=1

else:
   xV[:]=xP2[:]
   yV[:]=yP2[:]
   iconV[:,:]=iconP2[:,:]
   xP[:]=xP1[:]
   yP[:]=yP1[:]
   iconP[:,:]=iconP1[:,:]

NfemV=ndofV*NV
NfemP=ndofP*NP
Nfem=NfemV+NfemP

print("build coords,icon arrays: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# compute element center coordinates
#------------------------------------------------------------------------------
start = timing.time()

xc = np.zeros(nel,dtype=np.float64)
yc = np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    xc[iel]= (xP1[iconP1[0,iel]]+xP1[iconP1[1,iel]]+xP1[iconP1[2,iel]])/3
    yc[iel]= (yP1[iconP1[0,iel]]+yP1[iconP1[1,iel]]+yP1[iconP1[2,iel]])/3

print("     -> xc (m,M) %.6e %.6e " %(np.min(xc),np.max(xc)))
print("     -> yc (m,M) %.6e %.6e " %(np.min(yc),np.max(yc)))

print("compute element center coords: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# flag elements
#------------------------------------------------------------------------------
start = timing.time()

inclusion=np.zeros(nel,dtype=np.bool)

for iel in range(0,nel):
    inclusion[iel]=is_in_inclusion(case,xc[iel],yc[iel],a_incl,b_incl,phi_incl)

print("flag elements in inclusion: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# compute area of elements
#------------------------------------------------------------------------------
start = timing.time()

area=np.zeros(nel,dtype=np.float64)

for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNV=NNV(rq,sq,CR)
        dNNNVdr=dNNVdr(rq,sq,CR)
        dNNNVds=dNNVds(rq,sq,CR)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcob = np.linalg.det(jcb)
        area[iel]+=jcob*weightq
    #end for
#end for

print("     -> area (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total area (meas) %.6f " %(area.sum()/Lx/Ly))

print("compute elements areas: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# define boundary conditions
#------------------------------------------------------------------------------
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)
bc_val=np.zeros(NfemV,dtype=np.float64)

if bc_type=='pureshear':

   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = v_bc
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV]   = -v_bc
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -v_bc
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = v_bc

elif bc_type=='simpleshear':

   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1]   = 0
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV] = -v_bc
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV]   = True ; bc_val[i*ndofV] = v_bc
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 

elif bc_type=='solvi':

   for i in range(0, NV):
       ui=u_th(xV[i]-0.5,yV[i]-0.5)
       vi=v_th(xV[i]-0.5,yV[i]-0.5)
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi
       if yV[i]/Ly>(1-eps):
          bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV+0] = ui
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vi

else:

    exit('bc_type unknown')

print("define bc: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#------------------------------------------------------------------------------
start = timing.time()

A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
#A_sparse = np.zeros((Nfem,Nfem),dtype=np.float64)
f_rhs = np.zeros(NfemV,dtype=np.float64)            # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)            # right hand side h 
b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)     # gradient matrix B 
N_mat = np.zeros((3,ndofP*mP),dtype=np.float64)     # matrix N 
NNNV = np.zeros(mV,dtype=np.float64)                # shape functions V
NNNP = np.zeros(mP,dtype=np.float64)                # shape functions P
dNNNVdx = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)             # shape functions derivatives
u = np.zeros(NV,dtype=np.float64)                   # x-component velocity
v = np.zeros(NV,dtype=np.float64)                   # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64)

xq = np.zeros(nel*nqel,dtype=np.float64)   
yq = np.zeros(nel*nqel,dtype=np.float64)   
jcobq = np.zeros(nel*nqel,dtype=np.float64)   
etaq = np.zeros(nel*nqel,dtype=np.float64)   
counterq=0

for iel in range(0,nel):

    f_el=np.zeros((mV*ndofV),dtype=np.float64)
    K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP),dtype=np.float64)

    for kq in range(0,nqel):

        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]

        NNNV[0:mV]=NNV(rq,sq,CR)
        dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
        dNNNVds[0:mV]=dNNVds(rq,sq,CR)

        jcb=np.zeros((ndim,ndim),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
        jcobq[counterq] = np.linalg.det(jcb)
        jcbi = np.linalg.inv(jcb)

        for k in range(0,mV):
            xq[counterq]+=NNNV[k]*xV[iconV[k,iel]]
            yq[counterq]+=NNNV[k]*yV[iconV[k,iel]]
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

        # construct 3x8 b_mat matrix
        for i in range(0,mV):
            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                     [0.        ,dNNNVdy[i]],
                                     [dNNNVdy[i],dNNNVdx[i]]]

        etaq[counterq]=viscosity(case,xq[counterq],yq[counterq],a_incl,b_incl,phi_incl,eta_B)

        # compute elemental a_mat matrix
        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*weightq*jcobq[counterq]

        # compute elemental rhs vector
        #for i in range(0,mV):
        #    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*rho[iel]

        NNNP[0:mP]=NNP(rq,sq)

        for i in range(0,mP):
            N_mat[0,i]=NNNP[i]
            N_mat[1,i]=NNNP[i]
            N_mat[2,i]=0.

        G_el-=b_mat.T.dot(N_mat)*weightq*jcobq[counterq]

        counterq+=1

    #end for kq

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

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

            f_rhs[m1]+=f_el[ikk]

    for k2 in range(0,mP):
        m2=iconP[k2,iel]
        h_rhs[m2]+=h_el[k2]
    
#end for 

print("build FE matrix: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# solve system
#------------------------------------------------------------------------------
start = timing.time()

rhs=np.zeros(Nfem,dtype=np.float64)
rhs[0:NfemV]=f_rhs
rhs[NfemV:Nfem]=h_rhs

sol=sps.linalg.spsolve(sps.csr_matrix(A_sparse),rhs)

print("solve time: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# put solution into separate x,y velocity arrays
#------------------------------------------------------------------------------
start = timing.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

print("split vel into u,v: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# normalise pressure
#------------------------------------------------------------------------------
start = timing.time()

int_p=0
for iel in range(0,nel):
    for kq in range (0,nqel):
        rq=qcoords_r[kq]
        sq=qcoords_s[kq]
        weightq=qweights[kq]
        NNNP[0:mP]=NNP(rq,sq)
        p_q=NNNP[0:mP].dot(p[iconP[0:mP,iel]])
        int_p+=p_q*weightq*jcobq[iel*nqel+kq]
    #end for
#end for

avrg_p=int_p/Lx/Ly

print("     -> int_p %e " %(int_p))
print("     -> avrg_p %e " %(avrg_p))

p[:]-=avrg_p

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# project pressure onto P2 mesh for plotting
#------------------------------------------------------------------------------
start = timing.time()

q = np.zeros(NV,dtype=np.float64)            # right hand side h 

for iel in range(0,nel):
    q[iconV[0,iel]]=p[iconP[0,iel]]
    q[iconV[1,iel]]=p[iconP[1,iel]]
    q[iconV[2,iel]]=p[iconP[2,iel]]
    q[iconV[3,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[4,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[5,iel]]=(p[iconP[2,iel]]+p[iconP[0,iel]])*0.5

print("project pressure onto V grid: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# compute nodal strainrate
#  02              02
#  ||\\            ||\\
#  || \\           || \\
#  ||  \\          ||  \\
#  05   04         ||   \\
#  || 06 \\        ||    \\
#  ||     \\       ||     \\
#  00==03==01      00======01
#------------------------------------------------------------------------------
start = timing.time()

e_nodal    = np.zeros(NV,dtype=np.float64)  
e_xx_nodal = np.zeros(NV,dtype=np.float64)  
e_yy_nodal = np.zeros(NV,dtype=np.float64)  
e_xy_nodal = np.zeros(NV,dtype=np.float64)  
cc         = np.zeros(NV,dtype=np.float64)

rVnodes=[0,1,0,0.5,0.5,0,1./3.] # valid for CR and P2P1
sVnodes=[0,0,1,0,0.5,0.5,1./3.]

#u[:]=xV[:]**2
#v[:]=yV[:]**2

for iel in range(0,nel):
    for kk in range(0,mV):
        inode=iconV[kk,iel]
        rq = rVnodes[kk]
        sq = sVnodes[kk]
        NNNV[0:mV]=NNV(rq,sq,CR)
        dNNNVdr[0:mV]=dNNVdr(rq,sq,CR)
        dNNNVds[0:mV]=dNNVds(rq,sq,CR)
        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
            jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
            jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
            jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)
        for k in range(0,mV):
            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
        for k in range(0,mV):
            e_xx_nodal[inode] += dNNNVdx[k]*u[iconV[k,iel]]
            e_yy_nodal[inode] += dNNNVdy[k]*v[iconV[k,iel]]
            e_xy_nodal[inode] += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
        cc[inode]+=1
    #end for
#end for
e_xx_nodal/=cc
e_yy_nodal/=cc
e_xy_nodal/=cc

e_nodal=np.sqrt(0.5*(e_xx_nodal**2+e_yy_nodal**2)+e_xy_nodal**2)

print("     -> e_xx_nodal   (m,M) %.6e %.6e " %(np.min(e_xx_nodal),np.max(e_xx_nodal)))
print("     -> e_yy_nodal   (m,M) %.6e %.6e " %(np.min(e_yy_nodal),np.max(e_yy_nodal)))
print("     -> e_xy_nodal   (m,M) %.6e %.6e " %(np.min(e_xy_nodal),np.max(e_xy_nodal)))
    
#np.savetxt('sr_cartesian.ascii',np.array([xV,yV,e_xx_nodal,e_yy_nodal,e_xy_nodal,e_nodal,cc]).T)

print("compute sr and stress: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# plot of solution
# the 7-node P2+ element does not exist in vtk, but the 6-node one does, i.e. type=22. 
#------------------------------------------------------------------------------
start = timing.time()

if True:
    filename = 'solution.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" %((p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]])/3 ))
    vtufile.write("</DataArray>\n")
    if case==0:
       #--
       vtufile.write("<DataArray type='Float32' Name='pressure (th)' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10e \n" % (p_th((xc[iel]-0.5),(yc[iel]-0.5))))
       vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='eta(q)' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (etaq[iel*nqel+5]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='inclusion' Format='ascii'> \n")
    for iel in range(0,nel):
        if inclusion[iel]:
           val=1
        else:
           val=0
        vtufile.write("%10e \n" %val)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %e_xx_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %e_yy_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %e_xy_nodal[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %e_nodal[i])
    vtufile.write("</DataArray>\n")

    if case==0:
       #--
       vtufile.write("<DataArray type='Float32' Name='pressure (th)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" % (p_th((xV[i]-0.5),(yV[i]-0.5))))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
       for i in range(0,NV):
           ui=u_th((xV[i]-0.5),(yV[i]-0.5)) 
           vi=v_th((xV[i]-0.5),(yV[i]-0.5)) 
           vtufile.write("%10e %10e %10e \n" %(ui,vi,0.))
       vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='p / (2 eta_B D_B)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" % (q[i]/2/eta_B/D_B))
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='fix_u' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if bc_fix[i*2]:
    #       val=1
    #    else:
    #       val=0
    #    vtufile.write("%10e \n" %val)
    #vtufile.write("</DataArray>\n")
    #--  
    #vtufile.write("<DataArray type='Float32' Name='fix_v' Format='ascii'> \n")
    #for i in range(0,NV):
    #    if bc_fix[i*2+1]:
    #       val=1
    #    else:
    #       val=0
    #    vtufile.write("%10e \n" %val)
    #vtufile.write("</DataArray>\n")
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                              iconV[3,iel],iconV[4,iel],iconV[5,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*6))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %22)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

    print("write data: %.3fs" % (timing.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
