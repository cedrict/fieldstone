import sys as sys
import numpy as np
import time as clock
import scipy.sparse as sps
from scipy.sparse import lil_matrix
from vj3 import bx,by,u_th,v_th,p_th,exx_th,eyy_th,exy_th
from numpy import linalg 
import random 

###############################################################################

def basis_functions_V(r,s,A,mx,my):
    if serendipity>0:
       N1=(1-r)*(1-s)*(-r-s-1)*0.25
       N2=(1+r)*(1-s)*(r-s-1) *0.25
       N3=(1+r)*(1+s)*(r+s-1) *0.25
       N4=(1-r)*(1+s)*(-r+s-1)*0.25
       N5=(1-r**2)*(1-s)*0.5
       N6=(1+r)*(1-s**2)*0.5
       N7=(1-r**2)*(1+s)*0.5
       N8=(1-r)*(1-s**2)*0.5
       if serendipity==2: #eq 29 of zhxi20
          E=(1-r**2)*(1-s**2)
          denom=4*(4*A**2+mx**2+my**2)
          N3+=(mx**2-mx*my+my**2)/denom*E
          N4+=(mx**2+mx*my+my**2)/denom*E
          N1+=(mx**2-mx*my+my**2)/denom*E
          N2+=(mx**2+mx*my+my**2)/denom*E
          denom=4*A*(4*A**2+mx**2+my**2)
          N7-=mx*( 2*A*mx+my**2)/denom*E
          N8-=my*( 2*A*my+mx**2)/denom*E
          N5+=mx*(-2*A*mx+my**2)/denom*E
          N6+=my*(-2*A*my+mx**2)/denom*E
       return np.array([N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)
    else:
       N0= 0.5*r*(r-1) * 0.5*s*(s-1)
       N1= 0.5*r*(r+1) * 0.5*s*(s-1)
       N2= 0.5*r*(r+1) * 0.5*s*(s+1)
       N3= 0.5*r*(r-1) * 0.5*s*(s+1)
       N4=    (1-r**2) * 0.5*s*(s-1)
       N5= 0.5*r*(r+1) *    (1-s**2)
       N6=    (1-r**2) * 0.5*s*(s+1)
       N7= 0.5*r*(r-1) *    (1-s**2)
       N8=    (1-r**2) *    (1-s**2)
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s,A,mx,my):
    if serendipity>0:
       dNdr1= -0.25*(s-1)*(2*r+s)
       dNdr2= -0.25*(s-1)*(2*r-s)
       dNdr3= 0.25*(s+1)*(2*r+s)
       dNdr4= 0.25*(s+1)*(2*r-s)
       dNdr5= r*(s-1)
       dNdr6= 0.5*(1-s**2)
       dNdr7= -r*(s+1)           
       dNdr8= -0.5*(1-s**2)
       if serendipity==2:
          dEdr=-2*r*(1-s**2)
          denom=4*(4*A**2+mx**2+my**2)
          dNdr3+=(mx**2-mx*my+my**2)/denom*dEdr
          dNdr4+=(mx**2+mx*my+my**2)/denom*dEdr
          dNdr1+=(mx**2-mx*my+my**2)/denom*dEdr
          dNdr2+=(mx**2+mx*my+my**2)/denom*dEdr
          denom=4*A*(4*A**2+mx**2+my**2)
          dNdr7-=mx*( 2*A*mx+my**2)/denom*dEdr
          dNdr8-=my*( 2*A*my+mx**2)/denom*dEdr
          dNdr5+=mx*(-2*A*mx+my**2)/denom*dEdr
          dNdr6+=my*(-2*A*my+mx**2)/denom*dEdr
       return np.array([dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)
    else:
       dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr4=       (-2.*r) * 0.5*s*(s-1)
       dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr6=       (-2.*r) * 0.5*s*(s+1)
       dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr8=       (-2.*r) *   (1.-s**2)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s,A,mx,my):
    if serendipity>0:
       dNds1= -0.25*(r-1)*(r+2*s)
       dNds2= -0.25*(r+1)*(r-2*s)
       dNds3= 0.25*(r+1)*(r+2*s)
       dNds4= 0.25*(r-1)*(r-2*s)
       dNds5= -0.5*(1-r**2)
       dNds6= -(r+1)*s
       dNds7= 0.5*(1-r**2)
       dNds8= (r-1)*s
       if serendipity==2:
          dEds=-2*s*(1-r**2)
          denom=4*(4*A**2+mx**2+my**2)
          dNds3+=(mx**2-mx*my+my**2)/denom*dEds
          dNds4+=(mx**2+mx*my+my**2)/denom*dEds
          dNds1+=(mx**2-mx*my+my**2)/denom*dEds
          dNds2+=(mx**2+mx*my+my**2)/denom*dEds
          denom=4*A*(4*A**2+mx**2+my**2)
          dNds7-=mx*( 2*A*mx+my**2)/denom*dEds
          dNds8-=my*( 2*A*my+mx**2)/denom*dEds
          dNds5+=mx*(-2*A*mx+my**2)/denom*dEds
          dNds6+=my*(-2*A*my+mx**2)/denom*dEds
       return np.array([dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)
    else:
       dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds5= 0.5*r*(r+1.) *       (-2.*s)
       dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds7= 0.5*r*(r-1.) *       (-2.*s)
       dNds8=    (1.-r**2) *       (-2.*s)
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################
###############################################################################

debug=False

ndim=2
ndof_V=2

Lx=1.
Ly=1.

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   serendipity = int(sys.argv[4])
else:
   nelx = 24 
   nely = 24
   visu = 1
   serendipity=2

nel=nelx*nely

if serendipity>0:
   nn_V=(nelx+1)*(nely+1)+nelx*(nely+1)+ (nelx+1)*nely
   m_V=8
   m_P=4
else:
   nn_V=(2*nelx+1)*(2*nely+1)
   m_V=9
   m_P=4

nn_P=(nelx+1)*(nely+1)
Nfem_V=nn_V*ndof_V
Nfem_P=nn_P
Nfem=Nfem_V+Nfem_P

hx=Lx/nelx
hy=Ly/nely

use_random=False
xi=0.1 # controls level of mesh randomness (between 0 and 0.5 max)

compute_eigenv=False

print("*******************************")
print("********** stone 052 **********")
print("*******************************")
print('nelx  =',nelx)
print('nely  =',nely)
print('nel   =',nel)
print('nn_V  =',nn_V)
print('nn_P  =',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('serendipity=',serendipity)
print('use_random=',use_random)
print("*******************************")

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

#4 qpoints rule does not change anything
#nq_per_dim=4
#qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
#qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
#qw4a=(18-np.sqrt(30.))/36.
#qw4b=(18+np.sqrt(30.))/36.
#qcoords=[-qc4a,-qc4b,qc4b,qc4a]
#qweights=[qw4a,qw4b,qw4b,qw4a]

eps=1e-8
eta=1.

sparse=False

if serendipity>0:
   r_V=[-1,1,1,-1,0,1,0,-1]
   s_V=[-1,-1,1,1,-1,0,1,0]
else:
   r_V=[-1,1,1,-1,0,1,0,-1,0]
   s_V=[-1,-1,1,1,-1,0,1,0,0]

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

if serendipity>0:
   counter = 0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x_V[counter]=i*hx
           y_V[counter]=j*hy
           counter += 1

   for j in range(nely):
       for i in range(0,nelx):
           x_V[counter]=i*hx+hx/2
           y_V[counter]=j*hy
           counter+=1
       for i in range(0,nelx+1):
           x_V[counter]=i*hx
           y_V[counter]=j*hy+hy/2
           counter+=1

   for i in range(0,nelx):
       x_V[counter]=i*hx+hx/2
       y_V[counter]=nely*hy
       counter+=1

else:

   counter = 0
   for j in range(0, 2*nely+1):
       for i in range(0, 2*nelx+1):
           x_V[counter]=i*hx/2.
           y_V[counter]=j*hy/2.
           counter += 1

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

if serendipity>0:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_V[0,counter] = i + j * (nelx + 1)
           icon_V[1,counter] = i + 1 + j * (nelx + 1)
           icon_V[2,counter] = i + 1 + (j + 1) * (nelx + 1)
           icon_V[3,counter] = i + (j + 1) * (nelx + 1)
           icon_V[4,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j
           icon_V[5,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)
           icon_V[6,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*(j+1) 
           icon_V[7,counter] = (nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)-1
           counter += 1
else:
   nnx=2*nelx+1
   nny=2*nely+1
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
           icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
           icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
           icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
           icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
           icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
           icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
           icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
           icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
           counter+=1

#for iel in range (0,nel):
#     print ("iel=",iel)
#     for i in range(0,m_V):
#         print ("node ",i,':',icon_V[i,iel],"at pos.",x_V[icon_V[i,iel]], y_V[icon_V[i,iel]])

###############################################################################
# add random perturbation
###############################################################################
start=clock.time()

if use_random:

   for i in range(0,nn_V):
       if x_V[i]>0 and x_V[i]<Lx-eps and y_V[i]>0 and y_V[i]<Ly-eps:
          x_V[i]+=random.uniform(-1.,+1)*hx*xi
          y_V[i]+=random.uniform(-1.,+1)*hy*xi
       #end if
   #end for

   for iel in range(0,nel):
       x_V[icon_V[4,iel]]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]])
       y_V[icon_V[4,iel]]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]])
       x_V[icon_V[5,iel]]=0.5*(x_V[icon_V[1,iel]]+x_V[icon_V[2,iel]])
       y_V[icon_V[5,iel]]=0.5*(y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])
       x_V[icon_V[6,iel]]=0.5*(x_V[icon_V[2,iel]]+x_V[icon_V[3,iel]])
       y_V[icon_V[6,iel]]=0.5*(y_V[icon_V[2,iel]]+y_V[icon_V[3,iel]])
       x_V[icon_V[7,iel]]=0.5*(x_V[icon_V[3,iel]]+x_V[icon_V[0,iel]])
       y_V[icon_V[7,iel]]=0.5*(y_V[icon_V[3,iel]]+y_V[icon_V[0,iel]])
       if serendipity==0:
          x_V[icon_V[8,iel]]=0.25*(x_V[icon_V[0,iel]]+x_V[icon_V[1,iel]]+\
                                   x_V[icon_V[2,iel]]+x_V[icon_V[3,iel]])
          y_V[icon_V[8,iel]]=0.25*(y_V[icon_V[0,iel]]+y_V[icon_V[1,iel]]+\
                                   y_V[icon_V[2,iel]]+y_V[icon_V[3,iel]])
       #end if
   #end for

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("add random V grid: %.3f s" % (clock.time()-start))

###############################################################################
# build pressure grid and connectivity array 
###############################################################################
start=clock.time()

x_P=np.zeros(nn_P,dtype=np.float64)   
y_P=np.zeros(nn_P,dtype=np.float64)    
icon_P=np.zeros((m_P,nel),dtype=np.int32)

if serendipity>0:
   icon_P[0:m_P,0:nel]=icon_V[0:m_P,0:nel]
   x_P[0:nn_P]=x_V[0:nn_P]
   y_P[0:nn_P]=y_V[0:nn_P]
else:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_P[0,counter]=i+j*(nelx+1)
           icon_P[1,counter]=i+1+j*(nelx+1)
           icon_P[2,counter]=i+1+(j+1)*(nelx+1)
           icon_P[3,counter]=i+(j+1)*(nelx+1)
           counter += 1
   for iel in range(0,nel):
       x_P[icon_P[0,iel]]=x_V[icon_V[0,iel]]
       x_P[icon_P[1,iel]]=x_V[icon_V[1,iel]]
       x_P[icon_P[2,iel]]=x_V[icon_V[2,iel]]
       x_P[icon_P[3,iel]]=x_V[icon_V[3,iel]]
       y_P[icon_P[0,iel]]=y_V[icon_V[0,iel]]
       y_P[icon_P[1,iel]]=y_V[icon_V[1,iel]]
       y_P[icon_P[2,iel]]=y_V[icon_V[2,iel]]
       y_P[icon_P[3,iel]]=y_V[icon_V[3,iel]]

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start=clock.time()

area    = np.zeros(nel,dtype=np.float64) 
A       = np.zeros(nel,dtype=np.float64) 
mx      = np.zeros(nel,dtype=np.float64) 
my      = np.zeros(nel,dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):

    x1=x_V[icon_V[2,iel]] ; y1=y_V[icon_V[2,iel]]
    x2=x_V[icon_V[3,iel]] ; y2=y_V[icon_V[3,iel]]
    x3=x_V[icon_V[0,iel]] ; y3=y_V[icon_V[0,iel]]
    x4=x_V[icon_V[1,iel]] ; y4=y_V[icon_V[1,iel]]

    A[iel]=0.5*((x1-x3)*(y2-y4)-(x2-x4)*(y1-y3))
    mx[iel]=(x1-x4)*(y2-y3)-(x2-x3)*(y1-y4)
    my[iel]=(x3-x4)*(y1-y2)-(x1-x2)*(y3-y4)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq,A[iel],mx[iel],my[iel])
            dNdr_V=basis_functions_V_dr(rq,sq,A[iel],mx[iel],my[iel])
            dNds_V=basis_functions_V_ds(rq,sq,A[iel],mx[iel],my[iel])
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq

print("     -> area (m,M) %.4e %.4e " %(np.min(area),np.max(area)))
print("     -> total area %.6f " %(area.sum()))

print("compute elements areas: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps):
       bc_fix[i*ndof_V]   = True ; bc_val[i*ndof_V]   = 0.
       bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

if sparse:
   A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

f_rhs   = np.zeros(Nfem_V,dtype=np.float64)        # right hand side f 
h_rhs   = np.zeros(Nfem_P,dtype=np.float64)        # right hand side h 

B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat= np.zeros((3,m_P),dtype=np.float64) # matrix  
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_P=basis_functions_P(rq,sq)
            N_V=basis_functions_V(rq,sq,A[iel],mx[iel],my[iel])
            dNdr_V=basis_functions_V_dr(rq,sq,A[iel],mx[iel],my[iel])
            dNds_V=basis_functions_V_ds(rq,sq,A[iel],mx[iel],my[iel])
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                f_el[ndof_V*i+1]+=N_V[i]*by(xq,yq)*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.

            G_el-=B.T.dot(N_mat)*JxWq

        # end for jq
    # end for iq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[:]-=G_el[ikk,:]*bc_val[m1]
               G_el[ikk,:]=0

    # assemble matrix and right hand side
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                if sparse:
                   A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]

print("build FE matrix: %.3fs - %d elts" % (clock.time()-start, nel))

###############################################################################
# compute min,max eigenvalues of K matrix
###############################################################################
start=clock.time()

if compute_eigenv:
   eigvals, eigvecs = linalg.eig(K_mat)
   print('eigenvalues:',nel,eigvals.min(),eigvals.max())
   print('condition number:', nel,linalg.cond(K_mat))

print("eigenvalues and cond nb: %.3f s" % (clock.time() - start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs

if not sparse:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64) 
   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T

print("assemble blocks: %.3f s" % (clock.time() - start))

###############################################################################
# assign extra pressure b.c. to remove null space
###############################################################################

if sparse:
   A_sparse[Nfem-1,:]=0
   A_sparse[:,Nfem-1]=0
   A_sparse[Nfem-1,Nfem-1]=1
   rhs[Nfem-1]=0
else:
   idof=Nfem_V+icon_P[2,nel-1] #Nfem-1
   #print (idof)
   a_mat[idof,:]=0
   a_mat[:,idof]=0
   a_mat[idof,idof]=1
   rhs[idof]=0

###############################################################################
# solve system
###############################################################################
start=clock.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (clock.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# normalise pressure field 
###############################################################################
start=clock.time()

avrg_p=0.
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_P=basis_functions_P(rq,sq)
            N_V=basis_functions_V(rq,sq,A[iel],mx[iel],my[iel])
            dNdr_V=basis_functions_V_dr(rq,sq,A[iel],mx[iel],my[iel])
            dNds_V=basis_functions_V_ds(rq,sq,A[iel],mx[iel],my[iel])
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            pq=np.dot(N_P,p[icon_P[:,iel]])
            avrg_p+=pq*JxWq

print('avrg pressure',avrg_p, 'exact value: -1.25')

#p[:]-=avrg_p/Lx/Ly

if debug: np.savetxt('pressure_normalised.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# project pressure onto velocity grid
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)
c=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,m_V):
        N_P=basis_functions_P(r_V[i],s_V[i])
        q[icon_V[i,iel]]+=np.dot(p[icon_P[:,iel]],N_P)
        c[icon_V[i,iel]]+=1.
    # end for i
# end for iel

q/=c

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute L2 errors
###############################################################################
start=clock.time()

#u[:]=x_V[:]**3
#v[:]=y_V[:]**3

errv=0.
errp=0.
errq=0.
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_P=basis_functions_P(rq,sq)
            N_V=basis_functions_V(rq,sq,A[iel],mx[iel],my[iel])
            dNdr_V=basis_functions_V_dr(rq,sq,A[iel],mx[iel],my[iel])
            dNds_V=basis_functions_V_ds(rq,sq,A[iel],mx[iel],my[iel])
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            qq=np.dot(N_V,q[icon_V[:,iel]])
            errv+=((uq-u_th(xq,yq))**2+(vq-v_th(xq,yq))**2)*JxWq
            errq+=(qq-p_th(xq,yq))**2*JxWq

            xq=np.dot(N_P,x_P[icon_P[:,iel]])
            yq=np.dot(N_P,y_P[icon_P[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])
            errp+=(pq-p_th(xq,yq))**2*JxWq

        # end for jq
    # end for iq
# end for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq=np.sqrt(errq)

print("     -> nel= %6d ; errv= %.8e ; errp= %.8e ; errq= %.8e" %(nel,errv,errp,errq))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# export solution to vtu file
###############################################################################

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (area[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='A' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (A[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mx' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (mx[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='my' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%e\n" % (my[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='denom' Format='ascii'> \n")
    for iel in range (0,nel):
        denom=4*A[iel]**2+mx[iel]**2+my[iel]**2
        vtufile.write("%e\n" % (denom))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e %e %e \n" %(u_th(x_V[i],y_V[i]),v_th(x_V[i],y_V[i]),0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e \n" %(p_th(x_V[i],y_V[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e \n" %(exx_th(x_V[i],y_V[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e \n" %(eyy_th(x_V[i],y_V[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
    for i in range(0,nn_V):
        vtufile.write("%e \n" %(exy_th(x_V[i],y_V[i])))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],\
                                                     icon_V[2,iel],icon_V[3,iel],\
                                                     icon_V[4,iel],icon_V[5,iel],\
                                                     icon_V[6,iel],icon_V[7,iel]))
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
    vtufile.close()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
