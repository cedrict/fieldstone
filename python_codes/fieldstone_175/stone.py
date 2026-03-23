import numpy as np
import sys as sys
import time as clock
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################
# 1) smooth zone 1d
# 2) narrow zone 1d
# 3) two narrow zones 2d
# 4) rotated zone 2d (2x1 domain)
# 5) rotated zone 2d (2x3 domain)

experiment=5

###############################################################################

def gx(x,y):
    return 0

def gy(x,y):
    return 0

###############################################################################

if experiment==4: theta=np.pi/6
if experiment==5: theta=np.pi/20
w3x=8.
w3y=12.
w4=12.

def ud(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0.5*(x-Lx/2 -Lx/2/np.pi*np.sin(2*np.pi*x/Lx) )
    #----------------
    if experiment==2:
       delta=Lx/8
       if x<=Lx/2-delta: 
          return -delta/2
       elif x<=Lx/2+delta:
          return 0.5*(x-Lx/2+delta/np.pi*np.sin(np.pi*(x-Lx/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==3: 
       delta=Lx/w3x
       if x<=Lx/2-delta: 
          return -delta/2
       elif x<=Lx/2+delta:
          return 0.5*(x-Lx/2+delta/np.pi*np.sin(np.pi*(x-Lx/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if xp<=Lx/2-delta4: 
          return -delta4/2*np.cos(theta)
       elif xp<=Lx/2+delta4:
          return 0.5*(xp-Lx/2+delta4/np.pi*np.sin(np.pi*(xp-Lx/2)/delta4))*np.cos(theta)
       else: 
          return delta4/2*np.cos(theta)


def dud_dx(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return (1-np.cos(2*np.pi*x/Lx))/2
    #----------------
    if experiment==2:
       delta=Lx/8
       if abs(x-Lx/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(x-Lx/2)/delta))
       else:
          return 0
    #----------------
    if experiment==3:
       delta=Lx/w3x
       if abs(x-Lx/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(x-Lx/2)/delta))
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return 0.5*(1+np.cos(np.pi*(xp-Lx/2)/delta4))*np.cos(theta)*np.cos(theta)
       else:
          return 0



def d2ud_dx2(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return np.pi/Lx*np.sin(2*np.pi*x/Lx)
    #----------------
    if experiment==2:
       delta=Lx/8
       if abs(x-Lx/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(x-Lx/2)/delta)
       else:
          return 0
    #----------------
    if experiment==3:
       delta=Lx/w3x
       if abs(x-Lx/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(x-Lx/2)/delta)
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.cos(theta)*np.cos(theta)*np.cos(theta)
       else:
          return 0

def d2ud_dy2(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.sin(theta)*np.cos(theta)
       else:
          return 0


def d2ud_dxdy(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.cos(theta)*np.cos(theta)
       else:
          return 0

###############################################################################

def vd(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3: 
       delta=Ly/w3y
       if y<=Ly/2-delta: 
          return -delta/2
       elif y<=Ly/2+delta:
          return 0.5*(y-Ly/2+delta/np.pi*np.sin(np.pi*(y-Ly/2)/delta))
       else: 
          return delta/2
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if xp<=Lx/2-delta4: 
          return -delta4/2*np.sin(theta)
       elif xp<=Lx/2+delta4:
          return 0.5*(xp-Lx/2+delta4/np.pi*np.sin(np.pi*(xp-Lx/2)/delta4))*np.sin(theta)
       else: 
          return delta4/2*np.sin(theta)

          

def dvd_dy(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3:
       delta=Ly/w3y
       if abs(y-Ly/2)<=delta: 
          return 0.5*(1+np.cos(np.pi*(y-Ly/2)/delta))
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return 0.5*(1+np.cos(np.pi*(xp-Lx/2)/delta4))*np.sin(theta)*np.sin(theta)
       else:
          return 0

def d2vd_dy2(x,y,Lx,Ly):
    #----------------
    if experiment==1: 
       return 0
    #----------------
    if experiment==2: 
       return 0
    #----------------
    if experiment==3:
       delta=Ly/w3y
       if abs(y-Ly/2)<=delta: 
          return -0.5*np.pi/delta*np.sin(np.pi*(y-Ly/2)/delta)
       else:
          return 0
    #----------------
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.sin(theta)*np.sin(theta)
       else:
          return 0

def d2vd_dx2(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.cos(theta)*np.cos(theta)*np.sin(theta)
       else:
          return 0

def d2vd_dxdy(x,y,Lx,Ly):
    if experiment==1: return 0
    if experiment==2: return 0
    if experiment==3: return 0
    if experiment==4 or experiment==5: 
       delta4=Lx/w4
       xp=((x-Lx/2)*np.cos(theta)+(y-Ly/2)*np.sin(theta))+Lx/2
       if abs(xp-Lx/2)<=delta4: 
          return -0.5*np.pi/delta4*np.sin(np.pi*(xp-Lx/2)/delta4)*np.sin(theta)*np.cos(theta)*np.sin(theta)
       else:
          return 0

###############################################################################

def fx(x,y,Lx,Ly):
    return (4/3*d2ud_dx2(x,y,Lx,Ly) +d2ud_dy2(x,y,Lx,Ly) +1/3*d2vd_dxdy(x,y,Lx,Ly))

def fy(x,y,Lx,Ly):
    return (d2vd_dx2(x,y,Lx,Ly) +4/3*d2vd_dy2(x,y,Lx,Ly) +1/3*d2ud_dxdy(x,y,Lx,Ly))

###############################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,\
                     N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,\
                     dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,\
                     dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 175 **********")
print("*******************************")

ndim=2
m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

if int(len(sys.argv) == 2): 
   nelx = int(sys.argv[1])
else:
   nelx=48

nely=int(nelx/2)

Lx=2.
Ly=1.

if experiment==5: Ly=3
if experiment==5: nely*=3

eta=1.
rho=1.

###############################################################################

nnx=2*nelx+1             # number of nodes, x direction
nny=2*nely+1             # number of nodes, y direction
nn_V=nnx*nny             # total number of nodes
nel=nelx*nely            # total number of elements
Nfem_V=nn_V*ndof_V       # number of velocity dofs
Nfem_P=(nelx+1)*(nely+1) # number of pressure dofs
Nfem=Nfem_V+Nfem_P       # total number of dofs
hx=Lx/nelx               # mesh size in x direction
hy=Ly/nely               # mesh size in y direction

scaling_coeff=1

###############################################################################
# quadrature parameters
###############################################################################

nq_per_dim=3
nq_per_el=9
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

#nq_per_dim=4
#nq_per_el=16
#qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
#qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
#qw4a=(18-np.sqrt(30.))/36.
#qw4b=(18+np.sqrt(30.))/36.
#qcoords=[-qc4a,-qc4b,qc4b,qc4a]
#qweights=[qw4a,qw4b,qw4b,qw4a]

#nq_per_dim=5
#nq_per_el=25
#qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
#qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
#qc5c=0.
#qw5a=(322.-13.*np.sqrt(70.))/900.
#qw5b=(322.+13.*np.sqrt(70.))/900.
#qw5c=128./225.
#qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
#qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("hx",hx)
print("hy",hy)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter+=1

print("setup: grid points: %.3f s" % (clock.time()-start))

#################################################################
# build connectivity arrays for velocity and pressure
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
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
        counter += 1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

if experiment==1 or experiment==2:
   for i in range(0,nn_V):
       if abs(y_V[i]-Ly)/Ly<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0
       if abs(y_V[i])/Ly<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0 
          # fix 1 u dof to remove translational nullspace 
          if abs(x_V[i]-Lx/2)<eps:
             bc_fix_V[i*ndof_V] = True ; bc_val_V[i*ndof_V] = 0 

if experiment==3 or experiment==4 or experiment==5: 
   for i in range(0,nn_V):
       if abs(x_V[i])/Lx<eps or abs(x_V[i]-Lx)/Lx<eps or \
          abs(y_V[i])/Ly<eps or abs(y_V[i]-Ly)/Ly<eps :
          bc_fix_V[i*ndof_V+0] = True ; bc_val_V[i*ndof_V+0] = ud(x_V[i],y_V[i],Lx,Ly)
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vd(x_V[i],y_V[i],Lx,Ly)

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

u_th=np.zeros(nn_V,dtype=np.float64) 
v_th=np.zeros(nn_V,dtype=np.float64) 
dudx_th=np.zeros(nn_V,dtype=np.float64) 
dvdy_th=np.zeros(nn_V,dtype=np.float64) 
d2udx2_th=np.zeros(nn_V,dtype=np.float64) 
d2udy2_th=np.zeros(nn_V,dtype=np.float64) 
d2udxy_th=np.zeros(nn_V,dtype=np.float64) 
d2vdx2_th=np.zeros(nn_V,dtype=np.float64) 
d2vdy2_th=np.zeros(nn_V,dtype=np.float64) 
d2vdxy_th=np.zeros(nn_V,dtype=np.float64) 
fx_th=np.zeros(nn_V,dtype=np.float64) 
fy_th=np.zeros(nn_V,dtype=np.float64) 

for i in range(0,nn_V):
    u_th[i]=ud(x_V[i],y_V[i],Lx,Ly)
    v_th[i]=vd(x_V[i],y_V[i],Lx,Ly)
    dudx_th[i]=dud_dx(x_V[i],y_V[i],Lx,Ly)
    dvdy_th[i]=dvd_dy(x_V[i],y_V[i],Lx,Ly)
    d2udx2_th[i]=d2ud_dx2(x_V[i],y_V[i],Lx,Ly)
    d2udy2_th[i]=d2ud_dy2(x_V[i],y_V[i],Lx,Ly)
    d2udxy_th[i]=d2ud_dxdy(x_V[i],y_V[i],Lx,Ly)
    d2vdx2_th[i]=d2vd_dx2(x_V[i],y_V[i],Lx,Ly)
    d2vdy2_th[i]=d2vd_dy2(x_V[i],y_V[i],Lx,Ly)
    d2vdxy_th[i]=d2vd_dxdy(x_V[i],y_V[i],Lx,Ly)
    fx_th[i]=fx(x_V[i],y_V[i],Lx,Ly)
    fy_th[i]=fy(x_V[i],y_V[i],Lx,Ly)

np.savetxt('solution_th.ascii',np.array([x_V,y_V,u_th,v_th,dudx_th,dvdy_th,\
                                         d2udx2_th,d2udy2_th,d2udxy_th,\
                                         d2vdx2_th,d2vdy2_th,d2vdxy_th,
                                         fx_th,fy_th ]).T)

###############################################################################
# build FE matrix A and rhs 
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
# note that we *must* use the deviatoric C matrix!
###############################################################################

C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)    
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
N_mat=np.zeros((3,m_P),dtype=np.float64)
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0,nel):

    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
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

            for i in range(0,m_V):
                f_el[ndof_V*i+0]+=N_V[i]*(rho*gx(xq,yq)-fx(xq,yq,Lx,Ly))*JxWq
                f_el[ndof_V*i+1]+=N_V[i]*(rho*gy(xq,yq)-fy(xq,yq,Lx,Ly))*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.

            G_el-=B.T.dot(N_mat)*JxWq
             
            h_el[:]-=N_P[:]*(dud_dx(xq,yq,Lx,Ly)+dvd_dy(xq,yq,Lx,Ly))*JxWq

        # end for iq 
    # end for jq 

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for jkk
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0
            # end if 
        # end for i1 
    #end for k1 

    # assemble matrix and right hand side vector 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2] += K_el[ikk,jkk]
                #end for i2
            #end for k2
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]*scaling_coeff
                A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
            b_fem[m1]+=f_el[ikk]
            #end for k2
        #end for i1
    #end for k1 

    for k1 in range(0,m_P):
        m1=icon_P[k1,iel]
        b_fem[Nfem_V+m1]+=h_el[k1]*scaling_coeff
    #end for k1

# end for iel 

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble f, h into rhs and solve
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]*scaling_coeff

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

np.savetxt('solution.ascii',np.array([x_V,y_V,u,v]).T)

print("solve system: %.3f s - Nfem %d" % (clock.time()-start,Nfem))

###############################################################################
# normalise pressure
###############################################################################

if experiment==3 or experiment==4 or experiment==5:

   avrg_p=0.
   for iel in range(0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_P=basis_functions_P(rq,sq)
               dNdr_V=basis_functions_V_dr(rq,sq)
               dNds_V=basis_functions_V_ds(rq,sq)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               JxWq=np.linalg.det(jcb)*weightq
               pq=np.dot(N_P,p[icon_P[:,iel]])
               avrg_p+=pq*JxWq 
  
   p-=avrg_p/(Lx*Ly)

   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

###############################################################################
# interpolate pressure onto velocity grid points (for plotting)
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    q[icon_V[0,iel]]=p[icon_P[0,iel]]
    q[icon_V[1,iel]]=p[icon_P[1,iel]]
    q[icon_V[2,iel]]=p[icon_P[2,iel]]
    q[icon_V[3,iel]]=p[icon_P[3,iel]]
    q[icon_V[4,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
    q[icon_V[5,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
    q[icon_V[6,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
    q[icon_V[7,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
    q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+\
                      p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25

print("project p onto vel nodes: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate at center of element 
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
sr=np.zeros(nel,dtype=np.float64)  
p_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.
    sq = 0.
    N_V=basis_functions_V(rq,sq)
    N_P=basis_functions_P(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    p_e[iel]=np.dot(N_P,p[icon_P[:,iel]])
#end if

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> sr  (m,M) %.4e %.4e " %(np.min(sr),np.max(sr)))
print("     -> p_e (m,M) %.4e %.4e " %(np.min(p_e),np.max(p_e)))

np.savetxt('solution_e.ascii',np.array([x_e,y_e,p_e,exx,eyy,exx+eyy]).T)

print("compute elemental press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate on velocity grid
###############################################################################
start=clock.time()

exxn=np.zeros(nn_V,dtype=np.float64)
eyyn=np.zeros(nn_V,dtype=np.float64)
exyn=np.zeros(nn_V,dtype=np.float64)
srn=np.zeros(nn_V,dtype=np.float64)
divvn=np.zeros(nn_V,dtype=np.float64)
c=np.zeros(nn_V,dtype=np.float64)

r_V=[-1,+1,1,-1, 0,1,0,-1,0]
s_V=[-1,-1,1,+1,-1,0,1, 0,0]

for iel in range(0,nel):
    for i in range(0,m_V):
        dNdr_V=basis_functions_V_dr(r_V[i],s_V[i])
        dNds_V=basis_functions_V_ds(r_V[i],s_V[i])
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exxn[icon_V[i,iel]]+=np.dot(dNdx_V,u[icon_V[:,iel]])
        eyyn[icon_V[i,iel]]+=np.dot(dNdy_V,v[icon_V[:,iel]])
        exyn[icon_V[i,iel]]+=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                             np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
        c[icon_V[i,iel]]+=1.
    # end for i
# end for iel

exxn/=c
eyyn/=c
exyn/=c

divvn[:]=exxn[:]+eyyn[:]

srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

print("     -> exxn  (m,M) %.4e %.4e " %(np.min(exxn),np.max(exxn)))
print("     -> eyyn  (m,M) %.4e %.4e " %(np.min(eyyn),np.max(eyyn)))
print("     -> exyn  (m,M) %.4e %.4e " %(np.min(exyn),np.max(exyn)))
print("     -> srn   (m,M) %.4e %.4e " %(np.min(srn),np.max(srn)))
print("     -> divvn (m,M) %.4e %.4e " %(np.min(divvn),np.max(divvn)))

print("compute nod strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# compute discretisation errors 
###############################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])
            errv+=((uq-ud(xq,yq,Lx,Ly))**2+(vq-vd(xq,yq,Lx,Ly))**2)*JxWq
            errp+=(pq-0)**2*JxWq
        #end for
    #end for
#end for
errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv: %e ; errp: %e " %(nelx,errv,errp))

print("compute discr errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

filename = 'solution.vtu'
vtufile=open(filename,"w")
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
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
exx.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
eyy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
exy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='analytical velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u_th[i],v_th[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity-analytical' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i]-u_th[i],v[i]-v_th[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical dud_dx' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(dudx_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2ud_dx2' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2udx2_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2ud_dy2' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2udy2_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2ud_dxdy' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2udxy_th[i]))
vtufile.write("</DataArray>\n")

#--
vtufile.write("<DataArray type='Float32' Name='analytical dvd_dy' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(dvdy_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2vd_dx2' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2vdx2_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2vd_dy2' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2vdy2_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='analytical d2vd_dxdy' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %(d2vdxy_th[i]))
vtufile.write("</DataArray>\n")



#--
vtufile.write("<DataArray type='Float32' Name='analytical divv' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %( dudx_th[i]+dvdy_th[i]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                   icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                   icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*9))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu : %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
