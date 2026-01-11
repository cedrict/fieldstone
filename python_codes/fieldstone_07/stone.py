import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-rq)*(1.-sq)
    N1=0.25*(1.+rq)*(1.-sq)
    N2=0.25*(1.+rq)*(1.+sq)
    N3=0.25*(1.-rq)*(1.+sq)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-sq)
    dNdr1=+0.25*(1.-sq)
    dNdr2=+0.25*(1.+sq)
    dNdr3=-0.25*(1.+sq)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-rq)
    dNds1=-0.25*(1.+rq)
    dNds2=+0.25*(1.+rq)
    dNds3=+0.25*(1.-rq)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def viscosity(x,y):
    if (np.sqrt(x*x+y*y) < 0.2):
       val=1e3
    else:
       val=1.
    return val

###############################################################################

def solution(x,y):
    min_eta = 1.
    max_eta = 1.e3
    epsilon = 1.
    A=min_eta*(max_eta-min_eta)/(max_eta+min_eta)
    r_inclusion=0.2 
    r2_inclusion=r_inclusion*r_inclusion
    r2=x*x+y*y
    # phi, psi, dphi are complex
    z=x+y*1j
    if r2<r2_inclusion:
       phi=0+0.*1j
       dphi=0+0.*1j
       psi=-4*epsilon*(max_eta*min_eta/(min_eta+max_eta))*z
       visc=1e3
    else:
       phi=-2*epsilon*A*r2_inclusion/z
       dphi=-phi/z
       psi=-2*epsilon*(min_eta*z+A*r2_inclusion*r2_inclusion/(z*z*z))
       visc=1.

    v = (phi-z*np.conjugate(dphi)-np.conjugate(psi))/(2.*visc)
    vx=v.real
    vy=v.imag
    p=-2*epsilon*dphi.real

    return vx,vy,p

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 007 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 48
   nely = 48
   visu = 1
    
nnx=nelx+1       # number of V nodes, x direction
nny=nely+1       # number of V nodes, y direction
nn_V=nnx*nny     # number of V nodes
nel=nelx*nely    # total number of elements
Nfem_V=nn_V*ndof_V # Total number of degrees of freedom
Nfem=Nfem_V

penalty=1.e8  # penalty coefficient value

debug=False

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter+=1

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]= i + j * (nelx + 1)
        icon_V[1,counter]= i + 1 + j * (nelx + 1)
        icon_V[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon_V[3,counter]= i + (j + 1) * (nelx + 1)
        counter += 1

if debug:
   for iel in range (0,nel):
     print ("iel=",iel)
     print ("node 1",icon_V[0,iel],"at pos.",x_V[icon_V[0,iel]],y_V[icon_V[0,iel]])
     print ("node 2",icon_V[1,iel],"at pos.",x_V[icon_V[1,iel]],y_V[icon_V[1,iel]])
     print ("node 3",icon_V[2,iel],"at pos.",x_V[icon_V[2,iel]],y_V[icon_V[2,iel]])
     print ("node 4",icon_V[3,iel],"at pos.",x_V[icon_V[3,iel]],y_V[icon_V[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    ui,vi,pi=solution(x_V[i],y_V[i])
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=ui
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vi
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=ui
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vi
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=ui
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vi
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=ui
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vi

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=np.zeros((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)        # right hand side of Ax=b
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  # gradient matrix B 
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0, nel):

    # set 2 arrays to 0 
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V *ndof_V,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V=basis_functions_V(rq,sq)

            # calculate shape function derivatives
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate inverse of the jacobian matrix
            jcbi=np.linalg.inv(jcb)

            # calculate the determinant of the jacobian times weight
            JxWq=np.linalg.det(jcb)*weightq

            # compute coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental a_mat matrix
            A_el+=B.T.dot(C.dot(B))*viscosity(xq,yq)*JxWq

            # compute elemental rhs vector
            #for i in range(0, m):
            #    b_el[2*i  ]+=N[i]*jcob*wq*density(xq,yq)*gx
            #    b_el[2*i+1]+=N[i]*jcob*wq*density(xq,yq)*gy

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    JxWq=np.linalg.det(jcb)*weightq
    xq=np.dot(N_V,x_V[icon_V[:,iel]])
    yq=np.dot(N_V,y_V[icon_V[:,iel]])
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]
    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    # assemble matrix A_fem and right hand side b_fem
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
            b_fem[m1]+=b_el[ikk]

print("build FE matrix & rhs: %.3f s" % (clock.time()-start))

###############################################################################
# impose boundary conditions
###############################################################################
start=clock.time()

for i in range(0,Nfem):
    if bc_fix_V[i]:
       A_ref=A_fem[i,i]
       for j in range(0,Nfem):
           b_fem[j]-=A_fem[i,j]*bc_val_V[i]
           A_fem[i,j]=0.
           A_fem[j,i]=0.
           A_fem[i,i]=A_ref
       b_fem[i]=A_ref*bc_val_V[i]

print("impose boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve linear system: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split solution: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve pressure and strain rate in middle of elements
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N_V=basis_functions_V(rq,sq)
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

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    eta[iel]=viscosity(x_e[iel],y_e[iel])
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> e (m,M) %.4f %.4f " %(np.min(e),np.max(e)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> eta (m,M) %.4f %.4f " %(np.min(eta),np.max(eta)))

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute p and strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# smoothing pressure 
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q[icon_V[0,iel]]+=p[iel]
    q[icon_V[1,iel]]+=p[iel]
    q[icon_V[2,iel]]+=p[iel]
    q[icon_V[3,iel]]+=p[iel]
    count[icon_V[0,iel]]+=1
    count[icon_V[1,iel]]+=1
    count[icon_V[2,iel]]+=1
    count[icon_V[3,iel]]+=1

q/=count

print("project press on V grid: %.3f s" % (clock.time()-start))

###############################################################################
# extract velocity field at domain bottom and on diagonal
###############################################################################
start=clock.time()

xdiag=np.zeros(nnx,dtype=np.float64)  
udiag=np.zeros(nnx,dtype=np.float64)  
udiagth=np.zeros(nnx,dtype=np.float64)  

counter=0
for i in range(0,nn_V):
    if abs(x_V[i]-y_V[i])<eps:
       xdiag[counter]=x_V[i]
       ui,vi,qi=solution(x_V[i],y_V[i]) 
       udiag[counter]=u[i]
       udiagth[counter]=ui
       counter+=1
    #end if
#end for

xbot=np.zeros(nnx,dtype=np.float64)  
qbotth=np.zeros(nnx,dtype=np.float64)  
qbot=np.zeros(nnx,dtype=np.float64)  

counter=0
for i in range(0,nn_V):
    if abs(y_V[i])<eps:
       xbot[counter]=x_V[i]
       ui,vi,qi=solution(x_V[i],y_V[i]) 
       qbot[counter]=q[i]
       qbotth[counter]=qi
       counter+=1
   #end if
#end for

np.savetxt('bottom.ascii',np.array([xbot,qbot,qbotth]).T,header='# x,q')
np.savetxt('diag.ascii',np.array([xdiag,udiag,udiagth]).T,header='# x,u')

print("export measurements: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V):
    ui,vi,pi=solution(x_V[i],y_V[i]) 
    error_u[i]=u[i]-ui
    error_v[i]=v[i]-vi

for iel in range(0,nel): 
    ui,vi,pi=solution(x_e[iel],y_e[iel]) 
    error_p[iel]=p[iel]-pi
#end for

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N_V=basis_functions_V(rq,sq)
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
            ui,vi,pi=solution(xq,yq) 
            errv+=((uq-ui)**2+(vq-vi)**2)*JxWq
            errp+=(p[iel]-pi)**2*JxWq
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.10f ; errp= %.10f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

u_temp=np.reshape(u,(nny,nnx))
v_temp=np.reshape(v,(nny,nnx))
p_temp=np.reshape(p,(nely,nelx))
e_temp=np.reshape(e,(nely,nelx))
exx_temp=np.reshape(exx,(nely,nelx))
eyy_temp=np.reshape(eyy,(nely,nelx))
exy_temp=np.reshape(exy,(nely,nelx))
error_u_temp=np.reshape(error_u,(nny,nnx))
error_v_temp=np.reshape(error_v,(nny,nnx))
error_p_temp=np.reshape(error_p,(nely,nelx))
eta_temp=np.reshape(eta,(nely,nelx))

fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(18,18))

uextent=(np.amin(x_V),np.amax(x_V),np.amin(y_V),np.amax(y_V))
pextent=(np.amin(x_e),np.amax(x_e),np.amin(y_e),np.amax(y_e))

im = axes[0][0].imshow(u_temp,extent=uextent,cmap='Spectral_r',interpolation='nearest')
axes[0][0].set_title('$v_x$', fontsize=10, y=1.01)
axes[0][0].set_xlabel('x')
axes[0][0].set_ylabel('y')
fig.colorbar(im,ax=axes[0][0])

im = axes[0][1].imshow(v_temp,extent=uextent,cmap='Spectral_r',interpolation='nearest')
axes[0][1].set_title('$v_y$', fontsize=10, y=1.01)
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
fig.colorbar(im,ax=axes[0][1])

im = axes[0][2].imshow(p_temp,extent=pextent,cmap='RdGy_r',interpolation='nearest')
axes[0][2].set_title('$p$', fontsize=10, y=1.01)
axes[0][2].set_xlim(0,Lx)
axes[0][2].set_ylim(0,Ly)
axes[0][2].set_xlabel('x')
axes[0][2].set_ylabel('y')
fig.colorbar(im,ax=axes[0][2])

im = axes[1][0].imshow(exx_temp,extent=pextent, cmap='viridis',interpolation='nearest')
axes[1][0].set_title('$\dot{\epsilon}_{xx}$',fontsize=10, y=1.01)
axes[1][0].set_xlim(0,Lx)
axes[1][0].set_ylim(0,Ly)
axes[1][0].set_xlabel('x')
axes[1][0].set_ylabel('y')
fig.colorbar(im,ax=axes[1][0])

im = axes[1][1].imshow(eyy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][1].set_title('$\dot{\epsilon}_{yy}$',fontsize=10,y=1.01)
axes[1][1].set_xlim(0,Lx)
axes[1][1].set_ylim(0,Ly)
axes[1][1].set_xlabel('x')
axes[1][1].set_ylabel('y')
fig.colorbar(im,ax=axes[1][1])

im = axes[1][2].imshow(exy_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[1][2].set_title('$\dot{\epsilon}_{xy}$',fontsize=10,y=1.01)
axes[1][2].set_xlim(0,Lx)
axes[1][2].set_ylim(0,Ly)
axes[1][2].set_xlabel('x')
axes[1][2].set_ylabel('y')
fig.colorbar(im,ax=axes[1][2])

im = axes[2][0].imshow(error_u_temp,extent=uextent,cmap='Spectral_r',interpolation='nearest')
axes[2][0].set_title('$v_x-t^{th}_x$',fontsize=10,y=1.01)
axes[2][0].set_xlabel('x')
axes[2][0].set_ylabel('y')
fig.colorbar(im,ax=axes[2][0])

im = axes[2][1].imshow(error_v_temp,extent=uextent,cmap='Spectral_r',interpolation='nearest')
axes[2][1].set_title('$v_y-t^{th}_y$',fontsize=10,y=1.01)
axes[2][1].set_xlabel('x')
axes[2][1].set_ylabel('y')
fig.colorbar(im,ax=axes[2][1])

im = axes[2][2].imshow(error_p_temp, extent=uextent, cmap='RdGy_r',interpolation='nearest')
axes[2][2].set_title('$p-p^{th}$',fontsize=10,y=1.01)
axes[2][2].set_xlabel('x')
axes[2][2].set_ylabel('y')
fig.colorbar(im,ax=axes[2][2])

im = axes[3][0].imshow(eta_temp,extent=pextent,cmap='jet',interpolation='nearest',norm=LogNorm())
axes[3][0].set_title('$\eta$',fontsize=10, y=1.01)
axes[3][0].set_xlim(0,Lx)
axes[3][0].set_ylim(0,Ly)
axes[3][0].set_xlabel('x')
axes[3][0].set_ylabel('y')
fig.colorbar(im,ax=axes[3][0])


im = axes[3][2].imshow(e_temp,extent=pextent,cmap='jet',interpolation='nearest',norm=LogNorm())
axes[3][2].set_title('$\dot{\epsilon}$',fontsize=10, y=1.01)
axes[3][2].set_xlim(0,Lx)
axes[3][2].set_ylim(0,Ly)
axes[3][2].set_xlabel('x')
axes[3][2].set_ylabel('y')
fig.colorbar(im,ax=axes[3][2])

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
