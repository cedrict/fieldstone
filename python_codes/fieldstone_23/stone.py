import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as clock 
import matplotlib.pyplot as plt

###############################################################################

def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

def gx(x,y,ibench):
    if ibench==1: val=1/y
    if ibench==2: val = 1/np.cos(y)
    if ibench==3: val=-1
    if ibench==4: val=10
    if ibench==5: val=-1
    return val

def gy(x,y,ibench):
    if ibench==1: val=1/x
    if ibench==2: val=1/np.cos(x)
    if ibench==3: val=0
    if ibench==4: val=0
    if ibench==5: val=0
    return val

def density(x,y,ibench):
    if ibench==1: val=x*y
    if ibench==2: val = np.cos(x)*np.cos(y)
    if ibench==3: val=x
    if ibench==4: val = 1/(1-x)
    if ibench==5: val=np.cos(x)
    return val

def u_th(x,y,ibench):
    if ibench==1: val=1/x
    if ibench==2: val=1/np.cos(x)
    if ibench==3: val=1/x
    if ibench==4: val=1-x
    if ibench==5: val=1/np.cos(x)
    return val

def v_th(x,y,ibench):
    if ibench==1: val=1/y
    if ibench==2: val=1/np.cos(y)
    if ibench==3: val=0
    if ibench==4: val=0
    if ibench==5: val=0
    return val

def p_th(x,y,ibench):
    if ibench==1:
       val = -4/(3*x**2) - 4/(3*y**2) + x**2/2 + y**2/2  -1
    if ibench==2:
       val = 4*np.sin(x)/(3*np.cos(x)**2)+4*np.sin(y)/(3*np.cos(y)**2) \
           + np.sin(x) + np.sin(y) - 2+2*np.cos(1)-8/3*(1/np.cos(1)-1) 
    if ibench==3:
       val =  -4/(3*x**2) - x**2/2 + 11/6
    if ibench==4:
       val = 10*log(x-1) +290.9 
    if ibench==5:
       val = 4*np.sin(x)/(3*np.cos(x)**2) - np.sin(x) -0.674723
    return val

###############################################################################

def sr_xx_th(x,y,ibench):
    if ibench==1: val=-1/x**2
    if ibench==2: val=np.sin(x)/np.cos(x)**2
    return val

def sr_yy_th(x,y,ibench):
    if ibench==1: val=-1/y**2
    if ibench==2: val=np.sin(y)/np.cos(y)**2
    return val

def sr_xy_th(x,y,ibench):
    if ibench==1: val=0
    if ibench==2: val=0
    return val

###############################################################################

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=6, y=1.01)
    if (limitX != 0.0):
       axes[plotX][plotY].set_xlim(0,limitX)
    if (limitY != 0.0):
       axes[plotX][plotY].set_ylim(0,limitY)
    axes[plotX][plotY].set_xlabel(labelX)
    axes[plotX][plotY].set_ylabel(labelY)
    fig.colorbar(im,ax=axes[plotX][plotY])
    return

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

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
    
nnx=nelx+1         # number of nodes, x direction
nny=nely+1         # number of nodes, y direction
nn_V=nnx*nny       # number of nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nel         # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

hx=Lx/nelx
hy=Ly/nely

eta0=1. 

debug=False

# Available benchmarks 
# 1 - 2D Cartesian Linear
# 2 - 2D Cartesian Sinusoidal
# 3 - 1D Cartesian Linear
# 4 - Arie van den Berg
# 5 - 1D Cartesian Sinusoidal

ibench=1

if ibench==1:
   offsetx=1 
   offsety=1
if ibench==2:
   offsetx=0 
   offsety=0
if ibench==3:
   offsetx=1 
   offsety=1
if ibench==4:
   offsetx=20 
   offsety=20
if ibench==5:
   offsetx=0 
   offsety=0

pnormalise=True

write_blocks=False

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx + offsetx
        y_V[counter]=j*hy + offsety
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
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx + 1)
        icon_V[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]-offsetx<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_th(x_V[i],y_V[i],ibench)
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_th(x_V[i],y_V[i],ibench)
    if x_V[i]-offsetx>(Lx-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_th(x_V[i],y_V[i],ibench)
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_th(x_V[i],y_V[i],ibench)
    if y_V[i]-offsety<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_th(x_V[i],y_V[i],ibench)
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_th(x_V[i],y_V[i],ibench)
    if y_V[i]-offsety>(Ly-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=u_th(x_V[i],y_V[i],ibench)
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=v_th(x_V[i],y_V[i],ibench)

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# density setup
###############################################################################
start=clock.time()

rho_nodal=np.zeros(nn_V,dtype=np.float64)
gx_nodal=np.zeros(nn_V,dtype=np.float64)
gy_nodal=np.zeros(nn_V,dtype=np.float64)

for ip in range(0,nn_V):
    rho_nodal[ip]=density(x_V[ip],y_V[ip],ibench)
    gx_nodal[ip]=gx(x_V[ip],y_V[ip],ibench)
    gy_nodal[ip]=gy(x_V[ip],y_V[ip],ibench)

print("setup: rho,gx,gy nodal fields: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
f_rhs=np.zeros(Nfem_V,dtype=np.float64)          # right hand side f 
h_rhs=np.zeros(Nfem_P,dtype=np.float64)          # right hand side h 
Z_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix Z
B= np.zeros((3,ndof_V*m_V),dtype=np.float64)     # gradient matrix B 
C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    Z_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
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
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            #uq=np.dot(N_V,u[icon_V[:,iel]])
            #vq=np.dot(N_V,v[icon_V[:,iel]])
            rhoq=np.dot(N_V,rho_nodal[icon_V[:,iel]])

            drhodxq=np.dot(dNdx_V,rho_nodal[icon_V[:,iel]])
            drhodyq=np.dot(dNdy_V,rho_nodal[icon_V[:,iel]])

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta0*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*density(xq,yq,ibench)*gx(xq,yq,ibench) *JxWq 
                f_el[ndof_V*i+1]+=N_V[i]*density(xq,yq,ibench)*gy(xq,yq,ibench) *JxWq 
                G_el[ndof_V*i  ,0]-=dNdx_V[i] *JxWq 
                G_el[ndof_V*i+1,0]-=dNdy_V[i] *JxWq 
                Z_el[ndof_V*i  ,0]-=N_V[i]*drhodxq/rhoq *JxWq 
                Z_el[ndof_V*i+1,0]-=N_V[i]*drhodyq/rhoq *JxWq 

            h_el[0]+=0#(uq*drhodxq+vq*drhodyq)/rhoq*wq*jcob

        # end for jq
    # end for iq

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[0]-=G_el[ikk,0]*bc_val_V[m1]
               h_el[0]-=Z_el[ikk,0]*bc_val_V[m1]
               G_el[ikk,0]=0
               Z_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
            Z_mat[m1,iel]+=Z_el[ikk,0]
    h_rhs[iel]+=h_el[0]

print("     -> K (m,M) %.5f %.5f " %(np.min(K_mat),np.max(K_mat)))
print("     -> G (m,M) %.5f %.5f " %(np.min(G_mat),np.max(G_mat)))
print("     -> Z (m,M) %.5f %.5f " %(np.min(Z_mat),np.max(Z_mat)))
print("     -> f (m,M) %.5f %.5f " %(np.min(f_rhs),np.max(f_rhs)))
print("     -> h (m,M) %.5f %.5f " %(np.min(h_rhs),np.max(h_rhs)))

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
start=clock.time()

if write_blocks:

   f = open("K_mat.ascii","w")
   for i in range(0,Nfem_V):
       for j in range(0,Nfem_V):
           if K_mat[i,j]!=0:
              f.write("%i %i %10.6f\n" % (i,j,K_mat[i,j]))

   f = open("G_mat.ascii","w")
   for i in range(0,Nfem_V):
       for j in range(0,Nfem_P):
           if G_mat[i,j]!=0:
              f.write("%i %i %10.6f\n" % (i,j,G_mat[i,j]))

   f = open("f_rhs.ascii","w")
   for i in range(0,Nfem_V):
       f.write("%i %10.6f\n" % (i,f_rhs[i]))

   f = open("h_rhs.ascii","w")
   for i in range(0,Nfem_P):
       f.write("%i %10.6f\n" % (i,h_rhs[i]))

print("write blocks to file: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if pnormalise:
   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T+Z_mat.T
   a_mat[Nfem,Nfem_V:Nfem]=1
   a_mat[Nfem_V:Nfem,Nfem]=1
else:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T+Z_mat.T

rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

if pnormalise:
   print("     -> Lagrange multiplier: %.4es" % sol[Nfem])

if debug:
   np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  
exx_th=np.zeros(nel,dtype=np.float64)  
eyy_th=np.zeros(nel,dtype=np.float64)  
exy_th=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
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
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
             np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)
    exx_th[iel] = sr_xx_th(xc[iel],yc[iel],ibench)
    eyy_th[iel] = sr_yy_th(xc[iel],yc[iel],ibench)
    exy_th[iel] = sr_xy_th(xc[iel],yc[iel],ibench)

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal pressure
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    for k in range(0,m_V):
        q[icon_V[k,iel]]+=p[iel]
        count[icon_V[k,iel]]+=1

q/=count

if debug:
   np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("compute nodal pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute error fields for plotting
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_q=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=u[i]-u_th(x_V[i],y_V[i],ibench)
    error_v[i]=v[i]-v_th(x_V[i],y_V[i],ibench)
    error_q[i]=q[i]-p_th(x_V[i],y_V[i],ibench)

for iel in range(0,nel): 
    error_p[iel]=p[iel]-p_th(xc[iel],yc[iel],ibench)

print("compute error fields: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

errv=0.
errp=0.
errq=0.
for iel in range(0,nel):
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            qq=np.dot(N_V,q[icon_V[:,iel]])
            errv+=((uq-u_th(xq,yq,ibench))**2+(vq-v_th(xq,yq,ibench))**2)*JxWq
            errp+=(p[iel]-p_th(xq,yq,ibench))**2*JxWq
            errq+=(qq-p_th(xq,yq,ibench))**2*JxWq

errv=np.sqrt(errv)
errq=np.sqrt(errq)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %e ; errp= %e ; errq= %e" %(nel,errv,errp,errq))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

u_temp=np.reshape(u,(nny,nnx))
v_temp=np.reshape(v,(nny,nnx))
q_temp=np.reshape(q,(nny,nnx))
p_temp=np.reshape(p,(nely,nelx))
e_temp=np.reshape(e,(nely,nelx))
exx_temp=np.reshape(exx,(nely,nelx))
eyy_temp=np.reshape(eyy,(nely,nelx))
exy_temp=np.reshape(exy,(nely,nelx))
error_u_temp=np.reshape(error_u,(nny,nnx))
error_v_temp=np.reshape(error_v,(nny,nnx))
error_q_temp=np.reshape(error_q,(nny,nnx))
error_p_temp=np.reshape(error_p,(nely,nelx))
rho_temp=np.reshape(rho_nodal,(nny,nnx))
gx_temp=np.reshape(gx_nodal,(nny,nnx))
gy_temp=np.reshape(gy_nodal,(nny,nnx))
exx_th_temp=np.reshape(exx_th,(nely,nelx))
eyy_th_temp=np.reshape(eyy_th,(nely,nelx))
exy_th_temp=np.reshape(exy_th,(nely,nelx))

SMALL_SIZE = 6
MEDIUM_SIZE = 6
BIGGER_SIZE = 6
plt.rc('font',size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  

fig,axes=plt.subplots(nrows=5,ncols=4,figsize=(18,18))

uextent=(np.amin(x_V),np.amax(x_V),np.amin(y_V),np.amax(y_V))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

onePlot(u_temp,      0,0, "$v_x$",                "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(v_temp,      0,1, "$v_y$",                "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(p_temp,      0,2, "$p$",                  "x", "y", pextent, 0, 0, 'RdGy_r')
onePlot(q_temp,      0,3, "$q$",                  "x", "y", pextent, 0, 0, 'RdGy_r')
onePlot(error_u_temp,1,0, "$v_x-t^{th}_x$",       "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(error_v_temp,1,1, "$v_y-t^{th}_y$",       "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(error_p_temp,1,2, "$p-p^{th}$",           "x", "y", uextent, 0, 0, 'RdGy_r')
onePlot(error_q_temp,1,3, "$q-p^{th}$",           "x", "y", uextent, 0, 0, 'RdGy_r')
onePlot(exx_temp,    2,0, "$\dot{\epsilon}_{xx}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(eyy_temp,    2,1, "$\dot{\epsilon}_{yy}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(exy_temp,    2,2, "$\dot{\epsilon}_{xy}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(e_temp,      2,3, "$\dot{\epsilon}$",     "x", "y", pextent, 0, 0, 'viridis')
onePlot(exx_th_temp, 3,0, "$\dot{\epsilon}_{xx}^{th}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(eyy_th_temp, 3,1, "$\dot{\epsilon}_{yy}^{th}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(exy_th_temp, 3,2, "$\dot{\epsilon}_{xy}^{th}$","x", "y", pextent, 0, 0, 'viridis')
onePlot(rho_temp,    4,0, "density",              "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(gx_temp,     4,1, "$g_x$",                "x", "y", uextent, 0, 0, 'Spectral_r')
onePlot(gy_temp,     4,2, "$g_y$",                "x", "y", uextent, 0, 0, 'Spectral_r')

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
