import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import time as clock

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
       val = -4/(3*x**2) - 4/(3*y**2) + x*y  -11./12
    if ibench==2:
       val = 4*np.sin(x)/(3*np.cos(x)**2)+4*np.sin(y)/(3*np.cos(y)**2) \
           + np.sin(x) + np.sin(y) - 2+2*np.cos(1)-8/3*(1/np.cos(1)-1) 
    if ibench==3:
       val =  -4/(3*x**2) - x**2/2 + 11/6
    if ibench==4:
       val = 10*np.log(x-1) +290.9 
    if ibench==5:
       val = 4*np.sin(x)/(3*np.cos(x)**2) - np.sin(x) -0.674723
    return val

def sr_xx_th(x,y,ibench):
    if ibench==1: val=-1/x**2
    if ibench==2: val=np.sin(x)/np.cos(x)**2
    if ibench==3: val=-1/x**2
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

def sr_yy_th(x,y,ibench):
    if ibench==1: val=-1/y**2
    if ibench==2: val=np.sin(y)/np.cos(y)**2
    if ibench==3: val=0 
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

def sr_xy_th(x,y,ibench):
    if ibench==1: val=0
    if ibench==2: val=0
    if ibench==3: val=0 
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 023 **********")
print("*******************************")

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
   nelx = 17
   nely = nelx
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

print('ibench=',ibench)
print('nelx=',nelx)
print('nely=',nely)
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

    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    Z_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

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

            h_el[0,0]+=0#(uq*drhodxq+vq*drhodyq)/rhoq*wq*jcob

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
               h_el[0,0]-=G_el[ikk,0]*bc_val_V[m1]
               h_el[0,0]-=Z_el[ikk,0]*bc_val_V[m1]
               G_el[ikk,0]=0
               Z_el[ikk,0]=0
            #end if
        #end for
    #end for

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
                #end for
            #end for
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
            Z_mat[m1,iel]+=Z_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0,0]

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
   A_fem=np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   b_fem=np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T+Z_mat.T
   A_fem[Nfem,Nfem_V:Nfem]=1
   A_fem[Nfem_V:Nfem,Nfem]=1
else:
   A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T+Z_mat.T

b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

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

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  
exx_th=np.zeros(nel,dtype=np.float64)  
eyy_th=np.zeros(nel,dtype=np.float64)  
exy_th=np.zeros(nel,dtype=np.float64)  
press_th=np.zeros(nel,dtype=np.float64)  

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
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)
    exx_th[iel]=sr_xx_th(x_e[iel],y_e[iel],ibench)
    eyy_th[iel]=sr_yy_th(x_e[iel],y_e[iel],ibench)
    exy_th[iel]=sr_xy_th(x_e[iel],y_e[iel],ibench)
    press_th[iel]=p_th(x_e[iel],y_e[iel],ibench)

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug:
   np.savetxt('p.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

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

if debug: np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

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
    error_p[iel]=p[iel]-p_th(x_e[iel],y_e[iel],ibench)

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
# export to vtu 
###############################################################################
start=clock.time()

vtufile=open("solution.vtu","w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%.4e %.4e %.1e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
p.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
press_th.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
exx.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
eyy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
exy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx (th)' Format='ascii'> \n")
exx_th.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyy (th)' Format='ascii'> \n")
eyy_th.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy (th)' Format='ascii'> \n")
exy_th.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
e.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (th)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u_th(x_V[i],y_V[i],ibench),v_th(x_V[i],y_V[i],ibench),0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(gx_nodal[i],gy_nodal[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32'  Name='rho' Format='ascii'> \n")
rho_nodal.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*4))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %9)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
