import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import time as clock

###############################################################################

order=2

def basis_functions_V(r,s,order):
    if order==1:
       N0=0.25*(1.-r)*(1.-s)
       N1=0.25*(1.+r)*(1.-s)
       N2=0.25*(1.+r)*(1.+s)
       N3=0.25*(1.-r)*(1.+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)
    if order==2:
       N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
       N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
       N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
       N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
       N4=    (1.-r**2) * 0.5*s*(s-1.)
       N5= 0.5*r*(r+1.) *    (1.-s**2)
       N6=    (1.-r**2) * 0.5*s*(s+1.)
       N7= 0.5*r*(r-1.) *    (1.-s**2)
       N8=    (1.-r**2) *    (1.-s**2)
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)
   
def basis_functions_V_dr(r,s,order):
    if order==1:
       dNdr0=-0.25*(1.-s)
       dNdr1=+0.25*(1.-s)
       dNdr2=+0.25*(1.+s)
       dNdr3=-0.25*(1.+s)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)
    if order==2:
       dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
       dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
       dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
       dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
       dNdr4=       (-2.*r) * 0.5*s*(s-1)
       dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
       dNdr6=       (-2.*r) * 0.5*s*(s+1)
       dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
       dNdr8=       (-2.*r) *   (1.-s**2)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,\
                        dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s,order):
    if order==1:
       dNds0=-0.25*(1.-r)
       dNds1=-0.25*(1.+r)
       dNds2=+0.25*(1.+r)
       dNds3=+0.25*(1.-r)
       return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)
    if order==2:
       dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
       dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
       dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
       dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
       dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
       dNds5= 0.5*r*(r+1.) *       (-2.*s)
       dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
       dNds7= 0.5*r*(r-1.) *       (-2.*s)
       dNds8=    (1.-r**2) *       (-2.*s)
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,\
                        dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s,order):
    if order==1:
       N0=1
       return np.array([N0],dtype=np.float64)
    if order==2:
       N0=0.25*(1-r)*(1-s)
       N1=0.25*(1+r)*(1-s)
       N2=0.25*(1+r)*(1+s)
       N3=0.25*(1-r)*(1+s)
       return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

def gx(x,y,ibench):
    if ibench==0: 
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if ibench==1: val=1./x
    if ibench==2: val = 1/np.cos(y)
    if ibench==3: val=-1
    if ibench==4: val=-10
    if ibench==5: val=-1
    return val

def gy(x,y,ibench):
    if ibench==0: 
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if ibench==1: val=1/y
    if ibench==2: val=1/np.cos(x)
    if ibench==3: val=0
    if ibench==4: val=0
    if ibench==5: val=0
    return val

def density(x,y,ibench):
    if ibench==0: val=1 
    if ibench==1: val=x*y
    if ibench==2: val=np.cos(x)*np.cos(y)
    if ibench==3: val=x
    if ibench==4: val=1./(1-x)
    if ibench==5: val=np.cos(x)
    return val

def u_th(x,y,ibench):
    if ibench==0: val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if ibench==1: val=1/x
    if ibench==2: val=1/np.cos(x)
    if ibench==3: val=1/x
    if ibench==4: val=1-x
    if ibench==5: val=1/np.cos(x)
    return val

def v_th(x,y,ibench):
    if ibench==0: val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if ibench==1: val=1/y
    if ibench==2: val=1/np.cos(y)
    if ibench==3: val=0
    if ibench==4: val=0
    if ibench==5: val=0
    return val

def p_th(x,y,ibench):
    if ibench==0: val=x*(1.-x)-1./6. 
    if ibench==1:
       val = -4/(3*x**2) - 4/(3*y**2) + x*y  -11./12
    if ibench==2:
       val = 4*np.sin(x)/(3*np.cos(x)**2)+4*np.sin(y)/(3*np.cos(y)**2) \
           + np.sin(x) + np.sin(y) - 2+2*np.cos(1)-8/3*(1/np.cos(1)-1) 
    if ibench==3:
       val =  -4/(3*x**2) - x**2/2 + 11/6
    if ibench==4:
       val = 10*np.log(x-1) -29.7030486691745 
    if ibench==5:
       val = 4*np.sin(x)/(3*np.cos(x)**2) - np.sin(x) -0.674723
    return val

def sr_xx_th(x,y,ibench):
    if ibench==0: val=0 
    if ibench==1: val=-1/x**2
    if ibench==2: val=np.sin(x)/np.cos(x)**2
    if ibench==3: val=-1/x**2
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

def sr_yy_th(x,y,ibench):
    if ibench==0: val=0 
    if ibench==1: val=-1/y**2
    if ibench==2: val=np.sin(y)/np.cos(y)**2
    if ibench==3: val=0 
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

def sr_xy_th(x,y,ibench):
    if ibench==0: val=0 
    if ibench==1: val=0
    if ibench==2: val=0
    if ibench==3: val=0 
    if ibench==4: val=0 
    if ibench==5: val=0 
    return val

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 023 **********")
print("*******************************")

if order==1: m_V=4 # number of V-nodes making up an element
if order==2: m_V=9 # number of V-nodes making up an element
if order==1: m_P=1 # number of P-nodes making up an element
if order==2: m_P=4 # number of P-nodes making up an element
ndof_V=2           # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 16
   nely = nelx
   visu = 1
    
nnx=order*nelx+1                    # number of nodes, x direction
nny=order*nely+1                    # number of nodes, y direction
nn_V=nnx*nny                        # number of V-nodes
if order==1: nn_P=nelx*nely         # number of P-nodes
if order==2: nn_P=(nelx+1)*(nely+1) # number of P-nodes
nel=nelx*nely                       # number of elements, total
Nfem_V=nn_V*ndof_V                  # number of V-dofs
Nfem_P=nn_P                         # number of P-dofs
Nfem=Nfem_V+Nfem_P                  # total number of dofs

hx=Lx/nelx
hy=Ly/nely

eta0=1. 

debug=False

spy=False

pnormalise=True

# Available benchmarks 
# 0 - donea & huerta
# 1 - 2D Cartesian Linear
# 2 - 2D Cartesian Sinusoidal
# 3 - 1D Cartesian Linear
# 4 - Arie van den Berg
# 5 - 1D Cartesian Sinusoidal

ibench=1

if ibench==0: offsetx=0  ; offsety=0  ; compressible=False
if ibench==1: offsetx=1  ; offsety=1  ; compressible=True
if ibench==2: offsetx=0  ; offsety=0  ; compressible=True
if ibench==3: offsetx=1  ; offsety=1  ; compressible=True
if ibench==4: offsetx=20 ; offsety=20 ; compressible=True
if ibench==5: offsetx=0  ; offsety=0  ; compressible=True

if order==1: nq_per_dim=2
if order==2: nq_per_dim=3

if nq_per_dim==2:
   qcoords=[-1/np.sqrt(3.),1/np.sqrt(3.)]
   qweights=[1.,1.]

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

###############################################################################

print('order=',order)
print('ibench=',ibench)
print('nelx=',nelx)
print('nely=',nely)
print('nnx=',nnx)
print('nny=',nny)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
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
        x_V[counter]=i*hx/order + offsetx
        y_V[counter]=j*hy/order + offsety
        counter+=1


x_P=np.zeros(nn_P,dtype=np.float64)  # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)  # y coordinates

if order==1:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           x_P[counter]=(i+0.5)*hx + offsetx
           y_P[counter]=(j+0.5)*hx + offsety
           counter+=1

if order==2:
   counter = 0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x_P[counter]=i*hx + offsetx
           y_P[counter]=j*hy + offsety
           counter+=1

if debug: np.savetxt('grid_V.ascii',np.array([x_V,y_V]).T,header='# x,y')
if debug: np.savetxt('grid_P.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

if order==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_V[0,counter]=i+j*(nelx+1)
           icon_V[1,counter]=i+1+j*(nelx+1)
           icon_V[2,counter]=i+1+(j+1)*(nelx+1)
           icon_V[3,counter]=i+(j+1)*(nelx+1)
           icon_P[0,counter]=counter
           counter += 1

if order==2:
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
           icon_P[0,counter]=i+j*(nelx+1)
           icon_P[1,counter]=i+1+j*(nelx+1)
           icon_P[2,counter]=i+1+(j+1)*(nelx+1)
           icon_P[3,counter]=i+(j+1)*(nelx+1)
           counter+=1
       #end for
   #end for

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
# check basis fcts/mesh by calculating areas
###############################################################################
start=clock.time()

jcb=np.zeros((2,2),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
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

print('     -> toral area=',np.sum(area))

print("compute areas: %.3f s" % (clock.time()-start))

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
# [ K   G ][u]=[f]
# [GT+Z 0 ][p] [h]
###############################################################################
start=clock.time()

if pnormalise and order==1:
   A_fem=lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
   b_fem=np.zeros(Nfem+1,dtype=np.float64)
else:
   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
   b_fem=np.zeros(Nfem,dtype=np.float64)

N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)  
C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64) 
    Z_el=np.zeros((m_P,m_V*ndof_V),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_V=basis_functions_V(rq,sq,order)
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
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

            #rhoq=np.dot(N_V,rho_nodal[icon_V[:,iel]])
            rhoq=density(xq,yq,ibench)

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta0*JxWq

            N_mat[0,:]=N_P[:]
            N_mat[1,:]=N_P[:]

            #if order==1:
            #   for i in range(0,m_V):
            #       G_el[ndof_V*i  ,0]-=dNdx_V[i] *JxWq 
            #       G_el[ndof_V*i+1,0]-=dNdy_V[i] *JxWq 
            #if order==2:
            G_el-=B.T.dot(N_mat)*JxWq 

            if compressible:
               drhodxq=np.dot(dNdx_V,rho_nodal[icon_V[:,iel]])
               drhodyq=np.dot(dNdy_V,rho_nodal[icon_V[:,iel]])
               #if order==1:
               #   for i in range(0,m_V):
               #       Z_el[0,ndof_V*i  ]-=N_V[i]*drhodxq/rhoq *JxWq 
               #       Z_el[0,ndof_V*i+1]-=N_V[i]*drhodyq/rhoq *JxWq 

               #if order==2:
               for i in range(0,m_V):
                   for j in range(0,m_P):
                       Z_el[j,ndof_V*i  ]-=N_P[j]*N_V[i]*drhodxq/rhoq *JxWq 
                       Z_el[j,ndof_V*i+1]-=N_P[j]*N_V[i]*drhodyq/rhoq *JxWq 

            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*rhoq*gx(xq,yq,ibench) *JxWq 
                f_el[ndof_V*i+1]+=N_V[i]*rhoq*gy(xq,yq,ibench) *JxWq 

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
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               h_el[:]-=Z_el[:,ikk]*bc_val_V[m1]
               G_el[ikk,:]=0
               Z_el[:,ikk]=0
            #end if 
        #end for 
    #end for 

    # assemble matrix and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            m1=ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2+i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,m_P):
                jkk=k2
                m2=icon_P[k2,iel]
                A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                A_fem[Nfem_V+m2,m1]+=Z_el[jkk,ikk]
            #end for 
            b_fem[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        b_fem[Nfem_V+m2]+=h_el[k2]
    #end for 

#end for iel

if pnormalise and order==1:
   A_fem[Nfem,Nfem_V:Nfem]=1
   A_fem[Nfem_V:Nfem,Nfem]=1

if spy:
   import matplotlib.pyplot as plt
   plt.spy(A_fem)
   plt.savefig('matrix_u.pdf', bbox_inches='tight')

print("build FE matrix: %.3f s" % (clock.time()-start))

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

if pnormalise and order==1:
   print("     -> Lagrange multiplier: %.4es" % sol[Nfem])


if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('p_raw.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# normalise pressure
###############################################################################
start=clock.time()

if True:
   int_p=0
   for iel in range(0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_P=basis_functions_P(rq,sq,order)
               pq=np.dot(N_P,p[icon_P[:,iel]])
               dNdr_V=basis_functions_V_dr(rq,sq,order)
               dNds_V=basis_functions_V_ds(rq,sq,order)
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               JxWq=np.linalg.det(jcb)*weightq
               int_p+=pq*JxWq

   avrg_p=int_p/(Lx*Ly)

   print("     -> int_p %e " %(int_p))
   print("     -> avrg_p %e " %(avrg_p))

   p[:]-=avrg_p

print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if debug: np.savetxt('p_normalised.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental strainrate 
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
    N_V=basis_functions_V(rq,sq,order)
    dNdr_V=basis_functions_V_dr(rq,sq,order)
    dNds_V=basis_functions_V_ds(rq,sq,order)
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
    exy[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
            +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)
    exx_th[iel]=sr_xx_th(x_e[iel],y_e[iel],ibench)
    eyy_th[iel]=sr_yy_th(x_e[iel],y_e[iel],ibench)
    exy_th[iel]=sr_xy_th(x_e[iel],y_e[iel],ibench)
    press_th[iel]=p_th(x_e[iel],y_e[iel],ibench)

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal pressure
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

if order==1:
   for iel in range(0,nel):
       for k in range(0,m_V):
           q[icon_V[k,iel]]+=p[iel]
           count[icon_V[k,iel]]+=1

   q/=count

if order==2:
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

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

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
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq,order)
            N_P=basis_functions_P(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
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
            qq=np.dot(N_V,q[icon_V[:,iel]])
            errv+=((uq-u_th(xq,yq,ibench))**2+(vq-v_th(xq,yq,ibench))**2)*JxWq
            errp+=(pq-p_th(xq,yq,ibench))**2*JxWq
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
if order==1:
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

vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='vel (error)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(error_u[i],error_v[i],0.))
vtufile.write("</DataArray>\n")


vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='gravity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(gx_nodal[i],gy_nodal[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32'  Name='rho' Format='ascii'> \n")
rho_nodal.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
if order==1:
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))

if order==2:
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                      icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                      icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*m_V))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
if order==1:
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
if order==2:
   for iel in range (0,nel):
       vtufile.write("%d \n" %28)
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
