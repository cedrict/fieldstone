import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock 
from scipy.sparse import lil_matrix
import sys as sys
import matplotlib.pyplot as plt

debug=False

bench=1

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

def basis_functions_u(r,s):
    N0=0.25*(1.-r)*(1.-s) 
    N1=0.25*(1.+r)*(1.-s) 
    N2=0.25*(1.+r)*(1.+s) 
    N3=0.25*(1.-r)*(1.+s) 
    N4=r*(1-r**2)*(1-s**2)
    return np.array([N0,N1,N2,N3,N4],dtype=np.float64)

def basis_functions_u_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    dNdr4=(1-3*r**2)*(1-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4],dtype=np.float64)

def basis_functions_u_ds(r,s):
    dNds0=-0.25*(1.-r) 
    dNds1=-0.25*(1.+r) 
    dNds2=+0.25*(1.+r) 
    dNds3=+0.25*(1.-r) 
    dNds4=r*(1-r**2)*(-2*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4],dtype=np.float64)

###############################################################################

def basis_functions_v(r,s):
    N0=0.25*(1.-r)*(1.-s) 
    N1=0.25*(1.+r)*(1.-s) 
    N2=0.25*(1.+r)*(1.+s) 
    N3=0.25*(1.-r)*(1.+s) 
    N4=s*(1-r**2)*(1-s**2)
    return np.array([N0,N1,N2,N3,N4],dtype=np.float64)

def basis_functions_v_dr(r,s):
    dNdr0=-0.25*(1.-s)
    dNdr1=+0.25*(1.-s)
    dNdr2=+0.25*(1.+s)
    dNdr3=-0.25*(1.+s)
    dNdr4=s*(-2*r)*(1-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4],dtype=np.float64)

def basis_functions_v_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    dNds4=(1-r**2)*(1-3*s**2)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4],dtype=np.float64)

###############################################################################

def bx(x, y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2:
       val=0
    return val

def by(x, y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2:
       val=0
    return val

###############################################################################

def velocity_x(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2:
       val=20*x*y**3
    return val

def velocity_y(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2:
       val=5*x**4-5*y**4
    return val

def pressure(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    if bench==2:
       val=60*x**2*y-20*y**3-5
    return val

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 162 **********")
print("*******************************")

ndim=2
m_V=5    # number of nodes making up an element 4+1
ndof_V=2 # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 16
   nely = 16
   visu = 1
    
nel=nelx*nely             # number of elements, total
nnx=nelx+1                # number of nodes, x direction
nny=nely+1                # number of nodes, y direction
nn_V=nnx*nny+nel          # number of nodes
Nfem_V=nnx*nny*ndof_V+nel # number of velocity dofs
Nfem_P=nel                # number of pressure dofs
Nfem=Nfem_V+Nfem_P        # total number of dofs

eta=1.  # dynamic viscosity 

Gscaling=eta/(Ly/nely)

hx=Lx/nelx
hy=Ly/nely

###############################################################################

nq_per_dim_K=3
nq_per_dim_G=2

if nq_per_dim_K==1:
   qcoordsK=[0.]
   qweightsK=[2.]

if nq_per_dim_K==2:
   qcoordsK=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweightsK=[1.,1.]

if nq_per_dim_K==3:
   qcoordsK=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweightsK=[5./9.,8./9.,5./9.]

if nq_per_dim_G==1:
   qcoordsG=[0.]
   qweightsG=[2.]

if nq_per_dim_G==2:
   qcoordsG=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweightsG=[1.,1.]

if nq_per_dim_G==3:
   qcoordsG=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweightsG=[5./9.,8./9.,5./9.]

###############################################################################

print('nelx=',nelx)
print('nely=',nely)
print('nn_V=',nn_V)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print('Nfem=',Nfem)
print('nq_per_dim_K',nq_per_dim_K)
print('nq_per_dim_G',nq_per_dim_G)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx
        y_V[counter]=j*hy
        counter+=1

counter=nnx*nny
for j in range(0,nely):
    for i in range(0,nelx):
        x_V[counter]=(i+0.5)*hx
        y_V[counter]=(j+0.5)*hy
        counter+=1

if debug: np.savetxt('grid.ascii',np.array([x_V,y_V]).T,header='# x,y')

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
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        icon_V[4,counter]=nnx*nny+counter
        counter+=1
    #end for
#end for

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 1",icon_V[0,iel],"at pos.",x_V[icon_V[0,iel]], y_V[icon_V[0,iel]])
       print ("node 2",icon_V[1,iel],"at pos.",x_V[icon_V[1,iel]], y_V[icon_V[1,iel]])
       print ("node 3",icon_V[2,iel],"at pos.",x_V[icon_V[2,iel]], y_V[icon_V[2,iel]])
       print ("node 4",icon_V[3,iel],"at pos.",x_V[icon_V[3,iel]], y_V[icon_V[3,iel]])
       print ("node 5",icon_V[4,iel],"at pos.",x_V[icon_V[4,iel]], y_V[icon_V[4,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nnx*nny):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=velocity_x(x_V[i],y_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=velocity_y(x_V[i],y_V[i])
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute element center coordinate 
###############################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    x_e[iel]=0.5*(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]])
    y_e[iel]=0.5*(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]])

print("compute elements center: %.3f s" % (clock.time()-start))

###############################################################################
# sanity check / compute areas 
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim_K):
        for jq in range(0,nq_per_dim_K):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[0:4,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[0:4,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[0:4,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[0:4,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("compute area: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

A_fem = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
rhs   = np.zeros(Nfem,dtype=np.float64)  # right hand side 
B=np.zeros((3,9),dtype=np.float64) # gradient matrix B size=3x9!
Nu    = np.zeros(m_V,dtype=np.float64)    # shape functions
Nv    = np.zeros(m_V,dtype=np.float64)    # shape functions
dNudx = np.zeros(m_V,dtype=np.float64)    # shape functions derivatives
dNudy = np.zeros(m_V,dtype=np.float64)    # shape functions derivatives
dNvdx = np.zeros(m_V,dtype=np.float64)    # shape functions derivatives
dNvdy = np.zeros(m_V,dtype=np.float64)    # shape functions derivatives
C= np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64)  #b

jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx ; jcbi[0,1]=0    
jcbi[1,0]=0    ; jcbi[1,1]=2/hy 
jcob=hx*hy/4

for iel in range(0,nel):

    K_el=np.zeros((9,9),dtype=np.float64)
    G_el=np.zeros((9,1),dtype=np.float64)
    f_el=np.zeros((9),dtype=np.float64)
    h_el=np.zeros((1),dtype=np.float64)

    for iq in range(0,nq_per_dim_K):
        for jq in range(0,nq_per_dim_K):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]
            JxWq=jcob*weightq

            Nu=basis_functions_u(rq,sq)
            Nv=basis_functions_v(rq,sq)
            dNudr=basis_functions_u_dr(rq,sq)
            dNuds=basis_functions_u_ds(rq,sq)
            dNvdr=basis_functions_v_dr(rq,sq)
            dNvds=basis_functions_v_ds(rq,sq)

            N_V=basis_functions_V(rq,sq)
            xq=np.dot(N_V[0:4],x_V[icon_V[0:4,iel]])
            yq=np.dot(N_V[0:4],y_V[icon_V[0:4,iel]])

            dNudx=jcbi[0,0]*dNudr+jcbi[0,1]*dNuds
            dNudy=jcbi[1,0]*dNudr+jcbi[1,1]*dNuds
            dNvdx=jcbi[0,0]*dNvdr+jcbi[0,1]*dNvds
            dNvdy=jcbi[1,0]*dNvdr+jcbi[1,1]*dNvds

            B[0,:]=[dNudx[0],0       ,dNudx[1],       0,dNudx[2],       0,dNudx[3],       0,dNudx[4]         ]
            B[1,:]=[       0,dNvdy[0],       0,dNvdy[1],       0,dNvdy[2],       0,dNvdy[3],dNvdy[4]         ]
            B[2,:]=[dNudy[0],dNvdx[0],dNudy[1],dNvdx[1],dNudy[2],dNvdx[2],dNudy[3],dNvdx[3],dNudy[4]+dNvdx[4]]

            K_el+=B.T.dot(C.dot(B))*eta*JxWq

            f_el[0]+=Nu[0]*bx(xq,yq)*JxWq
            f_el[1]+=Nv[0]*by(xq,yq)*JxWq
            f_el[2]+=Nu[1]*bx(xq,yq)*JxWq
            f_el[3]+=Nv[1]*by(xq,yq)*JxWq
            f_el[4]+=Nu[2]*bx(xq,yq)*JxWq
            f_el[5]+=Nv[2]*by(xq,yq)*JxWq
            f_el[6]+=Nu[3]*bx(xq,yq)*JxWq
            f_el[7]+=Nv[3]*by(xq,yq)*JxWq
            f_el[8]+=Nu[4]*bx(xq,yq)*JxWq\
                    +Nv[4]*by(xq,yq)*JxWq

        #end for jq
    #end for iq

    for iq in range(0,nq_per_dim_G):
        for jq in range(0,nq_per_dim_G):
            rq=qcoordsG[iq]
            sq=qcoordsG[jq]
            weightq=qweightsG[iq]*qweightsG[jq]
            JxWq=jcob*weightq

            Nu=basis_functions_u(rq,sq)
            Nv=basis_functions_v(rq,sq)
            dNudr=basis_functions_u_dr(rq,sq)
            dNuds=basis_functions_u_ds(rq,sq)
            dNvdr=basis_functions_v_dr(rq,sq)
            dNvds=basis_functions_v_ds(rq,sq)

            dNudx=jcbi[0,0]*dNudr+jcbi[0,1]*dNuds
            dNudy=jcbi[1,0]*dNudr+jcbi[1,1]*dNuds
            dNvdx=jcbi[0,0]*dNvdr+jcbi[0,1]*dNvds
            dNvdy=jcbi[1,0]*dNvdr+jcbi[1,1]*dNvds

            G_el[0,0]-=dNudx[0]*JxWq
            G_el[1,0]-=dNvdy[0]*JxWq
            G_el[2,0]-=dNudx[1]*JxWq
            G_el[3,0]-=dNvdy[1]*JxWq
            G_el[4,0]-=dNudx[2]*JxWq
            G_el[5,0]-=dNvdy[2]*JxWq
            G_el[6,0]-=dNudx[3]*JxWq
            G_el[7,0]-=dNvdy[3]*JxWq
            G_el[8,0]-=(dNudx[4]+dNvdy[4])*JxWq

        #end for jq
    #end for iq

    G_el*=Gscaling

    # impose b.c. 
    for k1 in range(0,4):          
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,9):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[0]-=G_el[ikk,0]*bc_val_V[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for ikk in range(0,9):
        match(ikk):
            case(0): m1=ndof_V*icon_V[0,iel]+0
            case(1): m1=ndof_V*icon_V[0,iel]+1
            case(2): m1=ndof_V*icon_V[1,iel]+0
            case(3): m1=ndof_V*icon_V[1,iel]+1
            case(4): m1=ndof_V*icon_V[2,iel]+0
            case(5): m1=ndof_V*icon_V[2,iel]+1
            case(6): m1=ndof_V*icon_V[3,iel]+0
            case(7): m1=ndof_V*icon_V[3,iel]+1
            case(8): m1=2*nnx*nny+iel
            #end match
        for jkk in range(0,9):
            match(jkk):
                case(0): m2=ndof_V*icon_V[0,iel]+0
                case(1): m2=ndof_V*icon_V[0,iel]+1
                case(2): m2=ndof_V*icon_V[1,iel]+0
                case(3): m2=ndof_V*icon_V[1,iel]+1
                case(4): m2=ndof_V*icon_V[2,iel]+0
                case(5): m2=ndof_V*icon_V[2,iel]+1
                case(6): m2=ndof_V*icon_V[3,iel]+0
                case(7): m2=ndof_V*icon_V[3,iel]+1
                case(8): m2=2*nnx*nny+iel
            #end match
            A_fem[m1,m2]+=K_el[ikk,jkk]
        #end for jkk 
        rhs[m1]+=f_el[ikk]
        A_fem[m1,Nfem_V+iel]+=G_el[ikk,0]
        A_fem[Nfem_V+iel,m1]+=G_el[ikk,0]
    #end for ikk 
    rhs[Nfem_V+iel]+=h_el[0]

#plt.spy(A_fem, markersize=0.6)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem.tocsr(),rhs)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:2*nnx*nny],(nnx*nny,2)).T
a=sol[2*nnx*nny:Nfem_V]
p=sol[Nfem_V:Nfem]*Gscaling

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
print("     -> a (m,M) %e %e " %(np.min(a),np.max(a)))
print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))

if debug:
   np.savetxt('velocity.ascii',np.array([x_V[0:nnx*nny],y_V[0:nnx*nny],u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')

print("split solution vector: %.3f s" % (clock.time()-start))

###############################################################################
# compute velocity field also in the middle of elt 
###############################################################################
start=clock.time()

r_V=[-1,1,1,-1,0]
s_V=[-1,-1,1,1,0]

uu=np.zeros(nn_V,dtype=np.float64)  
vv=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    for k in range(0,m_V):
        Nu=basis_functions_u(r_V[k],s_V[k])
        Nv=basis_functions_v(r_V[k],s_V[k])

        uu[icon_V[k,iel]]+=Nu[0]*u[icon_V[0,iel]]+\
                           Nu[1]*u[icon_V[1,iel]]+\
                           Nu[2]*u[icon_V[2,iel]]+\
                           Nu[3]*u[icon_V[3,iel]]+\
                           Nu[4]*a[iel]

        vv[icon_V[k,iel]]+=Nv[0]*v[icon_V[0,iel]]+\
                           Nv[1]*v[icon_V[1,iel]]+\
                           Nv[2]*v[icon_V[2,iel]]+\
                           Nv[3]*v[icon_V[3,iel]]+\
                           Nv[4]*a[iel]

        count[icon_V[k,iel]]+=1

uu/=count
vv/=count

if debug: np.savetxt('velocity2.ascii',np.array([x_V,y_V,uu,vv,count]).T,header='# x,y,q,count')

print("     -> uu (m,M) %e %e " %(np.min(uu),np.max(uu)))
print("     -> vv (m,M) %e %e " %(np.min(vv),np.max(vv)))

print("compute full velocity solution: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal pressure q
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    for k in range(0,m_V):
        q[icon_V[k,iel]]+=p[iel]
        count[icon_V[k,iel]]+=1

q/=count

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q,count]).T,header='# x,y,q,count')

print("compute nodal pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_q=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=uu[i]-velocity_x(x_V[i],y_V[i])
    error_v[i]=vv[i]-velocity_y(x_V[i],y_V[i])
    error_q[i]=q[i]-pressure(x_V[i],y_V[i])

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(x_e[i],y_e[i])

errv=0.
errp=0.
for iel in range(0,nel):

    for iq in range(0,nq_per_dim_K):
        for jq in range(0,nq_per_dim_K):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]
            JxWq=jcob*weightq

            Nu=basis_functions_u(rq,sq)
            Nv=basis_functions_v(rq,sq)
            dNudr=basis_functions_u_dr(rq,sq)
            dNuds=basis_functions_u_ds(rq,sq)
            dNvdr=basis_functions_v_dr(rq,sq)
            dNvds=basis_functions_v_ds(rq,sq)

            xq=np.dot(N_V[0:4],x_V[icon_V[0:4,iel]])
            yq=np.dot(N_V[0:4],y_V[icon_V[0:4,iel]])

            uq=Nu[0]*u[icon_V[0,iel]]+\
               Nu[1]*u[icon_V[1,iel]]+\
               Nu[2]*u[icon_V[2,iel]]+\
               Nu[3]*u[icon_V[3,iel]]+\
               Nu[4]*a[iel]

            vq=Nv[0]*v[icon_V[0,iel]]+\
               Nv[1]*v[icon_V[1,iel]]+\
               Nv[2]*v[icon_V[2,iel]]+\
               Nv[3]*v[icon_V[3,iel]]+\
               Nv[4]*a[iel]

            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq))**2*JxWq
        #end jq
    #end iq
#end iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution export to vtu format
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel*4,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(x_V[icon_V[i,iel]],y_V[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(u[icon_V[i,iel]],v[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(error_u[icon_V[i,iel]],error_v[icon_V[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" % q[icon_V[i,iel]])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" %p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" %error_p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   counter=0
   for iel in range(0,nel):
       vtufile.write("%d %d %d %d \n" %(counter,counter+1,counter+2,counter+3))
       counter+=4
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
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
