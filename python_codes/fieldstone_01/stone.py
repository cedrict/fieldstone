import numpy as np
import sys as sys
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as clock 
import matplotlib.pyplot as plt

###############################################################################
# bx and by are the body force components

def bx(x,y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x,y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

###############################################################################
# analytical solution

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
    return val

###############################################################################

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)

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

print("*******************************")
print("********** stone 001 **********")
print("*******************************")

# declare variables
print("variable declaration")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 16
   nely = 16
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nn_V=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

penalty=1.e7  # penalty coefficient value

viscosity=1.  # dynamic viscosity \eta

Nfem_V=nn_V*ndof_V # number of velocity degrees for freedom
Nfem=Nfem_V        # Total number of degrees of freedom

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*Lx/float(nelx)
        y_V[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# build connectivity array
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0,iel],"at pos.",x[icon[0,iel]], y[icon[0,iel]])
#     print ("node 2",icon[1,iel],"at pos.",x[icon[1,iel]], y[icon[1,iel]])
#     print ("node 3",icon[2,iel],"at pos.",x[icon[2,iel]], y[icon[2,iel]])
#     print ("node 4",icon[3,iel],"at pos.",x[icon[3,iel]], y[icon[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions - for this benchmark: no slip on all sides 
###############################################################################
start=clock.time()

bc_fix_V =np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V =np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V]  =True ; bc_val_V[i*ndof_V]  =0.
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix. r,s are the reduced coordinates in the [-1:1]x[-1:1] ref elt
###############################################################################
start=clock.time()

#A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)

A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)   # gradient matrix B 
N_V=np.zeros(m_V,dtype=np.float64)            # shape functions
dNdx_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNdy_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNdr_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
dNds_V=np.zeros(m_V,dtype=np.float64)            # shape functions derivatives
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            # construct 3x8 B matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental matrix
            A_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                b_el[2*i+1]+=N_V[i]*by(xq,yq)*JxWq

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.

    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate determinant of the jacobian
    JxWq=np.linalg.det(jcb)*weightq

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute gradient matrix
    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    # compute elemental matrix
    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    # assemble matrix and right hand side 
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

print("Build FE matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

###############################################################################
# impose boundary conditions
# for now it is done outside of the previous loop, we will see
# later in the course how it can be incorporated seamlessly in it.
###############################################################################
start=clock.time()

for i in range(0, Nfem):
    if bc_fix_V[i]:
       A_femref = A_fem[i,i]
       for j in range(0,Nfem):
           b_fem[j]-= A_fem[i, j] * bc_val_V[i]
           A_fem[i,j]=0.
           A_fem[j,i]=0.
           A_fem[i,i] = A_femref
       b_fem[i]=A_femref*bc_val_V[i]

#print("A_fem (m,M) = %.4e %.4e" %(np.min(A_fem),np.max(A_fem)))
#print("b_fem (m,M) = %.4e %.4e" %(np.min(b_fem),np.max(b_fem)))

print("impose b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol = sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("Solve linear system: %.5f s | Nfem= %d " % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# compute pressure and strain rate components in the middle of the elements
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    weightq = 2.0 * 2.0

    # calculate shape functions
    N_V[0]=0.25*(1.-rq)*(1.-sq)
    N_V[1]=0.25*(1.+rq)*(1.-sq)
    N_V[2]=0.25*(1.+rq)*(1.+sq)
    N_V[3]=0.25*(1.-rq)*(1.+sq)

    # calculate shape function derivatives
    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # calculate jacobian matrix
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate inverse of the jacobian matrix
    jcbi = np.linalg.inv(jcb)

    # compute dNdx & dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute coords of center 
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')

np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x_e,y_e,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_p=np.zeros(nel,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=u[i]-velocity_x(x_V[i],y_V[i])
    error_v[i]=v[i]-velocity_y(x_V[i],y_V[i])

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(x_e[i],y_e[i])

print("compute nodal error for plot: %.3f s" % (clock.time()-start))

###############################################################################
# compute error in L2 norm
###############################################################################
start=clock.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            jcbi = np.linalg.inv(jcb)
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq))**2*JxWq

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# naive depth averaging
###############################################################################
start=clock.time()

avrg_u_profile= np.zeros(nny,dtype=np.float64)
avrg_v_profile=np.zeros(nny,dtype=np.float64)
avrg_vel_profile=np.zeros(nny,dtype=np.float64)
avrg_y_profile=np.zeros(nny,dtype=np.float64)

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        avrg_y_profile[j]+=y_V[counter]/nnx
        avrg_u_profile[j]+=u[counter]/nnx
        avrg_v_profile[j]+=v[counter]/nnx
        avrg_vel_profile[j]+=np.sqrt(u[counter]**2+v[counter]**2)/nnx
        counter += 1
    #end for
#end for

np.savetxt('profiles.ascii',np.array([avrg_y_profile,avrg_u_profile,avrg_v_profile,avrg_vel_profile]).T)

print("compute vertical profiles: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################

u_temp=np.reshape(u,(nny,nnx))
v_temp=np.reshape(v,(nny,nnx))
p_temp=np.reshape(p,(nely,nelx))
exx_temp=np.reshape(exx,(nely,nelx))
eyy_temp=np.reshape(eyy,(nely,nelx))
exy_temp=np.reshape(exy,(nely,nelx))
error_u_temp=np.reshape(error_u,(nny,nnx))
error_v_temp=np.reshape(error_v,(nny,nnx))
error_p_temp=np.reshape(error_p,(nely,nelx))

fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(18,18))

uextent=(np.amin(x_V),np.amax(x_V),np.amin(y_V),np.amax(y_V))
pextent=(np.amin(x_e),np.amax(x_e),np.amin(y_e),np.amax(y_e))

onePlot(u_temp,       0, 0, "$v_x$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(v_temp,       0, 1, "$v_y$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(p_temp,       0, 2, "$p$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
onePlot(exx_temp,     1, 0, "$\dot{\epsilon}_{xx}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(eyy_temp,     1, 1, "$\dot{\epsilon}_{yy}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(exy_temp,     1, 2, "$\dot{\epsilon}_{xy}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(error_u_temp, 2, 0, "$v_x-t^{th}_x$",        "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(error_v_temp, 2, 1, "$v_y-t^{th}_y$",        "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(error_p_temp, 2, 2, "$p-p^{th}$",            "x", "y", uextent,  0,  0, 'RdGy_r')

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
