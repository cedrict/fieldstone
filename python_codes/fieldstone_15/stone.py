import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import time as clock
import matplotlib.pyplot as plt

###############################################################################

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

def NNV(r,s):
    N_0=0.25*(1.-r)*(1.-s)
    N_1=0.25*(1.+r)*(1.-s)
    N_2=0.25*(1.+r)*(1.+s)
    N_3=0.25*(1.-r)*(1.+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def dNNVdr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=+0.25*(1.+s) 
    dNdr_3=-0.25*(1.+s) 
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def dNNVds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.+r)
    dNds_3=+0.25*(1.-r)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###############################################################################

print("-----------------------------")
print("--------- stone 15 ----------")
print("-----------------------------")

m=4      # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 1
    
nnx=nelx+1       # number of elements, x direction
nny=nely+1       # number of elements, y direction
NV=nnx*nny       # number of nodes
nel=nelx*nely    # number of elements, total
NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

eta=1.  

use_SchurComplementApproach=True
niter_stokes=100
solver_tolerance=1e-8

eps=1.e-10
sqrt3=np.sqrt(3.)

###############################################################################

print('nelx=',nelx)
print('nely=',nely)
print('nnx=',nnx)
print('nny=',nny)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nfem=',Nfem)

###############################################################################
# grid point setup
###############################################################################
start = clock.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start = clock.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = clock.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = clock.time()

K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
jcb   = np.zeros((2,2),dtype=np.float64)
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m*ndofV),dtype=np.float64)
    K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
    G_el=np.zeros((m*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.

            # calculate shape functions
            N=NNV(rq,sq)
            dNdr=dNNVdr(rq,sq)
            dNds=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]= dNds[:].dot(y[icon[:,iel]])

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])

            # compute dNdx & dNdy
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*wq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                f_el[ndofV*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq

        #end for
    #end for

    # impose b.c. 
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0,0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0
            #end if
        #end for
    #end for

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,m):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for
    #end for
    h_rhs[iel]+=h_el[0,0]

#end for iel

print("     -> h_rhs (m,M) %.4e %.4e " %(np.min(h_rhs),np.max(h_rhs)))
print("     -> f_rhs (m,M) %.4e %.4e " %(np.min(f_rhs),np.max(f_rhs)))

print("build FE matrix: %.3f s" % (clock.time() - start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start = clock.time()

if use_SchurComplementApproach:

   conv_file=open("solver_convergence.ascii","w")

   # convert matrices to CSR format
   G_mat=sps.csr_matrix(G_mat)
   K_mat=sps.csr_matrix(K_mat)

   # declare necessary arrays
   solP=np.zeros(NfemP,dtype=np.float64)  
   solV=np.zeros(NfemV,dtype=np.float64)  
   rvect_k=np.zeros(NfemP,dtype=np.float64) 
   pvect_k=np.zeros(NfemP,dtype=np.float64) 
   ptildevect_k=np.zeros(NfemV,dtype=np.float64) 
   dvect_k=np.zeros(NfemV,dtype=np.float64) 
   
   # carry out solve
   solV=sps.linalg.spsolve(K_mat,f_rhs)
   rvect_k=G_mat.T.dot(solV)-h_rhs
   rvect_0=np.linalg.norm(rvect_k)
   pvect_k=rvect_k
   for k in range (0,niter_stokes):
       ptildevect_k=G_mat.dot(pvect_k)
       dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)
       alpha=(rvect_k.dot(rvect_k))/(ptildevect_k.dot(dvect_k))
       solP+=alpha*pvect_k
       solV-=alpha*dvect_k
       rvect_kp1=rvect_k-alpha*G_mat.T.dot(dvect_k)
       beta=(rvect_kp1.dot(rvect_kp1))/(rvect_k.dot(rvect_k))
       pvect_kp1=rvect_kp1+beta*pvect_k
       rvect_k=rvect_kp1
       pvect_k=pvect_kp1
       xi=np.linalg.norm(rvect_k)/rvect_0
       conv_file.write("%3d %6e \n"  %(k,xi))
       print('iteration= %3d, xi= %.4e ' %(k,xi))
       if xi<solver_tolerance:
          break 
   u,v=np.reshape(solV[0:NfemV],(NV,2)).T
   p=solP[0:NfemP]

else:
   sol =np.zeros(Nfem,dtype=np.float64)  # x coordinates
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("solve time: %.3f s" % (clock.time() - start))

###############################################################################
# compute strainrate 
###############################################################################
start = clock.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    N=NNV(rq,sq)
    dNdr=dNNVdr(rq,sq)
    dNds=dNNVds(rq,sq)
    jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
    jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]= dNds[:].dot(y[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]
    xc[iel]= N[:].dot(x[icon[:,iel]])
    yc[iel]= N[:].dot(y[icon[:,iel]])
    exx[iel]=dNdx[:].dot(u[icon[:,iel]])
    eyy[iel]=dNdy[:].dot(v[icon[:,iel]])
    exy[iel]=0.5*dNdy[:].dot(u[icon[:,iel]])+0.5*dNdx[:].dot(v[icon[:,iel]])
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute nodal pressure
###############################################################################
start = clock.time()

q=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    q[icon[3,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1

q=q/count

print("project press on V grid: %.3f s" % (clock.time() - start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_q = np.empty(NV,dtype=np.float64)
error_p = np.empty(nel,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
    error_q[i]=q[i]-pressure(x[i],y[i])

for iel in range(0,nel): 
    error_p[iel]=p[iel]-pressure(xc[iel],yc[iel])

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]= dNds[:].dot(y[icon[:,iel]])
            jcob=np.linalg.det(jcb)
            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])
            uq=N[:].dot(u[icon[:,iel]])
            vq=N[:].dot(v[icon[:,iel]])
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*wq*jcob
        #for jq
    #for iq
#for iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time() - start))

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

fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(18,18))

uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

onePlot(u_temp,       0, 0, "$v_x$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(v_temp,       0, 1, "$v_y$",                 "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(p_temp,       0, 2, "$p$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
onePlot(q_temp,       0, 3, "$q$",                   "x", "y", pextent, Lx, Ly, 'RdGy_r')
onePlot(exx_temp,     1, 0, "$\dot{\epsilon}_{xx}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(eyy_temp,     1, 1, "$\dot{\epsilon}_{yy}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(exy_temp,     1, 2, "$\dot{\epsilon}_{xy}$", "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(e_temp,       1, 3, "$\dot{\epsilon}$",      "x", "y", pextent, Lx, Ly, 'viridis')
onePlot(error_u_temp, 2, 0, "$v_x-t^{th}_x$",        "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(error_v_temp, 2, 1, "$v_y-t^{th}_y$",        "x", "y", uextent,  0,  0, 'Spectral_r')
onePlot(error_p_temp, 2, 2, "$p-p^{th}$",            "x", "y", uextent,  0,  0, 'RdGy_r')
onePlot(error_q_temp, 2, 3, "$q-p^{th}$",            "x", "y", uextent,  0,  0, 'RdGy_r')

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

###############################################################################
