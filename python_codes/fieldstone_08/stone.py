import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tkinter

#------------------------------------------------------------------------------

def viscosity(exx,eyy,exy):
    e2=np.sqrt(0.5*(exx*exx+eyy*eyy)+exy*exy)
    e2=max(1e-8,e2)
    sigmay=1.
    val=sigmay/2./e2
    val=min(1.e3,val)
    val=max(1.e-3,val)
    #val=1.
    return val

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

# declare variables
print("variable declaration")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=0.5  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 128
   nely = 64
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

penalty=1.e7  # penalty coefficient value

Nfem=nnp*ndof  # Total number of degrees of freedom

eps=1.e-10

sqrt3=np.sqrt(3.)

width=0.111111111
niter=100

gx=0.
gy=0.
density=1.

# declare arrays
print("declaring arrays")

#####################################################################
# grid point setup
#####################################################################

print("grid point setup")

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter+=1

#####################################################################
# connectivity
#####################################################################

print("connectivity")

icon =np.zeros((m, nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

#####################################################################
# define boundary conditions
#####################################################################

print("defining boundary conditions")

bc_fix = np.zeros(Nfem, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
    if y[i]<eps:
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if y[i]>(Ly-eps) and abs(x[i]-Lx/2.)<width:
       #bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1.

#####################################################################
   
Res_file=open('residual.ascii',"w")
u_file=open('u_stats.ascii',"w")
v_file=open('v_stats.ascii',"w")
diff_file=open('diff_uv.ascii',"w")

N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
Res   = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol   = np.zeros(Nfem,dtype=np.float64)         # solution vector 
u_old = np.zeros(nnp,dtype=np.float64)          # x-component velocity
v_old = np.zeros(nnp,dtype=np.float64)          # y-component velocity

#------------------------------------------------------------------------------
# non-linear iterations
#------------------------------------------------------------------------------

for iter in range(0,niter):

    print("--------------------------")
    print("iter=", iter)
    print("--------------------------")

    #################################################################
    # build FE matrix
    #################################################################

    print("building FE matrix")

    a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
    rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el = np.zeros(m * ndof)
        a_el = np.zeros((m * ndof, m * ndof), dtype=float)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

                # calculate jacobian matrix
                jcb = np.zeros((2, 2),dtype=float)
                for k in range(0,m):
                    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                    jcb[1, 1] += dNds[k]*y[icon[k,iel]]
                #end for

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                exxq=0.0
                eyyq=0.0
                exyq=0.0
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
                #end for

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]
                #end for

                # compute elemental a_mat matrix
                a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity(exxq,eyyq,exyq)*wq*jcob

                # compute elemental rhs vector
                for i in range(0,m):
                    b_el[2*i  ]+=N[i]*jcob*wq*density*gx
                    b_el[2*i+1]+=N[i]*jcob*wq*density*gy
                #end for
            #end for
        #end for

        # integrate penalty term at 1 point
        rq=0.
        sq=0.
        wq=2.*2.

        N[0]=0.25*(1.-rq)*(1.-sq)
        N[1]=0.25*(1.+rq)*(1.-sq)
        N[2]=0.25*(1.+rq)*(1.+sq)
        N[3]=0.25*(1.-rq)*(1.+sq)

        dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
        dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
        dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
        dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

        # compute the jacobian
        jcb=np.zeros((2,2),dtype=float)
        for k in range(0, m):
            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNds[k]*y[icon[k,iel]]
        #end for

        # calculate determinant of the jacobian
        jcob = np.linalg.det(jcb)

        # calculate the inverse of the jacobian
        jcbi = np.linalg.inv(jcb)

        # compute dNdx and dNdy
        for k in range(0,m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
        #end for

        # compute gradient matrix
        for i in range(0,m):
            b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                                  [0.     ,dNdy[i]],
                                  [dNdy[i],dNdx[i]]]
        #end for

        # compute elemental matrix
        a_el+=b_mat.T.dot(k_mat.dot(b_mat))*penalty*wq*jcob

        # assemble matrix a_mat and right hand side rhs
        for k1 in range(0,m):
            for i1 in range(0,ndof):
                ikk=ndof*k1          +i1
                m1 =ndof*icon[k1,iel]+i1
                for k2 in range(0,m):
                    for i2 in range(0,ndof):
                        jkk=ndof*k2          +i2
                        m2 =ndof*icon[k2,iel]+i2
                        a_mat[m1,m2]+=a_el[ikk,jkk]
                    #end for
                #end for
                rhs[m1]+=b_el[ikk]
            #end for
        #end for

    #end for

    #################################################################
    # impose boundary conditions
    #################################################################

    print("imposing boundary conditions")

    for i in range(0, Nfem):
        if bc_fix[i]:
           a_matref=a_mat[i,i]
           for j in range(0,Nfem):
               rhs[j]-=a_mat[i,j]*bc_val[i]
               a_mat[i,j]=0.
               a_mat[j,i]=0.
               a_mat[i,i]=a_matref
           rhs[i]=a_matref*bc_val[i]
        #end if
    #end for

    #################################################################
    # compute non-linear residual
    #################################################################

    Res=a_mat.dot(sol)-rhs

    Res2=np.linalg.norm(Res,2)

    if iter==0:
       Res2Init=Res2

    Res_file.write("%10e \n" % (Res2/Res2Init))

    print("Nonlinear residual (inf. norm) %.7e" % (Res2/Res2Init))

    #################################################################
    # solve system
    #################################################################

    start = time.time()
    sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
    print("solve time: %.3f s" % (time.time() - start))

    #####################################################################
    # put solution into separate x,y velocity arrays
    #####################################################################

    u,v=np.reshape(sol,(nnp,2)).T

    print("u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

    u_file.write("%10e %10e\n" % (np.min(u),np.max(u)))
    v_file.write("%10e %10e\n" % (np.min(v),np.max(v)))

    #####################################################################

    udiff=np.linalg.norm(u_old-u,2)
    vdiff=np.linalg.norm(v_old-v,2)
    if iter==0:
       udiffinit=udiff
       vdiffinit=vdiff
    
    diff_file.write("%10e %10e\n" % (udiff/udiffinit,vdiff/vdiffinit))

    u_old[:]=u[:]
    v_old[:]=v[:]

#end if

#------------------------------------------------------------------------------
# end of non-linear iterations
#------------------------------------------------------------------------------
   
u_file.close()
v_file.close()
Res_file.close()
diff_file.close()

#####################################################################
# retrieve pressure and elemental strain rate components
# in the middle of the element (1 integration point)
#####################################################################

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
p=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0
    wq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    #end for

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    #end for

    for k in range(0,m):
        xc[iel]+=N[k]*x[icon[k,iel]]
        yc[iel]+=N[k]*y[icon[k,iel]]
        exx[iel]+=dNdx[k]*u[icon[k,iel]]
        eyy[iel]+=dNdy[k]*v[icon[k,iel]]
        exy[iel]+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
    #end for

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
    eta[iel]=viscosity(exx[iel],eyy[iel],exy[iel])
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("eta (m,M) %.4f %.4f " %(np.min(eta),np.max(eta)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################
# computing stress tensor components
#####################################################################

sigmaxx=np.zeros(nel,dtype=np.float64)  
sigmayy=np.zeros(nel,dtype=np.float64)  
sigmaxy=np.zeros(nel,dtype=np.float64)  

sigmaxx=-p+2*eta*exx
sigmayy=-p+2*eta*eyy
sigmaxy=   2*eta*exy

#####################################################################
# smoothing pressure 
#####################################################################

q=np.zeros(nnp,dtype=np.float64)  
count=np.zeros(nnp,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    q[icon[3,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1
#end for

q=q/count

#####################################################################
# extract velocity field at domain top
#####################################################################

xtop=np.zeros(nnx,dtype=np.float64)  
utop=np.zeros(nnx,dtype=np.float64)  
vtop=np.zeros(nnx,dtype=np.float64)  
qtop=np.zeros(nnx,dtype=np.float64)  


counter=0
for i in range(0,nnp):
    if y[i]>Ly-eps:
       xtop[counter]=x[i]
       utop[counter]=u[i]
       vtop[counter]=v[i]
       qtop[counter]=q[i]
       counter+=1
   #end if
#end for

xctop=np.zeros(nelx,dtype=np.float64)  
ptop=np.zeros(nelx,dtype=np.float64)  
exxtop=np.zeros(nelx,dtype=np.float64)  
exytop=np.zeros(nelx,dtype=np.float64)  
etop=np.zeros(nelx,dtype=np.float64)  
sigmaxxtop=np.zeros(nelx,dtype=np.float64)  
sigmayytop=np.zeros(nelx,dtype=np.float64)  
sigmaxytop=np.zeros(nelx,dtype=np.float64)  
etatop=np.zeros(nelx,dtype=np.float64)  

counter=0
for iel in range(0,nel):
    if y[icon[3,iel]]>Ly-eps:
       xctop[counter]=xc[iel]
       ptop[counter]=p[iel]
       exxtop[counter]=exx[iel]
       exytop[counter]=exy[iel]
       etop[counter]=e[iel]
       sigmaxxtop[counter]=sigmaxx[iel]
       sigmayytop[counter]=sigmayy[iel]
       sigmaxytop[counter]=sigmaxy[iel]
       etatop[counter]=eta[iel]
       counter+=1
   #end if
#end for

np.savetxt('v_q_top.ascii',np.array([xtop,utop,vtop,qtop]).T,header='# x,y,u,v,q')
np.savetxt('p_sr_top.ascii',np.array([xctop,ptop,exxtop,exytop,etop]).T,header='# x,y,p,exx,exy,e')
np.savetxt('sigma_eta_top.ascii',np.array([xctop,sigmaxxtop,sigmayytop,sigmaxytop,etatop]).T,header='# x,y,sigmaxx,sigmaxy')

#####################################################################
# plot of solution
#####################################################################

u_temp=np.reshape(u,(nny,nnx))
np.flipud(u_temp)

v_temp=np.reshape(v,(nny,nnx))
p_temp=np.reshape(p,(nely,nelx))
q_temp=np.reshape(q,(nny,nnx))
exx_temp=np.reshape(exx,(nely,nelx))
eyy_temp=np.reshape(eyy,(nely,nelx))
exy_temp=np.reshape(exy,(nely,nelx))
eta_temp=np.reshape(eta,(nely,nelx))
e_temp=np.reshape(e,(nely,nelx))

fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(18,18))

uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

im = axes[0][0].imshow(u_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][0].set_title('$v_x$', fontsize=10, y=1.01)
axes[0][0].set_xlabel('x')
axes[0][0].set_ylabel('y')
fig.colorbar(im,ax=axes[0][0])

im = axes[0][1].imshow(v_temp,extent=uextent,cmap='Spectral',interpolation='nearest')
axes[0][1].set_title('$v_y$', fontsize=10, y=1.01)
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
fig.colorbar(im,ax=axes[0][1])

im = axes[0][2].imshow(p_temp,extent=pextent,cmap='RdGy',interpolation='nearest')
axes[0][2].set_title('$p$', fontsize=10, y=1.01)
axes[0][2].set_xlim(0,Lx)
axes[0][2].set_ylim(0,Ly)
axes[0][2].set_xlabel('x')
axes[0][2].set_ylabel('y')
fig.colorbar(im,ax=axes[0][2])

im = axes[1][0].imshow(exx_temp,extent=pextent,cmap='viridis',interpolation='nearest')
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

im = axes[2][0].imshow(e_temp,extent=pextent,cmap='viridis',interpolation='nearest')
axes[2][0].set_title('$\dot{\epsilon}$',fontsize=10,y=1.01)
axes[2][0].set_xlim(0,Lx)
axes[2][0].set_ylim(0,Ly)
axes[2][0].set_xlabel('x')
axes[2][0].set_ylabel('y')
fig.colorbar(im,ax=axes[2][0])

im = axes[2][1].imshow(eta_temp,extent=pextent,cmap='jet',interpolation='nearest',norm=LogNorm(vmin=1e-3,vmax=1e3))
axes[2][1].set_title('$\eta$',fontsize=10, y=1.01)
axes[2][1].set_xlim(0,Lx)
axes[2][1].set_ylim(0,Ly)
axes[2][1].set_xlabel('x')
axes[2][1].set_ylabel('y')
fig.colorbar(im,ax=axes[2][1])

im = axes[2][2].imshow(q_temp,extent=pextent,cmap='RdGy',interpolation='nearest')
axes[2][2].set_title('$p$ (nodal)',fontsize=10, y=1.01)
axes[2][2].set_xlim(0,Lx)
axes[2][2].set_ylim(0,Ly)
axes[2][2].set_xlabel('x')
axes[2][2].set_ylabel('y')
fig.colorbar(im,ax=axes[2][2])

im = axes[3][0].plot(xtop,utop)
axes[3][0].set_xlabel('$x$')
axes[3][0].set_ylabel('$u$')

im = axes[3][1].plot(xtop,vtop)
axes[3][1].set_xlabel('$x$')
axes[3][1].set_ylabel('$v$')

im = axes[3][2].plot(xctop,ptop)
axes[3][2].set_xlabel('$x$')
axes[3][2].set_ylabel('$p$')

plt.subplots_adjust(hspace=0.5)

if visu==1:
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
