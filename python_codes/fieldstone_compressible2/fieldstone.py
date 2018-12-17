import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2      # number of dimensions
m=4         # number of nodes making up an element
ndofV=ndim  # number of velocity degrees of freedom per node
ndofP=1     # number of pressure degrees of freedom 
ndofT=1     # number of temperature degrees of freedom 

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
   nelx = 32
   nely = 32
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs
NfemT=nnp        # number of T dofs

eps=1.e-10

sqrt3=np.sqrt(3.)

pnormalise=True

write_blocks=False


Di=0.5
Ra=1e5       # Rayleigh number
alphaT=1e-2   # thermal expansion coefficient
hcond=1.     # thermal conductivity
hcapa=1.     # heat capacity
hprod=0
rho0=1       # reference density
T0=0         # reference temperature
CFL=1.       # CFL number 
gx=0
gy=-Ra/alphaT # vertical component of gravity vector
viscosity=1
betaT=0
#alphaT=0.1*Di
#viscosity=Di/Ra

CFL_nb=0.5

nstep=10

incompressible=True

model_time=np.zeros(nstep,dtype=np.float64) 
Tvect = np.zeros(m,dtype=np.float64)   
vrms=np.zeros(nstep,dtype=np.float64) 
Nu=np.zeros(nstep,dtype=np.float64) 

if incompressible:
   betaT=0

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp, dtype=np.float64)  # x coordinates
y = np.empty(nnp, dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx) 
        y[counter]=j*Ly/float(nely) 
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter]=i+j*(nelx+1)
        icon[1,counter]=i+1+j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx + 1)
        icon[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fixV=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_valV=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]<eps:
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0
    if x[i]>(Lx-eps):
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0
    if y[i]<eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0
    if y[i]>(Ly-eps):
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0

bc_fixT=np.zeros(NfemT,dtype=np.bool) # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,nnp):
    if y[i]<eps:
       bc_fixT[i] = True ; bc_valT[i] = 1
    if y[i]>(Ly-eps):
       bc_fixT[i] = True ; bc_valT[i] = 0

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# nodal pressure setup
#################################################################
q=np.zeros(nnp,dtype=np.float64)  

for ip in range(0,nnp):
    q[ip]=rho0*np.abs(gy)*(Ly-y[ip])

#################################################################
# temperature and nodal density setup
#################################################################
T=np.zeros(nnp,dtype=np.float64)
T_prev=np.zeros(nnp,dtype=np.float64)
rho=np.zeros(nnp,dtype=np.float64)

for ip in range(0,nnp):
    T[ip]=(1-y[ip]) - 0.01*np.cos(np.pi*x[ip]/Lx)*np.sin(np.pi*y[ip]/Ly)
    rho[ip]=rho0*(1-alphaT*(T[ip]-T0)+betaT*q[ip])

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)

    #################################################################
    # build FE matrix
    #################################################################
    start = time.time()

    K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    Z_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix Z
    W_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix W

    b_mat = np.zeros((3,ndofV*m),dtype=np.float64)  # gradient matrix B 
    N     = np.zeros(m,dtype=np.float64)            # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
    u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
    v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
    p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
    c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0, nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m*ndofV),dtype=np.float64)
        K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
        G_el=np.zeros((m*ndofV,1),dtype=np.float64)
        W_el=np.zeros((m*ndofV,1),dtype=np.float64)
        Z_el=np.zeros((m*ndofV,1),dtype=np.float64)
        h_el=np.zeros((1,1),dtype=np.float64)

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
                jcb = np.zeros((2, 2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                    jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                    jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                    jcb[1, 1] += dNds[k]*y[icon[k,iel]]

                # calculate the determinant of the jacobian
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.
                yq=0.
                uq=0.
                vq=0.
                rhoq=0.
                drhodxq=0.
                drhodyq=0.
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    drhodxq+=dNdx[k]*rho[icon[k,iel]]
                    drhodyq+=dNdy[k]*rho[icon[k,iel]]

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    f_el[ndofV*i  ]+=N[i]*jcob*wq*rhoq*gx
                    f_el[ndofV*i+1]+=N[i]*jcob*wq*rhoq*gy
                    G_el[ndofV*i  ,0]-=dNdx[i]*jcob*wq
                    G_el[ndofV*i+1,0]-=dNdy[i]*jcob*wq
                    Z_el[ndofV*i  ,0]-=N[i]*jcob*wq*drhodxq/rhoq
                    Z_el[ndofV*i+1,0]-=N[i]*jcob*wq*drhodyq/rhoq
                    W_el[ndofV*i  ,0]-=N[i]*jcob*wq*gx*betaT
                    W_el[ndofV*i+1,0]-=N[i]*jcob*wq*gy*betaT

                h_el[0]+=0#(uq*drhodxq+vq*drhodyq)/rhoq*wq*jcob

        # impose b.c. 
        for k1 in range(0,m):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*icon[k1,iel]+i1
                if bc_fixV[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m*ndofV):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_valV[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                       K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_valV[m1]
                   h_el[0]-=G_el[ikk,0]*bc_valV[m1]
                   h_el[0]-=Z_el[ikk,0]*bc_valV[m1]
                   G_el[ikk,0]=0
                   Z_el[ikk,0]=0
                   W_el[ikk,0]=0

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
                Z_mat[m1,iel]+=Z_el[ikk,0]
                W_mat[m1,iel]+=W_el[ikk,0]
        h_rhs[iel]+=h_el[0]

    if incompressible:
       Z_mat[:,:]=0
       W_mat[:,:]=0

    print("     -> K (m,M) %.5f %.5f " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G (m,M) %.5f %.5f " %(np.min(G_mat),np.max(G_mat)))
    print("     -> Z (m,M) %.5f %.5f " %(np.min(Z_mat),np.max(Z_mat)))
    print("     -> W (m,M) %.5f %.5f " %(np.min(W_mat),np.max(W_mat)))
    print("     -> f (m,M) %.5f %.5f " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5f %.5f " %(np.min(h_rhs),np.max(h_rhs)))

    print("build FE matrix: %.3f s" % (time.time() - start))

    ######################################################################
    start = time.time()

    if write_blocks:

       f = open("K_mat.ascii","w")
       for i in range(0,NfemV):
           for j in range(0,NfemV):
               if K_mat[i,j]!=0:
                  f.write("%i %i %10.6f\n" % (i,j,K_mat[i,j]))

       f = open("G_mat.ascii","w")
       for i in range(0,NfemV):
           for j in range(0,NfemP):
               if G_mat[i,j]!=0:
                  f.write("%i %i %10.6f\n" % (i,j,G_mat[i,j]))

       f = open("f_rhs.ascii","w")
       for i in range(0,NfemV):
           f.write("%i %10.6f\n" % (i,f_rhs[i]))

       f = open("h_rhs.ascii","w")
       for i in range(0,NfemP):
           f.write("%i %10.6f\n" % (i,h_rhs[i]))

    print("write blocks to file: %.3f s" % (time.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = time.time()

    if pnormalise:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat+W_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T+Z_mat.T
       a_mat[Nfem,NfemV:Nfem]=1
       a_mat[NfemV:Nfem,Nfem]=1
    else:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
       rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat+W_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T+Z_mat.T

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (time.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = time.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

    print("solve time: %.3f s" % (time.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
    p=sol[NfemV:Nfem]

    print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

    if pnormalise:
       print("     -> Lagrange multiplier: %.4es" % sol[Nfem])

    np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute strainrate 
    ######################################################################
    start = time.time()

    xc = np.zeros(nel,dtype=np.float64)  
    yc = np.zeros(nel,dtype=np.float64)  
    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  
    e   = np.zeros(nel,dtype=np.float64)  

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

        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0, m):
            jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
            jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
            jcb[1,0]+=dNds[k]*x[icon[k,iel]]
            jcb[1,1]+=dNds[k]*y[icon[k,iel]]

        # calculate determinant of the jacobian
        jcob=np.linalg.det(jcb)

        # calculate the inverse of the jacobian
        jcbi=np.linalg.inv(jcb)

        for k in range(0,m):
            dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
            dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

        for k in range(0,m):
            xc[iel] += N[k]*x[icon[k,iel]]
            yc[iel] += N[k]*y[icon[k,iel]]
            exx[iel] += dNdx[k]*u[icon[k,iel]]
            eyy[iel] += dNdy[k]*v[icon[k,iel]]
            exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

    print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

    np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
    np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (time.time() - start))

    ######################################################################
    # compute vrms 
    ######################################################################

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
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                uq=0.
                vq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                vrms[istep]+=(uq**2+vq**2)*wq*jcob

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))

    print("time= %.6f ; vrms   = %.6f" %(model_time[istep],vrms[istep]))

    ######################################################################
    # compute nodal pressure
    ######################################################################

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

    q=q/count

    np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

    ######################################################################
    # compute time step value 
    ######################################################################

    dt1=CFL*min(Lx/nelx,Ly/nely)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL*min(Lx/nelx,Ly/nely)**2/(hcond/hcapa/rho0)

    dt=min(dt1,dt2)

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    print('dt1= %.6f' %dt1)
    print('dt2= %.6f' %dt2)
    print('dt = %.6f' %dt)

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################

    print("building temperature matrix and rhs")

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            Tvect[k]=T[icon[k,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

                # calculate shape functions
                N_mat[0,0]=0.25*(1.-rq)*(1.-sq)
                N_mat[1,0]=0.25*(1.+rq)*(1.-sq)
                N_mat[2,0]=0.25*(1.+rq)*(1.+sq)
                N_mat[3,0]=0.25*(1.-rq)*(1.+sq)

                # calculate shape function derivatives
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

                # calculate jacobian matrix
                jcb=np.zeros((2, 2),dtype=float)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]

                # calculate the determinant of the jacobian
                jcob=np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                a_el=MM+(Ka+Kd)*dt

                b_el=MM.dot(Tvect)

                # apply boundary conditions

                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    if bc_fixT[m1]:
                       Aref=a_el[k1,k1]
                       for k2 in range(0,m):
                           m2=icon[k2,iel]
                           b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                           a_el[k1,k2]=0
                           a_el[k2,k1]=0
                       a_el[k1,k1]=Aref
                       b_el[k1]=Aref*bc_valT[m1]


                # assemble matrix A_mat and right hand side rhs
                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    for k2 in range(0,m):
                        m2=icon[k2,iel]
                        A_mat[m1,m2]+=a_el[k1,k2]
                    rhs[m1]+=b_el[k1]

    print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    #################################################################
    # solve system
    #################################################################

    start = time.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve T time: %.3f s" % (time.time() - start))

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    #####################################################################
    # update density 
    #####################################################################

    for ip in range(0,nnp):
        rho[ip]=rho0*(1-alphaT*(T[ip]-T0)+betaT*q[ip])

    #####################################################################
    # plot of solution
    #####################################################################

    u_temp=np.reshape(u,(nny,nnx))
    v_temp=np.reshape(v,(nny,nnx))
    q_temp=np.reshape(q,(nny,nnx))
    p_temp=np.reshape(p,(nely,nelx))
    e_temp=np.reshape(e,(nely,nelx))
    exx_temp=np.reshape(exx,(nely,nelx))
    eyy_temp=np.reshape(eyy,(nely,nelx))
    exy_temp=np.reshape(exy,(nely,nelx))
    rho_temp=np.reshape(rho,(nny,nnx))
    T_temp=np.reshape(T,(nny,nnx))

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

    fig,axes=plt.subplots(nrows=3,ncols=4,figsize=(18,18))

    uextent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y))
    pextent=(np.amin(xc),np.amax(xc),np.amin(yc),np.amax(yc))

    onePlot(u_temp,      0,0, "$v_x$",                "x", "y", uextent, 0, 0, 'Spectral_r')
    onePlot(v_temp,      0,1, "$v_y$",                "x", "y", uextent, 0, 0, 'Spectral_r')
    onePlot(p_temp,      0,2, "$p$",                  "x", "y", pextent, 0, 0, 'RdGy_r')
    onePlot(q_temp,      0,3, "$q$",                  "x", "y", pextent, 0, 0, 'RdGy_r')
    onePlot(exx_temp,    1,0, "$\dot{\epsilon}_{xx}$","x", "y", pextent, 0, 0, 'viridis')
    onePlot(eyy_temp,    1,1, "$\dot{\epsilon}_{yy}$","x", "y", pextent, 0, 0, 'viridis')
    onePlot(exy_temp,    1,2, "$\dot{\epsilon}_{xy}$","x", "y", pextent, 0, 0, 'viridis')
    onePlot(e_temp,      1,3, "$\dot{\epsilon}$",     "x", "y", pextent, 0, 0, 'viridis')
    onePlot(rho_temp,    2,0, "density",              "x", "y", uextent, 0, 0, 'Spectral_r')
    onePlot(T_temp,      2,1, "temperature",          "x", "y", uextent, 0, 0, 'Spectral_r')

    plt.subplots_adjust(hspace=0.5)

    if visu==1:
       filename = 'solution_{:04d}.pdf'.format(istep) 
       plt.savefig(filename, bbox_inches='tight')
       #plt.show()

################################################################################################
# END OF TIMESTEPPING
################################################################################################
    
np.savetxt('vrms.ascii',np.array([model_time,vrms]).T,header='# t,vrms')

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
