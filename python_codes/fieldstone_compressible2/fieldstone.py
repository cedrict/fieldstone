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

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2      # number of dimensions
m=4         # number of nodes making up an element
ndofV=ndim  # number of velocity degrees of freedom per node
ndofP=1     # number of pressure degrees of freedom 
ndofT=1     # number of temperature degrees of freedom 

Lx=3e6  # horizontal extent of the domain 
Ly=3e6  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 64 
   nely = 64
   visu = 0

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

year=3.154e+7
eps=1.e-10
sqrt3=np.sqrt(3.)


Di=0.75       # dissipation number
Ra=1e4        # Rayleigh number
hcond=3.      # thermal conductivity
hcapa=1250.   # heat capacity
hprod=0       # heat production coeff
gx=0          
gy=-10
rho0=3000       # reference density
T0=273.15       # reference temperature
Delta_Temp=4000 # temperature difference 

alphaT=Di*hcapa/abs(gy)/Ly
viscosity=Di*hcapa**2*Delta_Temp*Ly**2*rho0**2/Ra/hcond
reftime=rho0*hcapa*Ly**2/hcond
refvel=Ly/reftime
refTemp=Delta_Temp

print ("     -> alphaT %e " % alphaT)
print ("     -> eta %e " %  viscosity)
print ("     -> reftime %e " %  reftime)
print ("     -> refvel %e " %  refvel)

betaT=0

CFL_nb=0.25

nstep=1000

incompressible=True
use_shearheating=False
pnormalise=True
write_blocks=False

if incompressible:
   betaT=0

#################################################################

model_time=np.zeros(nstep,dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
Nu=np.zeros(nstep,dtype=np.float64)
Tavrg=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
mass=np.zeros(nstep,dtype=np.float64)
viscdiss=np.zeros(nstep,dtype=np.float64)
work_grav=np.zeros(nstep,dtype=np.float64)
EK=np.zeros(nstep,dtype=np.float64)
EG=np.zeros(nstep,dtype=np.float64)
ET=np.zeros(nstep,dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp,dtype=np.float64)  # x coordinates
y = np.empty(nnp,dtype=np.float64)  # y coordinates

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
    if x[i]/Lx>1-eps:
       bc_fixV[i*ndofV  ] = True ; bc_valV[i*ndofV  ] = 0
    if y[i]<eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0
    if y[i]/Ly>1-eps:
       bc_fixV[i*ndofV+1] = True ; bc_valV[i*ndofV+1] = 0

bc_fixT=np.zeros(NfemT,dtype=np.bool) # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,nnp):
    if y[i]<eps:
       bc_fixT[i] = True ; bc_valT[i] = Delta_Temp+T0
    if y[i]/Ly>1-eps:
       bc_fixT[i] = True ; bc_valT[i] = T0

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# nodal pressure setup
#################################################################
start = time.time()

q=np.zeros(nnp,dtype=np.float64)  
q_prev=np.zeros(nnp,dtype=np.float64)  
dqdt=np.zeros(nnp,dtype=np.float64)

for ip in range(0,nnp):
    q[ip]=rho0*np.abs(gy)*(Ly-y[ip])

q_prev[:]=q[:]

print("setup: q: %.3f s" % (time.time() - start))

#################################################################
# temperature and nodal density setup
#################################################################
start = time.time()

T=np.zeros(nnp,dtype=np.float64)
rho=np.zeros(nnp,dtype=np.float64)
rho_prev=np.zeros(nnp,dtype=np.float64)
drhodt=np.zeros(nnp,dtype=np.float64)

for ip in range(0,nnp):
    T[ip]=((Ly-y[ip])/Ly - 0.01*np.cos(np.pi*x[ip]/Lx)*np.sin(np.pi*y[ip]/Ly))*Delta_Temp+T0
    rho[ip]=rho0*(1-alphaT*(T[ip]-T0)+betaT*q[ip])
    
rho_prev[:]=rho[:]

print("setup: T,rho: %.3f s" % (time.time() - start))

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

for istep in range(0,nstep):
    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

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
    Tvect = np.zeros(m,dtype=np.float64)   

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

    G_mat*=viscosity/Ly

    if incompressible:
       Z_mat[:,:]=0
       W_mat[:,:]=0

    print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
    print("     -> Z (m,M) %.5e %.5e " %(np.min(Z_mat),np.max(Z_mat)))
    print("     -> W (m,M) %.5e %.5e " %(np.min(W_mat),np.max(W_mat)))
    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

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

    print("solve: %.3f s" % (time.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
    p=sol[NfemV:Nfem]*(viscosity/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u)*year,np.max(u)*year))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v)*year,np.max(v)*year))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

    u_stats[istep,0]=np.min(u)*year ; u_stats[istep,1]=np.max(u)*year
    v_stats[istep,0]=np.min(v)*year ; v_stats[istep,1]=np.max(v)*year

    if pnormalise:
       print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

    np.savetxt('velocity.ascii',np.array([x,y,u*year,v*year]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute strainrate, temperature gradient and Nusselt number 
    ######################################################################
    start = time.time()

    xc  = np.zeros(nel,dtype=np.float64)  
    yc  = np.zeros(nel,dtype=np.float64)  
    exx = np.zeros(nel,dtype=np.float64)  
    eyy = np.zeros(nel,dtype=np.float64)  
    exy = np.zeros(nel,dtype=np.float64)  
    e   = np.zeros(nel,dtype=np.float64)  
    dTdx= np.zeros(nel,dtype=np.float64)  
    dTdy= np.zeros(nel,dtype=np.float64)  
    Phi = np.zeros(nel,dtype=np.float64)  
    drhodx=np.zeros(nel,dtype=np.float64)
    drhody=np.zeros(nel,dtype=np.float64)
    u_el=np.zeros(nel,dtype=np.float64)
    v_el=np.zeros(nel,dtype=np.float64)
    T_el=np.zeros(nel,dtype=np.float64)
    dqdx=np.zeros(nel,dtype=np.float64)
    dqdy=np.zeros(nel,dtype=np.float64)

    iel=0
    for iely in range(0,nely):
        for ielx in range(0,nelx):

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
                u_el[iel] += N[k]*u[icon[k,iel]]
                v_el[iel] += N[k]*v[icon[k,iel]]
                T_el[iel] += N[k]*T[icon[k,iel]]
                exx[iel] += dNdx[k]*u[icon[k,iel]]
                eyy[iel] += dNdy[k]*v[icon[k,iel]]
                exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
                dTdx[iel] += dNdx[k]*T[icon[k,iel]]
                dTdy[iel] += dNdy[k]*T[icon[k,iel]]
                dqdx[iel] += dNdx[k]*q[icon[k,iel]]
                dqdy[iel] += dNdy[k]*q[icon[k,iel]]
                drhodx[iel]+= dNdx[k]*rho[icon[k,iel]]
                drhody[iel]+= dNdy[k]*rho[icon[k,iel]]
            exxd=exx[iel]-(exx[iel]+eyy[iel])/3.
            eyyd=eyy[iel]-(exx[iel]+eyy[iel])/3.
            exyd=exy[iel]
            Phi[iel]=2.*viscosity*(exxd**2+eyyd**2+2*exyd**2)    

            if iely==nely-1:
               Nu[istep]+=abs(dTdy[iel])*Lx/nelx  * Ly  / (Lx*Delta_Temp)
            e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
            iel+=1

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
    print("     -> dTdx (m,M) %.4f %.4f " %(np.min(dTdx),np.max(dTdx)))
    print("     -> dTdy (m,M) %.4f %.4f " %(np.min(dTdy),np.max(dTdy)))

    print("     -> time= %.6f ; Nu= %.6f" %(model_time[istep]/year,Nu[istep]))

    np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
    np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute sr, Nu: %.3f s" % (time.time() - start))

    ######################################################################
    # compute vrms 
    ######################################################################
    start = time.time()

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
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                uq=0.
                vq=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                rhoq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
                vrms[istep]+=(uq**2+vq**2)*wq*jcob
                exxd=exxq-(exxq+eyyq)/3.
                eyyd=eyyq-(exxq+eyyq)/3.
                exyd=exyq
                viscdiss[istep]+=2.*viscosity*(exxd**2+eyyd**2+2*exyd**2)*wq*jcob 
                EK[istep]+=0.5*rhoq*(uq**2+vq**2)*wq*jcob 
                EG[istep]+=rhoq*gy*yq*wq*jcob 
                work_grav[istep]+=(rhoq-rho0)*gy*vq*wq*jcob 

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))

    print("     -> vrms= %.6e" % vrms[istep])

    print("compute vrms: %.3f s" % (time.time() - start))

    ######################################################################
    # compute time step value 
    ######################################################################
    start = time.time()

    dt1=CFL_nb*min(Lx/nelx,Ly/nely)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL_nb*min(Lx/nelx,Ly/nely)**2/(hcond/hcapa/rho0)

    dt=min(dt1,dt2)

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    dt_stats[istep]=dt 

    print('     -> dt1= %.6e dt2= %.6e dt= %.6e' % (dt1/year,dt2/year,dt/year))

    print("compute timestep: %.3f s" % (time.time() - start))

    ######################################################################
    # compute nodal pressure
    ######################################################################
    start = time.time()

    count=np.zeros(nnp,dtype=np.float64)  
    q[:]=0

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

    dqdt=(q[:]-q_prev[:])/dt

    np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute q: %.3f s" % (time.time() - start))

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    start = time.time()

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
        f_el=np.zeros(m*ndofT,dtype=np.float64)

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

                # compute dNdx & dNdy and Phi
                vel[0,0]=0.
                vel[0,1]=0.
                exxq=0.
                eyyq=0.
                exyq=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=(dNdy[k]*u[icon[k,iel]]+dNdx[k]*v[icon[k,iel]])*.5
                exxd=exxq-(exxq+eyyq)/3.
                eyyd=eyyq-(exxq+eyyq)/3.
                exyd=exyq
                Phiq=2.*viscosity*(exxd**2+eyyd**2+2*exyd**2)    

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                # compute shear heating rhs term
                if use_shearheating:
                   f_el[:]=N_mat[:,0]*Phiq*dt*jcob*wq
                else:
                   f_el[:]=0

                a_el=MM+(Ka+Kd)*dt

                b_el=MM.dot(Tvect) + f_el

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

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix T: %.3f s" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats[istep,0]=np.min(T) ; T_stats[istep,1]=np.max(T)

    print("solve T: %.3f s" % (time.time() - start))

    #####################################################################
    # update density 
    #####################################################################
    start = time.time()

    for ip in range(0,nnp):
        rho[ip]=rho0*(1-alphaT*(T[ip]-T0)+betaT*q[ip])

    drhodt=(rho[:]-rho_prev[:])/dt

    print("     -> rho (m,M) %.4f %.4f " %(np.min(rho),np.max(rho)))

    print("compute density: %.3f s" % (time.time() - start))

    #####################################################################
    # compute average of temperature, total mass 
    #####################################################################
    start = time.time()

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
                jcob=np.linalg.det(jcb)
                Tq=0.
                for k in range(0,m):
                    Tq+=N[k]*T[icon[k,iel]]
                    rhoq+=N[k]*rho[icon[k,iel]]
                Tavrg[istep]+=Tq*wq*jcob
                mass[istep]+=rhoq*wq*jcob
                ET[istep]+=rhoq*hcapa*Tq*wq*jcob

    Tavrg[istep]/=Lx*Ly

    print("     -> avrg T= %.6e" % Tavrg[istep])
    print("     -> mass  = %.6e" % mass[istep])

    print("compute <T>,M: %.3f s" % (time.time() - start))

    #####################################################################
    # plot of solution
    #####################################################################
    start = time.time()

    if visu==1 or istep%10==0:

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nnp):
          vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % dTdx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % dTdy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='drhodx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % drhodx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='drhody' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % drhody[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='alpha T v.gradp' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (alphaT*T_el[iel]*(u_el[iel]*dqdx[iel]+v_el[iel]*dqdy[iel])))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exx[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % eyy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % exy[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % (np.sqrt(exx[iel]**2+eyy[iel]**2+2*exy[iel]**2)))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Phi' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%10e\n" % Phi[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='rho' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % rho[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='drho/dt' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % drhodt[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='alpha T dp/dt' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (alphaT*T[i]*dqdt[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='T' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % (T[i]-T0))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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

    print("generate pdf & vtu: %.3f s" % (time.time() - start))

    #####################################################################
    # write to file 
    #####################################################################
    start = time.time()

    np.savetxt('vrms_Nu.ascii',np.array([model_time[0:istep]/year,vrms[0:istep],Nu[0:istep]]).T,header='# t/year,vrms,Nu')
    np.savetxt('vrms_Nu_adim.ascii',np.array([model_time[0:istep]/reftime,vrms[0:istep]/refvel,Nu[0:istep]]).T,header='# t/reftime,vrms/refvel,Nu')

    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep]/year,Tavrg[0:istep]]).T,header='# t/year,Tavrg')
    np.savetxt('Tavrg_adim.ascii',np.array([model_time[0:istep]/reftime,Tavrg[0:istep]/refTemp]).T,header='# t/reftime,Tavrg/refTemp')

    np.savetxt('M.ascii',np.array([model_time[0:istep]/year,mass[0:istep],mass[0:istep]/mass[0]]).T,header='# t/year,M,M/M0')
    np.savetxt('EK.ascii',np.array([model_time[0:istep]/year,EK[0:istep]]).T,header='# t/year,EK')
    np.savetxt('EG.ascii',np.array([model_time[0:istep]/year,EG[0:istep],EG[0:istep]-EG[0]]).T,header='# t/year,EG,EG-EG(0)')
    np.savetxt('ET.ascii',np.array([model_time[0:istep]/year,ET[0:istep]]).T,header='# t/year,ET')
    np.savetxt('viscous_dissipation.ascii',np.array([model_time[0:istep]/year,viscdiss[0:istep]]).T,header='# t/year,Phi')
    np.savetxt('work_grav.ascii',np.array([model_time[0:istep]/year,work_grav[0:istep]]).T,header='# t/year,W')

    np.savetxt('u_stats.ascii',np.array([model_time[0:istep]/year,u_stats[0:istep,0],u_stats[0:istep,1]]).T,header='# t/year,min(u),max(u)')
    np.savetxt('v_stats.ascii',np.array([model_time[0:istep]/year,v_stats[0:istep,0],v_stats[0:istep,1]]).T,header='# t/year,min(v),max(v)')
    np.savetxt('T_stats.ascii',np.array([model_time[0:istep]/year,T_stats[0:istep,0],T_stats[0:istep,1]]).T,header='# t/year,min(T),max(T)')

    np.savetxt('dt_stats.ascii',np.array([model_time[0:istep]/year,dt_stats[0:istep]]).T,header='# t/year,dt')

    print("output stats: %.3f s" % (time.time() - start))

    #####################################################################

    q_prev[:]=q[:]
    rho_prev[:]=rho[:]


################################################################################################
# END OF TIMESTEPPING
################################################################################################
    

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
