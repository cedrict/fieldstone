import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time

#------------------------------------------------------------------------------

def density(rho0,alpha,T,T0,case):
    val=rho0*(1.-alpha*(T-T0)) -rho0
    return val

def viscosity(T,exx,eyy,exy,y,gamma_T,gamma_y,sigma_y,eta_star,case):
    #-------------------
    # blankenbach et al, case 1
    #-------------------
    if case==0: 
       val=1.
    #-------------------
    # tosi et al, case 1
    #-------------------
    elif case==1:
       val=np.exp(-gamma_T*T)
    #-------------------
    # tosi et al, case 2
    #-------------------
    elif case==2:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       e=max(e,1e-8)
       eta_lin=np.exp(-gamma_T*T)
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 3
    #-------------------
    elif case==3:
       val=np.exp(-gamma_T*T+gamma_y*(1-y))
    #-------------------
    # tosi et al, case 4
    #-------------------
    elif case==4:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 5
    #-------------------
    elif case==5:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2        # number of dimensions
m=4           # number of nodes making up an element
ndofV=ndim    # number of velocity degrees of freedom per node
ndofP=1       # number of pressure degrees of freedom 
ndofT=1       # number of temperature degrees of freedom 

Lx=1.               # horizontal extent of the domain 
Ly=1.               # vertical extent of the domain 
eta_ref=1           # rheology parameter 
gamma_T=np.log(1e5) # rheology parameter 
gamma_y=np.log(1.)  # rheology parameter 
eta_star=1e-3       # rheology parameter 
alphaT=1e-4         # thermal expansion coefficient
hcond=1.            # thermal conductivity
hcapa=1.            # heat capacity
rho0=1.             # reference density
T0=0                # reference temperature

CFL_nb=0.5    # CFL number 
every=50      # vtu output frequency
nstep=10000   # maximum number of timestep   

#--------------------------------------

case=1

if case==0:
   Ra=1e4  
   sigma_y=0.

if case==1:
   Ra=1e2 
   sigma_y=1.

if case==2:
   Ra=1e2 
   sigma_y = 1

if case==3:
   Ra=1e2 

if case==4:
   Ra=1e2 
   sigma_y = 1

if case==5:
   Ra=1e2 
   sigma_y=4

gx=0.
gy=-Ra/alphaT  # vertical component of gravity vector

#--------------------------------------

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = 32
   visu = 0

#--------------------------------------

nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemV=nnp*ndofV   # number of velocity dofs
NfemP=nel*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs
NfemT=nnp        # number of T dofs
use_BA=True
pnormalise=True

hx=Lx/nelx
hy=Ly/nely

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################

model_time=np.zeros(nstep,dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
Nu=np.zeros(nstep,dtype=np.float64)
Tavrg=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)
heatflux_boundary=np.zeros(nstep,dtype=np.float64)

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
       bc_fixT[i] = True ; bc_valT[i] = 1. 
    if y[i]/Ly>1-eps:
       bc_fixT[i] = True ; bc_valT[i] = 0. 

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# initial temperature setup
#################################################################
start = time.time()

T=np.zeros(nnp,dtype=np.float64)

for i in range(0,nnp):
    T[i]=1.-y[i]-0.01*np.cos(np.pi*x[i])*np.sin(np.pi*y[i])

print("setup: T: %.3f s" % (time.time() - start))

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
    
c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

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
    b_mat = np.zeros((3,ndofV*m),dtype=np.float64)   # gradient matrix B 
    N     = np.zeros(m,dtype=np.float64)             # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    u     = np.zeros(nnp,dtype=np.float64)           # x-component velocity
    v     = np.zeros(nnp,dtype=np.float64)           # y-component velocity
    p     = np.zeros(nel,dtype=np.float64)           # y-component velocity
    etaq  = np.zeros(4*nel,dtype=np.float64)         # viscosity at q points
    rhoq  = np.zeros(4*nel,dtype=np.float64)         # density at q points

    iiq=0
    for iel in range(0, nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m*ndofV),dtype=np.float64)
        K_el =np.zeros((m*ndofV,m*ndofV),dtype=np.float64)
        G_el=np.zeros((m*ndofV,1),dtype=np.float64)
        h_el=np.zeros((1,1),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1, 1]:
            for jq in [-1, 1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate shape functions
                N[0:m]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)

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
                Tq=0.0
                exxq=0.
                eyyq=0.
                exyq=0.
                for k in range(0, m):
                    xq+=N[k]*x[icon[k,iel]]
                    yq+=N[k]*y[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    exxq+=dNdx[k]*u[icon[k,iel]]
                    eyyq+=dNdy[k]*v[icon[k,iel]]
                    exyq+=0.5*dNdy[k]*u[icon[k,iel]]+\
                          0.5*dNdx[k]*v[icon[k,iel]]
                rhoq[iiq]=density(rho0,alphaT,Tq,T0,case)
                etaq[iiq]=viscosity(Tq,exxq,eyyq,exyq,yq,gamma_T,gamma_y,sigma_y,eta_star,case)

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                             [0.     ,dNdy[i]],
                                             [dNdy[i],dNdx[i]]]

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[iiq]*weightq*jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    f_el[ndofV*i  ]+=N[i]*jcob*weightq*rhoq[iiq]*gx
                    f_el[ndofV*i+1]+=N[i]*jcob*weightq*rhoq[iiq]*gy
                    G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                    G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

                iiq+=1

            # end for jq
        # end for iq

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
                   G_el[ikk,0]=0

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

    # end for iel

    G_mat*=eta_ref/Ly

    print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    print("build FE matrix: %.3f s" % (time.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = time.time()

    if pnormalise:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
       a_mat[Nfem,NfemV:Nfem]=1
       a_mat[NfemV:Nfem,Nfem]=1
    else:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
       rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T

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
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

    u_stats[istep,0]=np.min(u) ; u_stats[istep,1]=np.max(u)
    v_stats[istep,0]=np.min(v) ; v_stats[istep,1]=np.max(v)

    if pnormalise:
       print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

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
    u_el=np.zeros(nel,dtype=np.float64)
    v_el=np.zeros(nel,dtype=np.float64)
    T_el=np.zeros(nel,dtype=np.float64)
    qtop=0.
    qbottom=0.
    qleft=0.
    qright=0.
 
    iel=0
    for iely in range(0,nely):
        for ielx in range(0,nelx):
            rq = 0.0
            sq = 0.0
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
            jcob=np.linalg.det(jcb)
            jcbi=np.linalg.inv(jcb)
            for k in range(0,m):
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                xc[iel] += N[k]*x[icon[k,iel]]
                yc[iel] += N[k]*y[icon[k,iel]]
                u_el[iel] += N[k]*u[icon[k,iel]]
                v_el[iel] += N[k]*v[icon[k,iel]]
                T_el[iel] += N[k]*T[icon[k,iel]]
                exx[iel] += dNdx[k]*u[icon[k,iel]]
                eyy[iel] += dNdy[k]*v[icon[k,iel]]
                exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+\
                            0.5*dNdx[k]*v[icon[k,iel]]
                dTdx[iel] += dNdx[k]*T[icon[k,iel]]
                dTdy[iel] += dNdy[k]*T[icon[k,iel]]
            if iely==0:
               qbottom+=-k*dTdy[iel]*hx *-1
            if iely==nely-1:
               qtop+=-k*dTdy[iel]*hx    *1 
               Nu[istep]-=dTdy[iel]*hx *Ly/(Lx*1)
            if ielx==0:
               qleft+=-k*dTdx[iel]*hy   *-1
            if ielx==nelx-1:
               qright+=-k*dTdx[iel]*hy  *1
            e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
            iel+=1

        # end for ielx
    # end for iely

    heatflux_boundary[istep]=qtop+qbottom+qleft+qright

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
    print("     -> dTdx (m,M) %.4f %.4f " %(np.min(dTdx),np.max(dTdx)))
    print("     -> dTdy (m,M) %.4f %.4f " %(np.min(dTdy),np.max(dTdy)))
    print("     -> time= %.6f ; Nu= %.6f" %(model_time[istep],Nu[istep]))

    print("compute sr, Nu: %.3f s" % (time.time() - start))

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

    print('     -> dt1= %.3e dt2= %.3e dt= %.4e' % (dt1,dt2,dt))

    print("compute timestep: %.3f s" % (time.time() - start))

    ######################################################################
    # compute nodal pressure
    ######################################################################
    start = time.time()

    count=np.zeros(nnp,dtype=np.float64)  
    q=np.zeros(nnp,dtype=np.float64)  

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

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute q: %.3f s" % (time.time() - start))

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    # ToDo: look at np.outer product in python so N_mat -> N

    start = time.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(m,dtype=np.float64)    

    iiq=0
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
                weghtq=1.*1.

                # calculate shape functions
                N_mat[0:m,0]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2, 2),dtype=float)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                Tq=0.
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    Tq+=N_mat[k,0]*T[icon[k,iel]]
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]

                if use_BA or use_EBA:
                   rho_lhs=rho0
                else:
                   rho_lhs=rhoq[iiq]

                # compute mass matrix
                MM+=N_mat.dot(N_mat.T)*rho_lhs*hcapa*weightq*jcob

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(vel.dot(B_mat))*rho_lhs*hcapa*weightq*jcob

                iiq+=1

            # end for jq
        # end for iq

        a_el=MM+(Ka+Kd)*dt

        b_el=MM.dot(Tvect) + f_el*dt

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

    # end for iel

    print("build FEM matrix T: %.3f s" % (time.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = time.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats[istep,0]=np.min(T) ; T_stats[istep,1]=np.max(T)

    print("solve T: %.3f s" % (time.time() - start))

    ######################################################################
    # compute vrms and Tavrg
    ######################################################################
    start = time.time()

    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N[0:m]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)
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
                Tq=0.
                for k in range(0,m):
                    uq+=N[k]*u[icon[k,iel]]
                    vq+=N[k]*v[icon[k,iel]]
                    Tq+=N[k]*T[icon[k,iel]]
                Tavrg[istep]+=Tq*weightq*jcob
                vrms[istep]+=(uq**2+vq**2)*weightq*jcob
            # end for jq
        # end for iq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))
    Tavrg[istep]/=Lx*Ly

    print("     -> vrms= %.6e ; Ra= %.6e " % (vrms[istep],Ra))
    print("     -> avrg T= %.6e" % Tavrg[istep])

    print("compute vrms,Tavrg : %.3f s" % (time.time() - start))


    #####################################################################
    # plot of solution
    #####################################################################
    start = time.time()

    if visu==1 or istep%every==0:

       rho_el= np.zeros(nel,dtype=np.float64)
       eta_el= np.zeros(nel,dtype=np.float64)
       for iel in range(0,nel):
           rho_el[iel]=(rhoq[iel*4]+rhoq[iel*4+1]+rhoq[iel*4+2]+rhoq[iel*4+3])*0.25
           eta_el[iel]=(etaq[iel*4]+etaq[iel*4+1]+etaq[iel*4+2]+etaq[iel*4+3])*0.25

       # compute depth-averaged profiles
       T_profile=np.zeros(nny,dtype=np.float64)
       y_profile=np.zeros(nny,dtype=np.float64)
       V_profile=np.zeros(nny,dtype=np.float64)
       counter = 0
       for j in range(0,nny):
           for i in range(0,nnx):
               T_profile[j]+=T[counter]/nnx
               y_profile[j]+=y[counter]/nnx
               V_profile[j]+=np.sqrt(u[counter]**2+v[counter]**2)/nnx
               counter += 1
       np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='# y,T')
       np.savetxt('V_profile.ascii',np.array([y_profile,V_profile]).T,header='# y,V')

       yc_profile=np.zeros(nely,dtype=np.float64)
       eta_profile=np.zeros(nely,dtype=np.float64)
       counter = 0
       for j in range(0,nely):
           for i in range(0,nelx):
               eta_profile[j]+=eta_el[counter]/nelx
               yc_profile[j]+=yc[counter]/nelx
               counter += 1
       np.savetxt('eta_profile.ascii',np.array([yc_profile,eta_profile]).T,header='# y,eta')

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
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='viscosity' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10f \n" %eta_el[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='density' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%10f \n" %rho_el[iel])
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
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
       for i in range(0,nnp):
           vtufile.write("%10e \n" % q[i])
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

    np.savetxt('vrms_Nu.ascii',np.array([model_time[0:istep],vrms[0:istep],Nu[0:istep]]).T,header='# t,vrms,Nu')
    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep],Tavrg[0:istep]]).T,header='# t,Tavrg')
    np.savetxt('heat_flux_boundary.ascii',np.array([model_time[0:istep],heatflux_boundary[0:istep]]).T,header='# t,q')
    np.savetxt('u_stats.ascii',np.array([model_time[0:istep],u_stats[0:istep,0],u_stats[0:istep,1]]).T,header='# t,m(u),M(u)')
    np.savetxt('v_stats.ascii',np.array([model_time[0:istep],v_stats[0:istep,0],v_stats[0:istep,1]]).T,header='# t,m(v),M(v)')
    np.savetxt('T_stats.ascii',np.array([model_time[0:istep],T_stats[0:istep,0],T_stats[0:istep,1]]).T,header='# t,m(T),M(T)')
    np.savetxt('dt_stats.ascii',np.array([model_time[0:istep],dt_stats[0:istep]]).T,header='# t,dt')

    print("output stats: %.3f s" % (time.time() - start))

################################################################################################
# END OF TIMESTEPPING
################################################################################################
    

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
