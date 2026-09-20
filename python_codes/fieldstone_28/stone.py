import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import time as clock
import numba

###############################################################################

@numba.njit
def density(rho0,alpha,T,T0,case):
    val=rho0*(1.-alpha*(T-T0)) -rho0
    return val

@numba.njit
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
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T)
       eta_plast=eta_star + sigma_y/(np.sqrt(2.)*e)
       val=2./(1./eta_lin + 1./eta_plast)
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
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    #-------------------
    # tosi et al, case 5
    #-------------------
    elif case==5:
       e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
       e=max(e,1e-12)
       eta_lin=np.exp(-gamma_T*T+gamma_y*(1-y))
       eta_plast=eta_star + sigma_y/(np.sqrt(2)*e)
       val=2/(1/eta_lin + 1/eta_plast)
    val=min(2.0,val)
    val=max(1.e-5,val)
    return val

###############################################################################

@numba.njit
def basis_functions_V(r,s):
    N0=0.25*(1.-r)*(1.-s)
    N1=0.25*(1.+r)*(1.-s)
    N2=0.25*(1.+r)*(1.+s)
    N3=0.25*(1.-r)*(1.+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

@numba.njit
def basis_functions_V_dr(r,s):
    dNdr0=-0.25*(1.-s) 
    dNdr1=+0.25*(1.-s) 
    dNdr2=+0.25*(1.+s) 
    dNdr3=-0.25*(1.+s) 
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3],dtype=np.float64)

@numba.njit
def basis_functions_V_ds(r,s):
    dNds0=-0.25*(1.-r)
    dNds1=-0.25*(1.+r)
    dNds2=+0.25*(1.+r)
    dNds3=+0.25*(1.-r)
    return np.array([dNds0,dNds1,dNds2,dNds3],dtype=np.float64)

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 028 **********")
print("*******************************")

m=4      # number of nodes making up an element
ndim=2   # number of dimensions
ndof_V=2 # number of velocity degrees of freedom per node

Lx=1.               # horizontal extent of the domain 
Ly=1.               # vertical extent of the domain 
eta_ref=1           # rheology parameter 
gamma_T=np.log(1e5) # rheology parameter 
eta_star=1e-3       # rheology parameter 
alphaT=1e-4         # thermal expansion coefficient
hcond=1.            # thermal conductivity
hcapa=1.            # heat capacity
rho0=1.             # reference density
T0=0                # reference temperature

CFL_nb=0.95   # CFL number 
every=10     # vtu output frequency
nstep=5000   # maximum number of timestep   
tol_nl=1.e-1 # nonlinear convergence coeff.

###############################################################################

case=5

if case==0:
   Ra=1e4  
   sigma_y=0.
   gamma_y=np.log(1.)
   niter_nl=1

if case==1:
   Ra=1e2 
   sigma_y=1.
   gamma_y=np.log(1.)
   niter_nl=1

if case==2:
   Ra=1e2 
   sigma_y = 1
   gamma_y=np.log(1.)
   niter_nl=100

if case==3:
   Ra=1e2 
   sigma_y = 1
   gamma_y=np.log(10.)
   niter_nl=100

if case==4:
   Ra=1e2 
   sigma_y = 1
   gamma_y=np.log(10.)
   niter_nl=100

if case==5:
   Ra=1e2 
   sigma_y=4.
   gamma_y=np.log(10.)
   niter_nl=100

gx=0.
gy=-Ra/alphaT  # vertical component of gravity vector

###############################################################################

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 32
   nely = nelx
   visu = 0

###############################################################################

nnx=nelx+1         # number of elements, x direction
nny=nely+1         # number of elements, y direction
nn_V=nnx*nny       # number of V,T nodes
nel=nelx*nely      # number of elements, total
Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nel         # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs
Nfem_T=nn_V        # number of T dofs

use_BA=True

hx=Lx/nelx
hy=Ly/nely
       
convfile=open("conv_nl.ascii","w")
niterfile=open("niter_nl.ascii","w")

###############################################################################

model_time=np.zeros(nstep,dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
Nu=np.zeros(nstep,dtype=np.float64)
Tavrg=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)
heatflux_boundary=np.zeros(nstep,dtype=np.float64)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) 
y_V=np.zeros(nn_V,dtype=np.float64) 

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx
        y_V[counter]=j*hy
        counter+=1
    #end for
#end for

print("node coordinates: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx + 1)
        icon_V[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1
    #end for
#end for

print("connectivity array: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0
    if x_V[i]/Lx>1-eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0
    if y_V[i]/Ly>1-eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0

bc_fix_T=np.zeros(Nfem_T,dtype=bool) # boundary condition, yes/no
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if y_V[i]<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = 1. 
    if y_V[i]/Ly>1-eps:
       bc_fix_T[i] = True ; bc_val_T[i] = 0. 

print("define boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature setup
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    T[i]=1.-y_V[i]-0.01*np.cos(np.pi*x_V[i])*np.sin(np.pi*y_V[i])

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# all elements are rectangles of size hx,hy
# so we can precompute the Jacobian-related quantities
###############################################################################

jcb=np.zeros((ndim,ndim),dtype=np.float64)
jcob=hx*hy/4
jcbi=np.zeros((2,2),dtype=np.float64)
jcbi[0,0]=2/hx
jcbi[1,1]=2/hy

###############################################################################
###############################################################################
# TIME STEPPING
###############################################################################
###############################################################################
    
C=np.array([[ 4/3,-2/3,0],
            [-2/3, 4/3,0],
            [   0,   0,1]],dtype=np.float64) 

for istep in range(0,nstep):

    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

    u=np.zeros(nn_V,dtype=np.float64)     # x-component velocity
    v=np.zeros(nn_V,dtype=np.float64)     # y-component velocity
    p=np.zeros(nel,dtype=np.float64)      # y-component velocity
    Res=np.zeros(Nfem+1,dtype=np.float64) # non-linear residual 
    sol=np.zeros(Nfem+1,dtype=np.float64) # solution vector 

    for iter_nl in range(0,niter_nl):

        print("__________________ iter_nl= ", iter_nl)

        #######################################################################
        # build FE matrix
        #######################################################################
        start=clock.time()

        K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
        G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
        f_rhs=np.zeros(Nfem_V,dtype=np.float64)          # right hand side f 
        h_rhs=np.zeros(Nfem_P,dtype=np.float64)          # right hand side h 
        B=np.zeros((3,ndof_V*m),dtype=np.float64)        # gradient matrix B 
        etaq=np.zeros(4*nel,dtype=np.float64)            # viscosity at q points
        rhoq=np.zeros(4*nel,dtype=np.float64)            # density at q points

        iiq=0
        for iel in range(0, nel):

            K_el =np.zeros((m*ndof_V,m*ndof_V),dtype=np.float64)
            G_el=np.zeros((m*ndof_V,1),dtype=np.float64)
            f_el =np.zeros((m*ndof_V),dtype=np.float64)
            h_el=np.zeros((1,1),dtype=np.float64)

            for iq in [-1,1]:
                for jq in [-1,1]:

                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    weightq=1.*1.

                    N_V=basis_functions_V(rq,sq)
                    dNdr_V=basis_functions_V_dr(rq,sq)
                    dNds_V=basis_functions_V_ds(rq,sq)
                    #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                    #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                    #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                    #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                    #jcbi=np.linalg.inv(jcb)
                    #JxWq=np.linalg.det(jcb)*weightq
                    JxWq=jcob*weightq

                    xq=np.dot(N_V,x_V[icon_V[:,iel]])
                    yq=np.dot(N_V,y_V[icon_V[:,iel]])
                    Tq=np.dot(N_V,T[icon_V[:,iel]])

                    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                    exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                    eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                    exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                         np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

                    rhoq[iiq]=density(rho0,alphaT,Tq,T0,case)
                    etaq[iiq]=viscosity(Tq,exxq,eyyq,exyq,yq,gamma_T,gamma_y,sigma_y,eta_star,case)
        
                    #if np.isnan(etaq[iiq]): exit("etaq is NaN")
    
                    for i in range(0,m):
                        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                          [0.       ,dNdy_V[i]],
                                          [dNdy_V[i],dNdx_V[i]]]

                    K_el+=B.T.dot(C.dot(B))*etaq[iiq]*JxWq

                    for i in range(0, m):
                        f_el[ndof_V*i  ]+=N_V[i]*rhoq[iiq]*gx*JxWq
                        f_el[ndof_V*i+1]+=N_V[i]*rhoq[iiq]*gy*JxWq
                        G_el[ndof_V*i  ,0]-=dNdx_V[i]*JxWq
                        G_el[ndof_V*i+1,0]-=dNdy_V[i]*JxWq

                    iiq+=1

                # end for jq
            # end for iq

            # impose b.c. 
            for k1 in range(0,m):
                for i1 in range(0,ndof_V):
                    ikk=ndof_V*k1          +i1
                    m1 =ndof_V*icon_V[k1,iel]+i1
                    if bc_fix_V[m1]:
                       K_ref=K_el[ikk,ikk] 
                       for jkk in range(0,m*ndof_V):
                           f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                           K_el[ikk,jkk]=0
                           K_el[jkk,ikk]=0
                           K_el[ikk,ikk]=K_ref
                       #end for
                       f_el[ikk]=K_ref*bc_val_V[m1]
                       h_el[0]-=G_el[ikk,0]*bc_val_V[m1]
                       G_el[ikk,0]=0
                    #end if
                #end for
            #end for

            # assemble elemental matrix and right hand side vector
            for k1 in range(0,m):
                for i1 in range(0,ndof_V):
                    ikk=ndof_V*k1+i1
                    m1 =ndof_V*icon_V[k1,iel]+i1
                    for k2 in range(0,m):
                        for i2 in range(0,ndof_V):
                            jkk=ndof_V*k2+i2
                            m2 =ndof_V*icon_V[k2,iel]+i2
                            K_mat[m1,m2]+=K_el[ikk,jkk]
                        #end for
                    #end for
                    f_rhs[m1]+=f_el[ikk]
                    G_mat[m1,iel]+=G_el[ikk,0]
                #end for
            #end for

        # end for iel

        G_mat*=eta_ref/Ly

        print("build FE matrix: %.3f s" % (clock.time()-start))

        ######################################################################
        # assemble K, G, GT, f, h into A and rhs - not super elegant
        # pressure is normalised to zero w/ Lagrange multiplier
        ######################################################################
        start=clock.time()

        A_fem=np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
        b_fem=np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
        A_fem[0:Nfem_V,0:Nfem_V]=K_mat
        A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
        A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
        A_fem[Nfem,Nfem_V:Nfem]=1
        A_fem[Nfem_V:Nfem,Nfem]=1

        b_fem[0:Nfem_V]=f_rhs
        b_fem[Nfem_V:Nfem]=h_rhs

        print("assemble blocks: %.3f s" % (clock.time()-start))

        #######################################################################
        # compute non-linear residual
        #######################################################################
        start=clock.time()

        Res=A_fem.dot(sol)-b_fem

        if iter_nl==0: Res0=np.max(abs(Res))

        if case>0: print("      -> normalised nl residual %.3e" % (np.max(abs(Res))/Res0))
          
        convfile.write("%e %e \n" %( istep+iter_nl/200.,np.max(abs(Res))/Res0))
        convfile.flush()

        if np.max(abs(Res))/Res0 < tol_nl:
           print('******************')
           print("nl its  converged!")
           print('******************')
           niterfile.write("%d %d \n" %( istep,iter_nl))
           niterfile.flush()
           break 

        print("compute residual: %.3f s" % (clock.time()-start))

        ######################################################################
        # solve system
        ######################################################################
        start=clock.time()

        sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

        print("solve: %.3f s" % (clock.time()-start))

        ######################################################################
        # put solution into separate x,y velocity arrays
        ######################################################################
        start=clock.time()

        u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
        p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

        print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
        print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
        print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

        u_stats[istep,0]=np.min(u) ; u_stats[istep,1]=np.max(u)
        v_stats[istep,0]=np.min(v) ; v_stats[istep,1]=np.max(v)

        print("split vel into u,v: %.3f s" % (clock.time()-start))

    # end for nonlinear iterations
        
    print("__________________")

    ###########################################################################
    # compute strainrate, temperature gradient and Nusselt number 
    ###########################################################################
    start=clock.time()

    x_e=np.zeros(nel,dtype=np.float64)  
    y_e=np.zeros(nel,dtype=np.float64)  
    u_e=np.zeros(nel,dtype=np.float64)
    v_e=np.zeros(nel,dtype=np.float64)
    T_e=np.zeros(nel,dtype=np.float64)
    exx_e=np.zeros(nel,dtype=np.float64)  
    eyy_e=np.zeros(nel,dtype=np.float64)  
    exy_e=np.zeros(nel,dtype=np.float64)  
    dTdx_e=np.zeros(nel,dtype=np.float64)  
    dTdy_e=np.zeros(nel,dtype=np.float64)  

    qtop=0.
    qbottom=0.
    qleft=0.
    qright=0.

    rq=0.0
    sq=0.0
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
 
    iel=0
    for iely in range(0,nely):
        for ielx in range(0,nelx):
            #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            #jcbi=np.linalg.inv(jcb)
            #JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
            y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
            u_e[iel]=np.dot(N_V,u[icon_V[:,iel]])
            v_e[iel]=np.dot(N_V,v[icon_V[:,iel]])
            T_e[iel]=np.dot(N_V,T[icon_V[:,iel]])
            exx_e[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
            eyy_e[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
            exy_e[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                      +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
            dTdx_e[iel]=np.dot(dNdx_V[:],T[icon_V[:,iel]])
            dTdy_e[iel]=np.dot(dNdy_V[:],T[icon_V[:,iel]])

            if iely==0:      qbottom+=-hcond*dTdy_e[iel]*hx *-1
            if iely==nely-1: qtop   +=-hcond*dTdy_e[iel]*hx * 1 
            if ielx==0:      qleft  +=-hcond*dTdx_e[iel]*hy *-1
            if ielx==nelx-1: qright +=-hcond*dTdx_e[iel]*hy * 1

            if iely==nely-1: Nu[istep]-=dTdy_e[iel]*hx *Ly/(Lx*1)

            iel+=1

        # end for ielx
    # end for iely

    sr=np.sqrt(0.5*(exx_e**2+eyy_e**2)+exy_e**2)

    heatflux_boundary[istep]=qtop+qbottom+qleft+qright

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx_e),np.max(exx_e)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy_e),np.max(eyy_e)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy_e),np.max(exy_e)))
    print("     -> dTdx (m,M) %.4f %.4f " %(np.min(dTdx_e),np.max(dTdx_e)))
    print("     -> dTdy (m,M) %.4f %.4f " %(np.min(dTdy_e),np.max(dTdy_e)))
    print("     -> time= %.3e ; Nu= %.6f" %(model_time[istep],Nu[istep]))

    print("compute sr, Nu: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute time step value 
    ###########################################################################
    start=clock.time()

    dt1=CFL_nb*min(Lx/nelx,Ly/nely)/np.max(np.sqrt(u**2+v**2))

    dt2=CFL_nb*min(Lx/nelx,Ly/nely)**2/(hcond/hcapa/rho0)

    dt=min(dt1,dt2)

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    dt_stats[istep]=dt 

    print('     -> dt1= %.3e dt2= %.3e dt= %.4e' % (dt1,dt2,dt))

    print("compute timestep: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal pressure
    ###########################################################################
    start=clock.time()

    count=np.zeros(nn_V,dtype=np.float64)  
    q=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        q[icon_V[0,iel]]+=p[iel] ; count[icon_V[0,iel]]+=1
        q[icon_V[1,iel]]+=p[iel] ; count[icon_V[1,iel]]+=1
        q[icon_V[2,iel]]+=p[iel] ; count[icon_V[2,iel]]+=1
        q[icon_V[3,iel]]+=p[iel] ; count[icon_V[3,iel]]+=1

    q/=count

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute q: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build FE matrix for Temperature 
    ###########################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)          # FE rhs 
    B_mat=np.zeros((2,m),dtype=np.float64)           # gradient matrix B 
    N_mat= np.zeros((m,1),dtype=np.float64)          # shape functions

    for iel in range (0,nel):

        A_el=np.zeros((m,m),dtype=np.float64)
        b_el=np.zeros(m,dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)
        f_el=np.zeros(m,dtype=np.float64)

        Tvect=T[icon_V[0:m,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                N_V=basis_functions_V(rq,sq)
                dNdr_V=basis_functions_V_dr(rq,sq)
                dNds_V=basis_functions_V_ds(rq,sq)
                #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                #jcbi=np.linalg.inv(jcb)
                #JxWq=np.linalg.det(jcb)*weightq
                JxWq=jcob*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

                vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
                vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])

                B_mat[0,:]=dNdx_V
                B_mat[1,:]=dNdy_V
                N_mat[:,0]=N_V

                MM+=N_mat.dot(N_mat.T)*rho0*hcapa*JxWq
                Kd+=B_mat.T.dot(B_mat)*hcond*JxWq
                Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*JxWq

            # end for jq
        # end for iq

        #A_el=MM+(Ka+Kd)*dt
        #b_el=MM.dot(Tvect)+f_el*dt

        #Crank-Nicolson
        A_el+=MM+(Ka+Kd)*dt*0.5
        b_el+=(MM-(Ka+Kd)*dt*0.5).dot(Tvect)+f_el*dt

        # apply boundary conditions
        for k1 in range(0,m):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               #end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            #end for
        #end for

        # assemble matrix and right hand side vector
        for k1 in range(0,m):
            m1=icon_V[k1,iel]
            for k2 in range(0,m):
                m2=icon_V[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            #end for
            b_fem[m1]+=b_el[k1]
        #end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    T_stats[istep,0]=np.min(T) ; T_stats[istep,1]=np.max(T)

    print("solve T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute vrms and Tavrg
    ###########################################################################
    start=clock.time()

    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
                N_V=basis_functions_V(rq,sq)
                #dNdr_V=basis_functions_V_dr(rq,sq)
                #dNds_V=basis_functions_V_ds(rq,sq)
                #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                #jcbi=np.linalg.inv(jcb)
                #JxWq=np.linalg.det(jcb)*weightq
                JxWq=jcob*weightq
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                Tavrg[istep]+=Tq*JxWq
                vrms[istep]+=(uq**2+vq**2)*JxWq
            # end for jq
        # end for iq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))
    Tavrg[istep]/=Lx*Ly

    print("     -> vrms= %.6e ; Ra= %.6e " % (vrms[istep],Ra))
    print("     -> avrg T= %.6e" % Tavrg[istep])

    print("compute vrms,Tavrg : %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if visu==1 or istep%every==0:

       rho_e=np.zeros(nel,dtype=np.float64)
       eta_e=np.zeros(nel,dtype=np.float64)
       for iel in range(0,nel):
           rho_e[iel]=(rhoq[iel*4]+rhoq[iel*4+1]+rhoq[iel*4+2]+rhoq[iel*4+3])*0.25
           eta_e[iel]=(etaq[iel*4]+etaq[iel*4+1]+etaq[iel*4+2]+etaq[iel*4+3])*0.25
       
       # compute dev stress
       tauxx_e=2*eta_e*exx_e
       tauyy_e=2*eta_e*eyy_e
       tauxy_e=2*eta_e*exy_e

       # make sure <p>=0 at surface
       p-=np.sum(p[nel-nelx:nel])/nelx

       # compute full stress
       sigmaxx_e=-p+2*eta_e*exx_e
       sigmayy_e=-p+2*eta_e*eyy_e
       sigmaxy_e=   2*eta_e*exy_e

       # compute depth-averaged profiles
       T_profile=np.zeros(nny,dtype=np.float64)
       y_profile=np.zeros(nny,dtype=np.float64)
       V_profile=np.zeros(nny,dtype=np.float64)
       counter=0
       for j in range(0,nny):
           for i in range(0,nnx):
               T_profile[j]+=T[counter]/nnx
               y_profile[j]+=y_V[counter]/nnx
               V_profile[j]+=np.sqrt(u[counter]**2+v[counter]**2)/nnx
               counter += 1
           #end for
       #end for
       np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='# y,T')
       np.savetxt('V_profile.ascii',np.array([y_profile,V_profile]).T,header='# y,V')

       y_e_profile=np.zeros(nely,dtype=np.float64)
       eta_profile=np.zeros(nely,dtype=np.float64)
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               eta_profile[j]+=eta_e[counter]/nelx
               y_e_profile[j]+=y_e[counter]/nelx
               counter += 1
           #end for
       #end for
       np.savetxt('eta_profile.ascii',np.array([y_e_profile,eta_profile]).T,header='# y,eta')

       # export dynamic topography
       xx=np.zeros(nelx,dtype=np.float64)
       DT=np.zeros(nelx,dtype=np.float64)
       counter=0
       for j in range(0,nely):
           for i in range(0,nelx):
               if j==nely-1:
                  xx[i]=x_e[counter]
                  DT[i]=sigmayy_e[counter]/rho0/abs(gy)
               counter += 1
           #end for
       #end for
       np.savetxt('dynamic_topography_'+str(istep)+'.ascii',np.array([xx,DT]).T,header='# x,DT')

       # export to vtu
       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
          vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       p.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx' Format='ascii'> \n")
       dTdx_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy' Format='ascii'> \n")
       dTdy_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
       exx_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
       eyy_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
       exy_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tauxx' Format='ascii'> \n")
       tauxx_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tauyy' Format='ascii'> \n")
       tauyy_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='tauxy' Format='ascii'> \n")
       tauxy_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
       eta_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
       rho_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       divv=exx_e+eyy_e
       divv.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       sr.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       q.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       T.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m))
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

    print("export to vtu & ascii: %.3f s" % (clock.time()-start))

    ##########################################################################
    # write to file  | not the most elegant way, but does the job
    ##########################################################################
    start=clock.time()

    np.savetxt('vrms_Nu.ascii',np.array([model_time[0:istep],vrms[0:istep],Nu[0:istep]]).T,header='# t,vrms,Nu')
    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep],Tavrg[0:istep]]).T,header='# t,Tavrg')
    np.savetxt('heat_flux_boundary.ascii',np.array([model_time[0:istep],heatflux_boundary[0:istep]]).T,header='# t,q')
    np.savetxt('u_stats.ascii',np.array([model_time[0:istep],u_stats[0:istep,0],u_stats[0:istep,1]]).T,header='# t,m(u),M(u)')
    np.savetxt('v_stats.ascii',np.array([model_time[0:istep],v_stats[0:istep,0],v_stats[0:istep,1]]).T,header='# t,m(v),M(v)')
    np.savetxt('T_stats.ascii',np.array([model_time[0:istep],T_stats[0:istep,0],T_stats[0:istep,1]]).T,header='# t,m(T),M(T)')
    np.savetxt('dt_stats.ascii',np.array([model_time[0:istep],dt_stats[0:istep]]).T,header='# t,dt')

    print("output stats: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# END OF TIMESTEPPING
###############################################################################
###############################################################################

convfile.close()
niterfile.close()

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
