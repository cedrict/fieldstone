import time as clock 
import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_V(r,s,t):
    N0=0.125*(1.-r)*(1.-s)*(1.-t)
    N1=0.125*(1.+r)*(1.-s)*(1.-t)
    N2=0.125*(1.+r)*(1.+s)*(1.-t)
    N3=0.125*(1.-r)*(1.+s)*(1.-t)
    N4=0.125*(1.-r)*(1.-s)*(1.+t)
    N5=0.125*(1.+r)*(1.-s)*(1.+t)
    N6=0.125*(1.+r)*(1.+s)*(1.+t)
    N7=0.125*(1.-r)*(1.+s)*(1.+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def basis_functions_V_dr(r,s,t):
    dNdr0=-0.125*(1.-s)*(1.-t) 
    dNdr1=+0.125*(1.-s)*(1.-t)
    dNdr2=+0.125*(1.+s)*(1.-t)
    dNdr3=-0.125*(1.+s)*(1.-t)
    dNdr4=-0.125*(1.-s)*(1.+t)
    dNdr5=+0.125*(1.-s)*(1.+t)
    dNdr6=+0.125*(1.+s)*(1.+t)
    dNdr7=-0.125*(1.+s)*(1.+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)

def basis_functions_V_ds(r,s,t):
    dNds0=-0.125*(1.-r)*(1.-t) 
    dNds1=-0.125*(1.+r)*(1.-t)
    dNds2=+0.125*(1.+r)*(1.-t)
    dNds3=+0.125*(1.-r)*(1.-t)
    dNds4=-0.125*(1.-r)*(1.+t)
    dNds5=-0.125*(1.+r)*(1.+t)
    dNds6=+0.125*(1.+r)*(1.+t)
    dNds7=+0.125*(1.-r)*(1.+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def basis_functions_V_dt(r,s,t):
    dNdt0=-0.125*(1.-r)*(1.-s) 
    dNdt1=-0.125*(1.+r)*(1.-s)
    dNdt2=-0.125*(1.+r)*(1.+s)
    dNdt3=-0.125*(1.-r)*(1.+s)
    dNdt4=+0.125*(1.-r)*(1.-s)
    dNdt5=+0.125*(1.+r)*(1.-s)
    dNdt6=+0.125*(1.+r)*(1.+s)
    dNdt7=+0.125*(1.-r)*(1.+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

###############################################################################

eps=1.e-10
year=3.154e+7
Myear=1e6*year
sqrt3=np.sqrt(3.)
TKelvin=273.15

print("*******************************")
print("********** stone 020 **********")
print("*******************************")

ndim=3    # number of dimensions
m_V=8     # number of V nodes making up an element
m_T=8     # number of T nodes making up an element
ndof_V=3  # number of velocity degrees of freedom per node

Lx=1.0079*2700e3
Ly=0.6283*2700e3
Lz=1.0000*2700e3

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx=12
   nely=12
   nelz=12

# this is a requirement bc we measure at z=3Lz/4
assert (nelz%4==0), "nelz should be even and multiple of 4" 

do_jcb=False

gx=0
gy=0
gz=-10

visu=1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

nn_V=nnx*nny*nnz     # number of nodes
nn_T=nnx*nny*nnz     # number of nodes
nel=nelx*nely*nelz   # number of elements, total
Nfem_V=nn_V*ndof_V   # number of V dofs
NfemP=nel            # number of pressure dofs
Nfem=Nfem_V+NfemP    # total number of dofs
Nfem_T=nn_T          # number of T nodes

Temperature1=3700+TKelvin
Temperature2=   0+TKelvin

T0=TKelvin

rho0=3300.
eta0=8.0198e23
hcond=3.564
hcapa=1080
alpha=1.e-5

nstep=250

tol_Nu=1e-5

Ra=alpha*abs(gz)*(Temperature1-Temperature2)*Lz**3*rho0**2*hcapa/hcond/eta0

kappa=hcond/rho0/hcapa
reftime=Lz**2/kappa
tfinal=5*reftime

eta_ref=1.e23     
scaling_coeff=eta_ref/Lz # scaling of G block

r_V=[-1,1,1,-1,-1,1,1,-1]
s_V=[-1,-1,1,1,-1,-1,1,1]
t_V=[-1,-1,-1,-1,1,1,1,1]

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

relax_V=.9
relax_T=.9

debug=False

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("nn_V=",nn_V)
print("Ra=",Ra,3e5)
print("relax_V",relax_V)
print("relax_T",relax_T)
print("kappa=",kappa)
print("reftime=",reftime/Myear,'Myr') 
print("tfinal=",tfinal/Myear,'Myr') 
print("------------------------------")

###############################################################################

model_time=np.zeros(nstep,dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
Tavrg=np.zeros(nstep,dtype=np.float64)
Tm=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
w_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)
wmid_stats=np.zeros((nstep,4),dtype=np.float64) # velocities at z=Lz/2
Tmid_stats=np.zeros((nstep,4),dtype=np.float64) # temperatures at z=Lz/2
hf_stats=np.zeros((nstep,4),dtype=np.float64)

Nu_old=0.       
Nufile=open('Nu.ascii',"w")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x_V[counter]=i*hx
            y_V[counter]=j*hy
            z_V[counter]=k*hz
            counter += 1
        # end for k
    # end for j
# end for i

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_V[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon_V[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon_V[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon_V[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon_V[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon_V[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon_V[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon_V[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        # end for k
    # end for j
# end for i

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions velocity
# free slip on the sides, no slip at the bottom
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_valV=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V+0]=True ; bc_valV[i*ndof_V+0]=0
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_valV[i*ndof_V+0]=0
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_valV[i*ndof_V+1]=0
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1]=True ; bc_valV[i*ndof_V+1]=0
    if z_V[i]/Lz<eps:
       bc_fix_V[i*ndof_V+0]=True ; bc_valV[i*ndof_V+0]=0
       bc_fix_V[i*ndof_V+1]=True ; bc_valV[i*ndof_V+1]=0
       bc_fix_V[i*ndof_V+2]=True ; bc_valV[i*ndof_V+2]=0
    if z_V[i]/Lz>(1-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_valV[i*ndof_V+0]=0 
       bc_fix_V[i*ndof_V+1]=True ; bc_valV[i*ndof_V+1]=0 
       bc_fix_V[i*ndof_V+2]=True ; bc_valV[i*ndof_V+2]=0 
# end for

print("boundary conditions V: %.3f s" % (clock.time()-start))

###############################################################################
# temperature grid setup
###############################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64)  # x coordinates
y_T=np.zeros(nn_T,dtype=np.float64)  # y coordinates
z_T=np.zeros(nn_T,dtype=np.float64)  # z coordinates
icon_T=np.zeros((m_T,nel),dtype=np.int32)

x_T[:]=x_V[:]
y_T[:]=y_V[:]
z_T[:]=z_V[:]
icon_T[:,:]=icon_V[:,:]

print("build grid T: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions temperature: T1 at bottom, T2 at top
###############################################################################
start=clock.time()

bc_fix_T=np.zeros(Nfem_T,dtype=bool) # boundary condition, yes/no
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_T):
    if z_T[i]<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = Temperature1
    if z_T[i]/Lz>1-eps:
       bc_fix_T[i] = True ; bc_val_T[i] = Temperature2
# end for

print("boundary conditions T: %.3f s" % (clock.time()-start))

###############################################################################
# initial temperature field 
###############################################################################
start=clock.time()

T=np.zeros(nn_T,dtype=np.float64) 
T_old=np.zeros(nn_T,dtype=np.float64) 

for i in range(0,nn_T):
   T[i]=(Temperature2-Temperature1)/Lz*z_T[i]+Temperature1 \
       +100*(np.cos(np.pi*x_T[i]/Lx) + np.cos(np.pi*y_T[i]/Ly))*np.sin(np.pi*z_T[i]/Lz)
# end for

T_old[:]=T[:]

print("initial temperature: %.3f s" % (clock.time()-start))

#########################################################################################
#########################################################################################
# TIME STEPPING
#########################################################################################
#########################################################################################

C=np.array([[2,0,0,0,0,0],\
            [0,2,0,0,0,0],\
            [0,0,2,0,0,0],\
            [0,0,0,1,0,0],\
            [0,0,0,0,1,0],\
            [0,0,0,0,0,1]],dtype=np.float64) 

u_old=np.zeros(nn_V,dtype=np.float64) # x-component velocity
v_old=np.zeros(nn_V,dtype=np.float64) # y-component velocity
w_old=np.zeros(nn_V,dtype=np.float64) # y-component velocity
                    
#Note that instead of computing jcob and jcbi by means of the 
# determinant I instead assign it its value under the assumption that 
# the elements are all cuboids of size hxXhyXhz. 
jcob=hx*hy*hz/8
jcbi=np.zeros((ndim,ndim),dtype=np.float64)
jcbi[0,0]=2/hx ; jcbi[0,1]=0    ; jcbi[0,2]=0
jcbi[1,0]=0    ; jcbi[1,1]=2/hy ; jcbi[1,2]=0
jcbi[2,0]=0    ; jcbi[2,1]=0    ; jcbi[2,2]=2/hz

for istep in range(0,nstep):
    print("--------------------------------------------")
    print("istep= ", istep)
    print("--------------------------------------------")

    ###########################################################################
    # build FE matrix
    ###########################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    b_fem=np.zeros(Nfem,dtype=np.float64)       # right hand side of Ax=b
    f_rhs=np.zeros(Nfem_V,dtype=np.float64)     # right hand side f 
    h_rhs=np.zeros(NfemP,dtype=np.float64)      # right hand side h 
    B=np.zeros((6,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
    jcb=np.zeros((ndim,ndim),dtype=np.float64)

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
        h_el=0. #np.zeros((1,1),dtype=np.float64)

        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.

                    N_V=basis_functions_V(rq,sq,tq)
                    dNdr_V=basis_functions_V_dr(rq,sq,tq)
                    dNds_V=basis_functions_V_ds(rq,sq,tq)
                    dNdt_V=basis_functions_V_dt(rq,sq,tq)
                    if do_jcb:
                       jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                       jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                       jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                       jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                       jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                       jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                       jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                       jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                       jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                       jcbi=np.linalg.inv(jcb)
                       JxWq=np.linalg.det(jcb)*weightq
                    else:
                       JxWq=jcob**weightq
                    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                    for i in range(0,m_V):
                        B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                            [0.       ,dNdy_V[i],0.       ],
                                            [0.       ,0.       ,dNdz_V[i]],
                                            [dNdy_V[i],dNdx_V[i],0.       ],
                                            [dNdz_V[i],0.       ,dNdx_V[i]],
                                            [0.       ,dNdz_V[i],dNdy_V[i]]]

                    K_el+=B.T.dot(C.dot(B))*eta0*JxWq

                    Tq=np.dot(N_V,T[icon_V[:,iel]])

                    for i in range(0,m_V):
                        f_el[ndof_V*i+2]+=N_V[i]*JxWq*gz*rho0*(1-alpha*(Tq-T0))
                        G_el[ndof_V*i+0,0]-=dNdx_V[i]*jcob*weightq
                        G_el[ndof_V*i+1,0]-=dNdy_V[i]*jcob*weightq
                        G_el[ndof_V*i+2,0]-=dNdz_V[i]*jcob*weightq

                #end for kq
            #end for jq
        #end for iq

        # impose b.c. 
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1+i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                if bc_fix_V[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,m_V*ndof_V):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_valV[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for 
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_valV[m1]
                   h_el-=G_el[ikk,0]*bc_valV[m1]
                   G_el[ikk,0]=0
                #end if
            #end for 
        #end for 

        # assemble matrix and right hand side
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1+i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2+i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_fem[m1,m2]+=K_el[ikk,jkk]
                    #end for 
                #end for 
                b_fem[m1]+=f_el[ikk]
                #G_mat[m1,iel]+=G_el[ikk,0]*scaling_coeff
                A_fem[m1,Nfem_V+iel]+=G_el[ikk,0]*scaling_coeff
                A_fem[Nfem_V+iel,m1]+=G_el[ikk,0]*scaling_coeff
            #end for 
        #end for 
        b_fem[Nfem_V+iel]+=h_el*scaling_coeff

    #end iel

    print("build FE matrix Stokes: %.3f s" % (clock.time()-start))

    ###########################################################################
    # assemble K, G, GT, f, h into A and rhs
    ###########################################################################
    #start = clock.time()
    #if pnormalise:
    #   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
    #   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
    #   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
    #   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
    #   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
    #   a_mat[Nfem,Nfem_V:Nfem]=1
    #   a_mat[Nfem_V:Nfem,Nfem]=1
    #else:
    #   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    #   a_mat[0:Nfem_V,0:Nfem_V]=K_mat
    #   a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
    #   a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
    #rhs[0:Nfem_V]=f_rhs
    #rhs[Nfem_V:Nfem]=h_rhs
    #print("assemble blocks: %.3f s" % (clock.time() - start))
    ###########################################################################
    #a_mat[Nfem_V-1,:]=0
    #a_mat[:,Nfem_V-1]=0
    #a_mat[Nfem_V-1,Nfem_V-1]=1
    #rhs[Nfem_V-1]=0
    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

    print("solve Stokes system: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v,w=np.reshape(sol[0:Nfem_V],(nn_V,ndim)).T
    p=sol[Nfem_V:Nfem]*scaling_coeff

    p-=(np.sum(p)/nel) # pressure normalisation

    print("     -> u (m,M) %.4e %.4e " %(np.min(u)*year,np.max(u)*year))
    print("     -> v (m,M) %.4e %.4e " %(np.min(v)*year,np.max(v)*year))
    print("     -> w (m,M) %.4e %.4e " %(np.min(w)*year,np.max(w)*year))
    print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

    u_stats[istep,0]=np.min(u)
    u_stats[istep,1]=np.max(u)
    v_stats[istep,0]=np.min(v)
    v_stats[istep,1]=np.max(v)
    w_stats[istep,0]=np.min(w)
    w_stats[istep,1]=np.max(w)

    if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')

    print("transfer solution: %.3f s" % (clock.time()-start))

    ###########################################################################
    # relaxation step
    ###########################################################################
    start=clock.time()

    if istep>0:
       u=relax_V*u+(1-relax_V)*u_old
       v=relax_V*v+(1-relax_V)*v_old
       w=relax_V*w+(1-relax_V)*w_old

    print("relax velocity solution: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute time step value 
    ###########################################################################
    start=clock.time()

    #CFL_nb=0.005 # not needed since relaxation is used
    #dt1=CFL_nb*min(Lx/nelx,Ly/nely,Lz/nelz)/np.max(np.sqrt(u**2+v**2+w**2))
    #dt2=CFL_nb*min(Lx/nelx,Ly/nely,Lz/nelz)**2/(hcond/hcapa/rho0)
    #dt=min(dt1,dt2)
    #if istep==0:
    #   model_time[istep]=dt
    #else:
    #   model_time[istep]=model_time[istep-1]+dt
    #dt_stats[istep]=dt 
    #print('     -> dt1= %.6e dt2= %.6e dt= %.6e Myr' % (dt1/Myear,dt2/Myear,dt/Myear))

    model_time[istep]=istep

    print("compute timestep: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute vrms. 
    ###########################################################################
    start=clock.time()

    for iel in range(0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.
                    N_V=basis_functions_V(rq,sq,tq)

                    dNdr_V=basis_functions_V_dr(rq,sq,tq)
                    dNds_V=basis_functions_V_ds(rq,sq,tq)
                    dNdt_V=basis_functions_V_dt(rq,sq,tq)
                    if do_jcb:
                       jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                       jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                       jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                       jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                       jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                       jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                       jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                       jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                       jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                       JxWq=np.linalg.det(jcb)*weightq
                    else:
                       JxWq=jcob**weightq

                    uq=np.dot(N_V,u[icon_V[:,iel]])
                    vq=np.dot(N_V,v[icon_V[:,iel]])
                    wq=np.dot(N_V,w[icon_V[:,iel]])
                    Tq=np.dot(N_V,T[icon_T[:,iel]])
                    vrms[istep]+=(uq**2+vq**2+wq**2)*JxWq
                    Tavrg[istep]+=Tq*JxWq
                # end for ik
            # end for jk
        # end for kq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly*Lz))
    Tavrg[istep]/=Lx*Ly*Lz
    Tavrg[istep]-=TKelvin

    print("     -> vrms= %.4e ; vrmsdiff= %.3e " % (vrms[istep]/year,(vrms[istep]-vrms[istep-1])*year))

    print("compute vrms: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build FE matrix for Temperature 
    ###########################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)          # FE rhs 
    B=np.zeros((3,m_T),dtype=np.float64)             # gradient matrix B 
    NNNT_mat=np.zeros((m_T,1),dtype=np.float64)      # basis fcts matrix
    Tvect=np.zeros(m_T,dtype=np.float64)   

    for iel in range (0,nel):

        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        b_el=np.zeros(m_T,dtype=np.float64)
        Ka=np.zeros((m_T,m_T),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_T,m_T),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_T,m_T),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        Tvect[:]=T[icon_T[:,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:

                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.

                    N_V=basis_functions_V(rq,sq,tq)
                    NNNT_mat[:,0]=N_V[:]
                    dNdr_V=basis_functions_V_dr(rq,sq,tq)
                    dNds_V=basis_functions_V_ds(rq,sq,tq)
                    dNdt_V=basis_functions_V_dt(rq,sq,tq)
                    if do_jcb:
                       jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                       jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                       jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                       jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                       jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                       jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                       jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                       jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                       jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                       jcbi=np.linalg.inv(jcb)
                       JxWq=np.linalg.det(jcb)*weightq
                    else:
                       JxWq=jcob*weightq

                    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

                    vel[0,0]=np.dot(N_V,u[icon_V[:,iel]])
                    vel[0,1]=np.dot(N_V,v[icon_V[:,iel]])
                    vel[0,2]=np.dot(N_V,w[icon_V[:,iel]])

                    B[0,:]=dNdx_V[:]
                    B[1,:]=dNdy_V[:]
                    B[2,:]=dNdz_V[:]

                    # compute mass matrix
                    #MM=NNNT_mat.dot(NNNT_mat.T)*rho_lhs*hcapa*weightq*jcob

                    # compute diffusion matrix
                    Kd=B.T.dot(B)*hcond*JxWq

                    # compute advection matrix
                    Ka=NNNT_mat.dot(vel.dot(B))*rho0*hcapa*JxWq

                    #alphaT=0.5
                    #a_el+=MM+alphaT*(Ka+Kd)*dt
                    #b_el+=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvect)

                    A_el+=(Kd+Ka)

                #end for
            #end for
        #end for

        # apply boundary conditions

        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               # end for
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]
            # end if
        # end for

        # assemble matrix and right hand side
        for k1 in range(0,m_T):
            m1=icon_T[k1,iel]
            for k2 in range(0,m_T):
                m2=icon_T[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            # end for
            b_fem[m1]+=b_el[k1]
        # end for

    #end for

    #print("     -> A_fem (m,M) = %.4e %.4e" %(np.min(A_fem),np.max(A_fem)))
    #print("     -> b_fem (m,M) = %.4e %.4e" %(np.min(b_fem),np.max(b_fem)))

    print("build FE matrix T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T)-TKelvin,np.max(T)-TKelvin))

    T_stats[istep,0]=np.min(T)-TKelvin 
    T_stats[istep,1]=np.max(T)-TKelvin

    print("solve energy system: %.3f s" % (clock.time()-start))

    ###########################################################################
    # relax
    ###########################################################################
    start=clock.time()

    T=relax_T*T+(1-relax_T)*T_old

    print("relax temperature: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal strainrate on velocity grid
    # I should have separated the strainrate calculations from the 
    # temperature derivatives calculations for consistency but 
    # since the velocity and temperature basis functions are the same
    # and so are the connectivity arrays I save much time doing this.
    ###########################################################################
    start=clock.time()

    srn=np.zeros(nn_V,dtype=np.float64)
    exxn=np.zeros(nn_V,dtype=np.float64)
    eyyn=np.zeros(nn_V,dtype=np.float64)
    ezzn=np.zeros(nn_V,dtype=np.float64)
    exyn=np.zeros(nn_V,dtype=np.float64)
    exzn=np.zeros(nn_V,dtype=np.float64)
    eyzn=np.zeros(nn_V,dtype=np.float64)
    dTdxn=np.zeros(nn_T,dtype=np.float64)
    dTdyn=np.zeros(nn_T,dtype=np.float64)
    dTdzn=np.zeros(nn_T,dtype=np.float64)
    c=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,m_V):
            N_V=basis_functions_V(r_V[i],s_V[i],t_V[i])
            dNdr_V=basis_functions_V_dr(r_V[i],s_V[i],t_V[i])
            dNds_V=basis_functions_V_ds(r_V[i],s_V[i],t_V[i])
            dNdt_V=basis_functions_V_dt(r_V[i],s_V[i],t_V[i])
            if do_jcb:
               jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
               jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
               jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
               jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
               jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
               jcbi=np.linalg.inv(jcb)

            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
            dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V

            exxn[icon_V[i,iel]]+=np.dot(dNdx_V[:],u[icon_V[:,iel]])
            eyyn[icon_V[i,iel]]+=np.dot(dNdy_V[:],v[icon_V[:,iel]])
            ezzn[icon_V[i,iel]]+=np.dot(dNdz_V[:],w[icon_V[:,iel]])
            exyn[icon_V[i,iel]]+=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5+np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
            exzn[icon_V[i,iel]]+=np.dot(dNdz_V[:],u[icon_V[:,iel]])*0.5+np.dot(dNdx_V[:],w[icon_V[:,iel]])*0.5
            eyzn[icon_V[i,iel]]+=np.dot(dNdz_V[:],v[icon_V[:,iel]])*0.5+np.dot(dNdy_V[:],w[icon_V[:,iel]])*0.5

            dTdxn[icon_V[i,iel]]+=np.dot(dNdx_V[:],T[icon_V[:,iel]])
            dTdyn[icon_V[i,iel]]+=np.dot(dNdy_V[:],T[icon_V[:,iel]])
            dTdzn[icon_V[i,iel]]+=np.dot(dNdz_V[:],T[icon_V[:,iel]])
 
            c[icon_V[i,iel]]+=1.
        # end for i
    # end for iel
    exxn/=c
    eyyn/=c
    ezzn/=c
    exyn/=c
    exzn/=c
    eyzn/=c
    dTdxn/=c
    dTdyn/=c
    dTdzn/=c

    srn=np.sqrt(0.5*(exxn**2+eyyn**2+ezzn**2)+exyn**2+exzn**2+eyzn**2)

    print("     -> exx (m,M) %.4e %.4e " %(np.min(exxn),np.max(exxn)))
    print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyyn),np.max(eyyn)))
    print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezzn),np.max(ezzn)))
    print("     -> exy (m,M) %.4e %.4e " %(np.min(exyn),np.max(exyn)))
    print("     -> exz (m,M) %.4e %.4e " %(np.min(exzn),np.max(exzn)))
    print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyzn),np.max(eyzn)))
    print("     -> dTdx (m,M) %.4e %.4e " %(np.min(dTdxn),np.max(dTdxn)))
    print("     -> dTdy (m,M) %.4e %.4e " %(np.min(dTdyn),np.max(dTdyn)))
    print("     -> dTdz (m,M) %.4e %.4e " %(np.min(dTdzn),np.max(dTdzn)))

    print("compute nod strain rate: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal pressure
    ###########################################################################
    start=clock.time()

    q=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for k in range(0,m_V):
            q[icon_V[k,iel]]+=p[iel]
            count[icon_V[k,iel]]+=1
        # end for
    # end for

    q/=count

    if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute q: %.3f s" % (clock.time()-start))

    ###########################################################################
    # velocity and temperature at mid side edges
    ###########################################################################
    start=clock.time()

    for i in range(0,nn_V):
        if abs(z_V[i]-Lz/2.)/Lz<eps:
           if x_V[i]/Lx<eps and y_V[i]/Ly<eps: 
              wmid_stats[istep,0]=w[i]
              Tmid_stats[istep,0]=T[i]
           if x_V[i]/Lx>1-eps and y_V[i]/Ly<eps: 
              wmid_stats[istep,1]=w[i]
              Tmid_stats[istep,1]=T[i]
           if x_V[i]/Lx<eps and y_V[i]/Ly>1-eps: 
              wmid_stats[istep,2]=w[i]
              Tmid_stats[istep,2]=T[i]
           if x_V[i]/Lx>1-eps and y_V[i]/Ly>1-eps: 
              wmid_stats[istep,3]=w[i]
              Tmid_stats[istep,3]=T[i]
        # end if
    # end for

    for i in range(0,nn_T):
        if z_T[i]/Lz>1-eps:
           if x_T[i]/Lx<eps and y_T[i]/Ly<eps: 
              hf_stats[istep,0]=dTdzn[i]*hcond
           if x_T[i]/Lx>1-eps and y_T[i]/Ly<eps: 
              hf_stats[istep,1]=dTdzn[i]*hcond
           if x_T[i]/Lx<eps and y_T[i]/Ly>1-eps: 
              hf_stats[istep,2]=dTdzn[i]*hcond
           if x_T[i]/Lx>1-eps and y_T[i]/Ly>1-eps: 
              hf_stats[istep,3]=dTdzn[i]*hcond
        # end if
    # end for

    print("measurements: %.3f s" % (clock.time()-start))

    ###########################################################################
    # average temperature at z=3Lz/4
    ###########################################################################
    start=clock.time()

    ielztarget=3*nelz/4-1
 
    T_m=0.
    iel=0
    for ielx in range(nelx):
        for iely in range(nely):
            for ielz in range(nelz):
                if ielz==ielztarget:

                   for iq in [-1,1]:
                       for jq in [-1,1]:
                           rq=iq/sqrt3
                           sq=jq/sqrt3
                           weightq=1.*1.
                           N0=0.25*(1-rq)*(1-sq)
                           N1=0.25*(1+rq)*(1-sq)
                           N2=0.25*(1+rq)*(1+sq)
                           N3=0.25*(1-rq)*(1+sq)
                           Tq=N0*T[icon_T[4,iel]]+\
                              N1*T[icon_T[5,iel]]+\
                              N2*T[icon_T[6,iel]]+\
                              N3*T[icon_T[7,iel]]
                           jcobb=hx*hy/4
                           T_m+=Tq*jcobb*weightq
                # end if
                iel+=1
            # end for
        # end for
    # end for

    T_m/=(Lx*Ly)

    print("     -> avrg T at z=3Lz/4= %.4e " % (T_m))

    Tm[istep]=T_m

    print("compute avrg T at z=3Lz/4: %.3f s" % (clock.time()-start))

    ###########################################################################
    # Nusselt number 
    ###########################################################################
    start=clock.time()

    Nu=0.
    iel=0
    for ielx in range(nelx):
        for iely in range(nely):
            for ielz in range(nelz):
                if ielz==nelz-1:
                   for iq in [-1,1]:
                       for jq in [-1,1]:
                           rq=iq/sqrt3
                           sq=jq/sqrt3
                           weightq=1.*1.
                           N0=0.25*(1-rq)*(1-sq)
                           N1=0.25*(1+rq)*(1-sq)
                           N2=0.25*(1+rq)*(1+sq)
                           N3=0.25*(1-rq)*(1+sq)
                           dTdzq=N0*dTdzn[icon_T[4,iel]]+\
                                 N1*dTdzn[icon_T[5,iel]]+\
                                 N2*dTdzn[icon_T[6,iel]]+\
                                 N3*dTdzn[icon_T[7,iel]]
                           jcobb=hx*hy/4
                           Nu+=abs(dTdzq*jcobb*weightq)
                      # end for
                   # end for
                # end if
                iel+=1
            # end for
        # end for
    # end for

    Nu*=(Lz/(Lx*Ly*(Temperature1-273)))

    Nufile.write("%d %e\n" % (istep,Nu)) ; Nufile.flush()

    print("     -> Nu= %.6e  " % Nu)

    print("compute Nusselt number: %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if visu==1 and istep%20==0:

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%.5e %.5e %.5e \n" %(x_V[i],y_V[i],z_V[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       p.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f %10f %10f \n" %(u[i]*year,v[i]*year,w[i]*year))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T (K)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f \n" %(T[i]-TKelvin))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx(n)' Format='ascii'> \n")
       exxn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy(n)' Format='ascii'> \n")
       eyyn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='ezz(n)' Format='ascii'> \n")
       ezzn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy(n)' Format='ascii'> \n")
       exyn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exz(n)' Format='ascii'> \n")
       exzn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyz(n)' Format='ascii'> \n")
       eyzn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sr(n)' Format='ascii'> \n")
       srn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx(n)' Format='ascii'> \n")
       dTdxn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy(n)' Format='ascii'> \n")
       dTdyn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdz(n)' Format='ascii'> \n")
       dTdzn.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],\
                                                       icon_V[2,iel],icon_V[3,iel],\
                                                       icon_V[4,iel],icon_V[5,iel],\
                                                       icon_V[6,iel],icon_V[7,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %12)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("export to vtu: %.3f s" % (clock.time()-start))

    ###########################################################################
    start = clock.time()

    np.savetxt('vrms.ascii',np.array([model_time[0:istep]/Myear,\
                                      vrms[0:istep]*year]).T,header='# t/year,vrms')

    np.savetxt('u_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         u_stats[0:istep,0]*year,\
                                         u_stats[0:istep,1]*year]).T,header='# t/year,min(u),max(u)')

    np.savetxt('v_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         v_stats[0:istep,0]*year,\
                                         v_stats[0:istep,1]*year]).T,header='# t/year,min(v),max(v)')

    np.savetxt('w_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         w_stats[0:istep,0]*year,\
                                         w_stats[0:istep,1]*year]).T,header='# t/year,min(w),max(w)')

    np.savetxt('T_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         T_stats[0:istep,0],\
                                         T_stats[0:istep,1]]).T,header='# t/year,min(T),max(T)')

    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep]/Myear,\
                                       Tavrg[0:istep]]).T,header='# t/year,Tavrg')

    np.savetxt('Tm.ascii',np.array([model_time[0:istep]/Myear,\
                                    Tm[0:istep]]).T,header='# t/year,Tm at z=3Lz/4')

    np.savetxt('wmid_stats.ascii',np.array([model_time[0:istep]/Myear,
                                            wmid_stats[0:istep,0],\
                                            wmid_stats[0:istep,1],\
                                            wmid_stats[0:istep,2],\
                                            wmid_stats[0:istep,3]]).T,header='# t/year,w1,w2,w3,w4')

    np.savetxt('Tmid_stats.ascii',np.array([model_time[0:istep]/Myear,
                                            Tmid_stats[0:istep,0],\
                                            Tmid_stats[0:istep,1],\
                                            Tmid_stats[0:istep,2],\
                                            Tmid_stats[0:istep,3]]).T,header='# t/year,T1,T2,T3,T4')

    np.savetxt('hf_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                          hf_stats[0:istep,0],\
                                          hf_stats[0:istep,1],\
                                          hf_stats[0:istep,2],\
                                          hf_stats[0:istep,3]]).T,header='# t/year,hf1,hf2,hf3,hf4')

    print("export stats to ascii: %.3f s" % (clock.time()-start))

    ###########################################################################

    if np.abs(Nu-Nu_old)<tol_Nu and abs(Tm[istep]-Tm[istep-1])<1:
       print("Nu converged to 1e-6")
       break

    ###########################################################################

    u_old[:]=u[:]
    v_old[:]=v[:]
    w_old[:]=w[:]
    T_old[:]=T[:]
    Nu_old=Nu

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
