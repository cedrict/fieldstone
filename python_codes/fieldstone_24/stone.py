import numpy as np
import sys as sys
import time as clock 
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

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

cm=0.01
year=3.154e+7
cm_per_year=cm/year
eps=1.e-10
sqrt3=np.sqrt(3.)
TKelvin=273.15
MPa=1e6

print("*******************************")
print("********** stone 024 **********")
print("*******************************")

if order==1: m_V=4 # number of V-nodes making up an element
if order==2: m_V=9 # number of V-nodes making up an element
if order==1: m_P=1 # number of P-nodes making up an element
if order==2: m_P=4 # number of P-nodes making up an element
ndof_V=2           # number of velocity degrees of freedom per node

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   Ra   = float(sys.argv[4])
else:
   nelx = 24
   nely = 24
   visu = 0
   Ra   = 1e5 # Rayleigh number

nel=nelx*nely                       # number of elements, total
nnx=order*nelx+1                    # number of nodes, x direction
nny=order*nely+1                    # number of nodes, y direction
nn_V=nnx*nny                        # number of V-nodes
if order==1: nn_P=nelx*nely         # number of P-nodes
if order==2: nn_P=(nelx+1)*(nely+1) # number of P-nodes
Nfem_V=nn_V*ndof_V                  # number of velocity dofs
Nfem_P=nn_P                         # number of pressure dofs
Nfem=Nfem_V+Nfem_P                  # total number of dofs
Nfem_T=nn_V                         # number of T dofs

Lx=3000e3
Ly=3000e3
Di=0.5      # dissipation number
hcond=3.    # thermal conductivity
hcapa=1250. # heat capacity
hprod=0     # heat production coeff
gx=0          
gy=-10
rho0=3000       # reference density
T0=273.15       # reference temperature
Delta_Temp=4000 # temperature difference 
Tsurf=273.15    # reference temperature

kappa=hcond/rho0/hcapa
alphaT=Di*hcapa/abs(gy)/Ly
eta0=Di*hcapa**2*Delta_Temp*Ly**2*rho0**2/Ra/hcond
reftime=rho0*hcapa*Ly**2/hcond
refvel=Ly/reftime
refTemp=Delta_Temp
refPress=eta0*hcond/rho0/hcapa/Ly**2

print ("     -> kappa %e " % kappa)
print ("     -> alphaT %e " % alphaT)
print ("     -> eta %e " %  eta0)
print ("     -> reftime %e " %  reftime)
print ("     -> refvel %e " %  refvel)
print ("     -> refPress %e " %  refPress)

CFL_nb=0.75

nstep=1000

pnormalise=True

use_EBA=False
use_BA=True

debug=False

if use_BA:
   compressible=False
   use_shearheating=False
   use_adiabatic_heating=False

if use_EBA:
   compressible=False
   use_shearheating=True
   use_adiabatic_heating=True

betaT=0
if not compressible: betaT=0

hx=Lx/nelx
hy=Ly/nely

###############################################################################

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

print("*******************************")
print('order=',order)
print('nelx=',nelx)
print('nely=',nely)
print('nnx=',nnx)
print('nny=',nny)
print('nn_V=',nn_V)
print('nn_P=',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)
print("*******************************")

###############################################################################

model_time=np.zeros(nstep,dtype=np.float64) 
vrms=np.zeros(nstep,dtype=np.float64) 
Nu=np.zeros(nstep,dtype=np.float64)
Tavrg=np.zeros(nstep,dtype=np.float64)
u_stats=np.zeros((nstep,2),dtype=np.float64)
v_stats=np.zeros((nstep,2),dtype=np.float64)
T_stats=np.zeros((nstep,2),dtype=np.float64)
mass=np.zeros(nstep,dtype=np.float64)
visc_diss=np.zeros(nstep,dtype=np.float64)
work_grav=np.zeros(nstep,dtype=np.float64)
EK=np.zeros(nstep,dtype=np.float64)
EG=np.zeros(nstep,dtype=np.float64)
ET=np.zeros(nstep,dtype=np.float64)
dt_stats=np.zeros(nstep,dtype=np.float64)
heatflux_boundary=np.zeros((nstep,5),dtype=np.float64)
adiab_heating=np.zeros(nstep,dtype=np.float64)

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/order
        y_V[counter]=j*hy/order
        counter+=1

x_P=np.zeros(nn_P,dtype=np.float64)  # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)  # y coordinates

if order==1:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           x_P[counter]=(i+0.5)*hx
           y_P[counter]=(j+0.5)*hx
           counter+=1

if order==2:
   counter = 0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x_P[counter]=i*hx
           y_P[counter]=j*hy
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

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0
    if x_V[i]/Lx>1-eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0
    if y_V[i]/Ly>1-eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0

bc_fix_T=np.zeros(Nfem_T,dtype=bool) # boundary condition, yes/no
bc_val_T=np.zeros(Nfem_T,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if y_V[i]/Ly<eps:
       bc_fix_T[i] = True ; bc_val_T[i] = Delta_Temp+Tsurf
    if y_V[i]/Ly>1-eps:
       bc_fix_T[i] = True ; bc_val_T[i] = Tsurf

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# nodal pressure setup
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
q_prev=np.zeros(nn_V,dtype=np.float64)  
dqdt=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    q[i]=rho0*np.abs(gy)*(Ly-y_V[i])

q_prev[:]=q[:]

print("setup: q: %.3f s" % (clock.time()-start))

###############################################################################
# temperature and nodal density setup
###############################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64)
rho=np.zeros(nn_V,dtype=np.float64)
rho_prev=np.zeros(nn_V,dtype=np.float64)
drhodt=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    T[i]=((Ly-y_V[i])/Ly - 0.01*np.cos(np.pi*x_V[i]/Lx)*np.sin(np.pi*y_V[i]/Ly))*Delta_Temp+Tsurf
    rho[i]=rho0*(1-alphaT*(T[i]-T0)+betaT*q[i])
    
rho_prev[:]=rho[:]

print("setup: T,rho: %.3f s" % (clock.time()-start))

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

print('     -> toral area/Lx/Ly=',(np.sum(area)/Lx/Ly))

print("compute areas: %.3f s" % (clock.time()-start))

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

for istep in range(0,nstep):
    print("----------------------------------")
    print("istep= ", istep)
    print("----------------------------------")

    ###########################################################################
    # build FE matrix
    # [ K   G+W][u]=[f]
    # [GT+Z 0  ][p] [h]
    ###########################################################################
    start=clock.time()

    if pnormalise and order==1:
       A_fem=lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)
       b_fem=np.zeros(Nfem+1,dtype=np.float64)
    else:
       A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
       b_fem=np.zeros(Nfem,dtype=np.float64)
    N_mat=np.zeros((3,m_P),dtype=np.float64)
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
    C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 

    for iel in range(0,nel):

        K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64) 
        Z_el=np.zeros((m_P,m_V*ndof_V),dtype=np.float64)
        W_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
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
                rhoq=np.dot(N_V,rho[icon_V[:,iel]])              ##### use N_P to insure rho>0 ?
                if rhoq<0: exit('rhoq<0')

                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                K_el+=B.T.dot(C.dot(B))*eta0*JxWq

                N_mat[0,:]=N_P[:]
                N_mat[1,:]=N_P[:]

                G_el-=B.T.dot(N_mat)*JxWq 

                if compressible:
                   drhodxq=np.dot(dNdx_V,rho[icon_V[:,iel]])
                   drhodyq=np.dot(dNdy_V,rho[icon_V[:,iel]])
                   for i in range(0,m_V):
                       for j in range(0,m_P):
                           Z_el[j,ndof_V*i  ]-=N_P[j]*N_V[i]*drhodxq/rhoq *JxWq 
                           Z_el[j,ndof_V*i+1]-=N_P[j]*N_V[i]*drhodyq/rhoq *JxWq 
                   #for i in range(0, m):
                   #    W_el[ndof_V*i  ,0]-=N[i]*jcob*wq*gx*betaT ############################ FINISH
                   #    W_el[ndof_V*i+1,0]-=N[i]*jcob*wq*gy*betaT

                for i in range(0,m_V):
                    f_el[ndof_V*i  ]+=N_V[i]*rhoq*gx*JxWq
                    f_el[ndof_V*i+1]+=N_V[i]*rhoq*gy*JxWq

            # end jq
        # end iq

        G_el*=eta0/Ly

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
                    A_fem[Nfem_V+m2,m1]+=Z_el[jkk,ikk]  # W_el ?!?!
                #end for 
                b_fem[m1]+=f_el[ikk]
            #end for 
        #end for 
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            b_fem[Nfem_V+m2]+=h_el[k2]
        #end for 

    #end for iel

    if pnormalise and order==1: ######################## SCALE??!?!?
       A_fem[Nfem,Nfem_V:Nfem]=1
       A_fem[Nfem_V:Nfem,Nfem]=1

    print("build FE matrix: %.3f s" % (clock.time()-start))

    ###############################################################################
    # solve system
    ###############################################################################
    start=clock.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("solve time: %.3f s" % (clock.time()-start))

    ###########################################################################
    # put solution (u,v,p) into separate arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta0/Ly)

    print("     -> u (m,M) %.5e %.5e cm/yr" %(np.min(u)/cm_per_year,np.max(u)/cm_per_year))
    print("     -> v (m,M) %.5e %.5e cm/yr" %(np.min(v)/cm_per_year,np.max(v)/cm_per_year))
    print("     -> p (m,M) %.5e %.5e MPa" %(np.min(p)/MPa,np.max(p)/MPa))

    u_stats[istep,0]=np.min(u)/cm_per_year ; u_stats[istep,1]=np.max(u)/cm_per_year
    v_stats[istep,0]=np.min(v)/cm_per_year ; v_stats[istep,1]=np.max(v)/cm_per_year

    if pnormalise and order==1: print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

    if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

    print("split solution vector in u,v,p: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute time step value 
    ###########################################################################
    start=clock.time()

    dt1=CFL_nb*min(hx,hy)/np.max(np.sqrt(u**2+v**2))
    dt2=CFL_nb*min(hx,hy)**2/kappa
    dt=min(dt1,dt2)

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    dt_stats[istep]=dt 

    print('     -> dt1= %.6e yr ; dt2= %.6e yr ; dt= %.6e yr' % (dt1/year,dt2/year,dt/year))

    print("compute timestep: %.3f s" % (clock.time()-start))

    ###########################################################################
    # normalise pressure
    ###########################################################################
    start=clock.time()

    if order==2:
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
               #end for jq
           #end for iq
       #end for iel

       avrg_p=int_p/(Lx*Ly)

    print("     -> int_p %e " %(int_p))
    print("     -> avrg_p %e " %(avrg_p))

    p[:]-=avrg_p

    print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

    if debug: np.savetxt('p_normalised.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

    print("normalise pressure: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute nodal pressure
    ###########################################################################
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

    dqdt=(q[:]-q_prev[:])/dt

    print("     -> q (m,M) %.4e %.4e MPa" %(np.min(q)/MPa,np.max(q)/MPa))

    print("compute nodal pressure q: %.3f s" % (clock.time()-start))

    ###########################################################################
    # build FE matrix for Temperature 
    ###########################################################################
    start=clock.time()

    A_fem=np.zeros((Nfem_T,Nfem_T),dtype=np.float64) # FE matrix 
    b_fem=np.zeros(Nfem_T,dtype=np.float64)          # FE rhs 
    B_mat=np.zeros((2,m_V),dtype=np.float64)   
    N_mat=np.zeros((m_V,1),dtype=np.float64)  
    Tvect=np.zeros(m_V,dtype=np.float64)   

    for iel in range (0,nel):

        Ka=np.zeros((m_V,m_V),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m_V,m_V),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m_V,m_V),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,2),dtype=np.float64)
        f_el=np.zeros(m_V,dtype=np.float64)

        Tvect[:]=T[icon_V[:,iel]]

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
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                rhoq=np.dot(N_V,rho[icon_V[:,iel]])
                dqdxq=np.dot(dNdx_V,q[icon_V[:,iel]])
                dqdyq=np.dot(dNdy_V,q[icon_V[:,iel]])
                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5

                vel[0,0]=uq
                vel[0,1]=vq
                B_mat[0,:]=dNdx_V[:]
                B_mat[1,:]=dNdy_V[:]

                exxqd=exxq-(exxq+eyyq)/3.
                eyyqd=eyyq-(exxq+eyyq)/3.
                exyqd=exyq
                Phiq=2*eta0*(exxqd**2+eyyqd**2+2*exyqd**2)    

                if use_BA or use_EBA:
                   rho_lhs=rho0
                else:
                   rho_lhs=rhoq

                # compute mass matrix
                MM+=np.outer(N_V,N_V)*rho_lhs*hcapa*JxWq

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*JxWq

                # compute advection matrix
                Ka+=np.outer(N_V,vel.dot(B_mat))*rho_lhs*hcapa*JxWq

                # compute shear heating rhs term
                if use_shearheating: f_el[:]+=N_V[:]*Phiq*JxWq

                # compute adiabatic heating rhs term 
                if use_adiabatic_heating: f_el[:]+=N_V[:]*alphaT*Tq*(uq*dqdxq+vq*dqdyq)*JxWq

            # end jq
        # end iq

        A_el=MM+(Ka+Kd)*dt

        b_el=MM.dot(Tvect) + f_el*dt

        # apply boundary conditions
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            if bc_fix_T[m1]:
               Aref=A_el[k1,k1]
               for k2 in range(0,m_V):
                   m2=icon_V[k2,iel]
                   b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                   A_el[k1,k2]=0
                   A_el[k2,k1]=0
               A_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_val_T[m1]

        # assemble matrix and rhs
        for k1 in range(0,m_V):
            m1=icon_V[k1,iel]
            for k2 in range(0,m_V):
                m2=icon_V[k2,iel]
                A_fem[m1,m2]+=A_el[k1,k2]
            b_fem[m1]+=b_el[k1]

    # end iel

    print("build FEM matrix T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

    print("     -> T (m,M) %.4f %.4f C" %(np.min(T)-TKelvin,np.max(T)-TKelvin))

    T_stats[istep,0]=np.min(T)-TKelvin ; T_stats[istep,1]=np.max(T)-TKelvin

    print("solve T: %.3f s" % (clock.time()-start))

    ###########################################################################
    # update density 
    ###########################################################################
    start=clock.time()

    for i in range(0,nn_V):
        rho[i]=rho0*(1-alphaT*(T[i]-T0)+betaT*q[i])

    drhodt=(rho[:]-rho_prev[:])/dt

    print("     -> rho (m,M) %.4f %.4f kg/m3" %(np.min(rho),np.max(rho)))

    print("compute density: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute elemental strainrate, temperature gradient, ... (elemental) 
    ###########################################################################
    start=clock.time()

    e=np.zeros(nel,dtype=np.float64)  
    x_e=np.zeros(nel,dtype=np.float64)  
    y_e=np.zeros(nel,dtype=np.float64)  
    exx_e=np.zeros(nel,dtype=np.float64)  
    eyy_e=np.zeros(nel,dtype=np.float64)  
    exy_e=np.zeros(nel,dtype=np.float64)  
    dTdx_e=np.zeros(nel,dtype=np.float64)  
    dTdy_e=np.zeros(nel,dtype=np.float64)  
    Phi_e=np.zeros(nel,dtype=np.float64)  
    drhodx_e=np.zeros(nel,dtype=np.float64)
    drhody_e=np.zeros(nel,dtype=np.float64)
    u_e=np.zeros(nel,dtype=np.float64)
    v_e=np.zeros(nel,dtype=np.float64)
    T_e=np.zeros(nel,dtype=np.float64)
    dqdx_e=np.zeros(nel,dtype=np.float64)
    dqdy_e=np.zeros(nel,dtype=np.float64)

    qtop=0.
    qbottom=0.
    qleft=0.
    qright=0.
    iel=0
    for iely in range(0,nely):
        for ielx in range(0,nelx):
            rq=0.0
            sq=0.0
            N_V=basis_functions_V(rq,sq,order)
            dNdr_V=basis_functions_V_dr(rq,sq,order)
            dNds_V=basis_functions_V_ds(rq,sq,order)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
            y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
            exx_e[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyy_e[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
            exy_e[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5\
                      +np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            e[iel]=np.sqrt(0.5*(exx_e[iel]**2+eyy_e[iel]**2)+exy_e[iel]**2)
            u_e[iel]=np.dot(N_V,u[icon_V[:,iel]])
            v_e[iel]=np.dot(N_V,v[icon_V[:,iel]])
            T_e[iel]=np.dot(N_V,T[icon_V[:,iel]])
            dTdx_e[iel]=np.dot(dNdx_V,T[icon_V[:,iel]])
            dTdy_e[iel]=np.dot(dNdy_V,T[icon_V[:,iel]])
            dqdx_e[iel]=np.dot(dNdx_V,q[icon_V[:,iel]])
            dqdy_e[iel]=np.dot(dNdy_V,q[icon_V[:,iel]])
            drhodx_e[iel]=np.dot(dNdx_V,rho[icon_V[:,iel]])
            drhody_e[iel]=np.dot(dNdy_V,rho[icon_V[:,iel]])
            exxd=exx_e[iel]-(exx_e[iel]+eyy_e[iel])/3.
            eyyd=eyy_e[iel]-(exx_e[iel]+eyy_e[iel])/3.
            exyd=exy_e[iel]
            Phi_e[iel]=2.*eta0*(exxd**2+eyyd**2+2*exyd**2)    
            if iely==0:      qbottom+=-hcond*dTdy_e[iel]*hx *-1
            if iely==nely-1: qtop   +=-hcond*dTdy_e[iel]*hx * 1 
            if ielx==0:      qleft  +=-hcond*dTdx_e[iel]*hy *-1
            if ielx==nelx-1: qright +=-hcond*dTdx_e[iel]*hy * 1
            if iely==nely-1: Nu[istep]-=dTdy_e[iel]*hx *Ly/(Lx*Delta_Temp)
            iel+=1
        # end for ielx
    # end for iely

    heatflux_boundary[istep,0]=qtop+qbottom+qleft+qright
    heatflux_boundary[istep,1]=qtop
    heatflux_boundary[istep,2]=qbottom
    heatflux_boundary[istep,3]=qleft
    heatflux_boundary[istep,4]=qright

    print("     -> exx (m,M) %.5e %.5e s^-1" %(np.min(exx_e),np.max(exx_e)))
    print("     -> eyy (m,M) %.5e %.5e s^-1" %(np.min(eyy_e),np.max(eyy_e)))
    print("     -> exy (m,M) %.5e %.5e s^-1" %(np.min(exy_e),np.max(exy_e)))
    print("     -> dTdx (m,M) %.4e %.4e K/m" %(np.min(dTdx_e),np.max(dTdx_e)))
    print("     -> dTdy (m,M) %.4e %.4e K/m" %(np.min(dTdy_e),np.max(dTdy_e)))

    print("     -> time= %.6f ; Nu= %.6f" %(model_time[istep]/year,Nu[istep]))

    if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx_e,eyy_e,exy_e]).T,header='# x,y,exx,eyy,exy')

    print("compute sr, Nu: %.3f s" % (clock.time()-start))

    ###########################################################################
    # compute average of temperature, total mass 
    ###########################################################################
    start=clock.time()

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
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                Tq=np.dot(N_V,T[icon_V[:,iel]])
                rhoq=np.dot(N_V,rho[icon_V[:,iel]])
                dqdxq=np.dot(dNdx_V,q[icon_V[:,iel]])
                dqdyq=np.dot(dNdy_V,q[icon_V[:,iel]])
                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
                exxqd=exxq-(exxq+eyyq)/3.
                eyyqd=eyyq-(exxq+eyyq)/3.
                exyqd=exyq

                Tavrg[istep]+=Tq*JxWq
                mass[istep]+=rhoq*JxWq
                if use_BA or use_EBA:
                   ET[istep]+=rho0*hcapa*Tq*JxWq
                else:
                   ET[istep]+=rhoq*hcapa*Tq*JxWq
                EG[istep]+=rhoq*gy*yq*JxWq
                EK[istep]+=0.5*rhoq*(uq**2+vq**2)*JxWq
                adiab_heating[istep]+=alphaT*Tq*(uq*dqdxq+vq*dqdyq)*JxWq
                vrms[istep]+=(uq**2+vq**2)*JxWq
                visc_diss[istep]+=2.*eta0*(exxqd**2+eyyqd**2+2*exyqd**2)*JxWq
                work_grav[istep]+=(rhoq-rho0)*gy*vq*JxWq  # see text for justification 
            # end jq
        # end iq
    # end iel
    Tavrg[istep]/=Lx*Ly
    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly))/cm_per_year

    print("     -> vrms= %.6e ; Ra= %.6e ; vrmsdiff= %.6e " %\
         (vrms[istep],Ra,vrms[istep]-vrms[istep-1]))

    print("     -> avrg T= %.6e K" % Tavrg[istep])
    print("     -> mass  = %.6e kg/m" % mass[istep])

    print("compute <T>,M: %.3f s" % (clock.time()-start))

    ###########################################################################
    # plot of solution
    ###########################################################################
    start=clock.time()

    if visu==1 or istep%25==0:

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_V):
          vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       if order==1:
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
       vtufile.write("<DataArray type='Float32' Name='drhodx' Format='ascii'> \n")
       drhodx_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='drhody' Format='ascii'> \n")
       drhody_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='alpha T v.gradp' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (alphaT*T_e[iel]*(u_e[iel]*dqdx_e[iel]+v_e[iel]*dqdy_e[iel])))
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
       vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
       divv=exx_e+eyy_e
       divv.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='e (2nd inv.)' Format='ascii'> \n")
       e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Phi' Format='ascii'> \n")
       Phi_e.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e %10e %10e \n" %(u[i]/cm_per_year,v[i]/cm_per_year,0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='rho' Format='ascii'> \n")
       rho.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='drho/dt' Format='ascii'> \n")
       drhodt.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='q' Format='ascii'> \n")
       q.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='alpha T dp/dt' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e \n" % (alphaT*T[i]*dqdt[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='T (C)' Format='ascii'> \n")
       for i in range(0,nn_V):
           vtufile.write("%10e \n" % (T[i]-TKelvin))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
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
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m_V))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       if order==1:
          for iel in range (0,nel):
              vtufile.write("%d \n" %9)
       if order==2:
          for iel in range (0,nel):
              vtufile.write("%d \n" %28)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

    print("generate vtu: %.3f s" % (clock.time()-start))

    ###########################################################################
    # write to file 
    ###########################################################################
    start=clock.time()

    np.savetxt('vrms_Nu.ascii',np.array([model_time[0:istep]/year,vrms[0:istep],Nu[0:istep]]).T,header='# t/year,vrms,Nu')
    np.savetxt('vrms_Nu_adim.ascii',np.array([model_time[0:istep]/reftime,vrms[0:istep]/refvel,Nu[0:istep]]).T,header='# t/reftime,vrms/refvel,Nu')
    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep]/year,Tavrg[0:istep]]).T,header='# t/year,Tavrg')
    np.savetxt('Tavrg_adim.ascii',np.array([model_time[0:istep]/reftime,Tavrg[0:istep]/refTemp]).T,header='# t/reftime,Tavrg/refTemp')
    np.savetxt('M.ascii',np.array([model_time[0:istep]/year,mass[0:istep],mass[0:istep]/mass[0]]).T,header='# t/year,M,M/M0')
    np.savetxt('EK.ascii',np.array([model_time[0:istep]/year,EK[0:istep]]).T,header='# t/year,EK')
    np.savetxt('EG.ascii',np.array([model_time[0:istep]/year,EG[0:istep],EG[0:istep]-EG[0]]).T,header='# t/year,EG,EG-EG(0)')
    np.savetxt('ET.ascii',np.array([model_time[0:istep]/year,ET[0:istep]]).T,header='# t/year,ET')
    np.savetxt('viscous_dissipation.ascii',np.array([model_time[0:istep]/year,visc_diss[0:istep]]).T,header='# t/year,Phi')
    np.savetxt('work_grav.ascii',np.array([model_time[0:istep]/year,work_grav[0:istep]]).T,header='# t/year,W')
    np.savetxt('heat_flux_boundary.ascii',np.array([model_time[0:istep]/year,heatflux_boundary[0:istep,0]]).T,header='# t/year,q')
    np.savetxt('adiabatic_heating.ascii',np.array([model_time[0:istep]/year,adiab_heating[0:istep]]).T,header='# t/year,ad.heat.')
    np.savetxt('viscous_dissipation_adim.ascii',np.array([model_time[0:istep]/reftime,visc_diss[0:istep]/(refPress*refvel*Ly**2)]).T,header='# t/reftime,Phi')
    np.savetxt('work_grav_adim.ascii',np.array([model_time[0:istep]/reftime,work_grav[0:istep]/(refPress*refvel*Ly**2)]).T,header='# t/reftime,W')
    np.savetxt('conservation.ascii',np.array([model_time[0:istep]/year,visc_diss[0:istep],adiab_heating[0:istep],heatflux_boundary[0:istep,0]]).T,header='# t/reftime,W')
    np.savetxt('u_stats.ascii',np.array([model_time[0:istep]/year,u_stats[0:istep,0],u_stats[0:istep,1]]).T,header='# t/year,min(u),max(u)')
    np.savetxt('v_stats.ascii',np.array([model_time[0:istep]/year,v_stats[0:istep,0],v_stats[0:istep,1]]).T,header='# t/year,min(v),max(v)')
    np.savetxt('T_stats.ascii',np.array([model_time[0:istep]/year,T_stats[0:istep,0],T_stats[0:istep,1]]).T,header='# t/year,min(T),max(T)')
    np.savetxt('dt_stats.ascii',np.array([model_time[0:istep]/year,dt_stats[0:istep]]).T,header='# t/year,dt')

    if istep>0:
       np.savetxt('dETdt.ascii',np.array([model_time[1:istep]/year,  ((ET[1:istep]-ET[0:istep-1])/dt)   ]).T,header='# t/year,ET')

    print("output stats: %.3f s" % (clock.time() - start))

    #####################################################################

    q_prev[:]=q[:]
    rho_prev[:]=rho[:]

################################################################################################
# END OF TIMESTEPPING
################################################################################################

print("*******************************")
print("********** the end ************")
print("*******************************")
