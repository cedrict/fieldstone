import time as timing
import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
from shape_functionsV import NNV,dNNVdr,dNNVds,dNNVdt
from shape_functionsT import NNT,dNNTdr,dNNTds,dNNTdt

#------------------------------------------------------------------------------

eps=1.e-10
year=3.154e+7
Myear=1e6*year
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("--------fieldstone 20--------")
print("-----------------------------")

ndim=3   # number of dimensions
mV=8     # number of V nodes making up an element
mT=8     # number of T nodes making up an element
ndofV=3  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1  # number of temperature degrees of freedom 

Lx=1.0079*2700e3
Ly=0.6283*2700e3
Lz=1.0000*2700e3

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
else:
   nelx =16
   nely =10
   nelz =16

# this is a requirement bc we measure at z=3Lz/4
assert (nelz%4==0), "nelz should be even and multiple of 4" 

gx=0
gy=0
gz=-10

visu=1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnz=nelz+1  # number of elements, z direction

NV=nnx*nny*nnz  # number of nodes
NT=nnx*nny*nnz  # number of nodes

nel=nelx*nely*nelz  # number of elements, total

NfemV=NV*ndofV   # number of V dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs
NfemT=NT*ndofT   # number of T nodes

TKelvin=273.15
Temperature1=3700+TKelvin
Temperature2=   0+TKelvin

T0=TKelvin

rho0=3300.
eta0=8.0198e23
hcond=3.564
hcapa=1080
alpha=1.e-5
alphaT=0.5

nstep=250

Ra=alpha*abs(gz)*(Temperature1-Temperature2)*Lz**3*rho0**2*hcapa/hcond/eta0

kappa=hcond/rho0/hcapa
reftime=Lz**2/kappa
tfinal=5*reftime

eta_ref=1.e23     
scaling_coeff=eta_ref/Lz

rVnodes=[-1,1,1,-1,-1,1,1,-1]
sVnodes=[-1,-1,1,1,-1,-1,1,1]
tVnodes=[-1,-1,-1,-1,1,1,1,1]

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

relaxV=.9
relaxT=.9

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("NV=",NV)
print("Ra=",Ra,3e5)
print("kappa=",kappa)
print("reftime=",reftime/Myear,'Myr') 
print("tfinal=",tfinal/Myear,'Myr') 
print("------------------------------")

#################################################################

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

######################################################################
# grid point setup
######################################################################
start = timing.time()

xV = np.zeros(NV,dtype=np.float64)  # x coordinates
yV = np.zeros(NV,dtype=np.float64)  # y coordinates
zV = np.zeros(NV,dtype=np.float64)  # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            xV[counter]=i*hx
            yV[counter]=j*hy
            zV[counter]=k*hz
            counter += 1
        # end for k
    # end for j
# end for i

print("grid points setup: %.3f s" % (timing.time() - start))

######################################################################
# connectivity
######################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            iconV[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            iconV[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            iconV[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            iconV[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            iconV[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            iconV[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            iconV[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            iconV[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        # end for k
    # end for j
# end for i

print("build connectivity: %.3f s" % (timing.time() - start))

######################################################################
# define boundary conditions velocity
######################################################################
start = timing.time()

bc_fixV=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_valV=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fixV[i*ndofV+0]=True ; bc_valV[i*ndofV+0]=0
    if xV[i]/Lx>(1-eps):
       bc_fixV[i*ndofV+0]=True ; bc_valV[i*ndofV+0]=0
    if yV[i]/Ly<eps:
       bc_fixV[i*ndofV+1]=True ; bc_valV[i*ndofV+1]=0
    if yV[i]/Ly>(1-eps):
       bc_fixV[i*ndofV+1]=True ; bc_valV[i*ndofV+1]=0
    if zV[i]/Lz<eps:
       bc_fixV[i*ndofV+0]=True ; bc_valV[i*ndofV+0]=0
       bc_fixV[i*ndofV+1]=True ; bc_valV[i*ndofV+1]=0
       bc_fixV[i*ndofV+2]=True ; bc_valV[i*ndofV+2]=0
    if zV[i]/Lz>(1-eps):
       bc_fixV[i*ndofV+0]=True ; bc_valV[i*ndofV+0]=0 
       bc_fixV[i*ndofV+1]=True ; bc_valV[i*ndofV+1]=0 
       bc_fixV[i*ndofV+2]=True ; bc_valV[i*ndofV+2]=0 
# end for

print("boundary conditions V: %.3f s" % (timing.time() - start))

######################################################################
# temperature grid setup
######################################################################
start = timing.time()

xT=np.zeros(NT,dtype=np.float64)  # x coordinates
yT=np.zeros(NT,dtype=np.float64)  # y coordinates
zT=np.zeros(NT,dtype=np.float64)  # z coordinates
iconT=np.zeros((mT,nel),dtype=np.int32)

xT[:]=xV[:]
yT[:]=yV[:]
zT[:]=zV[:]
iconT[:,:]=iconV[:,:]

print("build grid T: %.3f s" % (timing.time() - start))

######################################################################
# define boundary conditions temperature
######################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=bool) # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,NT):
    if zT[i]<eps:
       bc_fixT[i] = True ; bc_valT[i] = Temperature1
    if zT[i]/Lz>1-eps:
       bc_fixT[i] = True ; bc_valT[i] = Temperature2
# end for

print("boundary conditions T: %.3f s" % (timing.time() - start))

######################################################################
# initial temperature field 
######################################################################

T=np.zeros(NT,dtype=np.float64) 
T_old=np.zeros(NT,dtype=np.float64) 

for i in range(0,NT):
   T[i]= (Temperature2-Temperature1)/Lz*zT[i]+Temperature1 \
       + 100*(np.cos(np.pi*xT[i]/Lx) + np.cos(np.pi*yT[i]/Ly))*np.sin(np.pi*zT[i]/Lz)
# end for

T_old=T

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################

c_mat = np.array([[2,0,0,0,0,0],\
                  [0,2,0,0,0,0],\
                  [0,0,2,0,0,0],\
                  [0,0,0,1,0,0],\
                  [0,0,0,0,1,0],\
                  [0,0,0,0,0,1]],dtype=np.float64) 

u=np.zeros(NV,dtype=np.float64)            # x-component velocity
v=np.zeros(NV,dtype=np.float64)            # y-component velocity
w=np.zeros(NV,dtype=np.float64)            # y-component velocity
p=np.zeros(nel,dtype=np.float64)           # pressure field
u_old=np.zeros(NV,dtype=np.float64)      # x-component velocity
v_old=np.zeros(NV,dtype=np.float64)      # y-component velocity
w_old=np.zeros(NV,dtype=np.float64)      # y-component velocity
                    
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

    ######################################################################
    # build FE matrix
    ######################################################################
    start = timing.time()

    A_sparse= lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    #K_mat   = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    #G_mat   = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    b_mat   = np.zeros((6,ndofV*mV),dtype=np.float64)  # gradient matrix B 
    NNNT    = np.zeros(mT,dtype=np.float64)            # shape functions
    NNNV    = np.zeros(mV,dtype=np.float64)            # shape functions
    dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdz = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdt = np.zeros(mV,dtype=np.float64)            # shape functions derivatives

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el=np.zeros((mV*ndofV),dtype=np.float64)
        K_el=np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
        h_el=0. #np.zeros((1,1),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:

                    # position & weight of quad. point
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.

                    # calculate shape functions
                    NNNV[0:mV]=NNV(rq,sq,tq)
                    NNNT[0:mT]=NNT(rq,sq,tq)
                    dNNNVdr[0:mV]=dNNVdr(rq,sq,tq)
                    dNNNVds[0:mV]=dNNVds(rq,sq,tq)
                    dNNNVdt[0:mV]=dNNVdt(rq,sq,tq)

                    # calculate jacobian matrix
                    #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                    #for k in range(0,mV):
                    #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    #    jcb[0,2] += dNNNVdr[k]*zV[iconV[k,iel]]
                    #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                    #    jcb[1,2] += dNNNVds[k]*zV[iconV[k,iel]]
                    #    jcb[2,0] += dNNNVdt[k]*xV[iconV[k,iel]]
                    #    jcb[2,1] += dNNNVdt[k]*yV[iconV[k,iel]]
                    #    jcb[2,2] += dNNNVdt[k]*zV[iconV[k,iel]]
                    #jcob=np.linalg.det(jcb)
                    #jcbi=np.linalg.inv(jcb)

                    # compute dNdx, dNdy, dNdz
                    for k in range(0,mV):
                        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]+jcbi[0,2]*dNNNVdt[k]
                        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]+jcbi[1,2]*dNNNVdt[k]
                        dNNNVdz[k]=jcbi[2,0]*dNNNVdr[k]+jcbi[2,1]*dNNNVds[k]+jcbi[2,2]*dNNNVdt[k]

                    Tq=0.0
                    for k in range(0,mT):
                        Tq+=NNNT[k]*T[iconT[k,iel]]

                    # construct 3x8 b_mat matrix
                    for i in range(0,mV):
                        b_mat[0:6, 3*i:3*i+3] = [[dNNNVdx[i],0.        ,0.        ],
                                                 [0.        ,dNNNVdy[i],0.        ],
                                                 [0.        ,0.        ,dNNNVdz[i]],
                                                 [dNNNVdy[i],dNNNVdx[i],0.        ],
                                                 [dNNNVdz[i],0.        ,dNNNVdx[i]],
                                                 [0.        ,dNNNVdz[i],dNNNVdy[i]]]

                    K_el += b_mat.T.dot(c_mat.dot(b_mat))*eta0*weightq*jcob

                    for i in range(0,mV):
                        f_el[ndofV*i+2]+=NNNV[i]*jcob*weightq*gz*rho0*(1-alpha*(Tq-T0))
                        G_el[ndofV*i+0,0]-=dNNNVdx[i]*jcob*weightq
                        G_el[ndofV*i+1,0]-=dNNNVdy[i]*jcob*weightq
                        G_el[ndofV*i+2,0]-=dNNNVdz[i]*jcob*weightq
                    #end for

                #end for kq
            #end for jq
        #end for iq

        # impose b.c. 
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1+i1
                m1 =ndofV*iconV[k1,iel]+i1
                if bc_fixV[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,mV*ndofV):
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

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1+i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2+i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        #K_mat[m1,m2]+=K_el[ikk,jkk]
                        A_sparse[m1,m2]+=K_el[ikk,jkk]
                    #end for 
                #end for 
                rhs[m1]+=f_el[ikk]
                #G_mat[m1,iel]+=G_el[ikk,0]*scaling_coeff
                A_sparse[m1,NfemV+iel]+=G_el[ikk,0]*scaling_coeff
                A_sparse[NfemV+iel,m1]+=G_el[ikk,0]*scaling_coeff
            #end for 
        #end for 
        rhs[NfemV+iel]+=h_el*scaling_coeff

    #end

    #print("f_rhs (m,M) = %.6e %.6e" %(np.min(f_rhs),np.max(f_rhs)))
    #print("h_rhs (m,M) = %.6e %.6e" %(np.min(h_rhs),np.max(h_rhs)))

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    #start = timing.time()
    #if pnormalise:
    #   a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
    #   rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
    #   a_mat[0:NfemV,0:NfemV]=K_mat
    #   a_mat[0:NfemV,NfemV:Nfem]=G_mat
    #   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #   a_mat[Nfem,NfemV:Nfem]=1
    #   a_mat[NfemV:Nfem,Nfem]=1
    #else:
    #   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
    #   a_mat[0:NfemV,0:NfemV]=K_mat
    #   a_mat[0:NfemV,NfemV:Nfem]=G_mat
    #   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #rhs[0:NfemV]=f_rhs
    #rhs[NfemV:Nfem]=h_rhs
    #print("assemble blocks: %.3f s" % (timing.time() - start))
    ######################################################################
    #a_mat[NfemV-1,:]=0
    #a_mat[:,NfemV-1]=0
    #a_mat[NfemV-1,NfemV-1]=1
    #rhs[NfemV-1]=0
    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol=sps.linalg.spsolve(A_sparse.tocsr(),rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u,v,w=np.reshape(sol[0:NfemV],(NV,ndim)).T
    p=sol[NfemV:Nfem]*scaling_coeff

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

    #np.savetxt('velocity.ascii',np.array([xV,yV,zV,u,v,w]).T,header='# x,y,z,u,v,w')

    print("transfer solution: %.3f s" % (timing.time() - start))

    #####################################################################
    # relaxation step
    #####################################################################

    u=relaxV*u+(1-relaxV)*u_old
    v=relaxV*v+(1-relaxV)*v_old
    w=relaxV*w+(1-relaxV)*w_old

    ######################################################################
    # compute time step value 
    ######################################################################
    start = timing.time()

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

    print("compute timestep: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute vrms. 
    ######################################################################
    start = timing.time()

    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:

                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.

                    NNNV[0:mV]=NNV(rq,sq,tq)
                    NNNT[0:mV]=NNT(rq,sq,tq)
 
                    uq=NNNV.dot(u[iconV[:,iel]])
                    vq=NNNV.dot(v[iconV[:,iel]])
                    wq=NNNV.dot(w[iconV[:,iel]])
                    Tq=NNNT.dot(T[iconT[:,iel]])

                    vrms[istep]+=(uq**2+vq**2+wq**2)*weightq*jcob

                    Tavrg[istep]+=Tq*weightq*jcob

                # end for ik
            # end for jk
        # end for kq
    # end for iel

    vrms[istep]=np.sqrt(vrms[istep]/(Lx*Ly*Lz))
    Tavrg[istep]/=Lx*Ly*Lz

    print("     -> vrms= %.6e ; vrmsdiff= %.6e " % (vrms[istep],vrms[istep]-vrms[0]))

    print("compute vrms: %.3f s" % (timing.time() - start))

    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((3,ndofT*mT),dtype=np.float64)     # gradient matrix B 
    NNNT_mat = np.zeros((mT,1),dtype=np.float64)         # shape functions
    Tvect = np.zeros(mT,dtype=np.float64)   
    dNNNTdx = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdy = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdz = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdr = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTds = np.zeros(mT,dtype=np.float64)            # shape functions derivatives
    dNNNTdt = np.zeros(mT,dtype=np.float64)            # shape functions derivatives

    for iel in range (0,nel):

        b_el=np.zeros(mT*ndofT,dtype=np.float64)
        a_el=np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64)
        Ka=np.zeros((mT,mT),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((mT,mT),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((mT,mT),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,mT):
            Tvect[k]=T[iconT[k,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:
                for kq in [-1,1]:

                    # position & weight of quad. point
                    rq=iq/sqrt3
                    sq=jq/sqrt3
                    tq=kq/sqrt3
                    weightq=1.*1.*1.

                    # calculate shape functions
                    NNNT_mat[0:mT,0]=NNT(rq,sq,tq)
                    dNNNTdr[0:mT]=dNNTdr(rq,sq,tq)
                    dNNNTds[0:mT]=dNNTds(rq,sq,tq)
                    dNNNTdt[0:mT]=dNNTdt(rq,sq,tq)
                    NNNV[0:mV]=NNV(rq,sq,tq)

                    # calculate jacobian matrix
                    #jcb=np.zeros((ndim,ndim),dtype=np.float64)
                    #for k in range(0,mT):
                    #    jcb[0,0] += dNNNTdr[k]*xT[iconT[k,iel]]
                    #    jcb[0,1] += dNNNTdr[k]*yT[iconT[k,iel]]
                    #    jcb[0,2] += dNNNTdr[k]*zT[iconT[k,iel]]
                    #    jcb[1,0] += dNNNTds[k]*xT[iconT[k,iel]]
                    #    jcb[1,1] += dNNNTds[k]*yT[iconT[k,iel]]
                    #    jcb[1,2] += dNNNTds[k]*zT[iconT[k,iel]]
                    #    jcb[2,0] += dNNNTdt[k]*xT[iconT[k,iel]]
                    #    jcb[2,1] += dNNNTdt[k]*yT[iconT[k,iel]]
                    #    jcb[2,2] += dNNNTdt[k]*zT[iconT[k,iel]]
                    # end for 
                    #jcob = np.linalg.det(jcb)
                    #jcbi = np.linalg.inv(jcb)

                    vel[0,:]=0.
                    for k in range(0,mV):
                        vel[0,0]+=NNNV[k]*u[iconV[k,iel]]
                        vel[0,1]+=NNNV[k]*v[iconV[k,iel]]
                        vel[0,2]+=NNNV[k]*w[iconV[k,iel]]
                    # end for 

                    # compute dNdx, dNdy, dNdz 
                    for k in range(0,mT):
                        dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]+jcbi[0,2]*dNNNTdt[k]
                        dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]+jcbi[1,2]*dNNNTdt[k]
                        dNNNTdz[k]=jcbi[2,0]*dNNNTdr[k]+jcbi[2,1]*dNNNTds[k]+jcbi[2,2]*dNNNTdt[k]
                        B_mat[0,k]=dNNNTdx[k]
                        B_mat[1,k]=dNNNTdy[k]
                        B_mat[2,k]=dNNNTdz[k]
                    # end for 

                    # compute mass matrix
                    #MM=NNNT_mat.dot(NNNT_mat.T)*rho_lhs*hcapa*weightq*jcob

                    # compute diffusion matrix
                    Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                    # compute advection matrix
                    Ka=NNNT_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

                    #a_el=MM+alphaT*(Ka+Kd)*dt
                    #b_el=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvect)

                    a_el+=(Kd+Ka)

                #end for
            #end for
        #end for

        # apply boundary conditions

        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,mT):
                   m2=iconT[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               # end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            # end if
        # end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,mT):
            m1=iconT[k1,iel]
            for k2 in range(0,mT):
                m2=iconT[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            # end for
            rhs[m1]+=b_el[k1]
        # end for

    #end for

    print("     -> A_mat (m,M) = %.6e %.6e" %(np.min(A_mat),np.max(A_mat)))
    print("     -> rhs   (m,M) = %.6e %.6e" %(np.min(rhs),np.max(rhs)))

    print("build FEM matrix T: %.3f s" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T)-TKelvin,np.max(T)-TKelvin))

    T_stats[istep,0]=np.min(T)-TKelvin 
    T_stats[istep,1]=np.max(T)-TKelvin

    print("solve T: %.3f s" % (timing.time() - start))

    #################################################################
    # relax
    #################################################################

    T=relaxT*T+(1-relaxT)*T_old

    #####################################################################
    # compute nodal strainrate on velocity grid
    #####################################################################
    # I should have separated the strainrate calculations from the 
    # temperature derivatives calculations for consistency but 
    # since the velocity and temperature shape functions are the same
    # and so are the connectivity arrays I save much time doing this.
    #####################################################################
    start = timing.time()

    exxn=np.zeros(NV,dtype=np.float64)
    eyyn=np.zeros(NV,dtype=np.float64)
    ezzn=np.zeros(NV,dtype=np.float64)
    exyn=np.zeros(NV,dtype=np.float64)
    exzn=np.zeros(NV,dtype=np.float64)
    eyzn=np.zeros(NV,dtype=np.float64)
    srn=np.zeros(NV,dtype=np.float64)
    dTdxn=np.zeros(NT,dtype=np.float64)
    dTdyn=np.zeros(NT,dtype=np.float64)
    dTdzn=np.zeros(NT,dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)

    for iel in range(0,nel):
        for i in range(0,mV):
            NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i],tVnodes[i])
            dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i],tVnodes[i])
            dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i],tVnodes[i])
            dNNNVdt[0:mV]=dNNVdt(rVnodes[i],sVnodes[i],tVnodes[i])
            #jcb=np.zeros((ndim,ndim),dtype=np.float64)
            #for k in range(0,mV):
            #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
            #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
            #    jcb[0,2] += dNNNVdr[k]*zV[iconV[k,iel]]
            #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
            #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            #    jcb[1,2] += dNNNVds[k]*zV[iconV[k,iel]]
            #    jcb[2,0] += dNNNVdt[k]*xV[iconV[k,iel]]
            #    jcb[2,1] += dNNNVdt[k]*yV[iconV[k,iel]]
            #    jcb[2,2] += dNNNVdt[k]*zV[iconV[k,iel]]
            #jcob=np.linalg.det(jcb)
            #jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]+jcbi[0,2]*dNNNVdt[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]+jcbi[1,2]*dNNNVdt[k]
                dNNNVdz[k]=jcbi[2,0]*dNNNVdr[k]+jcbi[2,1]*dNNNVds[k]+jcbi[2,2]*dNNNVdt[k]
            # end for
            e_xx=0.
            e_yy=0.
            e_zz=0.
            e_xy=0.
            e_xz=0.
            e_yz=0.
            dT_dx=0.
            dT_dy=0.
            dT_dz=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_zz += dNNNVdz[k]*w[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
                e_xz += 0.5*(dNNNVdz[k]*u[iconV[k,iel]]+dNNNVdx[k]*w[iconV[k,iel]])
                e_yz += 0.5*(dNNNVdz[k]*v[iconV[k,iel]]+dNNNVdy[k]*w[iconV[k,iel]])
                dT_dx += dNNNVdx[k]*T[iconV[k,iel]]
                dT_dy += dNNNVdy[k]*T[iconV[k,iel]]
                dT_dz += dNNNVdz[k]*T[iconV[k,iel]]
            # end for
            exxn[iconV[i,iel]]+=e_xx
            eyyn[iconV[i,iel]]+=e_yy
            ezzn[iconV[i,iel]]+=e_zz
            exyn[iconV[i,iel]]+=e_xy
            exzn[iconV[i,iel]]+=e_xz
            eyzn[iconV[i,iel]]+=e_yz
            dTdxn[iconV[i,iel]]+=dT_dx
            dTdyn[iconV[i,iel]]+=dT_dy
            dTdzn[iconV[i,iel]]+=dT_dz
            c[iconV[i,iel]]+=1.
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

    srn=np.sqrt(0.5*(exxn**2+eyyn**2+ezzn**2+exyn**2+exzn**2+eyzn**2))

    print("     -> exx (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
    print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
    print("     -> ezz (m,M) %.6e %.6e " %(np.min(ezzn),np.max(ezzn)))
    print("     -> exy (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
    print("     -> exz (m,M) %.6e %.6e " %(np.min(exzn),np.max(exzn)))
    print("     -> eyz (m,M) %.6e %.6e " %(np.min(eyzn),np.max(eyzn)))
    print("     -> dTdx (m,M) %.6e %.6e " %(np.min(dTdxn),np.max(dTdxn)))
    print("     -> dTdy (m,M) %.6e %.6e " %(np.min(dTdyn),np.max(dTdyn)))
    print("     -> dTdz (m,M) %.6e %.6e " %(np.min(dTdzn),np.max(dTdzn)))

    print("compute nod strain rate: %.3f s" % (timing.time() - start))

    ######################################################################
    # compute nodal pressure
    ######################################################################
    start = timing.time()

    q=np.zeros(NV,dtype=np.float64)  
    count=np.zeros(NV,dtype=np.float64)  

    for iel in range(0,nel):
        for k in range(0,mV):
            q[iconV[k,iel]]+=p[iel]
            count[iconV[k,iel]]+=1
        # end for
    # end for

    q=q/count

    #np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

    print("     -> q (m,M) %.4e %.4e " %(np.min(q),np.max(q)))

    print("compute nod pressure: %.3f s" % (timing.time() - start))

    #####################################################################
    # velocity and temperature at mid side edges
    #####################################################################

    for i in range(0,NV):
        if abs(zV[i]-Lz/2.)/Lz<eps:
           if xV[i]/Lx<eps and yV[i]/Ly<eps: 
              wmid_stats[istep,0]=w[i]
              Tmid_stats[istep,0]=T[i]
           if xV[i]/Lx>1-eps and yV[i]/Ly<eps: 
              wmid_stats[istep,1]=w[i]
              Tmid_stats[istep,1]=T[i]
           if xV[i]/Lx<eps and yV[i]/Ly>1-eps: 
              wmid_stats[istep,2]=w[i]
              Tmid_stats[istep,2]=T[i]
           if xV[i]/Lx>1-eps and yV[i]/Ly>1-eps: 
              wmid_stats[istep,3]=w[i]
              Tmid_stats[istep,3]=T[i]
        # end if
    # end for

    for i in range(0,NT):
        if zT[i]/Lz>1-eps:
           if xT[i]/Lx<eps and yT[i]/Ly<eps: 
              hf_stats[istep,0]=dTdzn[i]*hcond
           if xT[i]/Lx>1-eps and yT[i]/Ly<eps: 
              hf_stats[istep,1]=dTdzn[i]*hcond
           if xT[i]/Lx<eps and yT[i]/Ly>1-eps: 
              hf_stats[istep,2]=dTdzn[i]*hcond
           if xT[i]/Lx>1-eps and yT[i]/Ly>1-eps: 
              hf_stats[istep,3]=dTdzn[i]*hcond
        # end if
    # end for

    #####################################################################
    # average temperature at z=3Lz/4
    #####################################################################
    start = timing.time()

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
                           Tq=N0*T[iconT[4,iel]]+\
                              N1*T[iconT[5,iel]]+\
                              N2*T[iconT[6,iel]]+\
                              N3*T[iconT[7,iel]]
                           #print (zT[iconT[4:7,iel]],Lz*0.75)  
                           jcobb=hx*hy/4
                           T_m+=Tq*jcobb*weightq
                # end if
                iel+=1
            # end for
        # end for
    # end for

    T_m/=(Lx*Ly)

    print('     -> avrg T at z=3Lz/4 =',T_m)

    Tm[istep]=T_m

    print("compute avrt T at z=3Lz/4: %.3f s" % (timing.time() - start))

    #####################################################################
    # Nusselt number 
    #####################################################################

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
                           dTdzq=N0*dTdzn[iconT[4,iel]]+\
                                 N1*dTdzn[iconT[5,iel]]+\
                                 N2*dTdzn[iconT[6,iel]]+\
                                 N3*dTdzn[iconT[7,iel]]
                           #print (zT[iconT[4:7,iel]],Lz*0.75)  
                           jcobb=hx*hy/4
                           Nu+=abs(dTdzq*jcobb*weightq)
                # end if
                iel+=1
            # end for
        # end for
    # end for

    Nu*=(Lz/(Lx*Ly*(Temperature1-273)))

    Nufile.write("%d %e\n" % (istep,Nu))
    Nufile.flush()

    print("     -> Nu= %.6e  " % Nu)

    #####################################################################
    # plot of solution
    #####################################################################
    start = timing.time()

    if visu==1 and istep%20==0:

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e %.6e %.6e \n" %(xV[i],yV[i],zV[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<CellData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%.6e\n" % p[iel])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</CellData>\n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f %10f %10f \n" %(u[i]*year,v[i]*year,w[i]*year))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" %(T[i]-TKelvin))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10f \n" %q[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exx(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %exxn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyy(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %eyyn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='ezz(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %ezzn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exy(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %exyn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='exz(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %exzn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='eyz(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %eyzn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='sr(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %srn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdx(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %dTdxn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdy(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %dTdyn[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='dTdz(n)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%.6e \n" %dTdzn[i])
       vtufile.write("</DataArray>\n")

       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],\
                                                       iconV[2,iel],iconV[3,iel],\
                                                       iconV[4,iel],iconV[5,iel],\
                                                       iconV[6,iel],iconV[7,iel]))
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
       print("export to vtu: %.3f s" % (timing.time() - start))

    #####################################################################
    start = timing.time()

    np.savetxt('vrms.ascii',np.array([model_time[0:istep]/Myear,\
                                      vrms[0:istep]]).T,header='# t/year,vrms')

    np.savetxt('u_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         u_stats[0:istep,0],\
                                         u_stats[0:istep,1]]).T,header='# t/year,min(u),max(u)')

    np.savetxt('v_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         v_stats[0:istep,0],\
                                         v_stats[0:istep,1]]).T,header='# t/year,min(v),max(v)')

    np.savetxt('w_stats.ascii',np.array([model_time[0:istep]/Myear,\
                                         w_stats[0:istep,0],\
                                         w_stats[0:istep,1]]).T,header='# t/year,min(w),max(w)')

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

    print("export stats to ascii: %.3f s" % (timing.time() - start))

    #####################################################################

    if np.abs(Nu-Nu_old)<1.e-6:
       print("Nu converged to 1e-6")
       break

    #####################################################################

    u_old[:]=u[:]
    v_old[:]=v[:]
    w_old[:]=w[:]
    T_old[:]=T[:]
    Nu_old=Nu

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

