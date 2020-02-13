import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as timing

#------------------------------------------------------------------------------

def NNV(rq,sq):
    NV_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    NV_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    NV_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    NV_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    NV_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    NV_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    NV_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    NV_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    NV_8=     (1.-rq**2) *     (1.-sq**2)
    return NV_0,NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8

def dNNVdr(rq,sq):
    dNVdr_0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNVdr_1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNVdr_2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNVdr_3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNVdr_4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNVdr_5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNVdr_6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNVdr_7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNVdr_8=       (-2.*rq) *    (1.-sq**2)
    return dNVdr_0,dNVdr_1,dNVdr_2,dNVdr_3,dNVdr_4,dNVdr_5,dNVdr_6,dNVdr_7,dNVdr_8

def dNNVds(rq,sq):
    dNVds_0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNVds_1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNVds_2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNVds_3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNVds_4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNVds_5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNVds_6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNVds_7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNVds_8=     (1.-rq**2) *       (-2.*sq)
    return dNVds_0,dNVds_1,dNVds_2,dNVds_3,dNVds_4,dNVds_5,dNVds_6,dNVds_7,dNVds_8

def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

#------------------------------------------------------------------------------

cm=0.01
year=365.*24.*3600.

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
ndofT=1


# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 50
   nely = 50
   visu = 1

eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

pnormalise=True

benchmark=2

if benchmark==1:
   nelx=16
   nely=16
   Lx=100e3  
   Ly=100e3  
   gx=0.
   gy=0
   dt=100*year
   rho1=0
   rho2=0
   mu1=1e10
   mu2=0
   eta1=1e21
   eta2=0
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=0
   Z1=etaeff1/mu1/dt
   Z2=0
   nstep=200
   tfinal=0
   filter_compositions=False

if benchmark==2:
   nelx=50
   nely=50
   Lx=1000e3  # horizontal extent of the domain 
   Ly=1000e3  # vertical extent of the domain 
   gx=0.
   gy=-10
   dt=200*year
   rho1=4000
   rho2=1
   eta1=1e27
   eta2=1e21
   mu1=1e10
   mu2=1e20
   etaeff1=eta1*dt/(dt+eta1/mu1)
   etaeff2=eta2*dt/(dt+eta2/mu2)
   Z1=etaeff1/mu1/dt
   Z2=etaeff2/mu2/dt
   tfinal=20e3*year
   nstep=20
   filter_compositions=True

nnx=2*nelx+1                  # number of Vnodes, x direction
nny=2*nely+1                  # number of Vnodes, y direction
NV=nnx*nny                    # number of Vnodes
nel=nelx*nely                 # number of elements
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
NfemT=NV                      # number of field dofs 
hx=Lx/nelx
hy=Ly/nely

eta_ref=1e25
scaling_coeff=eta_ref/Ly
   
rVnodes=[-1,1,1,-1,0,1,0,-1,0]
sVnodes=[-1,-1,1,1,-1,0,1,0,0]

alpha=0.5

time=0.

#################################################################

stats_exx_file=open('exx.ascii',"w")
stats_eyy_file=open('eyy.ascii',"w")
stats_exy_file=open('exy.ascii',"w")
stats_oxy_file=open('oxy.ascii',"w")
stats_tauxx_file=open('tauxx.ascii',"w")
stats_tauyy_file=open('tauyy.ascii',"w")
stats_tauxy_file=open('tauxy.ascii',"w")
stats_C1_file=open('C1.ascii',"w")
stats_C2_file=open('C2.ascii',"w")
stats_u_file=open('u.ascii',"w")
stats_v_file=open('v.ascii',"w")
stats_Z_file=open('Z.ascii',"w")
stats_etaeff_file=open('etaeff.ascii',"w")
stats_C1pC2_file=open('C1+C2.ascii',"w")

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
if benchmark==1:
   print("etaeff1=",etaeff1)
   print("Z1=",Z1)

if benchmark==2:
   print("etaeff1=",etaeff1)
   print("etaeff2=",etaeff2)
   print("Z1=",Z1)
   print("Z2=",Z2)
   print("t_M 1=",eta1/mu1/year,"yr")
   print("t_M 2=",eta2/mu2/year,"yr")
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = timing.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        xV[counter]=i*hx/2.
        yV[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')

print("grid points: %.3f s" % (timing.time() - start))

#################################################################
# connectivity
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int16)
iconP=np.zeros((mP,nel),dtype=np.int16)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

#connectivity array for plotting
nel2=(nnx-1)*(nny-1)
iconQ1 =np.zeros((4,nel2),dtype=np.int16)
counter = 0
for j in range(0,nny-1):
    for i in range(0,nnx-1):
        iconQ1[0,counter]=i+j*nnx
        iconQ1[1,counter]=i+1+j*nnx
        iconQ1[2,counter]=i+1+(j+1)*nnx
        iconQ1[3,counter]=i+(j+1)*nnx
        counter += 1
    #end for
#end for

print("connectivity: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if benchmark==1:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -1*cm/year
       #end if
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +1*cm/year
       #end if
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = +1*cm/year
       #end if
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -1*cm/year
       #end if
   #end for

if benchmark==2:
   for i in range(0,NV):
       if xV[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
       if xV[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       #end if
       if yV[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
       if yV[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       #end if
   #end for

print("boundary conditions: %.3f s" % (timing.time() - start))

#################################################################
# material layout
#################################################################
start = timing.time()

C1=np.zeros(NV,dtype=np.float64)  
C2=np.zeros(NV,dtype=np.float64)  

if benchmark==1:
   C1[:]=1
   C2[:]=0

if benchmark==2:
   for i in range(0,NV):
       if xV[i]<=800e3 and np.abs(yV[i]-Ly/2)<=300e3:
          C1[i]=1
          C2[i]=0
       else:
          C1[i]=0
          C2[i]=1
       #end if
   #end for
    
stats_C1_file.write("%e %e %e \n" %(time,np.min(C1),np.max(C1))) ; stats_C1_file.flush()
stats_C2_file.write("%e %e %e \n" %(time,np.min(C2),np.max(C2))) ; stats_C2_file.flush()

print("material layout: %.3f s" % (timing.time() - start))

#################################################################
# initialise stress fields 
#################################################################

tauxx=np.zeros(NV,dtype=np.float64)  
tauyy=np.zeros(NV,dtype=np.float64)  
tauxy=np.zeros(NV,dtype=np.float64)  
oxy = np.zeros(NV,dtype=np.float64)  
etaeff = np.zeros(NV,dtype=np.float64)  
Z = np.zeros(NV,dtype=np.float64)  
rho = np.zeros(NV,dtype=np.float64)  
    
for i in range(0,NV):
    etaeff[i]=C1[i]*etaeff1+C2[i]*etaeff2
    Z[i]     =C1[i]*Z1     +C2[i]*Z2
    rho[i]   =C1[i]*rho1   +C2[i]*rho2

#==============================================================================
#==============================================================================
#==============================================================================
# time stepping loop
#==============================================================================
#==============================================================================
#==============================================================================

u = np.zeros(NV,dtype=np.float64)          # x-component velocity
v = np.zeros(NV,dtype=np.float64)          # y-component velocity
R = np.zeros(3,dtype=np.float64)           # shape functions V
c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 


for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    #filename = 'quadrature_points_values_{:04d}.ascii'.format(istep)
    #qpts_file=open(filename,"w")

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = timing.time()

    K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector

    b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNNNVdx = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdy = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVdr = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNNNVds = np.zeros(mV,dtype=np.float64)          # shape functions derivatives

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        N_N_NP= np.zeros(mP*ndofP,dtype=np.float64)   

        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:9]=NNV(rq,sq)
                dNNNVdr[0:9]=dNNVdr(rq,sq)
                dNNNVds[0:9]=dNNVds(rq,sq)
                NNNP[0:4]=NNP(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                #end for 
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compute dNdx & dNdy
                xq=0.
                yq=0.
                Zq=0.
                C1q=0.
                C2q=0.
                tauxxq=0.
                tauyyq=0.
                tauxyq=0.
                etaeffq=0.
                oxyq=0.
                rhoq=0.
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    C1q+=NNNV[k]*C1[iconV[k,iel]]
                    C2q+=NNNV[k]*C2[iconV[k,iel]]
                    Zq+=NNNV[k]*Z[iconV[k,iel]]
                    oxyq+=NNNV[k]*oxy[iconV[k,iel]]
                    rhoq+=NNNV[k]*rho[iconV[k,iel]]
                    tauxxq+=NNNV[k]*tauxx[iconV[k,iel]]
                    tauyyq+=NNNV[k]*tauyy[iconV[k,iel]]
                    tauxyq+=NNNV[k]*tauxy[iconV[k,iel]]
                    etaeffq+=NNNV[k]*etaeff[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                #end for 

                #rhoq=C1q*rho1+C2q*rho2
                #etaeffq=C1q*etaeff1+C2q*etaeff2
                #Zq=C1q*Z1+C2q*Z2

                #qpts_file.write("%e %e %e %e %e %e %e %e %e\n"\
                #                 %(xq,yq,rhoq,etaeffq,Zq,tauxxq,tauyyq,tauxyq,oxyq))

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.      ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]
                #end for 

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaeffq*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rhoq*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq*gy
                #end for 

                #compute elastic rhs

                R[0]=Zq*(tauxxq+dt*oxyq*(2*tauxyq))
                R[1]=Zq*(tauyyq+dt*oxyq*(-2*tauxyq))
                R[2]=Zq*(tauxyq+dt*oxyq*(tauyyq-tauxxq))
                
                f_el-=b_mat.T.dot(R)*weightq*jcob

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.
                #end for 

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                N_N_NP[:]+=NNNP[:]*jcob*weightq

            #end for jq
        #end for iq

        G_el*=scaling_coeff
        h_el*=scaling_coeff

        # impose b.c. 
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                if bc_fix[m1]:
                   K_ref=K_el[ikk,ikk] 
                   for jkk in range(0,mV*ndofV):
                       f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                       K_el[ikk,jkk]=0
                       K_el[jkk,ikk]=0
                   #end for 
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0
                #end if 
            #end for 
        #end for 

        # assemble matrix K_mat and right hand side rhs
        for k1 in range(0,mV):
            for i1 in range(0,ndofV):
                ikk=ndofV*k1          +i1
                m1 =ndofV*iconV[k1,iel]+i1
                for k2 in range(0,mV):
                    for i2 in range(0,ndofV):
                        jkk=ndofV*k2          +i2
                        m2 =ndofV*iconV[k2,iel]+i2
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    G_mat[m1,m2]+=G_el[ikk,jkk]
                #end for 
                f_rhs[m1]+=f_el[ikk]
            #end for 
        #end for 
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
            constr[m2]+=N_N_NP[k2]
        #end for 

    #end for iel

    print("     -> K_mat (m,M) %e %e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G_mat (m,M) %e %e " %(np.min(G_mat),np.max(G_mat)))

    print("build FE matrix: %.3f s" % (timing.time() - start))

    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = timing.time()

    if pnormalise:
       a_mat = np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
       rhs   = np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
       a_mat[Nfem,NfemV:Nfem]=constr
       a_mat[NfemV:Nfem,Nfem]=constr
    else:
       a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
       rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
       a_mat[0:NfemV,0:NfemV]=K_mat
       a_mat[0:NfemV,NfemV:Nfem]=G_mat
       a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
    #end if

    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs

    print("assemble blocks: %.3f s" % (timing.time() - start))

    ######################################################################
    # solve system
    ######################################################################
    start = timing.time()

    sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

    print("solve time: %.3f s" % (timing.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = timing.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*scaling_coeff

    print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))

    stats_u_file.write("%e %e %e \n" %(time,np.min(u),np.max(u))) ; stats_u_file.flush()
    stats_v_file.write("%e %e %e \n" %(time,np.min(v),np.max(v))) ; stats_v_file.flush()

    #np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (timing.time() - start))

    #################################################################
    # compute timestep
    #################################################################
    CFL=0.5

    dt_CFL=CFL*(Lx/nelx)/np.max(np.sqrt(u**2+v**2))

    print('     dt_CFL= %e year' %(dt_CFL/year))
    print('     dt    = %e year' %(dt/year))

    #####################################################################
    # compute nodal strainrate and heat flux 
    #####################################################################
    start = timing.time()
    
    count = np.zeros(NV,dtype=np.int16)  
    q=np.zeros(NV,dtype=np.float64)
    c=np.zeros(NV,dtype=np.float64)
    Lxx = np.zeros(NV,dtype=np.float64)  
    Lxy = np.zeros(NV,dtype=np.float64)  
    Lyx = np.zeros(NV,dtype=np.float64)  
    Lyy = np.zeros(NV,dtype=np.float64)  

    #u[:]=yV[:]
    #v[:]=xV[:]

    for iel in range(0,nel):
        for i in range(0,mV):
            rq=rVnodes[i]
            sq=sVnodes[i]
            NNNV[0:mV]=NNV(rq,sq)
            dNNNVdr[0:mV]=dNNVdr(rq,sq)
            dNNNVds[0:mV]=dNNVds(rq,sq)
            NNNP[0:mP]=NNP(rq,sq)
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
            #end for
            jcbi=np.linalg.inv(jcb)
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
            #end for
            L_xx=0.
            L_xy=0.
            L_yx=0.
            L_yy=0.
            for k in range(0,mV):
                L_xx += dNNNVdx[k]*u[iconV[k,iel]]
                L_xy += dNNNVdx[k]*v[iconV[k,iel]]
                L_yx += dNNNVdy[k]*u[iconV[k,iel]]
                L_yy += dNNNVdy[k]*v[iconV[k,iel]]
            #end for
            inode=iconV[i,iel]
            Lxx[inode]+=L_xx
            Lxy[inode]+=L_xy
            Lyx[inode]+=L_yx
            Lyy[inode]+=L_yy
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
    #end for
    Lxx/=count
    Lxy/=count
    Lyx/=count
    Lyy/=count
    q/=count

    print("     -> Lxx (m,M) %.6e %.6e " %(np.min(Lxx),np.max(Lxx)))
    print("     -> Lxy (m,M) %.6e %.6e " %(np.min(Lxy),np.max(Lxy)))
    print("     -> Lyx (m,M) %.6e %.6e " %(np.min(Lyx),np.max(Lyx)))
    print("     -> Lyy (m,M) %.6e %.6e " %(np.min(Lyy),np.max(Lyy)))

    #np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')
    #np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

    print("compute nodal p and L: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute nodal fields
    #####################################################################
    start = timing.time()

    exx = np.zeros(NV,dtype=np.float64)  
    eyy = np.zeros(NV,dtype=np.float64)  
    exy = np.zeros(NV,dtype=np.float64)  
    oxy = np.zeros(NV,dtype=np.float64)  
    Jxx = np.zeros(NV,dtype=np.float64)  
    Jyy = np.zeros(NV,dtype=np.float64)  
    Jxy = np.zeros(NV,dtype=np.float64)  

    exx[:]=Lxx[:]
    eyy[:]=Lyy[:]
    exy[:]=0.5*(Lxy[:]+Lyx[:])
    oxy[:]=0.5*(Lxy[:]-Lyx[:])

    Jxx[:]=2*tauxx[:]*oxy[:]
    Jyy[:]=-2*tauxy[:]*oxy[:]
    Jxy[:]=(tauyy[:]-tauxx[:])*oxy[:]

    print("     -> exx (m,M) %.6e %.6e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.6e %.6e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.6e %.6e " %(np.min(exy),np.max(exy)))
    print("     -> oxy (m,M) %.6e %.6e " %(np.min(oxy),np.max(oxy)))

    stats_exx_file.write("%e %e %e \n" %(time,np.min(exx),np.max(exx))) ; stats_exx_file.flush()
    stats_eyy_file.write("%e %e %e \n" %(time,np.min(eyy),np.max(eyy))) ; stats_eyy_file.flush()
    stats_exy_file.write("%e %e %e \n" %(time,np.min(exy),np.max(exy))) ; stats_exy_file.flush()
    stats_oxy_file.write("%e %e %e \n" %(time,np.min(oxy),np.max(oxy))) ; stats_oxy_file.flush()

    print("compute sr, rr and J: %.3f s" % (timing.time() - start))

    #####################################################################

    time+=dt

    #####################################################################
    # compute Z and J 
    #####################################################################
    start = timing.time()

    etaeff = np.zeros(NV,dtype=np.float64)  
    Z = np.zeros(NV,dtype=np.float64)  
    rho = np.zeros(NV,dtype=np.float64)  

    for i in range(0,NV):
        etaeff[i]=C1[i]*etaeff1+C2[i]*etaeff2
        Z[i]     =C1[i]*Z1     +C2[i]*Z2
        rho[i]   =C1[i]*rho1   +C2[i]*rho2
        if etaeff[i]<=0:
           print(i,xV[i],yV[i],C1[i],C2[i],etaeff1,etaeff2)
           exit("eta_eff<=0")
        if Z[i]<=0:
           print(i,xV[i],yV[i],C1[i],C2[i],Z1,Z2)
           exit("Z<=0")

    #etaeff[:]=C1[:]*etaeff1+C2[:]*etaeff2
    #Z[:]=C1[:]*Z1+C2[:]*Z2

    stats_Z_file.write("%e %e %e \n" %(time,np.min(Z),np.max(Z))) ; stats_Z_file.flush()
    stats_etaeff_file.write("%e %e %e \n" %(time,np.min(etaeff),np.max(etaeff))) ; stats_etaeff_file.flush()

    print("     -> etaeff (m,M) %.6e %.6e " %(np.min(etaeff),np.max(etaeff)))
    print("     -> Z      (m,M) %.6e %.6e " %(np.min(Z),np.max(Z)))

    print("compute nodal etaeff, Z: %.3f s" % (timing.time() - start))

    #####################################################################
    # update dev stress fields
    #####################################################################
    start = timing.time()

    tauxx=2*etaeff*exx+Z*tauxx+Z*dt*Jxx
    tauyy=2*etaeff*eyy+Z*tauyy+Z*dt*Jyy
    tauxy=2*etaeff*exy+Z*tauxy+Z*dt*Jxy

    print("     -> tauxx (m,M) %.6e %.6e " %(np.min(tauxx),np.max(tauxx)))
    print("     -> tauyy (m,M) %.6e %.6e " %(np.min(tauyy),np.max(tauyy)))
    print("     -> tauxy (m,M) %.6e %.6e " %(np.min(tauxy),np.max(tauxy)))

    stats_tauxx_file.write("%e %e %e \n" %(time,np.min(tauxx),np.max(tauxx))) ; stats_tauxx_file.flush()
    stats_tauyy_file.write("%e %e %e \n" %(time,np.min(tauyy),np.max(tauyy))) ; stats_tauyy_file.flush()
    stats_tauxy_file.write("%e %e %e \n" %(time,np.min(tauxy),np.max(tauxy))) ; stats_tauxy_file.flush()

    print("compute/update tau: %.3f s" % (timing.time() - start))

    #####################################################################
    # advect fields
    #####################################################################

    for ifield in range(0,6): # C1 C2 tauxx tauyy tauxy

        start = timing.time()

        A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
        rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
        B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
        N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions
        Tvect = np.zeros(mV,dtype=np.float64)   

        field=np.zeros(NV,dtype=np.float64)
        if ifield==0:
           field[:]=C1[:]
        if ifield==1:
           field[:]=C2[:]
        if ifield==2:
           field[:]=tauxx[:]
        if ifield==3:
           field[:]=tauyy[:]
        if ifield==4:
           field[:]=tauxy[:]

        for iel in range (0,nel):

            b_el=np.zeros(mV*ndofT,dtype=np.float64)
            a_el=np.zeros((mV*ndofT,mV*ndofT),dtype=np.float64)
            Ka=np.zeros((mV,mV),dtype=np.float64)   # elemental advection matrix 
            MM=np.zeros((mV,mV),dtype=np.float64)   # elemental mass matrix 
            vel=np.zeros((1,ndim),dtype=np.float64)

            for k in range(0,mV):
                Tvect[k]=field[iconV[k,iel]]
            #end for

            for iq in [0,1,2]:
                for jq in [0,1,2]:
                    rq=qcoords[iq]
                    sq=qcoords[jq]
                    weightq=qweights[iq]*qweights[jq]

                    N_mat[0:mV,0]=NNV(rq,sq)
                    dNNNVdr[0:mV]=dNNVdr(rq,sq)
                    dNNNVds[0:mV]=dNNVds(rq,sq)

                    # calculate jacobian matrix
                    jcb=np.zeros((ndim,ndim),dtype=np.float64)
                    for k in range(0,mV):
                        jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                        jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                        jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                        jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                    #end for
                    jcob = np.linalg.det(jcb)
                    jcbi = np.linalg.inv(jcb)

                    # compute dNdx & dNdy
                    vel[0,0]=0.
                    vel[0,1]=0.
                    for k in range(0,mV):
                        vel[0,0]+=N_mat[k,0]*u[iconV[k,iel]]
                        vel[0,1]+=N_mat[k,0]*v[iconV[k,iel]]
                        dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                        dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                        B_mat[0,k]=dNNNVdx[k]
                        B_mat[1,k]=dNNNVdy[k]
                    #end for

                    # compute mass matrix
                    MM+=N_mat.dot(N_mat.T)*weightq*jcob

                    # compute advection matrix
                    Ka+=N_mat.dot(vel.dot(B_mat))*weightq*jcob

                #end for
            #end for

            a_el+=MM+(Ka)*dt*alpha
            b_el=(MM-(Ka)*(1.-alpha)*dt).dot(Tvect)

            # assemble matrix A_mat and right hand side rhs
            for k1 in range(0,mV):
                m1=iconV[k1,iel]
                for k2 in range(0,mV):
                    m2=iconV[k2,iel]
                    A_mat[m1,m2]+=a_el[k1,k2]
                #end for
                rhs[m1]+=b_el[k1]
            #end for

        #end for iel

        field = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

        #filter composition with leka93 algorithm

        if ifield==0:
           print("     C1 (m,M) bef. %.4f %.4f " %(np.min(field),np.max(field)))
           stats_C1_file.write("%e %e %e \n" %(time,np.min(field),np.max(field))) ; stats_C1_file.flush()
        if ifield==1:
           print("     C2 (m,M) bef. %.4f %.4f " %(np.min(field),np.max(field)))
           stats_C2_file.write("%e %e %e \n" %(time,np.min(field),np.max(field))) ; stats_C2_file.flush()

        if filter_compositions and ifield<2:
           sum0=np.sum(field)
           minC=np.min(field)
           maxC=np.max(field)
           for i in range(0,NV):
               if field[i]<=np.abs(minC):
                  field[i]=0
               if field[i]>=2.-maxC:
                  field[i]=1
           sum1=np.sum(field)
           num=0
           for i in range(0,NV):
               if field[i]>0 and field[i]<1:
                  num+=1 
               #end if
           #end for
           for i in range(0,NV):
               if C1[i]>0 and C1[i]<1:
                  C1[i]+=(sum0-sum1)/num 
               #end if
           #end for
        #end if

        if ifield==0:
           print("     C1 (m,M) aft. %.4f %.4f " %(np.min(field),np.max(field)))
        if ifield==1:
           print("     C2 (m,M) aft. %.4f %.4f " %(np.min(field),np.max(field)))

        #end of filtering

        if ifield==0:
           C1[:]=field[:]
           print("advect C1 time: %.3f s" % (timing.time() - start))
           stats_C1_file.write("%e %e %e \n" %(time,np.min(C1),np.max(C1))) ; stats_C1_file.flush()

        if ifield==1:
           C2[:]=field[:]
           print("advect C2 time: %.3f s" % (timing.time() - start))
           stats_C2_file.write("%e %e %e \n" %(time,np.min(C2),np.max(C2))) ; stats_C2_file.flush()

        if ifield==2:
           tauxx[:]=field[:]
           print("advect tauxx time: %.3f s" % (timing.time() - start))
           stats_tauxx_file.write("%e %e %e \n" %(time,np.min(tauxx),np.max(tauxx))) ; stats_tauxx_file.flush()

        if ifield==3:
           tauyy[:]=field[:]
           print("advect tauyy time: %.3f s" % (timing.time() - start))
           stats_tauyy_file.write("%e %e %e \n" %(time,np.min(tauyy),np.max(tauyy))) ; stats_tauyy_file.flush()

        if ifield==4:
           tauxy[:]=field[:]
           print("advect tauxy time: %.3f s" % (timing.time() - start))
           stats_tauxy_file.write("%e %e %e \n" %(time,np.min(tauxy),np.max(tauxy))) ; stats_tauxy_file.flush()

    #end for ifield

    stats_C1pC2_file.write("%e %e %e \n" %(time,np.min(C1+C2),np.max(C1+C2))) ; stats_C1pC2_file.flush()

    #####################################################################
    # plot of solution
    #####################################################################

    filename = 'solution_{:04d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel2))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
    #for iel in range (0,nel):
    #    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
    #vtufile.write("</DataArray>\n")
    #vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/yr)' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='C1' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %C1[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='C2' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %C2[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='C1+C2' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(C1[i]+C2[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(C1[i]*rho1+C2[i]*rho2))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mu' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(C1[i]*mu1+C2[i]*mu2))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(C1[i]*eta1+C2[i]*eta2))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta_eff' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %etaeff[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='Z' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %Z[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(exx[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_yy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(eyy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='e_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(exy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='omega_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(oxy[i]))
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xx' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauxx[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_yy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauyy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='tau_xy' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10e \n" %(tauxy[i]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel2):
        vtufile.write("%d %d %d %d \n" %(iconQ1[0,iel],iconQ1[1,iel],iconQ1[2,iel],iconQ1[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel2):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel2):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
