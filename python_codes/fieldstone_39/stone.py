import numpy as np
import sys as sys
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as timing
from numpy import linalg as LA

#------------------------------------------------------------------------------

def gx(x,y):
    return 0

def gy(x,y):
    if benchmark==4 or benchmark==5:
       return 0
    else:
       return -10

#------------------------------------------------------------------------------
# benchmark=1: simple brick
# benchmark=2: Spiegelman et al (2016) 
# benchmark=3: Kaus (2010) brick
# benchmark=4: Gerya book (2019) 
# benchmark=5: Duretz et al (2018)

def ubc(x,y):
    if benchmark==1:
       vaal=1.e-15*(Lx/2.0)
    if benchmark==2:
       vaal=0.0025/year
    if benchmark==3: 
       vaal=-1.e-15*(Lx/2.0)
    if benchmark==4:
       vaal=5e-9
    if benchmark==5:
       vaal=5.e-15*(Lx/2.0)

    if x<Lx/2:
       val=vaal
    elif x>Lx/2:
       val=-vaal
    else:
       val=0
    return val

def vbc(x,y):
    if benchmark==4:
       vaal=5e-9
       if y<Ly/2:
          val=-vaal
       elif y>Ly/2:
          val=+vaal
       else:
          val=0
       return val 
    if benchmark==5:
       vaal=5.e-15*(Ly/2.0)
       if y<Ly/2:
          val=-vaal
       elif y>Ly/2:
          val=+vaal
       else:
          val=0
       return val 
    else:    
       return 0

def viscosity(exx,eyy,exy,pq,c,phi,iter,x,y,eta_m,eta_v):

    # deviatoric tensor E
    Exx=exx-(exx+eyy)/3
    Eyy=eyy-(exx+eyy)/3
    Ezz=   -(exx+eyy)/3

    #compute effective strain rate (sqrt of 2nd inv)
    e2=np.sqrt(0.5*(Exx**2+Eyy**2+Ezz**2)+exy**2)

    #-------------------------------------------------
    if benchmark==1: # simple brick
    #-------------------------------------------------
       if iter==0:
          e2=1e-15
          two_sin_psi=0.
       else:
          two_sin_psi=2.*np.sin(psi)
       #end if
       Y=max(pq,0)*np.sin(phi)+c*np.cos(phi)
       if 2*eta_v*e2<Y:
          val=eta_v
          eps_vp=0
          mech=1
       else:
          tau=(Y+2*eta_m*e2)/(1+eta_m/eta_v)
          eps_v=tau/2/eta_v
          eps_vp=e2-eps_v
          eta_vp=Y/(2.*eps_vp)+eta_m
          val=1./(1./eta_v + 1/eta_vp)
          mech=2
       #end if

       #regularised approximation
       #e2c=Y/2/eta_v
       #val=(1-np.exp(-e2/e2c))*(Y/(2.*e2)+eta_m)

    #end if

    #-------------------------------------------------
    if benchmark==2: # spmw16
    #-------------------------------------------------

       if y<8e3 or (abs(x-64e3)<2e3 and y<10e3):
          val=1e21
          two_sin_psi=0.
       else:
          if iter==0:
             e2=1.32e-15
             two_sin_psi=0.
          else:
             two_sin_psi=2.*np.sin(psi)
          #end if
          Y=pq*np.sin(phi)+c*np.cos(phi)
          if 2*eta_v*e2<Y:
             val=eta_v
          else:
             tau=(Y+2*eta_m*e2)/(1+eta_m/eta_v)
             eps_v=tau/2/eta_v
             eps_vp=e2-eps_v
             eta_vp=Y/(2.*eps_vp)+eta_m
             val=1./(1./eta_v + 1/eta_vp)
          #end if
          val=max(1e21,val)
       #end if

    #-------------------------------------------------
    if benchmark==3: # brick with seed
    #-------------------------------------------------

       if abs(x-20e3)<400 and y<400:
          val=1e20
          two_sin_psi=0.
       else:
          if iter==0:
             e2=1e-15
             two_sin_psi=0.
          else:
             two_sin_psi=2.*np.sin(psi)
          #end if
          etap=(pq*np.sin(phi)+c*np.cos(phi))/(2*e2)
          eta1=1e25
          #val=1./(1./(etap+1e20)+1./eta1)
          val=etap
          val=min(1.e25,val)
          val=max(1.e20,val)
       #end if
    #end if

    #-------------------------------------------------
    if benchmark==4: # shortening block
    #-------------------------------------------------

       if abs(x-Lx/2)<6.25e3 and abs(y-Ly/2)<6.25e3:
          val=1e17
          two_sin_psi=0.
       elif y<25e3 or y>75e3:
          val=1e17
          two_sin_psi=0.
       else:
          if iter==0:
             e2=1e-13
             two_sin_psi=0.
          else:
             two_sin_psi=2.*np.sin(psi)
          #end if

          Y=pq*np.sin(phi)+c*np.cos(phi)
          if 2*eta_v*e2<Y:
             val=eta_v
          else:
             tau=(Y+2*eta_m*e2)/(1+eta_m/eta_v)
             eps_v=tau/2/eta_v
             eps_vp=e2-eps_v
             eta_vp=Y/(2.*eps_vp)+eta_m
             val=1./(1./eta_v + 1/eta_vp)
          #end if

    #-------------------------------------------------
    if benchmark==5: # shortening block 2 (dusd18)
    #-------------------------------------------------

       if (x-Lx/2)**2 + (y-Ly/2)**2 < 100**2:
          val=1e17
          two_sin_psi=0.
       else:
          if iter==0:
             e2=1e-15
             two_sin_psi=0.
          else:
             two_sin_psi=2.*np.sin(psi)
          #end if

          Y=pq*np.sin(phi)+c*np.cos(phi)
          if 2*eta_v*e2<Y:
             val=eta_v
          else:
             tau=(Y+2*eta_m*e2)/(1+eta_m/eta_v)
             eps_v=tau/2/eta_v
             eps_vp=e2-eps_v
             eta_vp=Y/(2.*eps_vp)+eta_m
             val=1./(1./eta_v + 1/eta_vp)
          #end if
          val=max(1e21,val)

    return val,two_sin_psi,eps_vp,mech

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
# benchmark=1: brick with velocity discontinuity at bottom
# benchmark=2: Spiegelman et al, 2016. 
# benchmark=3: brick with seed Kaus (2010)
# benchmark=4: shortening block with sticky air - square inclusion (Gerya book)
# benchmark=5: shortening block - round inclusion,  Duretz et al (2018)
#------------------------------------------------------------------------------

cm=0.01
year=3600*24*365.
eps=1.e-10

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

if int(len(sys.argv) == 8):
   print("reading arguments")
   nelx = int(sys.argv[1])
   benchmark = int(sys.argv[2])
   phi = float(sys.argv[3])
   psi =  float(sys.argv[4])
   niter =  int(sys.argv[5])
   eta_m =  float(sys.argv[6])
   eta_v =  float(sys.argv[7])
   eta_v=10**eta_v
   eta_m=10**eta_m
   produce_nl_vtu=False
   name='_nelx'+sys.argv[1]+'_phi'+sys.argv[3]+'_psi'+sys.argv[4]+'_etam'+sys.argv[6]
   every=1000000
else:
   produce_nl_vtu=True
   benchmark=1
   every=1
   if benchmark==1:
      nelx = 128
      phi=30
      psi=30
      niter=250
      eta_v=1e25
      eta_m=1e20
      name=''

tol_nl=1e-6 # nonlinear tolerance

phi=phi/180*np.pi
psi=psi/180*np.pi

if benchmark==1: # simple brick
   Lx=80000. 
   Ly=10000.  
   rho=2800
   cohesion=1e7

if benchmark==2:  #----spmw16----
   Lx=128000. 
   Ly=32000.  
   rho=2700.
   cohesion=1e8

if benchmark==3:   #----kaus10----
   Lx=40e3
   Ly=10e3
   rho=2700.
   cohesion=40e6

if benchmark==4:  #----geryabook----
   Lx=100e3
   Ly=100e3
   rho=0
   cohesion=1e8

if benchmark==5: #----dusd18----
   Lx=4e3
   Ly=2e3
   rho=0
   cohesion=30e6

#################################################################

nely = int(nelx*Ly/Lx)        # number of elements y direction
nnx=2*nelx+1                  # number of nodes, x direction
nny=2*nely+1                  # number of nodes, y direction
NV=nnx*nny                    # total number of nodes
nel=nelx*nely                 # total number of elements
NfemV=NV*ndofV                # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
hx=Lx/nelx                    # mesh size in x direction
hy=Ly/nely                    # mesh size in y direction

#################################################################
# quadrature parameters

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nq=9*nel

#################################################################

eta_ref=1.e23      # scaling of G blocks
scaling_coeff=eta_ref/Ly
solver = 2 
if solver==1:
   use_SchurComplementApproach=True
   use_preconditioner=True
   niter_stokes=250
   solver_tolerance=1e-6
else:
   use_SchurComplementApproach=False
if use_SchurComplementApproach:
   ls_conv_file=open("linear_solver_convergence.ascii","w")
   ls_niter_file=open("linear_solver_niter.ascii","w")
   
use_srn_diff=True

#################################################################

ustats_file=open("stats_u"+name+".ascii","w")
vstats_file=open("stats_v"+name+".ascii","w")
pstats_file=open("stats_p"+name+".ascii","w")

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("NfemV=",NfemV)
print("NfemP=",NfemP)
print("Nfem=",Nfem)
print("hx",hx)
print("hy",hy)
print("niter",niter)
print("eta_m",eta_m)
print("eta_v",eta_v)
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

#np.savetxt('grid.ascii',np.array([xV,yV]).T,header='# x,y')

print("setup: grid points: %.3f s" % (timing.time() - start))

#################################################################
# build connectivity arrays for velocity and pressure
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################
start = timing.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
iconP=np.zeros((mP,nel),dtype=np.int32)

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

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (timing.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = timing.time()

u     =np.zeros(NV,dtype=np.float64)    # x-component velocity
v     =np.zeros(NV,dtype=np.float64)    # y-component velocity
bc_fix=np.zeros(NfemV,dtype=np.bool)    # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value

if benchmark==1: # simple brick
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
          u[i]=ubc(xV[i],yV[i])
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
          u[i] = ubc(xV[i],yV[i])
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(xV[i],yV[i])

if benchmark==2 or benchmark==3: # spmw16 & kaus10
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
          u[i]=ubc(xV[i],yV[i])
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
          u[i] = ubc(xV[i],yV[i])
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(xV[i],yV[i])
          v[i] = vbc(xV[i],yV[i])

if benchmark==4 or benchmark==5: # shortening blocks
   for i in range(0,NV):
       if xV[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
       if xV[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = ubc(xV[i],yV[i])
       if yV[i]/Ly<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(xV[i],yV[i])
       if yV[i]/Ly>1-eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = vbc(xV[i],yV[i])

print("setup: boundary conditions: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# non-linear iterations
#------------------------------------------------------------------------------
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

dNNNVdx = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdy = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVdr = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
dNNNVds = np.zeros(mV,dtype=np.float64)           # shape functions derivatives
p       = np.zeros(NfemP,dtype=np.float64)        # pressure field 
Res     = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 
sol     = np.zeros(Nfem,dtype=np.float64)         # solution vector 
b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
NNNV    = np.zeros(mV,dtype=np.float64)           # shape functions V
NNNP    = np.zeros(mP,dtype=np.float64)           # shape functions P
solP    = np.zeros(NfemP,dtype=np.float64)        # P solution vector 
solV    = np.zeros(NfemV,dtype=np.float64)        # V solution vector
exxn=np.zeros(NV,dtype=np.float64)
eyyn=np.zeros(NV,dtype=np.float64)
exyn=np.zeros(NV,dtype=np.float64)
srn=np.zeros(NV,dtype=np.float64)
etan=np.zeros(NV,dtype=np.float64)

convfile=open('conv'+name+'.ascii',"w")
vrmsfile=open('vrms'+name+'.ascii',"w")
avrgsrfile=open('avrgsr'+name+'.ascii',"w")

for iter in range(0,niter):

   print("--------------------------")
   print("iter=", iter)
   print("--------------------------")

   #################################################################
   # build FE matrix A and rhs 
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   #################################################################

   A_sparse= lil_matrix((Nfem,Nfem),dtype=np.float64) # FEM stokes matrix 
   rhs     = np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
   N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # N matrix  
   M_mat   = np.zeros((NfemP,NfemP),dtype=np.float64) # schur precond
   f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
   h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
   xq      = np.zeros(9*nel,dtype=np.float64)         # x coords of q points 
   yq      = np.zeros(9*nel,dtype=np.float64)         # y coords of q points 
   etaq    = np.zeros(9*nel,dtype=np.float64)         # viscosity of q points 
   pq      = np.zeros(9*nel,dtype=np.float64)         # pressure of q points 
   srq_T   = np.zeros(9*nel,dtype=np.float64)         # total strain rate of q points 
   srq_vp  = np.zeros(9*nel,dtype=np.float64)         # viscoplastic rate of q points 
   mechq   = np.zeros(9*nel,dtype=np.float64)         # deformation mechanism at q points 

   counter=0
   for iel in range(0,nel):

       # set arrays to 0 for each element 
       f_el =np.zeros((mV*ndofV),dtype=np.float64)
       K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
       G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
       h_el=np.zeros((mP*ndofP),dtype=np.float64)
       M_el=np.zeros((mP,mP),dtype=np.float64)  

       # integrate viscous term at 4 quadrature points
       for jq in [0,1,2]:
           for iq in [0,1,2]:

               # position & weight of quad. point
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]

               NNNV[0:mV]=NNV(rq,sq)
               dNNNVdr[0:mV]=dNNVdr(rq,sq)
               dNNNVds[0:mV]=dNNVds(rq,sq)
               NNNP[0:mP]=NNP(rq,sq)

               # calculate jacobian matrix
               #jcb=np.zeros((ndim,ndim),dtype=np.float64)
               #for k in range(0,mV):
               #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
               #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
               #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
               #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
               #jcob = np.linalg.det(jcb)
               #jcbi = np.linalg.inv(jcb)

               #only valid for rectangular elements!
               jcbi=np.zeros((ndim,ndim),dtype=np.float64)
               jcob=hx*hy/4
               jcbi[0,0] = 2/hx 
               jcbi[1,1] = 2/hy

               # compute dNdx & dNdy & strainrate
               exxq=0.0
               eyyq=0.0
               exyq=0.0
               for k in range(0,mV):
                   xq[counter]+=NNNV[k]*xV[iconV[k,iel]]
                   yq[counter]+=NNNV[k]*yV[iconV[k,iel]]
                   #dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                   #dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                   dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]
                   dNNNVdy[k]=jcbi[1,1]*dNNNVds[k]
                   exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                   eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                   exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

               if use_srn_diff:
                  exxq=0.0
                  eyyq=0.0
                  exyq=0.0
                  for k in range(0,mV):
                      exxq+=NNNV[k]*exxn[iconV[k,iel]]
                      eyyq+=NNNV[k]*eyyn[iconV[k,iel]]
                      exyq+=NNNV[k]*exyn[iconV[k,iel]]

               # effective strain rate at qpoint                
               srq_T[counter]=np.sqrt(0.5*(exxq*exxq+eyyq*eyyq)+exyq*exyq)

               # compute pressure at qpoint
               for k in range(0,mP):
                   pq[counter]+=NNNP[k]*p[iconP[k,iel]]

               # construct 3x8 b_mat matrix
               for i in range(0,mV):
                   b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                            [0.        ,dNNNVdy[i]],
                                            [dNNNVdy[i],dNNNVdx[i]]]

               # compute effective plastic viscosity
               etaq[counter],two_sin_psi,srq_vp[counter],mechq[counter]=\
                   viscosity(exxq,eyyq,exyq,pq[counter],cohesion,phi,\
                   iter,xq[counter],yq[counter],eta_m,eta_v)

               dilation_rate=two_sin_psi*srq_vp[counter]*0.5

               # compute elemental a_mat matrix
               K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob

               # compute elemental rhs vector
               for i in range(0,mV):
                   f_el[ndofV*i+0]+=NNNV[i]*jcob*weightq*gx(xq,yq)*rho
                   f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy(xq,yq)*rho

               #add to it dilation term
               for i in range(0,mV):
                   f_el[ndofV*i+0]-=2./3.*dNNNVdx[i]*jcob*weightq*etaq[counter]*dilation_rate
                   f_el[ndofV*i+1]-=2./3.*dNNNVdy[i]*jcob*weightq*etaq[counter]*dilation_rate

               for i in range(0,mP):
                   N_mat[0,i]=NNNP[i]
                   N_mat[1,i]=NNNP[i]
                   N_mat[2,i]=0.

               G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                
               h_el[:]-=NNNP[:]*dilation_rate*weightq*jcob

               for i in range(0,mP):
                   for j in range(0,mP):
                       M_el[i,j]+=NNNP[i]*NNNP[j]*weightq*jcob/etaq[counter]
                   # end for j
               # end for i

               counter+=1
           # end for iq 
       # end for jq 

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
                  #end for jkk
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val[m1]
                  G_el[ikk,:]=0
               # end if 
           # end for i1 
       #end for k1 

       # assemble matrix K_mat and right hand side rhs
       for k1 in range(0,mV):
           for i1 in range(0,ndofV):
               ikk=ndofV*k1          +i1
               m1 =ndofV*iconV[k1,iel]+i1
               for k2 in range(0,mV):
                   for i2 in range(0,ndofV):
                       jkk=ndofV*k2          +i2
                       m2 =ndofV*iconV[k2,iel]+i2
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                   #end for i2
               #end for k2
               for k2 in range(0,mP):
                   jkk=k2
                   m2 =iconP[k2,iel]
                   A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*scaling_coeff
                   A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*scaling_coeff
               f_rhs[m1]+=f_el[ikk]
               #end for k2
           #end for i1
       #end for k1 

       for k1 in range(0,mP):
           m1=iconP[k1,iel]
           h_rhs[m1]+=h_el[k1]*scaling_coeff
           for k2 in range(0,mP):
               m2=iconP[k2,iel]
               M_mat[m1,m2]+=M_el[k1,k2]
           #end for k2 
       #end for k1

   # end for iel 

   print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
   print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

   print("     -> etaq (m,M) %.5e %.5e " %(np.min(etaq),np.max(etaq)))

   print("build FE matrix: %.3f s" % (timing.time() - start))

   ######################################################################
   # pressure nullspace removal
   ######################################################################

   if benchmark==4 or benchmark==5: 
      for i in range(0,Nfem):
          A_sparse[Nfem-1,i]=0
          A_sparse[i,Nfem-1]=0
          A_sparse[Nfem-1,Nfem-1]=1
          h_rhs[NfemP-1]=0

   ######################################################################
   # assemble K, G, GT, f, h into A and rhs
   ######################################################################
   start = timing.time()

   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   sparse_matrix=A_sparse.tocsr()
   Res=sparse_matrix.dot(sol)-rhs
   sol=sps.linalg.spsolve(sparse_matrix,rhs)

   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   ustats_file.write("%d %8e %8e \n" %(iter,np.min(u),np.max(u)))
   vstats_file.write("%d %8e %8e \n" %(iter,np.min(v),np.max(v)))
   pstats_file.write("%d %8e %8e \n" %(iter,np.min(p),np.max(p)))
   ustats_file.flush()
   vstats_file.flush()
   pstats_file.flush()

   print("solve system: %.3f s - Nfem %d" % (timing.time() - start, Nfem))

   #################################################################
   #normalise pressure
   #################################################################
   start = timing.time()

   if benchmark==4:

      int_p=0
      for iel in range(0,nel):
          for jq in [0,1,2]:
              for iq in [0,1,2]:
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]
                  NNNP[0:mP]=NNP(rq,sq)
                  jcob=hx*hy/4
                  p_q=NNNP[0:mP].dot(p[iconP[0:mP,iel]])
                  int_p+=p_q*weightq*jcob
              #end for
          #end for
      #end for

      avrg_p=int_p/Lx/Ly

      print("     -> int_p %e " %(int_p))
      print("     -> avrg_p %e " %(avrg_p))

      p[:]-=avrg_p

      print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("normalise pressure: %.3f s" % (timing.time() - start))

   #################################################################
   # compute non-linear residual
   #################################################################
   start = timing.time()

   if iter==0:
      Res0_inf=LA.norm(Res,np.inf)
      Res0_two=LA.norm(Res,2)

   Res_inf=LA.norm(Res,np.inf)
   Res_two=LA.norm(Res,2)

   conv_inf=Res_inf/Res0_inf
   conv_two=Res_two/Res0_two

   print("     -> Nonlinear res. (2-norm) %7e" % (conv_two))

   if conv_two<tol_nl:
      break

   Res_u,Res_v=np.reshape(Res[0:NfemV],(NV,2)).T
   Res_p=Res[NfemV:Nfem]
   Ru=LA.norm(Res_u,2)
   Rv=LA.norm(Res_v,2)
   Rp=LA.norm(Res_p,2)

   convfile.write("%3d %10e %10e %10e %10e\n" %(iter,conv_two,Ru,Rv,Rp)) 
   convfile.flush()

   #np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
   #np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x,y,u,v]).T,header='# x,y,u,v')
   #np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')

   print("computing res norms: %.3f s" % (timing.time() - start))

   #####################################################################
   # interpolate pressure onto velocity grid points (for plotting)
   #####################################################################
   # velocity    pressure
   # 3---6---2   3-------2
   # |       |   |       |
   # 7   8   5   |       |
   # |       |   |       |
   # 0---4---1   0-------1
   #################################################################
   start = timing.time()

   q=np.zeros(NV,dtype=np.float64)
   Res_q=np.zeros(NV,dtype=np.float64)

   for iel in range(0,nel):
       q[iconV[0,iel]]=p[iconP[0,iel]]
       q[iconV[1,iel]]=p[iconP[1,iel]]
       q[iconV[2,iel]]=p[iconP[2,iel]]
       q[iconV[3,iel]]=p[iconP[3,iel]]
       q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
       q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
       q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
       q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
       q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+\
                        p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

   for iel in range(0,nel):
       Res_q[iconV[0,iel]]=Res_p[iconP[0,iel]]
       Res_q[iconV[1,iel]]=Res_p[iconP[1,iel]]
       Res_q[iconV[2,iel]]=Res_p[iconP[2,iel]]
       Res_q[iconV[3,iel]]=Res_p[iconP[3,iel]]
       Res_q[iconV[4,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]])*0.5
       Res_q[iconV[5,iel]]=(Res_p[iconP[1,iel]]+Res_p[iconP[2,iel]])*0.5
       Res_q[iconV[6,iel]]=(Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.5
       Res_q[iconV[7,iel]]=(Res_p[iconP[3,iel]]+Res_p[iconP[0,iel]])*0.5
       Res_q[iconV[8,iel]]=(Res_p[iconP[0,iel]]+Res_p[iconP[1,iel]]+\
                            Res_p[iconP[2,iel]]+Res_p[iconP[3,iel]])*0.25

   #np.savetxt('q_{:04d}.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute strainrate at center of element 
   ######################################################################
   start = timing.time()

   xc = np.zeros(nel,dtype=np.float64)  
   yc = np.zeros(nel,dtype=np.float64)  
   pc = np.zeros(nel,dtype=np.float64)  
   exx = np.zeros(nel,dtype=np.float64)  
   eyy = np.zeros(nel,dtype=np.float64)  
   exy = np.zeros(nel,dtype=np.float64)  
   sr  = np.zeros(nel,dtype=np.float64)  

   for iel in range(0,nel):

       rq = 0.
       sq = 0.

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
       jcob=np.linalg.det(jcb)
       jcbi=np.linalg.inv(jcb)

       for k in range(0,mV):
           dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
           dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

       for k in range(0,mV):
           xc[iel] += NNNV[k]*xV[iconV[k,iel]]
           yc[iel] += NNNV[k]*yV[iconV[k,iel]]
           exx[iel] += dNNNVdx[k]*u[iconV[k,iel]]
           eyy[iel] += dNNNVdy[k]*v[iconV[k,iel]]
           exy[iel] += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]

       sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

       for k in range(0,mP):
           pc[iel] += NNNP[k]*p[iconP[k,iel]]

   #end if

   print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
   print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
   print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))
   print("     -> sr  (m,M) %.5e %.5e " %(np.min(sr),np.max(sr)))
   print("     -> pc  (m,M) %.5e %.5e " %(np.min(pc),np.max(pc)))

   print("compute press & sr: %.3f s" % (timing.time() - start))

   #####################################################################
   # compute strainrate on velocity grid
   #####################################################################
   start = timing.time()

   exxn=np.zeros(NV,dtype=np.float64)
   eyyn=np.zeros(NV,dtype=np.float64)
   exyn=np.zeros(NV,dtype=np.float64)
   srn=np.zeros(NV,dtype=np.float64)
   c=np.zeros(NV,dtype=np.float64)

   rVnodes=[-1,+1,1,-1, 0,1,0,-1,0]
   sVnodes=[-1,-1,1,+1,-1,0,1, 0,0]

   for iel in range(0,nel):
       for i in range(0,mV):
           NNNV[0:mV]=NNV(rVnodes[i],sVnodes[i])
           dNNNVdr[0:mV]=dNNVdr(rVnodes[i],sVnodes[i])
           dNNNVds[0:mV]=dNNVds(rVnodes[i],sVnodes[i])
           jcb=np.zeros((ndim,ndim),dtype=np.float64)
           for k in range(0,mV):
               jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
               jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
               jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
               jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
           jcob=np.linalg.det(jcb)
           jcbi=np.linalg.inv(jcb)
           for k in range(0,mV):
               dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
               dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
           e_xx=0.
           e_yy=0.
           e_xy=0.
           for k in range(0,mV):
               e_xx += dNNNVdx[k]*u[iconV[k,iel]]
               e_yy += dNNNVdy[k]*v[iconV[k,iel]]
               e_xy += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                       0.5*dNNNVdx[k]*v[iconV[k,iel]]
           exxn[iconV[i,iel]]+=e_xx
           eyyn[iconV[i,iel]]+=e_yy
           exyn[iconV[i,iel]]+=e_xy
           c[iconV[i,iel]]+=1.
       # end for i
   # end for iel

   exxn/=c
   eyyn/=c
   exyn/=c

   srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

   print("     -> exxn (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
   print("     -> eyyn (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
   print("     -> exyn (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
   print("     -> srn  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

   print("compute nod strain rate: %.3f s" % (timing.time() - start))

   ######################################################################
   # diffuse nodal strain rate
   ######################################################################
   start = timing.time()

   if False: 

      NfemT=NV
      dt=1
      alphaT=0.5
      diffcoeff=250
      ndofT=1

      A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
      rhs_xx = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      rhs_yy = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      rhs_xy = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
      N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions

      counterq=0
      for iel in range (0,nel):

          exx_prev=np.zeros(mV,dtype=np.float64)
          eyy_prev=np.zeros(mV,dtype=np.float64)
          exy_prev=np.zeros(mV,dtype=np.float64)
          b_el_xx=np.zeros(mV*ndofT,dtype=np.float64)
          b_el_yy=np.zeros(mV*ndofT,dtype=np.float64)
          b_el_xy=np.zeros(mV*ndofT,dtype=np.float64)
          a_el=np.zeros((mV*ndofT,mV*ndofT),dtype=np.float64)
          Kd=np.zeros((mV,mV),dtype=np.float64)   # elemental diffusion matrix 
          MM=np.zeros((mV,mV),dtype=np.float64)   # elemental mass matrix 

          for k in range(0,mV):
               exx_prev[k]=exxn[iconV[k,iel]]
               eyy_prev[k]=eyyn[iconV[k,iel]]
               exy_prev[k]=exyn[iconV[k,iel]]
          #end for

          for iq in [0,1,2]:
              for jq in [0,1,2]:

                  # position & weight of quad. point
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]

                  NNNV[0:mV]=NNV(rq,sq)
                  dNNNVdr[0:mV]=dNNVdr(rq,sq)
                  dNNNVds[0:mV]=dNNVds(rq,sq)
                  N_mat[0:mV,0]=NNV(rq,sq)

                  #only valid for rectangular elements!
                  jcbi=np.zeros((ndim,ndim),dtype=np.float64)
                  jcob=hx*hy/4
                  jcbi[0,0] = 2/hx 
                  jcbi[1,1] = 2/hy
 
                  # compute dNdx & dNdy
                  for k in range(0,mV):
                      dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                      dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                      B_mat[0,k]=dNNNVdx[k]
                      B_mat[1,k]=dNNNVdy[k]
                  #end for

                  # compute mass matrix
                  MM=N_mat.dot(N_mat.T)*weightq*jcob

                  # compute diffusion matrix
                  Kd=B_mat.T.dot(B_mat)*diffcoeff*weightq*jcob

                  a_el+=MM+alphaT*Kd*dt
                  b_el_xx+=(MM-(1-alphaT)*Kd*dt).dot(exx_prev)
                  b_el_yy+=(MM-(1-alphaT)*Kd*dt).dot(eyy_prev)
                  b_el_xy+=(MM-(1-alphaT)*Kd*dt).dot(exy_prev)

                  counterq+=1
              #end for jq
          #end for iq

          # assemble matrix A_mat and right hand side rhs
          for k1 in range(0,mV):
              m1=iconV[k1,iel]
              for k2 in range(0,mV):
                  m2=iconV[k2,iel]
                  A_mat[m1,m2]+=a_el[k1,k2]
              #end for
              rhs_xx[m1]+=b_el_xx[k1]
              rhs_yy[m1]+=b_el_yy[k1]
              rhs_xy[m1]+=b_el_xy[k1]
          #end for

      #end for iel
    
      print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
      print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

      Txx = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xx)
      Tyy = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_yy)
      Txy = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xy)

      if use_srn_diff:
         exxn[:]=Txx[:]
         eyyn[:]=Tyy[:]
         exyn[:]=Txy[:]

      srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

      print("     -> exxn (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
      print("     -> eyyn (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
      print("     -> exyn (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
      print("     -> srn  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

      print("strain rate diffusion time: %.3f s" % (timing.time() - start))

   #end if use_srn_diff

   ######################################################################
   # compute nodal viscosity
   ######################################################################
   start = timing.time()

   for i in range(0,NV):
       etan[i],dum,dum,dum=viscosity(exxn[i],eyyn[i],exyn[i],q[i],cohesion,phi,\
                                     iter,xV[i],yV[i],eta_m,eta_v)

   print("     -> etan (m,M) %.6e %.6e " %(np.min(etan),np.max(etan)))

   #np.savetxt('etan_{:04d}.ascii'.format(iter),np.array([xV,yV,etan]).T,header='# x,y,eta')

   print("compute nodal viscosity: %.3f s" % (timing.time() - start))

   ######################################################################
   # compute vrms
   ######################################################################
   start = timing.time()

   vrms=0.
   avrg_sr=0.
   counterq=0
   for iel in range(0,nel):
       for iq in [0,1,2]:
           for jq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               NNNV[0:mV]=NNV(rq,sq)
               jcob=hx*hy/4 #only for rect elements!
               uq=0.
               vq=0.
               for k in range(0,mV):
                   uq+=NNNV[k]*u[iconV[k,iel]]
                   vq+=NNNV[k]*v[iconV[k,iel]]
               #end for
               vrms+=(uq**2+vq**2)*weightq*jcob
               avrg_sr+=srq_T[counterq]*weightq*jcob
               counterq+=1
           #end for
       #end for
   #end for

   vrms=np.sqrt(vrms/(Lx*Ly))
   avrg_sr=avrg_sr/(Lx*Ly)

   vrmsfile.write("%3d %10e\n" %(iter,vrms))
   vrmsfile.flush()
   if iter>0:
      avrgsrfile.write("%3d %10e\n" %(iter,avrg_sr))
      avrgsrfile.flush()

   print("     -> vrms= %.7e " %vrms)
   print("     -> <sr>= %.7e " %avrg_sr)

   print("compute vrms: %.3f s" % (timing.time() - start))

   ######################################################################
   # generate vtu output at every nonlinear iteration
   ######################################################################
   start = timing.time()

   if iter%every==0 and produce_nl_vtu:

      filename = 'solution_q_nl_{:04d}'.format(iter)+name+'.vtu'
      vtufile=open(filename,"w")
      vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
      vtufile.write("<UnstructuredGrid> \n")
      vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
      #####
      vtufile.write("<Points> \n")
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e %10e %10e \n" %(xq[iq],yq[iq],0.))
      vtufile.write("</DataArray>\n")
      vtufile.write("</Points> \n")
      #####
      vtufile.write("<PointData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % etaq[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain_rate (T)' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % srq_T[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain_rate (vp)' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % srq_vp[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='mechanism' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % mechq[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
      for iq in range(0,nq):
          vtufile.write("%10e \n" % pq[iq])
      vtufile.write("</DataArray>\n")
      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iq in range (0,nq):
          vtufile.write("%d\n" % iq ) 
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iq in range (0,nq):
          vtufile.write("%d \n" % (iq+1) )
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
      for iq in range (0,nq):
          vtufile.write("%d \n" % 1) 
      vtufile.write("</DataArray>\n")
      vtufile.write("</Cells>\n")
      #####
      vtufile.write("</Piece>\n")
      vtufile.write("</UnstructuredGrid>\n")
      vtufile.write("</VTKFile>\n")
      vtufile.close()

      filename = 'solution_g_nl_{:04d}.vtu'.format(iter)
      vtufile=open(filename,"w")
      vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
      vtufile.write("<UnstructuredGrid> \n")
      vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
      #####
      vtufile.write("<Points> \n")
      vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
      vtufile.write("</DataArray>\n")
      vtufile.write("</Points> \n")
      #####
      vtufile.write("<PointData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' Name='Angle' Format='ascii'> \n")
      for i in range(0,NV):
          if np.abs(xV[i]-Lx/2)>hx/10:
             theta=np.arctan(yV[i]/np.abs(xV[i]-Lx/2))/np.pi*180
          else:
             theta=90.
          vtufile.write("%10e \n" % theta)
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %Res_u[i])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %Res_v[i])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %Res_q[i])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
      for i in range(0,NV):
          vtufile.write("%10e \n" %q[i])
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%10e\n" % (etan[i]))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
      for i in range (0,NV):
          vtufile.write("%e\n" % (srn[i]))
      vtufile.write("</DataArray>\n")

      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                      iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*8))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %23)
      vtufile.write("</DataArray>\n")
      vtufile.write("</Cells>\n")
      #####
      vtufile.write("</Piece>\n")
      vtufile.write("</UnstructuredGrid>\n")
      vtufile.write("</VTKFile>\n")
      vtufile.close()

   print("write nl iter vtu file: %.3f s" % (timing.time() - start))

#------------------------------------------------------------------------------
# end of non-linear iterations
#------------------------------------------------------------------------------

#np.savetxt('sr_middle.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################
# extracting shear bands 
#####################################################################
start = timing.time()

if benchmark ==1 or benchmark==2 or benchmark==3:

   shear_band_L_file_1=open("shear_band_L_elt"+name+".ascii","w")
   shear_band_R_file_1=open("shear_band_R_elt"+name+".ascii","w")
   shear_band_L_file_2=open("shear_band_L_nod"+name+".ascii","w")
   shear_band_R_file_2=open("shear_band_R_nod"+name+".ascii","w")
   shear_band_L_file_3=open("shear_band_L_qpt"+name+".ascii","w")
   shear_band_R_file_3=open("shear_band_R_qpt"+name+".ascii","w")

   counter = 0
   for j in range(0,nely):
       srmaxL=0.
       srmaxR=0.
       for i in range(0,nelx):
           if i<=nelx/2 and sr[counter]>srmaxL:
              srmaxL=sr[counter]
              ilocL=counter
           # end if
           if i>=nelx/2 and sr[counter]>srmaxR:
              srmaxR=sr[counter]
              ilocR=counter
           # end if
           counter += 1
       # end for i
       shear_band_L_file_1.write("%6e %6e %6e \n"  % (xc[ilocL],yc[ilocL],sr[ilocL]) )
       shear_band_R_file_1.write("%6e %6e %6e \n"  % (xc[ilocR],yc[ilocR],sr[ilocR]) )
   # end for j

   counter = 0
   for j in range(0,nny):
       srmaxL=0.
       srmaxR=0.
       for i in range(0,nnx):
           if i<=nnx/2 and srn[counter]>srmaxL:
              srmaxL=srn[counter]
              ilocL=counter
           # end if
           if i>=nnx/2 and srn[counter]>srmaxR:
              srmaxR=srn[counter]
              ilocR=counter
           # end if
           counter += 1
       # end for i
       shear_band_L_file_2.write("%6e %6e %6e \n"  % (xV[ilocL],yV[ilocL],srn[ilocL]) )
       shear_band_R_file_2.write("%6e %6e %6e \n"  % (xV[ilocR],yV[ilocR],srn[ilocR]) )
   # end for j

   counter = 0
   for j in range(0,nely):
       srmaxL1=0.
       srmaxL2=0.
       srmaxL3=0.
       for i in range(0,nelx):
           for k in range(0,3):
               iq1=9*counter+k
               if i<=nelx/2 and srq_T[iq1]>srmaxL1:
                  srmaxL1=srq_T[iq1]
                  ilocL1=iq1
               # end if
               iq2=9*counter+k+3
               if i<=nelx/2 and srq_T[iq2]>srmaxL2:
                  srmaxL2=srq_T[iq2]
                  ilocL2=iq2
               # end if
               iq3=9*counter+k+6
               if i<=nelx/2 and srq_T[iq3]>srmaxL3:
                  srmaxL3=srq_T[iq3]
                  ilocL3=iq3
               # end if
           counter += 1
       # end for i
       shear_band_L_file_3.write("%6e %6e %6e \n"  % (xq[ilocL1],yq[ilocL1],srq_T[ilocL1]) )
       shear_band_L_file_3.write("%6e %6e %6e \n"  % (xq[ilocL2],yq[ilocL2],srq_T[ilocL2]) )
       shear_band_L_file_3.write("%6e %6e %6e \n"  % (xq[ilocL3],yq[ilocL3],srq_T[ilocL3]) )
   #end for j

   counter = 0
   for j in range(0,nely):
       srmaxR1=0.
       srmaxR2=0.
       srmaxR3=0.
       for i in range(0,nelx):
           for k in range(0,3):
               iq1=9*counter+k
               if i>=nelx/2 and srq_T[iq1]>srmaxR1:
                  srmaxR1=srq_T[iq1]
                  ilocR1=iq1
               # end if
               iq2=9*counter+k+3
               if i>=nelx/2 and srq_T[iq2]>srmaxR2:
                  srmaxR2=srq_T[iq2]
                  ilocR2=iq2
               # end if
               iq3=9*counter+k+6
               if i>=nelx/2 and srq_T[iq3]>srmaxR3:
                  srmaxR3=srq_T[iq3]
                  ilocR3=iq3
               # end if
           counter += 1
       # end for i
       shear_band_R_file_3.write("%6e %6e %6e \n"  % (xq[ilocR1],yq[ilocR1],srq_T[ilocR1]) )
       shear_band_R_file_3.write("%6e %6e %6e \n"  % (xq[ilocR2],yq[ilocR2],srq_T[ilocR2]) )
       shear_band_R_file_3.write("%6e %6e %6e \n"  % (xq[ilocR3],yq[ilocR3],srq_T[ilocR3]) )
   # end for j

   sr_file=open("line"+name+".ascii","w")
   counter=0
   for j in range(0,nny):
       for i in range(0,nnx):
           if abs(yV[counter]-0.5*Ly)<1:
              sr_file.write("%6e %6e %6e \n"  % (xV[counter],srn[counter],etan[counter]))
           counter += 1
       # end for i
   # end for i
   sr_file.close()


#else:

#sr_file=open("sr_line.ascii","w")
#counter=0
#for j in range(0,nny):
#    for i in range(0,nnx):
#        if abs(yV[counter]-11*Ly/16)<1:
#           sr_file.write("%6e %6e %6e \n"  % (xV[counter],srn[counter],yV[counter]))
#        counter += 1
#    # end for i
# end for i
#sr_file.close()

print("extracting shear bands: %.3f s" % (timing.time() - start))

######################################################################
# compute averaged elemental strainrate 
# I use a 5 point quadrature rule (per dimension) and compute the 
# average strain rate tensor components per element. 
######################################################################
start = timing.time()

exx_avrg = np.zeros(nel,dtype=np.float64)  
eyy_avrg = np.zeros(nel,dtype=np.float64)  
exy_avrg = np.zeros(nel,dtype=np.float64)  
sr_avrg  = np.zeros(nel,dtype=np.float64)  

qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.  
qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.  
qc5c=0.    
qw5a=(322.-13.*np.sqrt(70.))/900.
qw5b=(322.+13.*np.sqrt(70.))/900.
qw5c=128./225.
qcoords5=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
qweights5=[qw5a,qw5b,qw5c,qw5b,qw5a]

for iel in range(0,nel):
    for jq in [0,1,2,3,4]:
        for iq in [0,1,2,3,4]:
            # position & weight of quad. point
            rq=qcoords5[iq]
            sq=qcoords5[jq]
            weightq=qweights5[iq]*qweights5[jq]
            NNNV[0:9]=NNV(rq,sq)
            dNNNVdr[0:9]=dNNVdr(rq,sq)
            dNNNVds[0:9]=dNNVds(rq,sq)
            # calculate jacobian matrix
            jcb=np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
            # end for
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)
            # compute dNdx & dNdy & strainrate
            exxq=0.0
            eyyq=0.0
            exyq=0.0
            for k in range(0,mV):
                dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                exxq+=dNNNVdx[k]*u[iconV[k,iel]]
                eyyq+=dNNNVdy[k]*v[iconV[k,iel]]
                exyq+=0.5*dNNNVdy[k]*u[iconV[k,iel]]+ 0.5*dNNNVdx[k]*v[iconV[k,iel]]
            # end for
            exx_avrg[iel] += exxq*jcob*weightq
            eyy_avrg[iel] += eyyq*jcob*weightq
            exy_avrg[iel] += exyq*jcob*weightq
        # end for
    # end for
    exx_avrg[iel] /= (hx*hy) 
    eyy_avrg[iel] /= (hx*hy) 
    exy_avrg[iel] /= (hx*hy) 
    sr_avrg[iel]=np.sqrt(0.5*(exx_avrg[iel]**2+eyy_avrg[iel]**2)+exy_avrg[iel]**2)
#end for

print("     -> exx_avrg (m,M) %.6e %.6e " %(np.min(exx_avrg),np.max(exx_avrg)))
print("     -> eyy_avrg (m,M) %.6e %.6e " %(np.min(eyy_avrg),np.max(eyy_avrg)))
print("     -> exy_avrg (m,M) %.6e %.6e " %(np.min(exy_avrg),np.max(exy_avrg)))
print("     -> sr_avrg  (m,M) %.6e %.6e " %(np.min(sr_avrg),np.max(sr_avrg)))

print("compute avrg elemental strain rate: %.3f s" % (timing.time() - start))

#np.savetxt('sr_avrg.ascii',np.array([xc,yc,exx_avrg,eyy_avrg,exy_avrg]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

filename = 'solution'+name+'.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exx_avrg[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % exy_avrg[iel])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='strain rate (avrg)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (sr_avrg[iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    eta,dum,dum,dum= viscosity(exx[iel],eyy[iel],exy[iel],pc[iel],cohesion,phi,iter,xc[iel],yc[iel],eta_m,eta_v)
    vtufile.write("%10e\n" %eta) 
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='dilation rate (R)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (  2.*np.sin(psi)*sr[iel] ))
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Angle' Format='ascii'> \n")
for i in range(0,NV):
    if np.abs(xV[i]-Lx/2)>hx/10:
       theta=np.arctan(yV[i]/np.abs(xV[i]-Lx/2))/np.pi*180
    else:
       theta=90.
    vtufile.write("%10e \n" % theta)
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_u[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_v[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %Res_q[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exxn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %eyyn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %exyn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %srn[i])
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%.5e \n" %etan[i])
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*8))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %23)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
