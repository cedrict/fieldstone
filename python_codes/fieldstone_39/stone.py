import numpy as np
import sys as sys
import time as clock 
from numpy import linalg as LA
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import numba 


###############################################################################

def gx(x,y):
    return 0

def gy(x,y):
    if benchmark==4 or benchmark==5:
       return 0
    else:
       return -10

###############################################################################
# benchmark=1: simple brick
# benchmark=2: Spiegelman et al (2016) 
# benchmark=3: Kaus (2010) brick
# benchmark=4: Gerya book (2019) 
# benchmark=5: Duretz et al (2018)

def ubc(x,y):
    if benchmark==1: vaal=1.e-15*(Lx/2.0)
    if benchmark==2: vaal=0.0025/year
    if benchmark==3: vaal=-1.e-15*(Lx/2.0)
    if benchmark==4: vaal=5e-9
    if benchmark==5: vaal=5.e-15*(Lx/2.0)

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

###############################################################################

@numba.njit
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
          #see section 4.23
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

###############################################################################

def basis_functions_V(r,s):
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

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################
# benchmark=1: brick with velocity discontinuity at bottom
# benchmark=2: Spiegelman et al, 2016. 
# benchmark=3: brick with seed Kaus (2010)
# benchmark=4: shortening block with sticky air - square inclusion (Gerya book)
# benchmark=5: shortening block - round inclusion,  Duretz et al (2018)
###############################################################################

cm=0.01
year=3600*24*365.
eps=1.e-10

print("*******************************")
print("********** stone 039 **********")
print("*******************************")

ndim=2
m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

debug=True

if int(len(sys.argv) == 9):
   print("reading arguments")
   nelx = int(sys.argv[1])
   benchmark = int(sys.argv[2])
   phi = float(sys.argv[3])
   psi =  float(sys.argv[4])
   niter = int(sys.argv[5])
   eta_m = float(sys.argv[6]) ; eta_m=10**eta_m
   eta_v = float(sys.argv[7]) ; eta_v=10**eta_v
   method = str(sys.argv[8])
   produce_nl_vtu=False
   name='_nelx'+sys.argv[1]+'_phi'+sys.argv[3]+'_psi'+sys.argv[4]+'_etam'+sys.argv[6]+'_'+method
   print(name)
   every=1000000
else:
   #U (uncompensated): use dev strain rate in momentum eq, no rhs
   #C (compensated)  : use dev strain rate in momentum eq, additional rhs
   #M (modified)     : use full strain rate in momentum eq, no rhs
   method='U'
   produce_nl_vtu=True
   benchmark=1
   every=1
   if benchmark==1:
      nelx = 128
      phi=30
      psi=30
      niter=50
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

use_srn_diff=False # diffusion plasticity

###############################################################################

nely=int(nelx*Ly/Lx)     # number of elements y direction
nnx=2*nelx+1             # number of nodes, x direction
nny=2*nely+1             # number of nodes, y direction
nn_V=nnx*nny             # total number of nodes
nel=nelx*nely            # total number of elements
Nfem_V=nn_V*ndof_V       # number of velocity dofs
Nfem_P=(nelx+1)*(nely+1) # number of pressure dofs
Nfem=Nfem_V+Nfem_P       # total number of dofs
hx=Lx/nelx               # mesh size in x direction
hy=Ly/nely               # mesh size in y direction
ndof_V_el=m_V*ndof_V     # nb of V dofs per elt

###############################################################################
# quadrature parameters
###############################################################################

qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
nq=9*nel

###############################################################################
# scaling of G blocks
###############################################################################

eta_ref=1.e23     
scaling_coeff=eta_ref/Ly

###############################################################################

ustats_file=open("stats_u"+name+".ascii","w")
vstats_file=open("stats_v"+name+".ascii","w")
pstats_file=open("stats_p"+name+".ascii","w")
etaqstats_file=open("stats_etaq"+name+".ascii","w")
convfile=open('conv'+name+'.ascii',"w")
vrmsfile=open('vrms'+name+'.ascii',"w")
avrgsrfile=open('avrgsr'+name+'.ascii',"w")

###############################################################################

print("benchmark=",benchmark)
print("method=",method)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("hx",hx)
print("hy",hy)
print("niter",niter)
print("eta_m",eta_m)
print("eta_v",eta_v)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1

if debug: np.savetxt('grid.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# compute angle field that is only used in vtu output
###############################################################################

Angle=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V):
    if np.abs(x_V[i]-Lx/2)>hx/10:
       Angle[i]=np.arctan(y_V[i]/np.abs(x_V[i]-Lx/2))/np.pi*180
    else:
       Angle[i]=90.

###############################################################################
# build connectivity arrays for velocity and pressure
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0
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
        counter += 1

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

u=np.zeros(nn_V,dtype=np.float64)        # x-component velocity
v=np.zeros(nn_V,dtype=np.float64)        # y-component velocity
bc_fix=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

match benchmark:

 case 1: # simple brick
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = vbc(x_V[i],y_V[i]) ; v[i]=vbc(x_V[i],y_V[i])

 case 2 | 3: # spmw16 & kaus10
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = vbc(x_V[i],y_V[i]) ; v[i]=vbc(x_V[i],y_V[i])

 case 4 | 5: # shortening blocks
   for i in range(0,nn_V):
       if x_V[i]/Lx<eps:
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if x_V[i]/Lx>(1-eps):
          bc_fix[i*ndof_V  ] = True ; bc_val[i*ndof_V  ] = ubc(x_V[i],y_V[i]) ; u[i]=ubc(x_V[i],y_V[i])
       if y_V[i]/Ly<eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = vbc(x_V[i],y_V[i]) ; v[i]=vbc(x_V[i],y_V[i])
       if y_V[i]/Ly>1-eps:
          bc_fix[i*ndof_V+1] = True ; bc_val[i*ndof_V+1] = vbc(x_V[i],y_V[i]) ; v[i]=vbc(x_V[i],y_V[i])

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute array for assembly - see stone 181
###############################################################################
start=clock.time()

local_to_global_V=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            local_to_global_V[ikk,iel]=ndof_V*icon_V[k1,iel]+i1
                 
print("compute local_to_global_V: %.3f s" % (clock.time() - start))

###############################################################################
# fill I,J arrays - see stone 181
###############################################################################
start = clock.time()

bignb=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )
II_V=np.zeros(bignb,dtype=np.int32)    
JJ_V=np.zeros(bignb,dtype=np.int32)    
VV_V=np.zeros(bignb,dtype=np.float64)    

counter=0
for iel in range(0,nel):
    for ikk in range(ndof_V_el):
        m1=local_to_global_V[ikk,iel]
        for jkk in range(ndof_V_el):
            m2=local_to_global_V[jkk,iel]
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
        for jkk in range(0,m_P):
            m2 =icon_P[jkk,iel]+Nfem_V
            II_V[counter]=m1
            JJ_V[counter]=m2
            counter+=1
            II_V[counter]=m2
            JJ_V[counter]=m1
            counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# non-linear iterations
###############################################################################
###############################################################################

match method:
 case 'C' | 'U':
   C=np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64) 
 case 'M':
   C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

p=np.zeros(Nfem_P,dtype=np.float64)         # (old) pressure field (needed) 
sol=np.zeros(Nfem,dtype=np.float64)         # (old) solution vector (needed) 

#only valid for all elements being identical rectangles !
jcb=np.zeros((ndim,ndim),dtype=np.float64)  # jacobian matrix
jcbi=np.zeros((ndim,ndim),dtype=np.float64) # inverse of jcb
jcob=hx*hy/4
jcbi[0,0] = 2/hx 
jcbi[1,1] = 2/hy

for iter in range(0,niter):

   print("-------------------------------")
   print("iter=", iter)
   print("-------------------------------")

   ############################################################################
   # build FE matrix A and rhs 
   # [ K G ][u]=[f]
   # [GT 0 ][p] [h]
   ############################################################################

   A_fem = lil_matrix((Nfem,Nfem),dtype=np.float64)  # FEM stokes matrix 
   b_fem = np.zeros(Nfem,dtype=np.float64)           # right hand side of Ax=b
   N_mat = np.zeros((3,m_P),dtype=np.float64)        # N matrix  
   B     = np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 

   xq    = np.zeros(nq,dtype=np.float64) # x coords of q points 
   yq    = np.zeros(nq,dtype=np.float64) # y coords of q points 
   etaq  = np.zeros(nq,dtype=np.float64) # viscosity at q points 
   divvq = np.zeros(nq,dtype=np.float64) # div velocity at q points 
   pq    = np.zeros(nq,dtype=np.float64) # pressure at q points 
   Dq    = np.zeros(nq,dtype=np.float64) # dilation rate at q points 
   srq_T = np.zeros(nq,dtype=np.float64) # total strain rate at q points 
   srq_vp= np.zeros(nq,dtype=np.float64) # viscoplastic rate at q points 
   mechq = np.zeros(nq,dtype=np.float64) # deformation mechanism at q points 

   time_bc=0
   time_ass=0
   counter=0
   counterq=0
   for iel in range(0,nel):

       K_el=np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
       G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
       f_el=np.zeros((ndof_V_el),dtype=np.float64)
       h_el=np.zeros((m_P),dtype=np.float64)

       for jq in [0,1,2]:
           for iq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_V=basis_functions_V(rq,sq)
               N_P=basis_functions_P(rq,sq)
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
               xq[counterq]=np.dot(N_V,x_V[icon_V[:,iel]])
               yq[counterq]=np.dot(N_V,y_V[icon_V[:,iel]])
               exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
               eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
               exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                    np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
               pq[counterq]=np.dot(N_P,p[icon_P[:,iel]])
               #if use_srn_diff:
               #   exxq=np.dot(N_V,exxn[icon_V[:,iel]])
               #   eyyq=np.dot(N_V,eyyn[icon_V[:,iel]])
               #   exyq=np.dot(N_V,exyn[icon_V[:,iel]])
               divvq[counterq]=exxq+eyyq

               # effective strain rate at qpoint                
               srq_T[counterq]=np.sqrt(0.5*(exxq**2+eyyq**2)+exyq**2)     # dev?!!?

               # construct 3x8 B matrix
               for i in range(0,m_V):
                   B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                     [0.       ,dNdy_V[i]],
                                     [dNdy_V[i],dNdx_V[i]]]

               # compute effective plastic viscosity
               etaq[counterq],two_sin_psi,srq_vp[counterq],mechq[counterq]=\
                   viscosity(exxq,eyyq,exyq,pq[counterq],cohesion,phi,\
                   iter,xq[counterq],yq[counterq],eta_m,eta_v)

               Dq[counterq]=two_sin_psi*srq_vp[counterq] *0.5 ##why *0.5 ?!?!

               K_el+=B.T.dot(C.dot(B))*etaq[counterq]*JxWq

               for i in range(0,m_V):
                   f_el[ndof_V*i+0]+=N_V[i]*rho*gx(xq,yq)*JxWq
                   f_el[ndof_V*i+1]+=N_V[i]*rho*gy(xq,yq)*JxWq

               #add to rhs dilation term only if method is compensated
               if method=='C':
                  for i in range(0,m_V):
                      f_el[ndof_V*i+0]-=2./3.*dNdx_V[i]*etaq[counterq]*Dq[counterq]*JxWq
                      f_el[ndof_V*i+1]-=2./3.*dNdy_V[i]*etaq[counterq]*Dq[counterq]*JxWq

               N_mat[0,:]=N_P[:]
               N_mat[1,:]=N_P[:]
               G_el-=B.T.dot(N_mat)*JxWq
                
               h_el[:]-=N_P[:]*Dq[counterq]*JxWq

               counterq+=1
           # end for iq 
       # end for jq 

       G_el*=scaling_coeff
       h_el*=scaling_coeff

       # impose b.c. 
       startbc=clock.time()
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               if bc_fix[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,m_V*ndof_V):
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
       time_bc+=clock.time()-startbc

       # assembly
       startass=clock.time()
       for ikk in range(ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           for jkk in range(ndof_V_el):
               VV_V[counter]=K_el[ikk,jkk]
               counter+=1
           for jkk in range(0,m_P):
               VV_V[counter]=G_el[ikk,jkk]
               counter+=1
               VV_V[counter]=G_el[ikk,jkk]
               counter+=1
           b_fem[m1]+=f_el[ikk]
       for k2 in range(0,m_P):
           m2=icon_P[k2,iel]
           b_fem[Nfem_V+m2]+=h_el[k2]
       time_ass+=clock.time()-startass

   # end for iel 
   print("     -> time bc: %.2f s" %(time_bc))
   print("     -> time assembly: %.2f s" %(time_ass))

   print("     -> etaq (m,M) %.3e %.3e " %(np.min(etaq),np.max(etaq)))

   etaqstats_file.write("%d %8e %8e \n" %(iter,np.min(etaq),np.max(etaq))) ; etaqstats_file.flush()

   print("build FE matrix: %.3f s" % (clock.time()-start))

   ############################################################################
   # assemble f, h into rhs and solve
   ############################################################################
   start=clock.time()

   sparse_matrix=sps.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()
   Res=sparse_matrix.dot(sol)-b_fem
   sol=sps.linalg.spsolve(sparse_matrix,b_fem)

   u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
   p=sol[Nfem_V:Nfem]*scaling_coeff

   print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
   print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   ustats_file.write("%d %8e %8e \n" %(iter,np.min(u),np.max(u))) ; ustats_file.flush()
   vstats_file.write("%d %8e %8e \n" %(iter,np.min(v),np.max(v))) ; vstats_file.flush()
   pstats_file.write("%d %8e %8e \n" %(iter,np.min(p),np.max(p))) ; pstats_file.flush()

   print("solve system: %.3f s - Nfem %d" % (clock.time()-start,Nfem))

   ############################################################################
   # normalise pressure
   # I take a shortcut and assume elements are all same size rectangles hx,hy
   ############################################################################
   start=clock.time()

   if benchmark==4 or benchmark==5:

      int_p=0
      for iel in range(0,nel):
          for jq in [0,1,2]:
              for iq in [0,1,2]:
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]
                  N_P=basis_functions_P(rq,sq)
                  JxWq=jcob*weightq
                  p_q=np.dot(N_P,p[icon_P[:,iel]])
                  int_p+=p_q*JxWq
              #end for
          #end for
      #end for

      avrg_p=int_p/Lx/Ly

      print("     -> int_p %e " %(int_p))
      print("     -> avrg_p %e " %(avrg_p))

      p[:]-=avrg_p

      print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   print("normalise pressure: %.3f s" % (clock.time()-start))

   ############################################################################

   if debug:
      np.savetxt('etaq_{:04d}.ascii'.format(iter),np.array([xq,yq,etaq]).T,header='# x,y,eta')
      np.savetxt('velocity_{:04d}.ascii'.format(iter),np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
      np.savetxt('pq_{:04d}.ascii'.format(iter),np.array([xq,yq,pq]).T,header='# x,y,p')

   ############################################################################
   # compute non-linear residual
   ############################################################################
   start=clock.time()

   if iter==0: Res0_two=LA.norm(Res,2)

   Res_two=LA.norm(Res,2)

   conv_two=Res_two/Res0_two

   print("     -> Nonlinear res. (2-norm) %7e" % (conv_two))

   Res_u,Res_v=np.reshape(Res[0:Nfem_V],(nn_V,2)).T
   Ru=LA.norm(Res_u,2)
   Rv=LA.norm(Res_v,2)

   Res_p=Res[Nfem_V:Nfem]
   Rp=LA.norm(Res_p,2)

   convfile.write("%3d %10e %10e %10e %10e\n" %(iter,conv_two,Ru,Rv,Rp)) ; convfile.flush()

   if conv_two<tol_nl:
      break

   print("computing resduals norm: %.3f s" % (clock.time()-start))

   ############################################################################
   # interpolate pressure and pressure residual onto V nodes (for plotting)
   ############################################################################
   # velocity    pressure
   # 3---6---2   3-------2
   # |       |   |       |
   # 7   8   5   |       |
   # |       |   |       |
   # 0---4---1   0-------1
   ############################################################################
   start=clock.time()

   q=np.zeros(nn_V,dtype=np.float64)
   Res_q=np.zeros(nn_V,dtype=np.float64)

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

   for iel in range(0,nel):
       Res_q[icon_V[0,iel]]=Res_p[icon_P[0,iel]]
       Res_q[icon_V[1,iel]]=Res_p[icon_P[1,iel]]
       Res_q[icon_V[2,iel]]=Res_p[icon_P[2,iel]]
       Res_q[icon_V[3,iel]]=Res_p[icon_P[3,iel]]
       Res_q[icon_V[4,iel]]=(Res_p[icon_P[0,iel]]+Res_p[icon_P[1,iel]])*0.5
       Res_q[icon_V[5,iel]]=(Res_p[icon_P[1,iel]]+Res_p[icon_P[2,iel]])*0.5
       Res_q[icon_V[6,iel]]=(Res_p[icon_P[2,iel]]+Res_p[icon_P[3,iel]])*0.5
       Res_q[icon_V[7,iel]]=(Res_p[icon_P[3,iel]]+Res_p[icon_P[0,iel]])*0.5
       Res_q[icon_V[8,iel]]=(Res_p[icon_P[0,iel]]+Res_p[icon_P[1,iel]]+\
                             Res_p[icon_P[2,iel]]+Res_p[icon_P[3,iel]])*0.25

   if debug: np.savetxt('q_{:04d}.ascii'.format(iter),np.array([x_V,y_V,q]).T,header='# x,y,q')

   print("project p(Q1) onto vel(Q2) nodes: %.3f s" % (clock.time()-start))

   ############################################################################
   # compute strainrate and pressure at center of element 
   ############################################################################
   start=clock.time()

   x_e=np.zeros(nel,dtype=np.float64)  
   y_e=np.zeros(nel,dtype=np.float64)  
   p_e=np.zeros(nel,dtype=np.float64)  
   exx_e=np.zeros(nel,dtype=np.float64)  
   eyy_e=np.zeros(nel,dtype=np.float64)  
   exy_e=np.zeros(nel,dtype=np.float64)  
   sr_e=np.zeros(nel,dtype=np.float64)  

   rq=0.
   sq=0.
   N_V=basis_functions_V(rq,sq)
   N_P=basis_functions_P(rq,sq)
   dNdr_V=basis_functions_V_dr(rq,sq)
   dNds_V=basis_functions_V_ds(rq,sq)
   #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
   #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
   #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
   #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
   #jcbi=np.linalg.inv(jcb)
   dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
   dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

   for iel in range(0,nel):
       p_e[iel]=np.dot(N_P,p[icon_P[:,iel]])
       x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
       y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
       exx_e[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
       eyy_e[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
       exy_e[iel]=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                  np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
       sr_e[iel]=np.sqrt(0.5*(exx_e[iel]**2+eyy_e[iel]**2)+exy_e[iel]**2)

   divv_e=exx_e+eyy_e

   print("     -> exx_e (m,M) %.3e %.3e " %(np.min(exx_e),np.max(exx_e)))
   print("     -> eyy_e (m,M) %.3e %.3e " %(np.min(eyy_e),np.max(eyy_e)))
   print("     -> exy_e (m,M) %.3e %.3e " %(np.min(exy_e),np.max(exy_e)))
   print("     -> p_e   (m,M) %.3e %.3e " %(np.min(p_e),np.max(p_e)))

   print("compute press & sr: %.3f s" % (clock.time()-start))

   ############################################################################
   # compute elemental viscosity
   ############################################################################
   start=clock.time()

   eta_e=np.zeros(nel,dtype=np.float64)

   for iel in range(0,nel):
       eta_e[iel],dum,dum,dum=viscosity(exx_e[iel],eyy_e[iel],exy_e[iel],p_e[iel],\
                                        cohesion,phi,iter,x_e[iel],y_e[iel],eta_m,eta_v)

   print("     -> eta_e (m,M) %.3e %.3e " %(np.min(eta_e),np.max(eta_e)))

   if debug: 
      np.savetxt('eta_e_{:04d}.ascii'.format(iter),np.array([x_e,y_e,eta_e]).T,header='# x,y,eta')

   print("compute elemental viscosity: %.3f s" % (clock.time()-start))

   ############################################################################
   # compute strainrate on velocity grid
   ############################################################################
   start=clock.time()

   r_V=[-1,+1,1,-1, 0,1,0,-1,0]
   s_V=[-1,-1,1,+1,-1,0,1, 0,0]

   exx_n=np.zeros(nn_V,dtype=np.float64)
   eyy_n=np.zeros(nn_V,dtype=np.float64)
   exy_n=np.zeros(nn_V,dtype=np.float64)
   counter=np.zeros(nn_V,dtype=np.float64)

   for iel in range(0,nel):
       for i in range(0,m_V):
           dNdr_V=basis_functions_V_dr(r_V[i],s_V[i])
           dNds_V=basis_functions_V_ds(r_V[i],s_V[i])
           #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
           #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
           #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
           #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
           #jcbi=np.linalg.inv(jcb)
           dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
           dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
           e_xx=np.dot(dNdx_V,u[icon_V[:,iel]])
           e_yy=np.dot(dNdy_V,v[icon_V[:,iel]])
           e_xy=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
           exx_n[icon_V[i,iel]]+=e_xx
           eyy_n[icon_V[i,iel]]+=e_yy
           exy_n[icon_V[i,iel]]+=e_xy
           counter[icon_V[i,iel]]+=1.

   exx_n/=counter
   eyy_n/=counter
   exy_n/=counter

   divv_n=exx_n+eyy_n

   sr_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

   print("     -> exx_n  (m,M) %.3e %.3e " %(np.min(exx_n),np.max(exx_n)))
   print("     -> eyy_n  (m,M) %.3e %.3e " %(np.min(eyy_n),np.max(eyy_n)))
   print("     -> exy_n  (m,M) %.3e %.3e " %(np.min(exy_n),np.max(exy_n)))
   print("     -> divv_n (m,M) %.3e %.3e " %(np.min(divv_n),np.max(divv_n)))
   print("     -> sr_n   (m,M) %.3e %.3e " %(np.min(sr_n),np.max(sr_n)))

   print("compute nod strain rate: %.3f s" % (clock.time()-start))

   ############################################################################
   # diffuse nodal strain rate
   ############################################################################
   start=clock.time()

   #if use_srn_diff:
      #check additional python file diffuse_strainrate.py & reattach

   ############################################################################
   # compute nodal viscosity
   ############################################################################
   start=clock.time()

   eta_n=np.zeros(nn_V,dtype=np.float64)

   for i in range(0,nn_V):
       eta_n[i],dum,dum,dum=viscosity(exx_n[i],eyy_n[i],exy_n[i],q[i],cohesion,phi,\
                                      iter,x_V[i],y_V[i],eta_m,eta_v)

   print("     -> eta_n (m,M) %.3e %.3e " %(np.min(eta_n),np.max(eta_n)))

   if debug: 
      np.savetxt('eta_n_{:04d}.ascii'.format(iter),np.array([x_V,y_V,eta_n]).T,header='# x,y,eta')

   print("compute nodal viscosity: %.3f s" % (clock.time()-start))

   ############################################################################
   # compute vrms
   ############################################################################
   start=clock.time()

   vrms=0.
   avrg_sr=0.
   counterq=0
   for iel in range(0,nel):
       for iq in [0,1,2]:
           for jq in [0,1,2]:
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_V=basis_functions_V(rq,sq)
               #dNdr_V=basis_functions_V_dr(rq,sq)
               #dNds_V=basis_functions_V_ds(rq,sq)
               #jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
               #jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
               #jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
               #jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
               #JxWq=np.linalg.det(jcb)*weightq
               JxWq=jcob*weightq
               uq=np.dot(N_V,u[icon_V[:,iel]])
               vq=np.dot(N_V,v[icon_V[:,iel]])
               vrms+=(uq**2+vq**2)*JxWq
               avrg_sr+=srq_T[counterq]*JxWq
               counterq+=1
           #end for
       #end for
   #end for

   vrms=np.sqrt(vrms/(Lx*Ly))
   avrg_sr=avrg_sr/(Lx*Ly)

   vrmsfile.write("%3d %e\n" %(iter,vrms)) ; vrmsfile.flush()
   if iter>0:
      avrgsrfile.write("%3d %e\n" %(iter,avrg_sr)) ; avrgsrfile.flush()

   print("     -> vrms= %.7e " %vrms)
   print("     -> <sr>= %.7e " %avrg_sr)

   print("compute vrms: %.3f s" % (clock.time()-start))

   ############################################################################
   # generate vtu output at every nonlinear iteration
   ############################################################################
   start=clock.time()

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
      etaq.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
      divvq.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain_rate (T)' Format='ascii'> \n")
      srq_T.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain_rate (vp)' Format='ascii'> \n")
      srq_vp.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='mechanism' Format='ascii'> \n")
      mechq.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
      pq.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='D' Format='ascii'> \n")
      Dq.tofile(vtufile, sep=" ", format="%.4e")
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

      filename = 'solution_n_nl_{:04d}.vtu'.format(iter)
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
      vtufile.write("<PointData Scalars='scalars'>\n")
      vtufile.write("<DataArray type='Float32' Name='Angle' Format='ascii'> \n")
      Angle.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
      Res_u.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
      Res_v.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
      Res_q.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
      q.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
      eta_n.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
      sr_n.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
      divv_n.tofile(vtufile, sep=" ", format="%.4e")
      vtufile.write("</DataArray>\n")

      vtufile.write("</PointData>\n")
      #####
      vtufile.write("<Cells>\n")
      vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                         icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                         icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %((iel+1)*m_V))
      vtufile.write("</DataArray>\n")
      vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
      for iel in range (0,nel):
          vtufile.write("%d \n" %28)
      vtufile.write("</DataArray>\n")
      vtufile.write("</Cells>\n")
      #####
      vtufile.write("</Piece>\n")
      vtufile.write("</UnstructuredGrid>\n")
      vtufile.write("</VTKFile>\n")
      vtufile.close()

   print("write nl iter vtu file: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################
# end of non-linear iterations
###############################################################################
###############################################################################

print("-------------------------------")
print("end of nonlinear iterations")
print("-------------------------------")

if debug: 
   np.savetxt('sr_e.ascii',np.array([x_e,y_e,exx_e,eyy_e,exy_e,sr_e]).T,header='# x,y,exx,eyy,exy')
   np.savetxt('sr_n.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n,sr_n]).T,header='# x,y,exx,eyy,exy')

###############################################################################
# extracting shear bands 
###############################################################################
start=clock.time()

if benchmark ==1 or benchmark==2 or benchmark==3:

   #-----elemental -----
   shear_band_L_file_1=open("shear_band_L_elt"+name+".ascii","w")
   shear_band_R_file_1=open("shear_band_R_elt"+name+".ascii","w")
   counter = 0
   for j in range(0,nely):
       srmaxL=0.
       srmaxR=0.
       for i in range(0,nelx):
           if i<=nelx/2 and sr_e[counter]>srmaxL:
              srmaxL=sr_e[counter]
              ilocL=counter
           # end if
           if i>=nelx/2 and sr_e[counter]>srmaxR:
              srmaxR=sr_e[counter]
              ilocR=counter
           # end if
           counter += 1
       # end for i
       shear_band_L_file_1.write("%6e %6e %6e \n"  % (x_e[ilocL],y_e[ilocL],sr_e[ilocL]) )
       shear_band_R_file_1.write("%6e %6e %6e \n"  % (x_e[ilocR],y_e[ilocR],sr_e[ilocR]) )
   # end for j

   #----- nodal -----
   shear_band_L_file_2=open("shear_band_L_nod"+name+".ascii","w")
   shear_band_R_file_2=open("shear_band_R_nod"+name+".ascii","w")
   counter = 0
   for j in range(0,nny):
       srmaxL=0.
       srmaxR=0.
       for i in range(0,nnx):
           if i<=nnx/2 and sr_n[counter]>srmaxL:
              srmaxL=sr_n[counter]
              ilocL=counter
           # end if
           if i>=nnx/2 and sr_n[counter]>srmaxR:
              srmaxR=sr_n[counter]
              ilocR=counter
           # end if
           counter += 1
       # end for i
       shear_band_L_file_2.write("%6e %6e %6e \n"  % (x_V[ilocL],y_V[ilocL],sr_n[ilocL]) )
       shear_band_R_file_2.write("%6e %6e %6e \n"  % (x_V[ilocR],y_V[ilocR],sr_n[ilocR]) )
   # end for j

   #----- quadrature pts -----
   shear_band_L_file_3=open("shear_band_L_qpt"+name+".ascii","w")
   shear_band_R_file_3=open("shear_band_R_qpt"+name+".ascii","w")
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

   #----- horizontal line at mid height -----
   sr_file=open("line"+name+".ascii","w")
   counter=0
   for j in range(0,nny):
       for i in range(0,nnx):
           if abs(y_V[counter]/Ly-0.5)<eps:
              sr_file.write("%6e %6e %6e \n"  % (x_V[counter],sr_n[counter],eta_n[counter]))
           counter += 1
       # end for i
   # end for i
   sr_file.close()

print("extracting shear bands: %.3f s" % (clock.time()-start))

###############################################################################
# compute averaged elemental strainrate 
# I use a 5 point quadrature rule (per dimension) and compute the 
# average strain rate tensor components per element. 
###############################################################################
start=clock.time()

sr_e_avrg=np.zeros(nel,dtype=np.float64)  
exx_e_avrg=np.zeros(nel,dtype=np.float64)  
eyy_e_avrg=np.zeros(nel,dtype=np.float64)  
exy_e_avrg=np.zeros(nel,dtype=np.float64)  
area=np.zeros(nel,dtype=np.float64)  

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
            rq=qcoords5[iq]
            sq=qcoords5[jq]
            weightq=qweights5[iq]*qweights5[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
            exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
            eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
            exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
            exx_e_avrg[iel] += exxq*JxWq
            eyy_e_avrg[iel] += eyyq*JxWq
            exy_e_avrg[iel] += exyq*JxWq
            area[iel]+=JxWq
        # end for
    # end for
#end for

exx_e_avrg/=area
eyy_e_avrg/=area
exy_e_avrg/=area

sr_e_avrg=np.sqrt(0.5*(exx_e_avrg**2+eyy_e_avrg**2)+exy_e_avrg**2)

print("     -> exx_e_avrg (m,M) %.3e %.3e " %(np.min(exx_e_avrg),np.max(exx_e_avrg)))
print("     -> eyy_e_avrg (m,M) %.3e %.3e " %(np.min(eyy_e_avrg),np.max(eyy_e_avrg)))
print("     -> exy_e_avrg (m,M) %.3e %.3e " %(np.min(exy_e_avrg),np.max(exy_e_avrg)))
print("     -> sr_e_avrg  (m,M) %.3e %.3e " %(np.min(sr_e_avrg),np.max(sr_e_avrg)))

if debug:
   np.savetxt('sr_e_avrg.ascii',np.array([x_e,y_e,exx_e_avrg,eyy_e_avrg,exy_e_avrg,sr_e_avrg]).T,header='#x,y,exx,eyy,exy')

print("compute avrg elemental strain rate: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
###############################################################################
start=clock.time()

filename = 'solution'+name+'.vtu'
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
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
divv_e.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exx (avrg)' Format='ascii'> \n")
exx_e_avrg.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exy (avrg)' Format='ascii'> \n")
exy_e_avrg.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='strain rate (avrg)' Format='ascii'> \n")
sr_e_avrg.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
eta_e.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/s)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (m/year)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Angle' Format='ascii'> \n")
Angle.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
q.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (u)' Format='ascii'> \n")
Res_u.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (v)' Format='ascii'> \n")
Res_v.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='Res (p)' Format='ascii'> \n")
Res_q.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exxn' Format='ascii'> \n")
exx_n.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='eyyn' Format='ascii'> \n")
eyy_n.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='exyn' Format='ascii'> \n")
exy_n.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
sr_n.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
eta_n.tofile(vtufile, sep=" ", format="%.4e")
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                    icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                    icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*m_V))
vtufile.write("</DataArray>\n")
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
vtufile.write("</DataArray>\n")
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("produce final vtu file: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
