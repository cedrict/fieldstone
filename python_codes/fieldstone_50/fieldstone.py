import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import time as time
import matplotlib.pyplot as plt

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

def rho(xq,yq,imat,Tq):
    if imat==1:
       rho0=2800.
       alpha=2.5e-5
       T0=273.15
    elif imat==2:
       rho0=2900.
       alpha=2.5e-5
       T0=273.15
    elif imat==3:
       rho0=3300.
       alpha=2.5e-5
       T0=500+273.15
    elif imat==4:
       rho0=3300.
       alpha=2.5e-5
       T0=500+273.15
    val=rho0*(1.-alpha*(Tq-T0))
    return val

def eta(xq,yq,imat,exx,eyy,exy,p,T):
    Rgas=8.314
    eta_min=1.e19
    eta_max=1.e26

    if imat==1: # upper crust
       phi=20./180.*np.pi
       c=20e6
       A=8.57e-28
       Q=223e3
       n=4
       V=0
       f=1
    elif imat==2: # lower crust
       phi=20./180.*np.pi
       c=20e6
       A=7.13e-18
       Q=345e3
       n=3
       V=0
       f=1
    elif imat==3: # mantle
       phi=20./180.*np.pi
       c=20e6
       A=6.52e-16
       Q=530e3
       n=3.5
       V=18e-6
       f=1 
    else:  # seed
       phi=20./180.*np.pi
       c=20e6
       A=7.13e-18
       Q=345e3
       n=3 
       V=0 
       f=1 

    E2= np.sqrt( 0.5*(exx**2+eyy**2)+exy**2 )
    #print (exx,eyy,exy,imat,T,p)
    # compute effective viscous viscosity
    if E2<1e-20:
       etaeff_v=eta_max
    else:
       etaeff_v= 0.5 *f*A**(-1./n) * E2**(1./n-1.) * np.exp(max(Q+p*V,Q)/n/Rgas/T)
    # compute effective plastic viscosity
    if E2<1e-20: 
       etaeff_p=eta_max
    else:
       etaeff_p=( max(p*np.sin(phi)+c*np.cos(phi),c*np.cos(phi)) )/E2 * 0.5
    # blend the two viscosities
    etaeffq=2./(1./etaeff_p + 1./etaeff_v)
    etaeffq=min(etaeffq,eta_max)
    etaeffq=max(etaeffq,eta_min)
    return etaeffq

def bc_fct_left(x,y):
    return -0.25*cm/year
    
def bc_fct_right(x,y):
    return 0.25*cm/year

def bc_fct_bottom(x,y):
    return 0.125*cm/year

def onePlot(variable, plotX, plotY, title, labelX, labelY, extVal, limitX, limitY, colorMap):
    im = axes[plotX][plotY].imshow(np.flipud(variable),extent=extVal, cmap=colorMap, interpolation="nearest")
    axes[plotX][plotY].set_title(title,fontsize=10, y=1.01)

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

cm=0.01
year=31536000.

ndim=2
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=400e3  # horizontal extent of the domain 
Ly=100e3  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   niter = int(sys.argv[4])
else:
   nelx = 200
   nely = 50
   visu = 1
   niter= 5

    
nnx=2*nelx+1         # number of elements, x direction
nny=2*nely+1         # number of elements, y direction
NV=nnx*nny           # number of V nodes
NP=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely        # number of elements, total
NfemV=NV*ndofV       # number of velocity dofs
NfemP=NP*ndofP       # number of pressure dofs
Nfem=NfemV+NfemP     # total number of dofs

eps=1.e-8

nqel=9
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

gx=0.
gy=-9.81

eta_ref=1e23      # scaling of G blocks

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.zeros(NV,dtype=np.float64)  # x coordinates
yV=np.zeros(NV,dtype=np.float64)  # y coordinates

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*hx/2
        yV[counter]=j*hy/2
        counter+=1

xP=np.empty(NP,dtype=np.float64)     # x coordinates
yP=np.empty(NP,dtype=np.float64)     # y coordinates

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        xP[counter]=i*hx
        yP[counter]=j*hy
        counter+=1

#np.savetxt('gridV.ascii',np.array([xV,yV]).T,header='# x,y')
#np.savetxt('gridP.ascii',np.array([xP,yP]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

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
start = time.time()

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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0,NV):
    if xV[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = bc_fct_left(xV[i],yV[i])
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = bc_fct_right(xV[i],yV[i])
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = bc_fct_bottom(xV[i],yV[i])

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# define temperature field
#################################################################

T=np.zeros(NV,dtype=np.float64) 

for i in range(0,NV):
    if yV[i]>70.e3:
       T[i]= -(yV[i]-100.e3)*(500.)/30.e3+0
    else:
       T[i]= -(yV[i]-70.e3)*(700.)/70.e3+500

T[:]=T[:]+273.15

#################################################################
# define material layout
#################################################################

material=np.zeros(nel,dtype=np.int32) 

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    xc[iel]=xV[iconV[:,iel]].sum()/mV
    yc[iel]=yV[iconV[:,iel]].sum()/mV

    if yc[iel]>80.e3:
       material[iel]=1
    elif yc[iel]>70.e3:
       material[iel]=2
    else:
       material[iel]=3
 
    if yc[iel]<68.e3 and yc[iel]>60.e3 and xc[iel]>=198.e3 and xc[iel]<=202.e3:
       material[iel]=4

#========================================================================================
#========================================================================================
# non linear iterations
#========================================================================================
#========================================================================================
u      = np.zeros(NV,dtype=np.float64)            # x-component velocity
v      = np.zeros(NV,dtype=np.float64)            # y-component velocity
p      = np.zeros(NP,dtype=np.float64)            # y-component velocity
u_old  = np.zeros(NV,dtype=np.float64)            # x-component velocity
v_old  = np.zeros(NV,dtype=np.float64)            # y-component velocity
p_old  = np.zeros(NP,dtype=np.float64)            # y-component velocity
    
c_mat   = np.array([[2,0,0],\
                    [0,2,0],\
                    [0,0,1]],dtype=np.float64) 

for iter in range(0,niter):

    print('=======================')
    print('======= iter ',iter,'=======')
    print('=======================')

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = time.time()

    A_sparse= lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs     = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
    N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
    NNNV    = np.zeros(mV,dtype=np.float64)            # shape functions V
    NNNP    = np.zeros(mP,dtype=np.float64)            # shape functions P
    dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    dNNNVds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
    eta_el  = np.zeros(nel,dtype=np.float64)  
    rho_el  = np.zeros(nel,dtype=np.float64)  

    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)

        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:

                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]

                NNNV[0:mV]=NNV(rq,sq)
                dNNNVdr[0:mV]=dNNVdr(rq,sq)
                dNNNVds[0:mV]=dNNVds(rq,sq)
                NNNP[0:mP]=NNP(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((ndim,ndim),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)

                # compure pressure at q point
                pq=0.
                for k in range(0,mP):
                    pq+=NNNP[k]*p[iconP[k,iel]]

                # compute dNdx & dNdy
                xq=0.0
                yq=0.0
                Tq=0.0
                exxq=0.
                eyyq=0.
                exyq=0.
                for k in range(0,mV):
                    xq+=NNNV[k]*xV[iconV[k,iel]]
                    yq+=NNNV[k]*yV[iconV[k,iel]]
                    Tq+=NNNV[k]*T[iconV[k,iel]]
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    exxq += dNNNVdx[k]*u[iconV[k,iel]]
                    eyyq += dNNNVdy[k]*v[iconV[k,iel]]
                    exyq += 0.5*dNNNVdy[k]*u[iconV[k,iel]]+\
                            0.5*dNNNVdx[k]*v[iconV[k,iel]]

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                etaq=eta(xq,yq,material[iel],exxq,eyyq,exyq,pq,Tq)
                eta_el[iel]+=etaq/nqel

                rhoq=rho(xq,yq,material[iel],Tq)
                rho_el[iel]+=rhoq/nqel

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*rhoq*gx
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rhoq*gy

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

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
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val[m1]
                   G_el[ikk,:]=0

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
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*eta_ref/Ly
                    A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*eta_ref/Ly
                rhs[m1]+=f_el[ikk]
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            rhs[NfemV+m2]+=h_el[k2]*eta_ref/Ly

    print("build FE matrix: %.5f s nel= %d time/elt %.5f" % (time.time() - start, nel, (time.time()-start)/ nel,))

    ######################################################################
    # solve system
    ######################################################################
    start = time.time()

    sparse_matrix=A_sparse.tocsr()

    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (time.time() - start))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.4f %.4f cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("     -> v (m,M) %.4f %.4f cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
    print("     -> p (m,M) %.4f %.4f Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
    #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

    print("split vel into u,v: %.3f s" % (time.time() - start))


    ######################################################################

    xi_u=np.linalg.norm(u-u_old,2)/np.linalg.norm(u+u_old,2)
    xi_v=np.linalg.norm(v-v_old,2)/np.linalg.norm(v+v_old,2)
    xi_p=np.linalg.norm(p-p_old,2)/np.linalg.norm(p+p_old,2)

    print("conv: u,v,p: %.6f %.6f %.6f" %(xi_u,xi_v,xi_p))

    u_old=u
    v_old=v
    p_old=p


#========================================================================================
#========================================================================================

######################################################################
# compute strainrate 
######################################################################
start = time.time()

exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.0
    sq=0.0
    weightq=2.0*2.0

    NNNV[0:mV]=NNV(rq,sq)
    dNNNVdr[0:mV]=dNNVdr(rq,sq)
    dNNNVds[0:mV]=dNNVds(rq,sq)

    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
        jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
        jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
        jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
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

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################

q=np.zeros(NV,dtype=np.float64)

for iel in range(0,nel):
    q[iconV[0,iel]]=p[iconP[0,iel]]
    q[iconV[1,iel]]=p[iconP[1,iel]]
    q[iconV[2,iel]]=p[iconP[2,iel]]
    q[iconV[3,iel]]=p[iconP[3,iel]]
    q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
    q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
    q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
    q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
    q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25

np.savetxt('q.ascii',np.array([xV,yV,q]).T,header='# x,y,q')

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

if visu:
   filename = 'solution.vtu'
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
   #--
   vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % eta_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % rho_el[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='material' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % material[iel])
   vtufile.write("</DataArray>\n")

   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix u' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[i*ndofV  ]:
          vtufile.write("%10e \n" % 1.)
       else:
          vtufile.write("%10e \n" % 0.)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='bc_fix v' Format='ascii'> \n")
   for i in range(0,NV):
       if bc_fix[i*ndofV+1]:
          vtufile.write("%10e \n" % 1.)
       else:
          vtufile.write("%10e \n" % 0.)
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %(T[i]-273.15))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*8))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %23)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

   fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(18,18))
   extent=(np.amin(xV),np.amax(xV),np.amin(yV),np.amax(yV))

   onePlot(u.reshape((nny,nnx)),        0, 0, "$v_x$",                 "x", "y", extent,  0,  0, 'Spectral_r')
   onePlot(v.reshape((nny,nnx)),        0, 1, "$v_y$",                 "x", "y", extent,  0,  0, 'Spectral_r')
   onePlot(q.reshape((nny,nnx)),        0, 2, "$p$",                   "x", "y", extent, Lx, Ly, 'RdGy_r')
   onePlot(exx.reshape((nely,nelx)),    1, 0, "$\dot{\epsilon}_{xx}$", "x", "y", extent, Lx, Ly, 'viridis')
   onePlot(eyy.reshape((nely,nelx)),    1, 1, "$\dot{\epsilon}_{yy}$", "x", "y", extent, Lx, Ly, 'viridis')
   onePlot(exy.reshape((nely,nelx)),    1, 2, "$\dot{\epsilon}_{xy}$", "x", "y", extent, Lx, Ly, 'viridis')
   onePlot(rho_el.reshape((nely,nelx)), 2, 0, "density",               "x", "y", extent, Lx, Ly, 'RdGy_r')
   onePlot(eta_el.reshape((nely,nelx)), 2, 1, "viscosity",             "x", "y", extent, Lx, Ly, 'RdGy_r')
   onePlot(T.reshape((nny,nnx)),        2, 2, "$T$",                   "x", "y", extent, Lx, Ly, 'RdGy_r')

   plt.subplots_adjust(hspace=0.5)
   plt.savefig('solution.pdf', bbox_inches='tight')
   plt.show()

#==============================================================================
# end time stepping loop


print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
