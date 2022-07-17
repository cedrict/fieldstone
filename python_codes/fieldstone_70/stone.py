import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import time as time
from numpy import linalg as LA
import numba
from numba import jit

#------------------------------------------------------------------------------

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def NNP(rq,sq):
    NP_0=0.25*(1-rq)*(1-sq)
    NP_1=0.25*(1+rq)*(1-sq)
    NP_2=0.25*(1+rq)*(1+sq)
    NP_3=0.25*(1-rq)*(1+sq)
    return NP_0,NP_1,NP_2,NP_3

def material_model(xq,yq,exx,eyy,exy,p,T):
    #---------------------
    # material properties
    #---------------------
    A=3.1623e-26
    nnn=3.3
    Q=186.5e3
    Rgas=8.314
    phi=31./180.*np.pi #(atan(0.6) in paper)
    c=50e6
    eta_m=1e21
    eta_v=1e25
    #-------------------
    if iter==0:
       E2=1e-15
    else:
       E2=np.sqrt( 0.5*(exx**2+eyy**2)+exy**2 )
    #-------------------

    if (xq-Lx/2)**2+(yq-Ly/2)**2 < 4e6: # inclusion
       etaeff=1e20
       tau=2*etaeff*E2
       rh=0
       eps_v=E2
       eps_ds=0
       eps_vp=0
    else:

       #yield strength
       Y= max(p,0)*np.sin(phi) + c*np.cos(phi) 

       #initial guess for tau using E2/2 for both mechanisms
       eta_ds = 0.5*A**(-1./nnn)*np.exp(Q/nnn/Rgas/T)*(E2/2)**(1./nnn-1.)
       etaeff=1./(1./eta_v+1./eta_ds)
       tau=2*etaeff*E2

       #compute strain rate partitioning between linear viscous and dislocation creep
       it=0
       func=1
       taus=np.zeros(31,dtype=np.float64) 
       #print('---------------------------')
       while abs(func)>1e-20:
             eps_ds=A*tau**nnn*np.exp(-Q/Rgas/T)
             eps_v=tau/2/eta_v
             func=E2-eps_ds-eps_v
             funcp=-nnn*A*tau**(nnn-1)*np.exp(-Q/Rgas/T)-0.5/eta_v
             tau-=func/funcp
             if tau<0:
                print('visc branch',iter,it,tau)
                exit("tau<0")
             #print('->',iter,it,tau,eps_ds,func)
             taus[it]=tau
             it+=1
             if it>30:
                rh=3
                print('visc branch: iter=',iter,'tau=',tau)
                print('visc branch: not converged')
                print('visc branch: F=',func)
                for i in range(0,30):
                    print(i,taus[i])
                break
       #end while

       if tau <= Y: #-------viscous branch
          eta_ds = 0.5*A**(-1./nnn)*np.exp(Q/nnn/Rgas/T)*eps_ds**(1./nnn-1.)
          etaeff=1./(1./eta_v+1./eta_ds)
          rh=1
          eps_v=tau/2/eta_v
          eps_vp=0
       else: #----------------------visco-viscoplastic branch

          rh=2
          it=0
          func=1
          #print('---------------------------')
          taus=np.zeros(31,dtype=np.float64) 
          while abs(func)>1e-6:
             eps_ds=A*tau**nnn*np.exp(-Q/Rgas/T)
             eps_v=tau/2/eta_v
             func=Y + 2*(E2-eps_ds-eps_v)*eta_m - tau
             funcp=-eta_m/eta_v -2*eta_m*nnn*A*tau**(nnn-1)*np.exp(-Q/Rgas/T)-1
             tau-=func/funcp
             if tau<0:
                print(iter,it,tau,eps_ds,func,Y)
                exit("tau<0")
             #print(iter,it,tau,eps_ds,func,Y)
             taus[it]=tau
             it+=1
             if it>30:
                rh=3
                print('vp branch: iter=',iter,'tau=',tau)
                print('vp branch: not converged')
                print('vp branch: F=',func)
                for i in range(0,30):
                    print(i,taus[i])
                break
          #end while

          #now that we have tau we can use it to compute the 
          #viscosities and strain rates of v, ds and vp element
          eps_ds=A*tau**nnn*np.exp(-Q/Rgas/T)
          eta_ds = 0.5*A**(-1./nnn)*np.exp(Q/nnn/Rgas/T)*eps_ds**(1./nnn-1.)
          eps_v=tau/2/eta_v

          eps_vp=E2-eps_ds-eps_v
          if eps_vp<0:
             print("eps_vp<0",E2,eps_ds)
          eta_vp=Y/2/eps_vp+eta_m

          etaeff=1/(1/eta_v+1/eta_ds+1/eta_vp)

       #end if
    #end if
    return etaeff,rh,tau,eps_v,eps_ds,eps_vp

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

Lx=100e3  # horizontal extent of the domain 
Ly= 30e3  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 7):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   niter = int(sys.argv[4])
   bc    = int(sys.argv[5])
   vp    = int(sys.argv[6])
else:
   nelx = 100
   nely = 30
   visu = 1
   niter= 500
   bc   = 1
   vp   = 21

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
gy=-10

velbc=bc*1e-15*Lx/2

eta_ref=1e23      # scaling of G blocks
    
p_has_nullspace=True

rho0=2700

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("niter=",niter)
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
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = +velbc 
    if xV[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = -velbc 
    if yV[i]/Ly<eps:
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = -velbc*Ly/Lx 
    if yV[i]/Ly>(1-eps):
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = +velbc*Ly/Lx 

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# define temperature field
#################################################################

T=np.zeros(NV,dtype=np.float64) 

for i in range(0,NV):
    T[i]=20+(Ly-yV[i])*0.015+273.15

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
sol    = np.zeros(Nfem,dtype=np.float64)          # solution vector 
    
c_mat   = np.array([[2,0,0],\
                    [0,2,0],\
                    [0,0,1]],dtype=np.float64) 
    
convfile=open('conv.ascii',"w")
time_build_matrix=np.zeros(niter,dtype=np.float64)  

for iter in range(0,niter):

    print('=========================')
    print('======= iter ',iter,'=======')
    print('=========================')

    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = time.time()

    #A_sparse= sps.dok_matrix((Nfem,Nfem),dtype=np.float64)
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

    etaq  = np.zeros(nel*9,dtype=np.float64)         
    tauq  = np.zeros(nel*9,dtype=np.float64)         
    xq    = np.zeros(nel*9,dtype=np.float64)         
    yq    = np.zeros(nel*9,dtype=np.float64)         
    pq    = np.zeros(nel*9,dtype=np.float64)         
    rhq   = np.zeros(nel*9,dtype=np.float64)         
    srq   = np.zeros(nel*9,dtype=np.float64)         
    srq_ds = np.zeros(nel*9,dtype=np.float64)         
    srq_vp = np.zeros(nel*9,dtype=np.float64)         
    srq_v  = np.zeros(nel*9,dtype=np.float64)         

    counterq=0
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

                for k in range(0,mV):
                    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]

                # compute dNdx & dNdy
                xq[counterq]=np.sum(NNNV[:]*xV[iconV[:,iel]])
                yq[counterq]=np.sum(NNNV[:]*yV[iconV[:,iel]])
                Tq=np.sum(NNNV[:]*T[iconV[:,iel]])
                pq[counterq]=np.sum(NNNP[:]*p[iconP[:,iel]])
                exxq=np.sum(dNNNVdx[:]*u[iconV[:,iel]])
                eyyq=np.sum(dNNNVdy[:]*v[iconV[:,iel]])
                exyq=0.5*np.sum(dNNNVdx[:]*v[iconV[:,iel]])+\
                     0.5*np.sum(dNNNVdy[:]*u[iconV[:,iel]])
                srq[counterq]=np.sqrt( 0.5*(exxq**2+eyyq**2)+exyq**2 )

                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.     ],
                                             [0.        ,dNNNVdy[i]],
                                             [dNNNVdy[i],dNNNVdx[i]]]

                etaq[counterq],rhq[counterq],tauq[counterq],srq_v[counterq],srq_ds[counterq],srq_vp[counterq]=\
                material_model(xq[counterq],yq[counterq],exxq,eyyq,exyq,pq[counterq],Tq)

                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counterq]*weightq*jcob

                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*rho0*gy

                for i in range(0,mP):
                    N_mat[0,i]=NNNP[i]
                    N_mat[1,i]=NNNP[i]
                    N_mat[2,i]=0.

                G_el-=b_mat.T.dot(N_mat)*weightq*jcob

                counterq+=1
            #end jq
        #end iq

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

    print("build FE matrix: %.5f s" % (time.time()-start))

    time_build_matrix[iter]=time.time()-start

    np.savetxt('qpts.ascii',np.array([xq,yq,etaq,tauq,rhq,pq,srq]).T)

    ######################################################################
    # solve system
    ######################################################################
    start = time.time()

    #a_mat[Nfem,NfemV:Nfem]=constr
    #a_mat[NfemV:Nfem,Nfem]=constr

    if p_has_nullspace:
       #simple p boundary condition
       for i in range(0,Nfem):
           A_sparse[Nfem-1,i]=0
       A_sparse[Nfem-1,Nfem-1]=1
       rhs[Nfem-1]=0.

    sparse_matrix=A_sparse.tocsr()
    Res=sparse_matrix.dot(sol)-rhs
    sol=sps.linalg.spsolve(sparse_matrix,rhs)

    print("solve time: %.3f s" % (time.time() - start))

    if iter==0:
      Res0_two=LA.norm(Res,2)
    Res_two=LA.norm(Res,2)
    print("     -> Nonl. res. (2-norm) %.7e" % (Res_two/Res0_two))

    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()

    u,v=np.reshape(sol[0:NfemV],(NV,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.4e %.4e cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("     -> v (m,M) %.4e %.4e cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
    print("     -> p (m,M) %.4e %.4e Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

    #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    #normalise pressure: average on top boundary is zero
    #pressure is Q1 so I need a single quad point in the middle of the 
    #face to carry out integration on edge.
    ######################################################################
    start = time.time()

    if p_has_nullspace:

       np.savetxt('pressure_bef_normalisation.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

       intp=0
       for iel in range(0,nel):
           inode=iconV[6,iel]       #top middle node 
           if yV[inode]/Ly>(1-eps): #if on top boundary
              intp+=hx*(p[iconP[2,iel]]+p[iconP[3,iel]])/2
       intp/=Lx
       p-=intp

       print('     -> intp=',intp)
       print("     -> p (m,M) %.4e %.4e Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

       np.savetxt('pressure_aft_normalisation.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

       print("normalise pressure: %.3f s" % (time.time() - start))

    ######################################################################
    start = time.time()

    xi_u=np.linalg.norm(u-u_old,2)/np.linalg.norm(u+u_old,2)
    xi_v=np.linalg.norm(v-v_old,2)/np.linalg.norm(v+v_old,2)
    xi_p=np.linalg.norm(p-p_old,2)/np.linalg.norm(p+p_old,2)

    print("     -> conv: u,v,p,Res: %.6f %.6f %.6f %.6f" %(xi_u,xi_v,xi_p,Res_two/Res0_two))

    convfile.write("%3d %10e %10e %10e %10e\n" %(iter,xi_u,xi_v,xi_p,Res_two/Res0_two)) 
    convfile.flush()

    print("assess convergence: %.3f s" % (time.time() - start))

    ######################################################################
    # generate vtu output at every nonlinear iteration
    ######################################################################
    start = time.time()

    if iter%1==0:
       nq=9*nel
       filename = 'qpts_{:04d}.vtu'.format(iter) 
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
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % etaq[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % tauq[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % pq[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rh' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % rhq[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % srq[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_v' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % srq_v[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_vp' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % srq_vp[iq])
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_ds' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%10e \n" % srq_ds[iq])
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
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

    print("produce nonlinear it vtu file: %.3f s" % (time.time() - start))

    u_old=u
    v_old=v
    p_old=p

#end for iter

#========================================================================================
#========================================================================================

######################################################################
# compute elemental strainrate 
######################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
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

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

#####################################################################
# compute nodal strainrate and pressure on V grid. 
#
# 3--6--2
# |  |  |
# 7--8--5
# |  |  |
# 0--4--1
#####################################################################
start = time.time()

rVnodes=[-1,+1,+1,-1,0,+1,0,-1,0]
sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]
    
exx_n = np.zeros(NV,dtype=np.float64)  
eyy_n = np.zeros(NV,dtype=np.float64)  
exy_n = np.zeros(NV,dtype=np.float64)  
count = np.zeros(NV,dtype=np.int32)  
q=np.zeros(NV,dtype=np.float64)

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
            e_xx=0.
            e_yy=0.
            e_xy=0.
            for k in range(0,mV):
                e_xx += dNNNVdx[k]*u[iconV[k,iel]]
                e_yy += dNNNVdy[k]*v[iconV[k,iel]]
                e_xy += 0.5*(dNNNVdy[k]*u[iconV[k,iel]]+dNNNVdx[k]*v[iconV[k,iel]])
            #end for
            inode=iconV[i,iel]
            exx_n[inode]+=e_xx
            eyy_n[inode]+=e_yy
            exy_n[inode]+=e_xy
            q[inode]+=np.dot(p[iconP[0:mP,iel]],NNNP[0:mP])
            count[inode]+=1
        #end for
#end for
    
exx_n/=count
eyy_n/=count
exy_n/=count
q/=count

print("     -> exx nodal (m,M) %.6e %.6e " %(np.min(exx_n),np.max(exx_n)))
print("     -> eyy nodal (m,M) %.6e %.6e " %(np.min(eyy_n),np.max(eyy_n)))
print("     -> exy nodal (m,M) %.6e %.6e " %(np.min(exy_n),np.max(exy_n)))
print("     -> press nodal (m,M) %.5e %.5e " %(np.min(q),np.max(q)))

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')
#np.savetxt('strainrate.ascii',np.array([xV,yV,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

print("compute nodal press & sr: %.3f s" % (time.time() - start))

#####################################################################
start = time.time()

pfile=open('profile.ascii',"w")

for i in range(0,NV):
    if abs(yV[i]-2*Ly/3)<eps*Ly:
       pfile.write("%10e %10e %10e %10e %10e \n" %(xV[i],q[i],exx_n[i],eyy_n[i],exy_n[i]))

print("write out profile: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 
#####################################################################
start = time.time()

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
   vtufile.write("<DataArray type='Float32' Name='etaq (middle)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (etaq[iel*9+4]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='tauq (middle)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%10e\n" % (tauq[iel*9+4]))
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
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %exx_n[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %eyy_n[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %exy_n[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for i in range(0,NV):
       e=np.sqrt(0.5*(exx_n[i]*exx_n[i]+eyy_n[i]*eyy_n[i])+exy_n[i]*exy_n[i])
       vtufile.write("%10e \n" %e)
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
       vtufile.write("%d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel],\
                                                   iconV[4,iel],iconV[5,iel],iconV[6,iel],iconV[7,iel]))
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

print("write vtu file: %.3f s" % (time.time() - start))

#==============================================================================
# end time stepping loop

print("-----------------------------")

print("avrg time build FEM matrix=",np.sum(time_build_matrix)/niter)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
