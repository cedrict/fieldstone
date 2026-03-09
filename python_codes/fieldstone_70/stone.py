import numpy as np
import sys as sys
#import scipy
import scipy.sparse as sps
#from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix,lil_matrix
import time as clock 
from numpy import linalg as LA
#import numba
from numba import jit

###############################################################################

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

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
    #eta_m=1e21
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

###############################################################################

cm=0.01
year=365.25*24*3600
eps=1.e-8

print("*******************************")
print("********** stone 070 **********")
print("*******************************")

ndim=2
m_V=9    # number of velocity nodes making up an element
m_P=4    # number of pressure nodes making up an element
ndof_V=2 # number of velocity degrees of freedom per node

Lx=100e3  # horizontal extent of the domain 
Ly= 30e3  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 7):
   nelx  = int(sys.argv[1])
   nely  = int(sys.argv[2])
   visu  = int(sys.argv[3])
   niter = int(sys.argv[4])
   bc    = int(sys.argv[5])
   eta_m = int(sys.argv[6])
   print(sys.argv)
else:
   nelx = 100
   nely = 30
   visu = 1
   niter= 10
   bc   = 1
   eta_m= 21

eta_m=10**eta_m

nnx=2*nelx+1           # number of V nodes, x direction
nny=2*nely+1           # number of V nodes, y direction
nn_V=nnx*nny           # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely          # number of elements, total
Nfem_V=nn_V*ndof_V     # number of velocity dofs
Nfem_P=nn_P            # number of pressure dofs
Nfem=Nfem_V+Nfem_P     # total number of dofs

nq_per_dim=3
nq_per_el=nq_per_dim**ndim
nq=nel*nq_per_el
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

rho0=2700
gy=-10

velbc=bc*1e-15*Lx/2

eta_ref=1e23      # scaling of G blocks
    
p_has_nullspace=True

debug=False

#################################################################
#################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("niter=",niter)
print("------------------------------")

#################################################################
# grid point setup
#################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)
y_V=np.zeros(nn_V,dtype=np.float64)

counter=0    
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2
        y_V[counter]=j*hy/2
        counter+=1

x_P=np.zeros(nn_P,dtype=np.float64)
y_P=np.zeros(nn_P,dtype=np.float64)

counter=0    
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter+=1

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')
if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

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
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

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
        counter+=1

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter+=1

print("setup: connectivity: %.3f s" % (clock.time()-start))

#################################################################
# define boundary conditions
#################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=+velbc 
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=-velbc 
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=-velbc*Ly/Lx 
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=+velbc*Ly/Lx 

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# define temperature field
#################################################################
start=clock.time()

T=np.zeros(nn_V,dtype=np.float64) 

for i in range(0,nn_V):
    T[i]=20+(Ly-y_V[i])*0.015+273.15

print("setup: initial temperature: %.3f s" % (clock.time()-start))

#========================================================================================
#========================================================================================
# non linear iterations
#========================================================================================
#========================================================================================
u      = np.zeros(nn_V,dtype=np.float64) # x-component velocity
v      = np.zeros(nn_V,dtype=np.float64) # y-component velocity
p      = np.zeros(nn_P,dtype=np.float64) # y-component velocity
u_old  = np.zeros(nn_V,dtype=np.float64) # x-component velocity
v_old  = np.zeros(nn_V,dtype=np.float64) # y-component velocity
p_old  = np.zeros(nn_P,dtype=np.float64) # y-component velocity
sol    = np.zeros(Nfem,dtype=np.float64) # solution vector 
    
C= np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
    
convfile=open('conv.ascii',"w")
convfile.write("#iter,xi_u,xi_v,xi_p,Res_two/Res0_two") 
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
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    b_fem=np.zeros(Nfem,dtype=np.float64) 
    B=np.zeros((3,ndof_V*m_V),dtype=np.float64)
    N_mat=np.zeros((3,m_P),dtype=np.float64)  
    jcb=np.zeros((ndim,ndim),dtype=np.float64)

    xq=np.zeros(nq,dtype=np.float64)         
    yq=np.zeros(nq,dtype=np.float64)         
    pq=np.zeros(nq,dtype=np.float64)         
    Tq=np.zeros(nq,dtype=np.float64)         
    rhq=np.zeros(nq,dtype=np.float64)         
    srq=np.zeros(nq,dtype=np.float64)         
    etaq=np.zeros(nq,dtype=np.float64)         
    tauq=np.zeros(nq,dtype=np.float64)         
    srq_v=np.zeros(nq,dtype=np.float64)         
    srq_ds=np.zeros(nq,dtype=np.float64)         
    srq_vp=np.zeros(nq,dtype=np.float64)         

    counterq=0
    for iel in range(0,nel):

        # set arrays to 0 every loop
        f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
        K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
        G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
        h_el=np.zeros((m_P),dtype=np.float64)

        for iq in range(0,nq_per_dim):
            for jq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                N_V=basis_functions_V(rq,sq)
                N_P=basis_functions_P(rq,sq)
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
                for i in range(0,m_V):
                    B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                      [0.       ,dNdy_V[i]],
                                      [dNdy_V[i],dNdx_V[i]]]

                xq[counterq]=np.dot(N_V,x_V[icon_V[:,iel]])
                yq[counterq]=np.dot(N_V,y_V[icon_V[:,iel]])
                Tq[counterq]=np.dot(N_V,T[icon_V[:,iel]])
                pq[counterq]=np.dot(N_P,p[icon_P[:,iel]])

                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                exyq=np.dot(dNdx_V,v[icon_V[:,iel]])*0.5+\
                     np.dot(dNdy_V,u[icon_V[:,iel]])*0.5
                srq[counterq]=np.sqrt( 0.5*(exxq**2+eyyq**2)+exyq**2 )

                etaq[counterq],rhq[counterq],tauq[counterq],srq_v[counterq],srq_ds[counterq],srq_vp[counterq]=\
                material_model(xq[counterq],yq[counterq],exxq,eyyq,exyq,pq[counterq],Tq[counterq])

                K_el+=B.T.dot(C.dot(B))*etaq[counterq]*JxWq

                for i in range(0,m_V):
                    f_el[ndof_V*i+1]+=N_V[i]*rho0*gy*JxWq

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=0.

                G_el-=B.T.dot(N_mat)*JxWq

                counterq+=1
            #end jq
        #end iq

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
                   K_el[ikk,ikk]=K_ref
                   f_el[ikk]=K_ref*bc_val_V[m1]
                   h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                   G_el[ikk,:]=0

        # assemble matrix and right hand side
        for k1 in range(0,m_V):
            for i1 in range(0,ndof_V):
                ikk=ndof_V*k1          +i1
                m1 =ndof_V*icon_V[k1,iel]+i1
                for k2 in range(0,m_V):
                    for i2 in range(0,ndof_V):
                        jkk=ndof_V*k2          +i2
                        m2 =ndof_V*icon_V[k2,iel]+i2
                        A_fem[m1,m2] += K_el[ikk,jkk]
                for k2 in range(0,m_P):
                    jkk=k2
                    m2 =icon_P[k2,iel]
                    A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]*eta_ref/Ly
                    A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]*eta_ref/Ly
                b_fem[m1]+=f_el[ikk]
        for k2 in range(0,m_P):
            m2=icon_P[k2,iel]
            b_fem[Nfem_V+m2]+=h_el[k2]*eta_ref/Ly

    time_build_matrix[iter]=clock.time()-start

    if debug: np.savetxt('qpts.ascii',np.array([xq,yq,etaq,tauq,rhq,pq,srq]).T)
    
    print("build FE matrix: %.5f s" % (clock.time()-start))

    ###########################################################################
    # solve system
    ###########################################################################
    start=clock.time()

    #a_mat[Nfem,NfemV:Nfem]=constr
    #a_mat[NfemV:Nfem,Nfem]=constr

    if p_has_nullspace:
       #simple p boundary condition
       for i in range(0,Nfem):
           A_fem[Nfem-1,i]=0
       A_fem[Nfem-1,Nfem-1]=1
       b_fem[Nfem-1]=0.

    sparse_matrix=A_fem.tocsr()
    Res=sparse_matrix.dot(sol)-b_fem
    sol=sps.linalg.spsolve(sparse_matrix,b_fem)

    print("solve time: %.3f s" % (clock.time()-start))

    if iter==0: Res0_two=LA.norm(Res,2)
    Res_two=LA.norm(Res,2)
    print("     -> Nonl. res. (2-norm) %.7e" % (Res_two/Res0_two))

    ###########################################################################
    # put solution into separate x,y velocity arrays
    ###########################################################################
    start=clock.time()

    u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
    p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

    print("     -> u (m,M) %.4e %.4e cm/yr" %(np.min(u)/cm*year,np.max(u)/cm*year))
    print("     -> v (m,M) %.4e %.4e cm/yr" %(np.min(v)/cm*year,np.max(v)/cm*year))
    print("     -> p (m,M) %.4e %.4e Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

    if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

    print("split vel into u,v: %.3f s" % (clock.time()-start))

    ###########################################################################
    #normalise pressure: average on top boundary is zero
    #pressure is Q1 so I need a single quad point in the middle of the 
    #face to carry out integration on edge.
    ###########################################################################
    start=clock.time()

    if p_has_nullspace:

       if debug: np.savetxt('pressure_bef_normalisation.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

       intp=0
       for iel in range(0,nel):
           inode=icon_V[6,iel]       #top middle node 
           if y_V[inode]/Ly>(1-eps): #if on top boundary
              intp+=hx*(p[icon_P[2,iel]]+p[icon_P[3,iel]])/2
       intp/=Lx
       p-=intp

       print('     -> intp=',intp)
       print("     -> p (m,M) %.4e %.4e Mpa  " %(np.min(p)/1e6,np.max(p)/1e6))

       if debug: np.savetxt('pressure_aft_normalisation.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

       print("normalise pressure: %.3f s" % (clock.time()-start))

    ###########################################################################
    start=clock.time()

    xi_u=np.linalg.norm(u-u_old,2)/np.linalg.norm(u+u_old,2)
    xi_v=np.linalg.norm(v-v_old,2)/np.linalg.norm(v+v_old,2)
    xi_p=np.linalg.norm(p-p_old,2)/np.linalg.norm(p+p_old,2)

    print("     -> conv: u,v,p,Res: %.6f %.6f %.6f %.6f" %(xi_u,xi_v,xi_p,Res_two/Res0_two))

    convfile.write("%3d %e %e %e %e\n" %(iter,xi_u,xi_v,xi_p,Res_two/Res0_two)) 
    convfile.flush()

    print("assess convergence: %.3f s" % (clock.time()-start))

    ###########################################################################
    # generate vtu output at every nonlinear iteration
    ###########################################################################
    start=clock.time()

    if iter%5==0:
       filename = 'qpts_{:04d}.vtu'.format(iter) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nq,nq))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for iq in range(0,nq):
           vtufile.write("%e %e %e \n" %(xq[iq],yq[iq],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
       etaq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
       tauq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
       pq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       Tq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='rh' Format='ascii'> \n")
       rhq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr' Format='ascii'> \n")
       srq.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_v' Format='ascii'> \n")
       srq_v.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_vp' Format='ascii'> \n")
       srq_vp.tofile(vtufile,sep=' ',format='%.4e')
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='sr_ds' Format='ascii'> \n")
       srq_ds.tofile(vtufile,sep=' ',format='%.4e')
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

    print("produce nonlinear it vtu file: %.3f s" % (clock.time()-start))

    u_old=np.copy(u)
    v_old=np.copy(v)
    p_old=np.copy(p)

#end for iter

#========================================================================================
#========================================================================================
# end nonlinear iterations
#========================================================================================
#========================================================================================

###############################################################################
# compute elemental strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.0
    sq=0.0
    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

if debug: 
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute elemental strain rate: %.3f s" % (clock.time()-start))

#####################################################################
# compute nodal strainrate and pressure on V grid. 
#
# 3--6--2
# |  |  |
# 7--8--5
# |  |  |
# 0--4--1
#####################################################################
start=clock.time()

r_V=[-1,+1,+1,-1,0,+1,0,-1,0]
s_V=[-1,-1,+1,+1,-1,0,+1,0,0]
    
q=np.zeros(nn_V,dtype=np.float64)
exx_n=np.zeros(nn_V,dtype=np.float64)  
eyy_n=np.zeros(nn_V,dtype=np.float64)  
exy_n=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.int32)  

for iel in range(0,nel):
    for i in range(0,m_V):
        rq=r_V[i]
        sq=s_V[i]
        N_V=basis_functions_V(rq,sq)
        N_P=basis_functions_P(rq,sq)
        dNdr_V=basis_functions_V_dr(rq,sq)
        dNds_V=basis_functions_V_ds(rq,sq)
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        e_xx=np.dot(dNdx_V[:],u[icon_V[:,iel]])
        e_yy=np.dot(dNdy_V[:],v[icon_V[:,iel]])
        e_xy=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
        inode=icon_V[i,iel]
        exx_n[inode]+=e_xx
        eyy_n[inode]+=e_yy
        exy_n[inode]+=e_xy
        q[inode]+=np.dot(N_P,p[icon_P[:,iel]])
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

if debug: 
   np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')
   np.savetxt('strainrate.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

print("compute nodal press & sr: %.3f s" % (clock.time()-start))

#####################################################################
start=clock.time()

pfile=open('profile.ascii',"w")

for i in range(0,nn_V):
    if abs(y_V[i]-2*Ly/3)<eps*Ly:
       pfile.write("%e %e %e %e %e \n" %(x_V[i],q[i],exx_n[i],eyy_n[i],exy_n[i]))

print("write out profile: %.3f s" % (clock.time()-start))

#####################################################################
# plot of solution
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 
#####################################################################
start=clock.time()

if visu:

   filename = 'solution.vtu'
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
   vtufile.write("<DataArray type='Float32' Name='etaq (middle)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (etaq[iel*9+4]))
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='tauq (middle)' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%e\n" % (tauq[iel*9+4]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='bc_fix u' Format='ascii'> \n")
   for i in range(0,nn_V):
       if bc_fix_V[i*ndof_V  ]:
          vtufile.write("%e \n" % 1.)
       else:
          vtufile.write("%e \n" % 0.)
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='bc_fix v' Format='ascii'> \n")
   for i in range(0,nn_V):
       if bc_fix_V[i*ndof_V+1]:
          vtufile.write("%e \n" % 1.)
       else:
          vtufile.write("%e \n" % 0.)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   q.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   exx_n.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   eyy_n.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   exy_n.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for i in range(0,nn_V):
       e=np.sqrt(0.5*(exx_n[i]*exx_n[i]+eyy_n[i]*eyy_n[i])+exy_n[i]*exy_n[i])
       vtufile.write("%e \n" %e)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e \n" %(T[i]-273.15))
   vtufile.write("</DataArray>\n")
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
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

print("write vtu file: %.3f s" % (clock.time()-start))

#==============================================================================
# end time stepping loop

print("-----------------------------")

print("avrg time build FEM matrix=",np.sum(time_build_matrix)/niter)

print("*******************************")
print("********** the end ************")
print("*******************************")
