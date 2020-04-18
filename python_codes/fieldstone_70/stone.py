import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import time as time
from numpy import linalg as LA

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

def rho(xq,yq,Tq):
    return 2700.

def eta(xq,yq,exx,eyy,exy,p,T):
    #-------------------
    A=3.1623e-26
    nnn=3.3
    Q=186.5e3
    Rgas=8.314
    eta_min=1.e19
    eta_max=1.e25
    phi=31./180.*np.pi #(atan(0.6) in paper)
    c=5e7
    E2=np.sqrt( 0.5*(exx**2+eyy**2)+exy**2 )
    #-------------------
    if iter==0:
       E2=1e-15
    #E2=max(1e-25,E2)

    eta_v = A**(-1./nnn)*np.exp(Q/nnn/Rgas/T)*E2**(1./nnn-1.)
    sigmayield= p*np.sin(phi) + c*np.cos(phi) 
    etaeff = sigmayield /(2.*E2)+ eta_vp
    etaeff=1./(1./eta_v+1./etaeff)
    if (xq-Lx/2)**2+(yq-Ly/2)**2 < 4e6:
       etaeff=1e20
    etaeff=min(etaeff,eta_max)
    etaeff=max(etaeff,eta_min)

    #if iter==0:
    #   etaeff=1e20
    return etaeff

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
   niter= 10
   bc   = +1
   vp   = 0
    
eta_vp=10.**vp

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

#################################################################
#################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("NV=",NV)
print("niter=",niter)
print("eta_vp=",eta_vp)
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
    T[i]=20+(Ly-yV[i])*15/1e3+ 273.15

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
sol    = np.zeros(Nfem,dtype=np.float64)         # solution vector 
    
c_mat   = np.array([[2,0,0],\
                    [0,2,0],\
                    [0,0,1]],dtype=np.float64) 
    
convfile=open('conv.ascii',"w")

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

                etaq=eta(xq,yq,exxq,eyyq,exyq,pq,Tq)

                rhoq=rho(xq,yq,Tq)

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

    print("build FE matrix: %.5f s" % (time.time()-start))

    ######################################################################
    # solve system
    ######################################################################
    start = time.time()

    #a_mat[Nfem,NfemV:Nfem]=constr
    #a_mat[NfemV:Nfem,Nfem]=constr

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
    #normalise pressure
    #pressure is Q1 so I need a single quad point in the middle of the 
    #face to carry out integration on edge.
    ######################################################################
    start = time.time()

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

    xi_u=np.linalg.norm(u-u_old,2)/np.linalg.norm(u+u_old,2)
    xi_v=np.linalg.norm(v-v_old,2)/np.linalg.norm(v+v_old,2)
    xi_p=np.linalg.norm(p-p_old,2)/np.linalg.norm(p+p_old,2)

    print("conv: u,v,p: %.6f %.6f %.6f" %(xi_u,xi_v,xi_p))

    convfile.write("%3d %10e %10e %10e %10e\n" %(iter,xi_u,xi_v,xi_p,Res_two/Res0_two)) 
    convfile.flush()

    u_old=u
    v_old=v
    p_old=p


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
count = np.zeros(NV,dtype=np.int16)  
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
pfile=open('profile.ascii',"w")

for i in range(0,NV):
    if abs(yV[i]-2*Ly/3)<eps*Ly:
       pfile.write("%10e %10e %10e %10e %10e \n" %(xV[i],q[i],exx_n[i],eyy_n[i],exy_n[i]))

#####################################################################

eta_n = np.zeros(NV,dtype=np.float64)  
tau_n = np.zeros(NV,dtype=np.float64)  

for i in range(0,NV):
    eta_n[i]=eta(xV[i],yV[i],exx_n[i],eyy_n[i],exy_n[i],q[i],T[i])
    e=np.sqrt(0.5*(exx_n[i]*exx_n[i]+eyy_n[i]*eyy_n[i])+exy_n[i]*exy_n[i])
    tau_n[i]=2.*eta_n[i]*e

np.savetxt('tau.ascii',np.array([xV,yV,tau_n]).T,header='# x,y,tau')

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

if visu:

   #np.savetxt('pressure.ascii',np.array([xP,yP,p]).T,header='# x,y,p')

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
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % eta_el[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='rho' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % rho_el[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % exx[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % eyy[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % exy[iel])
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2) )
   #vtufile.write("</DataArray>\n")
   #--
   #vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
   #for iel in range (0,nel):
   #    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
   #vtufile.write("</DataArray>\n")
   #vtufile.write("</CellData>\n")
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
   vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %eta_n[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='tau' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e \n" %tau_n[i])
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

#==============================================================================
# end time stepping loop


print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
