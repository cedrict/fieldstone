import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time
#------------------------------------------------------------------------------
def gx(x,y):
    return 0
def gy(x,y):
    return -grav
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
print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
Lx=50e3  # horizontal extent of the domain 
Ly=10e3  # vertical extent of the domain 
# allowing for argument parsing through command line
if int(len(sys.argv) == 7):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 125
   nely = 25
   visu = 1
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
nq=9*nel
NfemV=nnp*ndofV               # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs
eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]
hx=Lx/nelx
hy=Ly/nely
pnormalise=True
eta_ref=1e21      # scaling of G blocks
grav=9.81

#material 1: eta_salt=1e17, rho_salt=2200
#material 2: eta_sed=1e21, #rho_sed=2600
rho_mat = np.array([2200,2600],dtype=np.float64) 
eta_mat = np.array([1e17,1.e20],dtype=np.float64) 
salt_thickness=2e3
amplitude=500

avrg=3
nstep_change=25
nstep=50
CFL_nb=0.5
nmarker_per_dim=10
rk=2

nmarker_per_element=nmarker_per_dim**2
nmarker=nmarker_per_element*nel
year=3600.*24.*365.
#################################################################
#################################################################
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnp=",nnp)
print("------------------------------")
#################################################################
# grid point setup
#################################################################
start = time.time()
x=np.empty(nnp,dtype=np.float64)  # x coordinates
y=np.empty(nnp,dtype=np.float64)  # y coordinates
counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx/2.
        y[counter]=j*hy/2.
        counter += 1
np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')
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
# marker setup
#################################################################
swarm_x=np.empty(nmarker,dtype=np.float64)  # x coordinates
swarm_y=np.empty(nmarker,dtype=np.float64)  # y coordinates
swarm_mat=np.empty(nmarker,dtype=np.int16)  # material id 
swarm_u=np.empty(nmarker,dtype=np.float64)  # x velocity
swarm_v=np.empty(nmarker,dtype=np.float64)  # y velocity
swarm_paint=np.empty(nmarker,dtype=np.float64)  # paint
swarm_x0=np.empty(nmarker,dtype=np.float64)  # x coordinates
swarm_y0=np.empty(nmarker,dtype=np.float64)  # y coordinates
counter=0
for iel in range(0,nel):
    x1=x[iconV[0,iel]] ; y1=y[iconV[0,iel]]
    x2=x[iconV[1,iel]] ; y2=y[iconV[1,iel]]
    x3=x[iconV[2,iel]] ; y3=y[iconV[2,iel]]
    x4=x[iconV[3,iel]] ; y4=y[iconV[3,iel]]
    for j in range(0,nmarker_per_dim):
        for i in range(0,nmarker_per_dim):
            r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
            s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
            N1=0.25*(1-r)*(1-s)
            N2=0.25*(1+r)*(1-s)
            N3=0.25*(1+r)*(1+s)
            N4=0.25*(1-r)*(1+s)
            swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
            swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
            counter+=1

swarm_x0[0:nmarker]=swarm_x[0:nmarker]
swarm_y0[0:nmarker]=swarm_y[0:nmarker]

print("     -> swarm_x (m,M) %.4f %.4f " %(np.min(swarm_x),np.max(swarm_x)))
print("     -> swarm_y (m,M) %.4f %.4f " %(np.min(swarm_y),np.max(swarm_y)))
#################################################################
# assign material id to markers 
#################################################################
for im in range (0,nmarker):
    if swarm_y[im]>salt_thickness-amplitude*np.cos(2*np.pi*swarm_x[im]/Lx):
       swarm_mat[im]=2
    else:
       swarm_mat[im]=1
#################################################################
# paint markers 
#################################################################
for im in range (0,nmarker):
    swarm_paint[im]=(np.sin(2*np.pi*swarm_x[im]/Lx*20)*\
                     np.sin(2*np.pi*swarm_y[im]/Ly*4))

np.savetxt('markers.ascii',np.array([swarm_x,swarm_y,swarm_mat]).T,header='# x,y,mat')
#################################################################
# define boundary conditions
#################################################################
start = time.time()
bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0, nnp):
    if x[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
       #bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # no slip
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]/Ly>(1-eps):
       #bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0. # free slip
print("setup: boundary conditions: %.3f s" % (time.time() - start))
#--------------------------------------------------------------------------------------------------
# time stepping loop
#--------------------------------------------------------------------------------------------------
for istep in range(0,nstep):
    print ('----------------------------------------istep= %i' %istep)
    #################################################################
    # compute elemental averagings 
    #################################################################
    start = time.time()
    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    nmarker_in_element=np.zeros(nel,dtype=np.float64) 
    for im in range(0,nmarker):
        #compute ielx
        ielx=int(swarm_x[im]/Lx*nelx)
        if ielx<0:
           print ('ielx<0',ielx)
        if ielx>nelx-1:
           print ('ielx>nelx-1')
        #compute iely
        iely=int(swarm_y[im]/Ly*nely)
        if iely<0:
           print ('iely<0')
        if iely>nely-1:
           print ('iely>nely-1')
        #compute element id
        iel=nelx*(iely)+ielx
        if iel<0:
           print ('iel<0')
        if iel>nel-1:
           print ('iel>nel-1')
        rho_elemental[iel]+=rho_mat[swarm_mat[im]-1]
        nmarker_in_element[iel]+=1
        if avrg==1: # arithmetic
           eta_elemental[iel]+=eta_mat[swarm_mat[im]-1]
        if avrg==2: # geometric
           eta_elemental[iel]+=math.log(eta_mat[swarm_mat[im]-1],10)
        if avrg==3: # harmonic
           eta_elemental[iel]+=1./eta_mat[swarm_mat[im]-1]
    for iel in range(0,nel):
        rho_elemental[iel]/=nmarker_in_element[iel]
        if avrg==1:
            eta_elemental[iel]/=nmarker_in_element[iel]
        if avrg==2:
            eta_elemental[iel]=10.**(eta_elemental[iel]/nmarker_in_element[iel])
        if avrg==3:
            eta_elemental[iel]=nmarker_in_element[iel]/eta_elemental[iel]
    print("-> nmarker_in_elt(m,M) %.5e %.5e " %(np.min(nmarker_in_element),np.max(nmarker_in_element)))
    print("-> rho_elemental (m,M) %.5e %.5e " %(np.min(rho_elemental),np.max(rho_elemental)))
    print("-> eta_elemental (m,M) %.5e %.5e " %(np.min(eta_elemental),np.max(eta_elemental)))
    print("projection elemental: %.3f s" % (time.time() - start))
    #################################################################
    # build FE matrix
    # [ K G ][u]=[f]
    # [GT 0 ][p] [h]
    #################################################################
    start = time.time()
    K_mat = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
    G_mat = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
    f_rhs = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
    h_rhs = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
    constr= np.zeros(NfemP,dtype=np.float64)         # constraint matrix/vector
    b_mat = np.zeros((3,ndofV*mV),dtype=np.float64) # gradient matrix B 
    N_mat = np.zeros((3,ndofP*mP),dtype=np.float64) # matrix  
    NV    = np.zeros(mV,dtype=np.float64)           # shape functions V
    NP    = np.zeros(mP,dtype=np.float64)           # shape functions P
    dNVdx  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNVdy  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNVdr  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    dNVds  = np.zeros(mV,dtype=np.float64)          # shape functions derivatives
    u     = np.zeros(nnp,dtype=np.float64)          # x-component velocity
    v     = np.zeros(nnp,dtype=np.float64)          # y-component velocity
    c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
    xq   = np.zeros(nq,dtype=np.float64)          # 
    yq   = np.zeros(nq,dtype=np.float64)          # 
    rhoq = np.zeros(nq,dtype=np.float64)          # 
    etaq = np.zeros(nq,dtype=np.float64)          # 
    counter=0
    for iel in range(0,nel):
        # set arrays to 0 every loop
        f_el =np.zeros((mV*ndofV),dtype=np.float64)
        K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
        G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
        h_el=np.zeros((mP*ndofP),dtype=np.float64)
        NNNP= np.zeros(mP*ndofP,dtype=np.float64)   
        # integrate viscous term at 4 quadrature points
        for iq in [0,1,2]:
            for jq in [0,1,2]:
                # position & weight of quad. point
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NV[0:9]=NNV(rq,sq)
                dNVdr[0:9]=dNNVdr(rq,sq)
                dNVds[0:9]=dNNVds(rq,sq)
                NP[0:4]=NNP(rq,sq)
                # calculate jacobian matrix
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNVdr[k]*x[iconV[k,iel]]
                    jcb[0,1] += dNVdr[k]*y[iconV[k,iel]]
                    jcb[1,0] += dNVds[k]*x[iconV[k,iel]]
                    jcb[1,1] += dNVds[k]*y[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                jcbi = np.linalg.inv(jcb)
                # compute dNdx & dNdy
                for k in range(0,mV):
                    xq[counter]+=NV[k]*x[iconV[k,iel]]
                    yq[counter]+=NV[k]*y[iconV[k,iel]]
                    dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                    dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]
                #end for
                rhoq[counter]=rho_elemental[iel]
                etaq[counter]=eta_elemental[iel]
                # construct 3x8 b_mat matrix
                for i in range(0,mV):
                    b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.     ],
                                             [0.      ,dNVdy[i]],
                                             [dNVdy[i],dNVdx[i]]]
                # compute elemental a_mat matrix
                K_el+=b_mat.T.dot(c_mat.dot(b_mat))*etaq[counter]*weightq*jcob
                # compute elemental rhs vector
                for i in range(0,mV):
                    f_el[ndofV*i  ]+=NV[i]*jcob*weightq*gx(xq[counter],yq[counter])*rhoq[counter]
                    f_el[ndofV*i+1]+=NV[i]*jcob*weightq*gy(xq[counter],yq[counter])*rhoq[counter]
                for i in range(0,mP):
                    N_mat[0,i]=NP[i]
                    N_mat[1,i]=NP[i]
                    N_mat[2,i]=0.
                G_el-=b_mat.T.dot(N_mat)*weightq*jcob
                NNNP[:]+=NP[:]*jcob*weightq
                counter+=1
            #end for
        #end for

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
                        K_mat[m1,m2]+=K_el[ikk,jkk]
                for k2 in range(0,mP):
                    jkk=k2
                    m2 =iconP[k2,iel]
                    G_mat[m1,m2]+=G_el[ikk,jkk]
                f_rhs[m1]+=f_el[ikk]
        for k2 in range(0,mP):
            m2=iconP[k2,iel]
            h_rhs[m2]+=h_el[k2]
            constr[m2]+=NNNP[k2]

    G_mat*=eta_ref/Ly
    h_rhs*=eta_ref/Ly

    print("     -> K (m,M) %.5e %.5e " %(np.min(K_mat),np.max(K_mat)))
    print("     -> G (m,M) %.5e %.5e " %(np.min(G_mat),np.max(G_mat)))
    print("     -> f (m,M) %.5e %.5e " %(np.min(f_rhs),np.max(f_rhs)))
    print("     -> h (m,M) %.5e %.5e " %(np.min(h_rhs),np.max(h_rhs)))

    np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
    np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')
    print("build FE matrix: %.3f s" % (time.time() - start))
    ######################################################################
    # assemble K, G, GT, f, h into A and rhs
    ######################################################################
    start = time.time()
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
    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs
    print("assemble blocks: %.3f s" % (time.time() - start))
    ######################################################################
    # solve system
    ######################################################################
    start = time.time()
    sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
    print("solve time: %.3f s" % (time.time() - start))
    ######################################################################
    # put solution into separate x,y velocity arrays
    ######################################################################
    start = time.time()
    u,v=np.reshape(sol[0:NfemV],(nnp,2)).T
    p=sol[NfemV:Nfem]*(eta_ref/Ly)
    print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
    print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))
    np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
    print("split vel into u,v: %.3f s" % (time.time() - start))

    ######################################################################
    # compute timestep 
    ######################################################################

    dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))

    print("dt= %.3e yr" %(dt/year))

    ######################################################################
    # advect markers 
    ######################################################################
    if istep<nstep_change:
       sign=+1
    else:
       sign=-1

    if rk==1:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NV[0:mV]= NNV(r,s)
           swarm_u[im]=sum(NV[0:mV]*u[iconV[0:mV,iel]])
           swarm_v[im]=sum(NV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=swarm_u[im]*dt*sign
           swarm_y[im]+=swarm_v[im]*dt*sign

    elif rk==2:
       for im in range(0,nmarker):
           ielx=int(swarm_x[im]/Lx*nelx)
           iely=int(swarm_y[im]/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(swarm_x[im]-x0)/hx
           s=-1+2*(swarm_y[im]-y0)/hy
           NV[0:mV]= NNV(r,s)
           um=sum(NV[0:mV]*u[iconV[0:mV,iel]])
           vm=sum(NV[0:mV]*v[iconV[0:mV,iel]])
           xm=swarm_x[im]+um*dt/2*sign
           ym=swarm_y[im]+vm*dt/2*sign

           ielx=int(xm/Lx*nelx)
           iely=int(ym/Ly*nely)
           iel=nelx*(iely)+ielx
           x0=x[iconV[0,iel]]
           y0=y[iconV[0,iel]]
           r=-1+2*(xm-x0)/hx
           s=-1+2*(ym-y0)/hy
           NV[0:mV]= NNV(r,s)
           swarm_u[im]=sum(NV[0:mV]*u[iconV[0:mV,iel]])
           swarm_v[im]=sum(NV[0:mV]*v[iconV[0:mV,iel]])
           swarm_x[im]+=swarm_u[im]*dt*sign
           swarm_y[im]+=swarm_v[im]*dt*sign

    ######################################################################
    # compute strainrate 
    ######################################################################

    start = time.time()

    xc = np.zeros(nel,dtype=np.float64)
    yc = np.zeros(nel,dtype=np.float64)
    exx = np.zeros(nel,dtype=np.float64)
    eyy = np.zeros(nel,dtype=np.float64)
    exy = np.zeros(nel,dtype=np.float64)
    e   = np.zeros(nel,dtype=np.float64)
    rho = np.zeros(nel,dtype=np.float64)

    for iel in range(0,nel):

        rq = 0.0
        sq = 0.0
        weightq = 2.0 * 2.0

        NV[0:9]=NNV(rq,sq)
        dNVdr[0:9]=dNNVdr(rq,sq)
        dNVds[0:9]=dNNVds(rq,sq)

        jcb=np.zeros((2,2),dtype=np.float64)
        for k in range(0,mV):
            jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
            jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
            jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
            jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
        jcob=np.linalg.det(jcb)
        jcbi=np.linalg.inv(jcb)

        for k in range(0,mV):
            dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
            dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

        for k in range(0,mV):
            xc[iel] += NV[k]*x[iconV[k,iel]]
            yc[iel] += NV[k]*y[iconV[k,iel]]
            exx[iel] += dNVdx[k]*u[iconV[k,iel]]
            eyy[iel] += dNVdy[k]*v[iconV[k,iel]]
            exy[iel] += 0.5*dNVdy[k]*u[iconV[k,iel]]+ 0.5*dNVdx[k]*v[iconV[k,iel]]

        e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

    print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
    print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
    print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

    np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

    print("compute press & sr: %.3f s" % (time.time() - start))


    #####################################################################
    # interpolate pressure onto velocity grid points
    #####################################################################
    q=np.zeros(nnp,dtype=np.float64)

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

    np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')
    #####################################################################
    # plot of solution
    #####################################################################
    # the 9-node Q2 element does not exist in vtk, but the 8-node one 
    # does, i.e. type=23. 
    filename = 'solution_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<CellData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (rho_elemental[iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%10e\n" % (eta_elemental[iel]))
    vtufile.write("</DataArray>\n")
    vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e %10e %10e \n" %(u[i]*year,v[i]*year,0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10e \n" %q[i])
    vtufile.write("</DataArray>\n")
    #--
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

    filename = 'markers_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_mat[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_paint[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='velocity' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%5e %5e %5e \n" %(swarm_u[i],swarm_v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='displacement' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%5e %5e %5e \n" %(swarm_x[i]-swarm_x0[i],swarm_y[i]-swarm_y0[i],0.))
    vtufile.write("</DataArray>\n")

    vtufile.write("</PointData>\n")
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nmarker):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nmarker):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()







#--------------------------------------------------------------------------------------------------
# end time stepping loop
#--------------------------------------------------------------------------------------------------
print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
