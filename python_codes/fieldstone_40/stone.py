import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time

#------------------------------------------------------------------------------

def gx(x,y):
    return 0

def gy(x,y):
    return -grav

#------------------------------------------------------------------------------

def density(x,y):
    if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
       val=3300
    else:
       val=3000
    return val

def viscosity(x,y,eta1,eta2):
    if y>256e3+amplitude*np.cos(2*np.pi*x/llambda):
       val=eta1
    else:
       val=eta2
    return val

#------------------------------------------------------------------------------

def vy_th(phi1,phi2,rho1,rho2):
    c11 = (eta1*2*phi1**2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        - (2*phi2**2)/(np.cosh(2*phi2)-1-2*phi2**2)
    d12 = (eta1*(np.sinh(2*phi1) -2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        + (np.sinh(2*phi2)-2*phi2)/(np.cosh(2*phi2)-1-2*phi2**2)
    i21 = (eta1*phi2*(np.sinh(2*phi1)+2*phi1))/(eta2*(np.cosh(2*phi1)-1-2*phi1**2)) \
        + (phi2*(np.sinh(2*phi2)+2*phi2))/(np.cosh(2*phi2)-1-2*phi2**2) 
    j22 = (eta1*2*phi1**2*phi2)/(eta2*(np.cosh(2*phi1)-1-2*phi1**2))\
        - (2*phi2**3)/(np.cosh(2*phi2)-1-2*phi2**2)
    K=-d12/(c11*j22-d12*i21)
    val=K*(rho1-rho2)/2/eta2*(Ly/2.)*grav*amplitude
    return val

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

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=9     # number of velocity nodes making up an element
mP=4     # number of pressure nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=512e3  # horizontal extent of the domain 
Ly=512e3  # vertical extent of the domain 

if int(len(sys.argv) == 7):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   llambda = float(sys.argv[4])
   amplitude = float(sys.argv[5])
   eta2 = float(sys.argv[6])
else:
   nelx = 24
   nely = 24
   visu = 1
   llambda=256e3
   amplitude=4000
   eta2=1e21

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

grav=10
phi1=2.*np.pi*(Ly/2.)/llambda
phi2=2.*np.pi*(Ly/2.)/llambda
eta1=1e21
rho1=3300
rho2=3000

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

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# deform grid 
#################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
#################################################################

for i in range(0, nnp):
    if abs(y[i]-Ly/2.)/Ly<eps:
       y[i]+=amplitude*np.cos(2*np.pi*x[i]/llambda)
    #end if
#end for

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        ya=256e3+amplitude*np.cos(2*np.pi*x[counter]/llambda)
        if j<nny/2:
           dy=ya/(nely/2)/2
           y[counter]=j*dy
        else:
           dy=(Ly-ya)/(nely/2)/2
           y[counter]=ya+(j-2*nely/2)*dy
        counter+=1

for iel in range(0,nel):
    y[iconV[7,iel]]=(y[iconV[0,iel]]+y[iconV[3,iel]])/2.
    y[iconV[5,iel]]=(y[iconV[1,iel]]+y[iconV[2,iel]])/2.
    y[iconV[8,iel]]=(y[iconV[4,iel]]+y[iconV[6,iel]])/2.
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0, nnp):
    if x[i]/Lx<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
    if x[i]/Lx>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # free slip
    if y[i]/Ly<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # no slip
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]/Ly>(1-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0. # no slip
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

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

            rhoq[counter]=density(xq[counter],yq[counter])
            etaq[counter]=viscosity(xq[counter],yq[counter],eta1,eta2)

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

#end for iel

G_mat*=eta_ref/Ly
h_rhs*=eta_ref/Ly

#np.savetxt('rhoq.ascii',np.array([xq,yq,rhoq]).T,header='# x,y,rho')
#np.savetxt('etaq.ascii',np.array([xq,yq,etaq]).T,header='# x,y,eta')

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

print("     -> vy/ampl, phi1 %.5e %.5e %.5e" %(np.max(abs(v)),phi1,vy_th(phi1,phi2,rho1,rho2)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (time.time() - start))

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

    rho[iel]=density(xc[iel],yc[iel])
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

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

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

#####################################################################
# plot of solution
#####################################################################
# the 9-node Q2 element does not exist in vtk, but the 8-node one 
# does, i.e. type=23. 

filename = 'solution.vtu'
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
    vtufile.write("%10e\n" % (rho[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
