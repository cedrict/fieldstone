import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import time as time

#------------------------------------------------------------------------------
def bx(x, y):
    val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
         (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
         (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
         1.-4.*y+12.*y*y-8.*y*y*y)
    return val

def by(x, y):
    val=((8.-48.*y+48.*y*y)*x*x*x+
         (-12.+72.*y-72.*y*y)*x*x+
         (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
         12.*y*y+24.*y*y*y-12.*y**4)
    return val

def uth(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def vth(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pth(x,y):
    val=x*(1.-x)-1./6.
    return val

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

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

assert (Lx>0.), "Lx should be positive" 
assert (Ly>0.), "Ly should be positive" 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 24
   nely = 24
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

nnp=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

viscosity=1.  # dynamic viscosity \mu

NfemV=nnp*ndofV               # number of velocity dofs
NfemP=(nelx+1)*(nely+1)*ndofP # number of pressure dofs
Nfem=NfemV+NfemP              # total number of dofs

eps=1.e-10
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

pnormalise=True

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

#for iel in range (0,nel):
#    print ("iel=",iel)
#    print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#    print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#    print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#    print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])
#    print ("node 2",icon[4][iel],"at pos.",x[icon[4][iel]], y[icon[4][iel]])
#    print ("node 2",icon[5][iel],"at pos.",x[icon[5][iel]], y[icon[5][iel]])
#    print ("node 2",icon[6][iel],"at pos.",x[icon[6][iel]], y[icon[6][iel]])
#    print ("node 2",icon[7][iel],"at pos.",x[icon[7][iel]], y[icon[7][iel]])
#    print ("node 2",icon[8][iel],"at pos.",x[icon[8][iel]], y[icon[8][iel]])

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconP[0,counter]=i+j*(nelx+1)
        iconP[1,counter]=i+1+j*(nelx+1)
        iconP[2,counter]=i+1+(j+1)*(nelx+1)
        iconP[3,counter]=i+(j+1)*(nelx+1)
        counter += 1

# for iel in range (0,nel):
#     print ("iel=",iel)
#     print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
#     print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
#     print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
#     print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0, nnp):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

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
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                dNVdx[k]=jcbi[0,0]*dNVdr[k]+jcbi[0,1]*dNVds[k]
                dNVdy[k]=jcbi[1,0]*dNVdr[k]+jcbi[1,1]*dNVds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNVdx[i],0.     ],
                                         [0.      ,dNVdy[i]],
                                         [dNVdy[i],dNVdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NV[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=NV[i]*jcob*weightq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NP[i]
                N_mat[1,i]=NP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*weightq*jcob

            NNNP[:]+=NP[:]*jcob*weightq

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
p=sol[NfemV:Nfem]

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

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

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute error
######################################################################
start = time.time()

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            NV[0:9]=NNV(rq,sq)
            dNVdr[0:9]=dNNVdr(rq,sq)
            dNVds[0:9]=dNNVds(rq,sq)
            NP[0:4]=NNP(rq,sq)

            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNVdr[k]*x[iconV[k,iel]]
                jcb[0,1]+=dNVdr[k]*y[iconV[k,iel]]
                jcb[1,0]+=dNVds[k]*x[iconV[k,iel]]
                jcb[1,1]+=dNVds[k]*y[iconV[k,iel]]
            jcob=np.linalg.det(jcb)

            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=NV[k]*x[iconV[k,iel]]
                yq+=NV[k]*y[iconV[k,iel]]
                uq+=NV[k]*u[iconV[k,iel]]
                vq+=NV[k]*v[iconV[k,iel]]
            errv+=((uq-uth(xq,yq))**2+\
                   (vq-vth(xq,yq))**2)*weightq*jcob

            pq=0.0
            for k in range(0,mP):
                pq+=NP[k]*p[iconP[k,iel]]
            errp+=(pq-pth(xq,yq))**2*weightq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

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
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" %pth(x[i],y[i]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
