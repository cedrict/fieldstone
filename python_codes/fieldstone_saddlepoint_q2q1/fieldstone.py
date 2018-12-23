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

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
    return val

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
   nelx = 10
   nely = 10
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
sqrt3=np.sqrt(3.)
qcoords=[-sqrt(3./5.),0.,+sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

hx=Lx/nelx
hy=Ly/nely

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

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter]=(i-1)*2+1+(j-1)*2*nnx
        icon[1,counter]=(i-1)*2+2+(j-1)*2*nnx
        icon[2,counter]=(i-1)*2+3+(j-1)*2*nnx
        icon[3,counter]=(i-1)*2+1+(j-1)*2*nnx+nnx
        icon[4,counter]=(i-1)*2+2+(j-1)*2*nnx+nnx
        icon[5,counter]=(i-1)*2+3+(j-1)*2*nnx+nnx
        icon[6,counter]=(i-1)*2+1+(j-1)*2*nnx+nnx*2
        icon[7,counter]=(i-1)*2+2+(j-1)*2*nnx+nnx*2
        icon[8,counter]=(i-1)*2+3+(j-1)*2*nnx+nnx*2
        counter += 1

for iel in range (0,nel):
    print ("iel=",iel)
    print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel]], y[icon[0][iel]])
    print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel]], y[icon[1][iel]])
    print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel]], y[icon[2][iel]])
    print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel]], y[icon[3][iel]])
    print ("node 2",icon[4][iel],"at pos.",x[icon[4][iel]], y[icon[4][iel]])
    print ("node 2",icon[5][iel],"at pos.",x[icon[5][iel]], y[icon[5][iel]])
    print ("node 2",icon[6][iel],"at pos.",x[icon[6][iel]], y[icon[6][iel]])
    print ("node 2",icon[7][iel],"at pos.",x[icon[7][iel]], y[icon[7][iel]])
    print ("node 2",icon[8][iel],"at pos.",x[icon[8][iel]], y[icon[8][iel]])

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
p     = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
    h_el=np.zeros((mP*ndofP,mP*ndofP),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [0,1,2]:
        for jq in [0,1,2]:

            # position & weight of quad. point
            rq=qcoords[iq]
            sq=qcoords[jq]
            wq=qweights[iq]*qweights[jq]

            # calculate shape functions
            N[0]= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
            N[1]=     (1.-rq**2) * 0.5*sq*(sq-1.)
            N[2]= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
            N[3]= 0.5*rq*(rq-1.) *     (1.-sq**2)
            N[4]=     (1.-rq**2) *     (1.-sq**2)
            N[5]= 0.5*rq*(rq+1.) *     (1.-sq**2)
            N[6]= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
            N[7]=     (1.-rq**2) * 0.5*sq*(sq+1.)
            N[8]= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)

            # calculate shape function derivatives
            dNdr[0]= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
            dNdr[1]=       (-2.*rq) * 0.5*sq*(sq-1)
            dNdr[2]= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
            dNdr[3]= 0.5*(2.*rq-1.) *    (1.-sq**2)
            dNdr[4]=       (-2.*rq) *    (1.-sq**2)
            dNdr[5]= 0.5*(2.*rq+1.) *    (1.-sq**2)
            dNdr[6]= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
            dNdr[7]=       (-2.*rq) * 0.5*sq*(sq+1)
            dNdr[8]= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)

            dNds[0]= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
            dNds[1]=     (1.-rq**2) * 0.5*(2.*sq-1.)
            dNds[2]= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
            dNds[3]= 0.5*rq*(rq-1.) *       (-2.*sq)
            dNds[4]=     (1.-rq**2) *       (-2.*sq)
            dNds[5]= 0.5*rq*(rq+1.) *       (-2.*sq)
            dNds[6]= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
            dNds[7]=     (1.-rq**2) * 0.5*(2.*sq+1.)
            dNds[8]= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)

            NP[0]=0.25*(1-rq)*(1-st)
            NP[1]=0.25*(1+rq)*(1-st)
            NP[2]=0.25*(1+rq)*(1+st)
            NP[3]=0.25*(1-rq)*(1+st)

            # calculate jacobian matrix
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,mV):
                jcb[0,0] += dNdr[k]*x[icon[k,iel]]
                jcb[0,1] += dNdr[k]*y[icon[k,iel]]
                jcb[1,0] += dNds[k]*x[icon[k,iel]]
                jcb[1,1] += dNds[k]*y[icon[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*wq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=N[i]*jcob*wq*bx(xq,yq)
                f_el[ndofV*i+1]+=N[i]*jcob*wq*by(xq,yq)

            for i in range(0,mP):
                N_mat[0,i]=NP[i]
                N_mat[1,i]=NP[i]
                N_mat[2,i]=0.

            G_el-=b_mat.T.dot(N_mat)*wq*jcob

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*icon[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
    h_rhs[iel]+=h_el[0]

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start = time.time()

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
    wq = 2.0 * 2.0

    N[0]= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N[1]=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N[2]= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N[3]= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N[4]=     (1.-rq**2) *     (1.-sq**2)
    N[5]= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N[6]= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N[7]=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N[8]= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)

    dNdr[0]= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr[1]=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr[2]= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr[3]= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr[4]=       (-2.*rq) *    (1.-sq**2)
    dNdr[5]= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr[6]= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr[7]=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr[8]= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)

    dNds[0]= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds[1]=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds[2]= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds[3]= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds[4]=     (1.-rq**2) *       (-2.*sq)
    dNds[5]= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds[6]= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds[7]=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds[8]= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0,mV):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]

    for k in range(0,mV):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute error
######################################################################
start = time.time()

error_u = np.empty(nnp,dtype=np.float64)
error_v = np.empty(nnp,dtype=np.float64)
error_q = np.empty(nnp,dtype=np.float64)
error_p = np.empty(nel,dtype=np.float64)

for i in range(0,nnp): 
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
    error_q[i]=q[i]-pressure(x[i],y[i])

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i])

errv=0.
errp=0.
for iel in range (0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            wq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*wq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*wq*jcob

errv=np.sqrt(errv)
errp=np.sqrt(errp)


print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution
#####################################################################


#if visu==1:
#   plt.savefig('solution.pdf', bbox_inches='tight')
#   plt.show()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
