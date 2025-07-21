import numpy as np
import sys as sys
import time as time
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

#------------------------------------------------------------------------------
# bx and by are the body force components

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

#------------------------------------------------------------------------------
# analytical solution

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
print("--------- stone 176 ---------")
print("-----------------------------")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 16
   nely = 16
   visu = 1
    
nnx=nelx+1    # number of elements, x direction
nny=nely+1    # number of elements, y direction
NV=nnx*nny    # number of nodes
nel=nelx*nely # number of elements, total
Nfem=NV*ndof  # Total number of degrees of freedom

viscosity=1.  # dynamic viscosity \eta
penalty=1.e7  # penalty coefficient value

eps=1.e-10
sqrt3=np.sqrt(3.)

new_assembly=False

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# build connectivity array
#################################################################
start = time.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter]= i + j * (nelx + 1)
        icon[1,counter]= i + 1 + j * (nelx + 1)
        icon[2,counter]= i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter]= i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
# for this benchmark: no slip. 
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if x[i]>(Lx-eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if y[i]<eps:
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
    if y[i]>(Ly-eps):
       bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
       bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# r,s are the reduced coordinates in the [-1:1]x[-1:1] ref elt
#################################################################
start = time.time()

b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
jcb   = np.zeros((2,2),dtype=np.float64)
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
dofs  = np.zeros(ndof*m,dtype=np.int32) 

if new_assembly:
   row = [] 
   col = []
   a_mat = []
else:
   a_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)

time_ass=0.

for iel in range(0,nel):

    a_el=np.zeros((m*ndof,m*ndof),dtype=np.float64)
    b_el=np.zeros(m*ndof,dtype=np.float64)

    for k in range(0,m):
        dofs[k*ndof+0]=icon[k,iel]*ndof+0
        dofs[k*ndof+1]=icon[k,iel]*ndof+1

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]= dNds[:].dot(y[icon[:,iel]])

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])
            dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
            dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            a_el+=b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*weightq*bx(xq,yq)
                b_el[2*i+1]+=N[i]*jcob*weightq*by(xq,yq)

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
    jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]= dNds[:].dot(y[icon[:,iel]])
    jcob = np.linalg.det(jcb)
    jcbi = np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]

    # compute elemental matrix
    a_el+=b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

    # apply boundary conditions
    for k1 in range(0,m):
        for i1 in range(0,ndof):
            m1 =ndof*icon[k1,iel]+i1
            if bc_fix[m1]: 
               fixt=bc_val[m1]
               ikk=ndof*k1+i1
               aref=a_el[ikk,ikk]
               for jkk in range(0,m*ndof):
                   b_el[jkk]-=a_el[jkk,ikk]*fixt
                   a_el[ikk,jkk]=0.
                   a_el[jkk,ikk]=0.
               #end for
               a_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix a_mat and right hand side rhs
    start2=time.time()
    if new_assembly:
        for i_local,idof in enumerate(dofs):
            for j_local,jdof in enumerate(dofs):
                row.append(idof)
                col.append(jdof)
                a_mat.append(a_el[i_local,j_local])
            #end for
            rhs[idof]+=b_el[i_local]
        #end for
    else:
       for k1 in range(0,m):
           for i1 in range(0,ndof):
               ikk=ndof*k1          +i1
               m1 =ndof*icon[k1,iel]+i1
               for k2 in range(0,m):
                   for i2 in range(0,ndof):
                       jkk=ndof*k2          +i2
                       m2 =ndof*icon[k2,iel]+i2
                       a_mat[m1,m2]+=a_el[ikk,jkk]
                   #end for
               #end for
               rhs[m1]+=b_el[ikk]
           #end for
       #end for
    #end if
    time_ass+=time.time()-start2
#end for iel

print('     -> time assembly=',time_ass,Nfem)

start3=time.time()
if new_assembly:
   A=sps.csr_matrix((a_mat,(row,col)),shape=(Nfem,Nfem))
else:
   A=sps.csr_matrix(a_mat)

print("Convert to csr format: %.5f s | Nfem= %d " % (time.time() - start3, Nfem))

print("Build FE matrix: %.5f s | Nfem= %d" % (time.time() - start,Nfem))

###############################################################################
# solve system
###############################################################################
start = time.time()

sol=sps.linalg.spsolve(A,rhs)

print("Solve linear system: %.5f s | Nfem= %d " % (time.time() - start, Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))

print("split vel into u,v: %.6f s | Nfem %d " % (time.time()-start,Nfem))

#####################################################################
# we compute the pressure and strain rate components in the middle 
# of the elements.
#####################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
p  = np.zeros(nel,dtype=np.float64)  
exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq = 0.0
    sq = 0.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb[0,0]= dNdr[:].dot(x[icon[:,iel]])
    jcb[0,1]= dNdr[:].dot(y[icon[:,iel]])
    jcb[1,0]= dNds[:].dot(x[icon[:,iel]])
    jcb[1,1]= dNds[:].dot(y[icon[:,iel]])
    jcbi=np.linalg.inv(jcb)

    dNdx[:]=jcbi[0,0]*dNdr[:]+jcbi[0,1]*dNds[:]
    dNdy[:]=jcbi[1,0]*dNdr[:]+jcbi[1,1]*dNds[:]

    xc[iel]=N[:].dot(x[icon[:,iel]])
    yc[iel]=N[:].dot(y[icon[:,iel]])

    exx[iel]=dNdx[:].dot(u[icon[:,iel]])
    eyy[iel]=dNdy[:].dot(v[icon[:,iel]])
    exy[iel]=0.5*dNdy[:].dot(u[icon[:,iel]])+\
             0.5*dNdx[:].dot(v[icon[:,iel]])
    p[iel]=-penalty*(exx[iel]+eyy[iel])

print("     -> p   (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.5f s | Nfem: %d" % (time.time() - start,Nfem))

#################################################################
# compute error
#################################################################
#start = time.time()

#error_u=np.empty(NV,dtype=np.float64)
#error_v=np.empty(NV,dtype=np.float64)
#error_p=np.empty(nel,dtype=np.float64)

#for i in range(0,NV): 
#    error_u[i]=u[i]-velocity_x(x[i],y[i])
#    error_v[i]=v[i]-velocity_y(x[i],y[i])

#for i in range(0,nel): 
#    error_p[i]=p[i]-pressure(xc[i],yc[i])

#print("compute nodal error for plot: %.3f s" % (time.time() - start))

#################################################################
# compute error in L2 norm
#################################################################
start = time.time()

errv=0.
errp=0.
for iel in range(0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.
            N[0]=0.25*(1.-rq)*(1.-sq)
            N[1]=0.25*(1.+rq)*(1.-sq)
            N[2]=0.25*(1.+rq)*(1.+sq)
            N[3]=0.25*(1.-rq)*(1.+sq)
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
            jcb[0,0]=dNdr[:].dot(x[icon[:,iel]])
            jcb[0,1]=dNdr[:].dot(y[icon[:,iel]])
            jcb[1,0]=dNds[:].dot(x[icon[:,iel]])
            jcb[1,1]=dNds[:].dot(y[icon[:,iel]])
            jcob=np.linalg.det(jcb)
            xq=N[:].dot(x[icon[:,iel]])
            yq=N[:].dot(y[icon[:,iel]])
            uq=N[:].dot(u[icon[:,iel]])
            vq=N[:].dot(v[icon[:,iel]])
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

#np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')
#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
