import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as time
import matplotlib.pyplot as plt
import random

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

def area_triangle(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    ABx=x2-x1
    ABy=y2-y1
    ABz=z2-z1
    ACx=x3-x1
    ACy=y3-y1
    ACz=z3-z1
    # w1 = u2 v3 - u3 v2
    # w2 = u3 v1 - u1 v3
    # w3 = u1 v2 - u2 v1
    nx=ABy*ACz-ABz*ACy
    ny=ABz*ACx-ABx*ACz
    nz=ABx*ACy-ABy*ACx
    norm=np.sqrt(nx**2+ny**2+nz**2)
    area=0.5*norm
    #nx=nx/norm
    #ny=ny/norm
    #nz=nz/norm
    return area

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

print("variable declaration")

m=4     # number of nodes making up an element
ndof=2  # number of degrees of freedom per node

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
   nelx = 32
   nely = nelx
   visu = 1

assert (nelx>0.), "nnx should be positive" 
assert (nely>0.), "nny should be positive" 
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

NV=nnx*nny  # number of nodes

nel=nelx*nely  # number of elements, total

penalty=1.e6  # penalty coefficient value

viscosity=1.  # dynamic viscosity \mu

Nfem=NV*ndof  # Total number of degrees of freedom

eps=1.e-10

sqrt3=np.sqrt(3.)

hx=Lx/float(nelx)
hy=Ly/float(nely)

xi=0.2 # controls level of mesh randomness (between 0 and 0.5 max)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (time.time() - start))


for i in range(0,NV):
    if x[i]>0 and x[i]<Lx and y[i]>0 and y[i]<Ly:
       x[i]+=random.uniform(-1.,+1)*hx*xi
       y[i]+=random.uniform(-1.,+1)*hy*xi
    #end if
#end for

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
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
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
#################################################################
start = time.time()

a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
b_mat = np.zeros((3,ndof*m),dtype=np.float64)   # gradient matrix B 
rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
N     = np.zeros(m,dtype=np.float64)            # shape functions
dNdx  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)          # x-component velocity
v     = np.zeros(NV,dtype=np.float64)          # y-component velocity
area  = np.zeros(nel,dtype=np.float64)  
k_mat = np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set 2 arrays to 0 every loop
    b_el = np.zeros(m * ndof)
    a_el = np.zeros((m * ndof, m * ndof), dtype=float)

    # integrate viscous term at 4 quadrature points
    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            jcb = np.zeros((2, 2),dtype=float)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]
            #end for

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0, m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
            #end for

            area[iel]+=jcob*weightq 

            # construct 3x8 b_mat matrix
            for i in range(0, m):
                b_mat[0:3, 2*i:2*i+2] = [[dNdx[i],0.     ],
                                         [0.     ,dNdy[i]],
                                         [dNdy[i],dNdx[i]]]
            #end for

            # compute elemental a_mat matrix
            a_el += b_mat.T.dot(c_mat.dot(b_mat))*viscosity*weightq*jcob

            # compute elemental rhs vector
            for i in range(0, m):
                b_el[2*i  ]+=N[i]*jcob*weightq*bx(xq,yq)
                b_el[2*i+1]+=N[i]*jcob*weightq*by(xq,yq)
            #end for

        #end for
    #end for

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

    # compute the jacobian
    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]

    # calculate determinant of the jacobian
    jcob = np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi = np.linalg.inv(jcb)

    # compute dNdx and dNdy
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    #end for

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]
    #end for

    # compute elemental matrix
    a_el += b_mat.T.dot(k_mat.dot(b_mat))*penalty*weightq*jcob

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

#end for

#print(np.sum(area))

print("building FE matrix: %.3f s" % (time.time() - start))

#################################################################
# impose boundary conditions
#################################################################
#start = time.time()

#for i in range(0, Nfem):
#    if bc_fix[i]:
#       a_matref = a_mat[i,i]
#       for j in range(0,Nfem):
#           rhs[j]-= a_mat[i, j] * bc_val[i]
#           a_mat[i,j]=0.
#           a_mat[j,i]=0.
#           a_mat[i,i] = a_matref
#       #end for
#       rhs[i]=a_matref*bc_val[i]
#    #end if
#end for

#print("a_mat (m,M) = %.4f %.4f" %(np.min(a_mat),np.max(a_mat)))
#print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

#print("impose boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# solve system
#################################################################

start = time.time()
sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.5f %.5f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5f %.5f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

#####################################################################
# retrieve pressure from velocity divergence
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
    weightq = 2.0 * 2.0

    N[0]=0.25*(1.-rq)*(1.-sq)
    N[1]=0.25*(1.+rq)*(1.-sq)
    N[2]=0.25*(1.+rq)*(1.+sq)
    N[3]=0.25*(1.-rq)*(1.+sq)

    dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
    dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
    dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
    dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

    jcb=np.zeros((2,2),dtype=float)
    for k in range(0, m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    #end for

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    #end for

    for k in range(0, m):
        xc[iel] += N[k]*x[icon[k,iel]]
        yc[iel] += N[k]*y[icon[k,iel]]
        exx[iel] += dNdx[k]*u[icon[k,iel]]
        eyy[iel] += dNdy[k]*v[icon[k,iel]]
        exy[iel] += 0.5*dNdy[k]*u[icon[k,iel]]+ 0.5*dNdx[k]*v[icon[k,iel]]
    #end for

    p[iel]=-penalty*(exx[iel]+eyy[iel])

#end for iel

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

#####################################################################
# smoothing pressure 
# q1 is the nodal pressure obtained by smoothing the elemental p
#####################################################################
start = time.time()

q1=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q1[icon[0,iel]]+=p[iel]
    q1[icon[1,iel]]+=p[iel]
    q1[icon[2,iel]]+=p[iel]
    q1[icon[3,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1
#end for

q1/=count

np.savetxt('q1.ascii',np.array([x,y,q1]).T,header='# x,y,q1')

print("     -> q1 (m,M) %.5f %.5f " %(np.min(q1),np.max(q1)))

print("compute q1: %.3f s" % (time.time() - start))

#####################################################################
# smoothing pressure 
# q4 is the nodal pressure obtained by smoothing the elemental p
# using element areas
#####################################################################
start = time.time()

q4=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q4[icon[0,iel]]+=p[iel]*area[iel]
    q4[icon[1,iel]]+=p[iel]*area[iel]
    q4[icon[2,iel]]+=p[iel]*area[iel]
    q4[icon[3,iel]]+=p[iel]*area[iel]
    count[icon[0,iel]]+=area[iel]
    count[icon[1,iel]]+=area[iel]
    count[icon[2,iel]]+=area[iel]
    count[icon[3,iel]]+=area[iel]
#end for

q4/=count

np.savetxt('q4.ascii',np.array([x,y,q4]).T,header='# x,y,q4')

print("     -> q4 (m,M) %.5f %.5f " %(np.min(q4),np.max(q4)))

print("compute q4: %.3f s" % (time.time() - start))

#####################################################################
# smoothing pressure 
# q5 is the nodal pressure obtained by smoothing the elemental p
# using inverse element areas
#####################################################################
start = time.time()

q5=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q5[icon[0,iel]]+=p[iel]/area[iel]
    q5[icon[1,iel]]+=p[iel]/area[iel]
    q5[icon[2,iel]]+=p[iel]/area[iel]
    q5[icon[3,iel]]+=p[iel]/area[iel]
    count[icon[0,iel]]+=1./area[iel]
    count[icon[1,iel]]+=1./area[iel]
    count[icon[2,iel]]+=1./area[iel]
    count[icon[3,iel]]+=1./area[iel]
#end for

q5/=count

np.savetxt('q5.ascii',np.array([x,y,q5]).T,header='# x,y,q5')

print("     -> q5 (m,M) %.5f %.5f " %(np.min(q5),np.max(q5)))

print("compute q5: %.3f s" % (time.time() - start))

#####################################################################
# smoothing pressure 
# q6 is the nodal pressure obtained by smoothing the elemental p
# using triangles (scheme 2 of sagl81a)
# AtrX is the area of the triangle attached to node X
#
# 3-----2
# |     |
# |     |
# 0-----1
#
#####################################################################
start = time.time()

q6=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    i0=icon[0,iel]
    i1=icon[1,iel]
    i2=icon[2,iel]
    i3=icon[3,iel]
    Atr0=area_triangle(x[i3],y[i3],0,x[i0],y[i0],0,x[i1],y[i1],0)
    Atr1=area_triangle(x[i0],y[i0],0,x[i1],y[i1],0,x[i2],y[i2],0)
    Atr2=area_triangle(x[i1],y[i1],0,x[i2],y[i2],0,x[i3],y[i3],0)
    Atr3=area_triangle(x[i2],y[i2],0,x[i3],y[i3],0,x[i0],y[i0],0)
    q6[i0]+=p[iel]*Atr0
    q6[i1]+=p[iel]*Atr1
    q6[i2]+=p[iel]*Atr2
    q6[i3]+=p[iel]*Atr3
    count[i0]+=Atr0
    count[i1]+=Atr1
    count[i2]+=Atr2
    count[i3]+=Atr3
#end for

q6/=count

np.savetxt('q6.ascii',np.array([x,y,q6]).T,header='# x,y,q6')

print("     -> q6 (m,M) %.5f %.5f " %(np.min(q6),np.max(q6)))

print("compute q6: %.3f s" % (time.time() - start))

#####################################################################
# smoothing pressure 
# q2 is the nodal pressure obtained by solving FE system 
#####################################################################
start = time.time()

q2=np.zeros(NV,dtype=np.float64)  
q3=np.zeros(NV,dtype=np.float64)  
a_mat2=np.zeros((NV,NV),dtype=np.float64)  
a_mat3=np.zeros((NV,NV),dtype=np.float64)  
rhs=np.zeros(NV,dtype=np.float64)        
velvect=np.zeros(m*ndof,dtype=np.float64)  
Nmat=np.zeros((m,3),dtype=np.float64)  

for iel in range(0,nel):

    velvect[0]=u[icon[0,iel]]
    velvect[1]=v[icon[0,iel]]
    velvect[2]=u[icon[1,iel]]
    velvect[3]=v[icon[1,iel]]
    velvect[4]=u[icon[2,iel]]
    velvect[5]=v[icon[2,iel]]
    velvect[6]=u[icon[3,iel]]
    velvect[7]=v[icon[3,iel]]

    M_el=np.zeros((m,m),dtype=np.float64)  
    B_el=np.zeros(m,dtype=np.float64)  

    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            jcb = np.zeros((2, 2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]
            #end for

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)

            for i in range(0,m):
                for j in range(0,m):
                    M_el[i,j]+=N[i]*N[j]*weightq*jcob
                #end for
            #end for

        #end for jq
    #end for iq

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

    jcb=np.zeros((2,2),dtype=np.float64)
    for k in range(0,m):
        jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
        jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
        jcb[1,0]+=dNds[k]*x[icon[k,iel]]
        jcb[1,1]+=dNds[k]*y[icon[k,iel]]
    #end for

    # calculate determinant of the jacobian
    jcob=np.linalg.det(jcb)

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    for k in range(0, m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
    #end for

    # compute gradient matrix
    for i in range(0,m):
        b_mat[0:3,2*i:2*i+2]=[[dNdx[i],0.     ],
                              [0.     ,dNdy[i]],
                              [dNdy[i],dNdx[i]]]
    #end for

    for k in range(0,m):
        Nmat[k,0]=N[k]
        Nmat[k,1]=N[k]
        Nmat[k,2]=0.
    #end for

    # compute elemental matrices
    Del= Nmat.dot(b_mat)*weightq*jcob
    B_el=-Del.dot(velvect)*penalty

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        ik=icon[k1,iel]
        for k2 in range(0,m):
            jk=icon[k2,iel]
            a_mat2[ik,jk]+=M_el[k1,k2]
            a_mat3[ik,ik]+=M_el[k1,k2] # lumping while assembling
        #end for
        rhs[ik]+=B_el[k1]
    #end for

#end for

q2 = sps.linalg.spsolve(sps.csr_matrix(a_mat2),rhs)
q3 = sps.linalg.spsolve(sps.csr_matrix(a_mat3),rhs)

np.savetxt('q2.ascii',np.array([x,y,q2]).T,header='# x,y,q2')
np.savetxt('q3.ascii',np.array([x,y,q3]).T,header='# x,y,q3')

print("     -> q2 (m,M) %.5f %.5f " %(np.min(q2),np.max(q2)))
print("     -> q3 (m,M) %.5f %.5f " %(np.min(q3),np.max(q3)))

print("compute q2,q3: %.3f s" % (time.time() - start))

#####################################################################
# normalise pressure fields 
#####################################################################
start = time.time()

avrg_p=0
avrg_q1=0
avrg_q2=0
avrg_q3=0
avrg_q4=0
avrg_q5=0
avrg_q6=0

for iel in range(0,nel):
    for iq in [-1, 1]:
        for jq in [-1, 1]:

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
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]
            #end for

            # calculate the determinant of the jacobian
            jcob = np.linalg.det(jcb)
      
            p_q=p[iel]
            avrg_p+=p_q*weightq*jcob

            q1_q=0.0
            q2_q=0.0
            q3_q=0.0
            q4_q=0.0
            q5_q=0.0
            q6_q=0.0
            for k in range(0,m):
                q1_q+=N[k]*q1[icon[k,iel]]
                q2_q+=N[k]*q2[icon[k,iel]]
                q3_q+=N[k]*q3[icon[k,iel]]
                q4_q+=N[k]*q4[icon[k,iel]]
                q5_q+=N[k]*q5[icon[k,iel]]
                q6_q+=N[k]*q6[icon[k,iel]]
            #end for

            avrg_q1+=q1_q*weightq*jcob            
            avrg_q2+=q2_q*weightq*jcob            
            avrg_q3+=q3_q*weightq*jcob            
            avrg_q4+=q4_q*weightq*jcob            
            avrg_q5+=q5_q*weightq*jcob            
            avrg_q6+=q6_q*weightq*jcob            

        #end for jq
    #end for iq
#end for iel

print ("     -> avrg(p )= %.19f" % avrg_p)
print ("     -> avrg(q1)= %.19f" % avrg_q1)
print ("     -> avrg(q2)= %.19f" % avrg_q2)
print ("     -> avrg(q3)= %.19f" % avrg_q3)
print ("     -> avrg(q4)= %.19f" % avrg_q4)
print ("     -> avrg(q5)= %.19f" % avrg_q5)
print ("     -> avrg(q6)= %.19f" % avrg_q6)

p-=avrg_p
q1-=avrg_q1
q2-=avrg_q2
q3-=avrg_q3
q4-=avrg_q4
q5-=avrg_q5
q6-=avrg_q6

print("normalise p fields: %.3fs" % (time.time() - start))

#####################################################################
# compute error
#####################################################################
start = time.time()

error_p = np.empty(nel,dtype=np.float64)
error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_q1 = np.empty(NV,dtype=np.float64)
error_q2 = np.empty(NV,dtype=np.float64)
error_q3 = np.empty(NV,dtype=np.float64)
error_q4 = np.empty(NV,dtype=np.float64)
error_q5 = np.empty(NV,dtype=np.float64)
error_q6 = np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
    error_q1[i]=q1[i]-pressure(x[i],y[i])
    error_q2[i]=q2[i]-pressure(x[i],y[i])
    error_q3[i]=q3[i]-pressure(x[i],y[i])
    error_q4[i]=q4[i]-pressure(x[i],y[i])
    error_q5[i]=q5[i]-pressure(x[i],y[i])
    error_q6[i]=q6[i]-pressure(x[i],y[i])
#end for

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i])
#end for

errv=0.
errp=0.
errq1=0.
errq2=0.
errq3=0.
errq4=0.
errq5=0.
errq6=0.
for iel in range (0,nel):
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
            jcb=np.zeros((2,2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            #end for
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            q1_q=0.0
            q2_q=0.0
            q3_q=0.0
            q4_q=0.0
            q5_q=0.0
            q6_q=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
                q1_q+=N[k]*q1[icon[k,iel]]
                q2_q+=N[k]*q2[icon[k,iel]]
                q3_q+=N[k]*q3[icon[k,iel]]
                q4_q+=N[k]*q4[icon[k,iel]]
                q5_q+=N[k]*q5[icon[k,iel]]
                q6_q+=N[k]*q6[icon[k,iel]]
            #end for
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
            errq1+=(q1_q-pressure(xq,yq))**2*weightq*jcob
            errq2+=(q2_q-pressure(xq,yq))**2*weightq*jcob
            errq3+=(q3_q-pressure(xq,yq))**2*weightq*jcob
            errq4+=(q4_q-pressure(xq,yq))**2*weightq*jcob
            errq5+=(q5_q-pressure(xq,yq))**2*weightq*jcob
            errq6+=(q6_q-pressure(xq,yq))**2*weightq*jcob
        #end for jq
    #end for iq
#end for iel 

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq1=np.sqrt(errq1)
errq2=np.sqrt(errq2)
errq3=np.sqrt(errq3)
errq4=np.sqrt(errq4)
errq5=np.sqrt(errq5)
errq6=np.sqrt(errq6)

print("nel= %6d ; v: %.8f ; p: %.8f ; q1: %.8f ; q2: %.8f ; q3: %.8f ; q4: %.8f ; q5: %.8f ; q6: %.8f"\
 %(nel,errv,errp,errq1,errq2,errq3,errq4,errq5,errq6))

print("compute discr. errors: %.3f s" % (time.time() - start))

#####################################################################
# plot of solution export to vtu format
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % area[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q4' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q4[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q5' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q5[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q6' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q6[i])
   vtufile.write("</DataArray>\n")



   #--
   vtufile.write("<DataArray type='Float32' Name='error q1' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q1[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q2' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q2[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q3' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q3[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q4' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q4[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q5' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q5[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q6' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q6[i])
   vtufile.write("</DataArray>\n")





   #--
   vtufile.write("<DataArray type='Float32' Name='q5' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q5[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*4))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %9)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()
   print("export to vtu: %.3f s" % (time.time() - start))

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
