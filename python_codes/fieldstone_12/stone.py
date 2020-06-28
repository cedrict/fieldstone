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
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    else:
       val=0
    return val

def by(x, y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    else:
       val=0
    return val

def velocity_x(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    else:
       val=0
    return val

def velocity_y(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    else:
       val=0
    return val

def pressure(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    else:
       val=0
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
    return 0.5*norm

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
   nelx = 64
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

xi=0. # controls level of mesh randomness (between 0 and 0.5 max)

#bench=1: donea huerta
#bench=2: ldc 
#bench=3: punch
bench=3

use_filter=True

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx+random.uniform(-1.,+1)*hx*xi
        y[counter]=j*hy+random.uniform(-1.,+1)*hy*xi
        if i==0:
           x[counter]=0
        if i==nnx-1:
           x[counter]=Lx
        if j==0:
           y[counter]=0
        if j==nny-1:
           y[counter]=Ly
        counter += 1
    #end for
#end for

print("grid nodes setup: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m,nel),dtype=np.int32)

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

print("connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix = np.zeros(Nfem, dtype=np.bool)  # boundary condition, yes/no
bc_val = np.zeros(Nfem, dtype=np.float64)  # boundary condition, value

if bench==1:
   for i in range(0,NV):
       if y[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
   #end for
elif bench==2:
   for i in range(0,NV):
       if y[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 1. #x[i]**2*(1-x[i])**2
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
   #end for
else:
   for i in range(0,NV):
       if y[i]<eps:
          #bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if y[i]>(Ly-eps) and abs(x[i]-0.5)<0.19999:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = -1.
       if x[i]<eps:
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndof]   = True ; bc_val[i*ndof]   = 0.
          bc_fix[i*ndof+1] = True ; bc_val[i*ndof+1] = 0.
   #end for



print("boundary conditions: %.3f s" % (time.time() - start))

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
# solve system
#################################################################
start = time.time()

sol = sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)

print("solve time: %.3f s" % (time.time() - start))

#####################################################################
# put solution into separate x,y velocity arrays
#####################################################################
start = time.time()

u,v=np.reshape(sol,(NV,2)).T

print("     -> u (m,M) %.5f %.5f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5f %.5f " %(np.min(v),np.max(v)))

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("reshape: %.3f s" % (time.time() - start))

#####################################################################
# retrieve elemental pressure & strain rate from velocity divergence
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

if bench==1 or bench==2:
   int_p=area.dot(p)
   print("     -> avrg pressure=",int_p/Lx/Ly)
   p[:]=p[:]-int_p
else:
   int_p=0.
   
print("     -> rawp (m,M,avrg) %.5f %.5f %.5f %d" %(np.min(p),np.max(p),int_p,nel))

np.savetxt('p_top.ascii',np.array([xc[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,p')

np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute p,exx,eyy,exy: %.3f s" % (time.time() - start))

#####################################################################
# +1/-1 checkerboard automatic removal 
#####################################################################
#start = time.time()

pcb  = np.zeros(nel,dtype=np.float64)  
for iely in range(0,nely):
    for ielx in range(0,nelx):
        pcb[iely*nelx+ielx]=(-1)**ielx*(-1)**iely

if use_filter:

   int_p=0
   for iel in range(0,nel):
       int_p+=p[iel]*pcb[iel]*area[iel]
   print('     -> amplitude of checkerboard=',int_p)

   for iel in range(0,nel):
       p[iel]-=int_p*pcb[iel]

   print("     -> raw2 (m,M,avrg) %.5f %.5f %.5f %d" %(np.min(p),np.max(p),int_p,nel))

   np.savetxt('pressure2.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')

#end if

print("filter p with pcb: %.3f s" % (time.time() - start))

#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

print("******************************")
print("PROJECTING PRESSURE ONTO NODES")
print("******************************")

#####################################################################
# Scheme 1
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

np.savetxt('q1_top.ascii',np.array([x[NV-nnx:NV],q1[NV-nnx:NV]]).T,header='# x,q1')

print("     -> q1 (m,M) %.5f %.5f " %(np.min(q1),np.max(q1)))

print("compute q1: %.3f s" % (time.time() - start))

#####################################################################
# Scheme 2
#####################################################################
start = time.time()

q2=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q2[icon[0,iel]]+=p[iel]*area[iel]
    q2[icon[1,iel]]+=p[iel]*area[iel]
    q2[icon[2,iel]]+=p[iel]*area[iel]
    q2[icon[3,iel]]+=p[iel]*area[iel]
    count[icon[0,iel]]+=area[iel]
    count[icon[1,iel]]+=area[iel]
    count[icon[2,iel]]+=area[iel]
    count[icon[3,iel]]+=area[iel]
#end for

q2/=count

np.savetxt('q2.ascii',np.array([x,y,q2]).T,header='# x,y,q2')

np.savetxt('q2_top.ascii',np.array([x[NV-nnx:NV],q2[NV-nnx:NV]]).T,header='# x,q2')

print("     -> q2 (m,M) %.5f %.5f " %(np.min(q2),np.max(q2)))

print("compute q2: %.3f s" % (time.time() - start))

#####################################################################
# Scheme 3 
# AtrX is the area of the triangle attached to node X
# 3-----2
# |     |
# |     |
# 0-----1
#####################################################################
start = time.time()

q3=np.zeros(NV,dtype=np.float64)  
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
    q3[i0]+=p[iel]*Atr0
    q3[i1]+=p[iel]*Atr1
    q3[i2]+=p[iel]*Atr2
    q3[i3]+=p[iel]*Atr3
    count[i0]+=Atr0
    count[i1]+=Atr1
    count[i2]+=Atr2
    count[i3]+=Atr3
#end for

q3/=count

np.savetxt('q3.ascii',np.array([x,y,q3]).T,header='# x,y,q3')

np.savetxt('q3_top.ascii',np.array([x[NV-nnx:NV],q3[NV-nnx:NV]]).T,header='# x,q3')

print("     -> q3 (m,M) %.5f %.5f " %(np.min(q3),np.max(q3)))

print("compute q3: %.3f s" % (time.time() - start))

#####################################################################
# Scheme 4
#####################################################################
start = time.time()

q4=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q4[icon[0,iel]]+=p[iel]/area[iel]
    q4[icon[1,iel]]+=p[iel]/area[iel]
    q4[icon[2,iel]]+=p[iel]/area[iel]
    q4[icon[3,iel]]+=p[iel]/area[iel]
    count[icon[0,iel]]+=1./area[iel]
    count[icon[1,iel]]+=1./area[iel]
    count[icon[2,iel]]+=1./area[iel]
    count[icon[3,iel]]+=1./area[iel]
#end for

q4/=count

np.savetxt('q4.ascii',np.array([x,y,q4]).T,header='# x,y,q4')

print("     -> q4 (m,M) %.5f %.5f " %(np.min(q4),np.max(q4)))

np.savetxt('q4_top.ascii',np.array([x[NV-nnx:NV],q4[NV-nnx:NV]]).T,header='# x,q4')

print("compute q4: %.3f s" % (time.time() - start))

#####################################################################
# Scheme 6: consistent recovery 
#####################################################################
start = time.time()

q6=np.zeros(NV,dtype=np.float64)  
q7=np.zeros(NV,dtype=np.float64)  
a_mat6=np.zeros((NV,NV),dtype=np.float64)  
a_mat7=np.zeros((NV,NV),dtype=np.float64)  
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

    #I here decide to use p directly so that 
    #the filter that has been applied does 
    #alter the q6 and q7 fields.
    B_el[0]=N[0]*p[iel]*weightq*jcob
    B_el[1]=N[1]*p[iel]*weightq*jcob
    B_el[2]=N[2]*p[iel]*weightq*jcob
    B_el[3]=N[3]*p[iel]*weightq*jcob

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m):
        ik=icon[k1,iel]
        for k2 in range(0,m):
            jk=icon[k2,iel]
            a_mat6[ik,jk]+=M_el[k1,k2]
            a_mat7[ik,ik]+=M_el[k1,k2] # lumping while assembling
        #end for
        rhs[ik]+=B_el[k1]
    #end for

#end for

q6 = sps.linalg.spsolve(sps.csr_matrix(a_mat6),rhs)
q7 = sps.linalg.spsolve(sps.csr_matrix(a_mat7),rhs)

np.savetxt('q6.ascii',np.array([x,y,q6]).T,header='# x,y,q6')
np.savetxt('q7.ascii',np.array([x,y,q7]).T,header='# x,y,q7')

np.savetxt('q6_top.ascii',np.array([x[NV-nnx:NV],q6[NV-nnx:NV]]).T,header='# x,q6')
np.savetxt('q7_top.ascii',np.array([x[NV-nnx:NV],q7[NV-nnx:NV]]).T,header='# x,q7')

print("     -> q6 (m,M) %.5f %.5f " %(np.min(q6),np.max(q6)))
print("     -> q7 (m,M) %.5f %.5f " %(np.min(q7),np.max(q7)))

print("compute q6,q7: %.3f s" % (time.time() - start))


#####################################################################
# Scheme 8
#####################################################################
start = time.time()

q8=np.zeros(NV,dtype=np.float64)  
mat=np.zeros((4,4),dtype=np.float64)  
rhs=np.zeros(4,dtype=np.float64)  

for iely in range(0,nely-1):
    for ielx in range(0,nelx-1):
        iel1=iely*nelx+ielx
        iel2=iel1+1
        iel3=iel1+nelx
        iel4=iel1+nelx+1
        mat[0,0]=1 ; mat[0,1]=xc[iel1] ; mat[0,2]=yc[iel1] ; mat[0,3]=xc[iel1]*yc[iel1]
        mat[1,0]=1 ; mat[1,1]=xc[iel2] ; mat[1,2]=yc[iel2] ; mat[1,3]=xc[iel2]*yc[iel2]
        mat[2,0]=1 ; mat[2,1]=xc[iel3] ; mat[2,2]=yc[iel3] ; mat[2,3]=xc[iel3]*yc[iel3]
        mat[3,0]=1 ; mat[3,1]=xc[iel4] ; mat[3,2]=yc[iel4] ; mat[3,3]=xc[iel4]*yc[iel4]
        rhs[0]=p[iel1]
        rhs[1]=p[iel2]
        rhs[2]=p[iel3]
        rhs[3]=p[iel4]
        sol = sps.linalg.spsolve(sps.csr_matrix(mat),rhs)
        inode=icon[2,iel1] 

        q8[inode]=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]

        if ielx==0 and iely==0: #lower left corner
           inode=icon[0,iel1] 
           q8[inode]=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if ielx==nelx-2 and iely==0: #lower right corner
           inode=icon[1,iel2] 
           q8[inode]=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if ielx==0 and iely==nely-2: #upper left corner
           inode=icon[3,iel3] 
           q8[inode]=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if ielx==nelx-2 and iely==nely-2: #upper right corner
           inode=icon[2,iel4] 
           q8[inode]=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]

        if ielx==0: # left side without corners
           inode=icon[3,iel1] 
           q8[inode]+=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if ielx==nelx-2: # right side without corners
           inode=icon[2,iel2] 
           q8[inode]+=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if iely==0: # left side without corners
           inode=icon[1,iel1] 
           q8[inode]+=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]
        if iely==nely-2: # left side without corners
           inode=icon[2,iel3] 
           q8[inode]+=sol[0]+sol[1]*x[inode]+sol[2]*y[inode]+sol[3]*x[inode]*y[inode]

    #end for
#end for

np.savetxt('q8.ascii',np.array([x,y,q8]).T,header='# x,y,q8')

np.savetxt('q8_top.ascii',np.array([x[NV-nnx:NV],q8[NV-nnx:NV]]).T,header='# x,q8')

print("     -> q8 (m,M) %.5f %.5f " %(np.min(q8),np.max(q8)))

#####################################################################
# normalise pressure fields 
#####################################################################
start = time.time()

avrg_p=0
avrg_q1=0
avrg_q2=0
avrg_q3=0
avrg_q4=0
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
            q6_q=0.0
            for k in range(0,m):
                q1_q+=N[k]*q1[icon[k,iel]]
                q2_q+=N[k]*q2[icon[k,iel]]
                q3_q+=N[k]*q3[icon[k,iel]]
                q4_q+=N[k]*q4[icon[k,iel]]
                q6_q+=N[k]*q6[icon[k,iel]]
            #end for

            avrg_q1+=q1_q*weightq*jcob            
            avrg_q2+=q2_q*weightq*jcob            
            avrg_q3+=q3_q*weightq*jcob            
            avrg_q4+=q4_q*weightq*jcob            
            avrg_q6+=q6_q*weightq*jcob            

        #end for jq
    #end for iq
#end for iel

print ("     -> avrg(p )= %.19f" % avrg_p)
print ("     -> avrg(q1)= %.19f" % avrg_q1)
print ("     -> avrg(q2)= %.19f" % avrg_q2)
print ("     -> avrg(q3)= %.19f" % avrg_q3)
print ("     -> avrg(q4)= %.19f" % avrg_q4)
print ("     -> avrg(q6)= %.19f" % avrg_q6)

#p-=avrg_p
#q1-=avrg_q1
#q2-=avrg_q2
#q3-=avrg_q3
#q4-=avrg_q4
#q6-=avrg_q6

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
error_q6 = np.empty(NV,dtype=np.float64)
error_q7 = np.empty(NV,dtype=np.float64)
error_q8 = np.empty(NV,dtype=np.float64)

for i in range(0,NV): 
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
    error_q1[i]=q1[i]-pressure(x[i],y[i])
    error_q2[i]=q2[i]-pressure(x[i],y[i])
    error_q3[i]=q3[i]-pressure(x[i],y[i])
    error_q4[i]=q4[i]-pressure(x[i],y[i])
    error_q6[i]=q6[i]-pressure(x[i],y[i])
    error_q7[i]=q7[i]-pressure(x[i],y[i])
    error_q8[i]=q8[i]-pressure(x[i],y[i])
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
errq6=0.
errq7=0.
errq8=0.
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
            q6_q=0.0
            q7_q=0.0
            q8_q=0.0
            for k in range(0,m):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
                q1_q+=N[k]*q1[icon[k,iel]]
                q2_q+=N[k]*q2[icon[k,iel]]
                q3_q+=N[k]*q3[icon[k,iel]]
                q4_q+=N[k]*q4[icon[k,iel]]
                q6_q+=N[k]*q6[icon[k,iel]]
                q7_q+=N[k]*q7[icon[k,iel]]
                q8_q+=N[k]*q8[icon[k,iel]]
            #end for
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
            errq1+=(q1_q-pressure(xq,yq))**2*weightq*jcob
            errq2+=(q2_q-pressure(xq,yq))**2*weightq*jcob
            errq3+=(q3_q-pressure(xq,yq))**2*weightq*jcob
            errq4+=(q4_q-pressure(xq,yq))**2*weightq*jcob
            errq6+=(q6_q-pressure(xq,yq))**2*weightq*jcob
            errq7+=(q7_q-pressure(xq,yq))**2*weightq*jcob
            errq8+=(q8_q-pressure(xq,yq))**2*weightq*jcob
        #end for jq
    #end for iq
#end for iel 

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errq1=np.sqrt(errq1)
errq2=np.sqrt(errq2)
errq3=np.sqrt(errq3)
errq4=np.sqrt(errq4)
errq6=np.sqrt(errq6)
errq7=np.sqrt(errq7)
errq8=np.sqrt(errq8)

print("nel= %6d ; v: %.8f ; p: %.8f ; q1: %.8f ; q2: %.8f ; q3: %.8f ; q4: %.8f ; q6: %.8f ; q7: %.8f ; q8: %.8f"\
 %(nel,errv,errp,errq1,errq2,errq3,errq4,errq6,errq7,errq8))

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
   vtufile.write("<DataArray type='Float32' Name='error_p' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % error_p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='pcb' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%f\n" % pcb[iel])
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
   vtufile.write("<DataArray type='Float32' Name='q6' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q7[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='q7' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %q8[i])
   vtufile.write("</DataArray>\n")
   vtufile.write("<DataArray type='Float32' Name='q8' Format='ascii'> \n")
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
   vtufile.write("<DataArray type='Float32' Name='error q6' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q6[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q7' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q7[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q8' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10f \n" %error_q8[i])
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
