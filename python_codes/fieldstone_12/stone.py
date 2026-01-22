import numpy as np
import sys as sys
import time as clock 
import random
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def bx(x,y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    else:
       val=0
    return val

def by(x,y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    else:
       val=0
    return val

###############################################################################

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

###############################################################################

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

###############################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 012 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 64
   nely = nelx
   visu = 1
    
nnx=nelx+1          # number of V-nodes, x direction
nny=nely+1          # number of V-nodes, y direction
nn_V=nnx*nny        # number of V-nodes
nel=nelx*nely       # total number of elements
Nfem_V=nn_V*ndof_V  # number of velocity degrees of freedom
Nfem=Nfem_V         # Total number of degrees of freedom

viscosity=1.  # dynamic viscosity \mu
penalty=1.e6  # penalty coefficient value

hx=Lx/float(nelx)
hy=Ly/float(nely)

xi=0. # controls level of mesh randomness (between 0 and 0.5 max)

#bench=1: donea huerta
#bench=2: ldc 
#bench=3: punch
#bench=4: regularized ldc 
bench=1

use_filter=False

debug=False

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx+random.uniform(-1.,+1)*hx*xi
        y_V[counter]=j*hy+random.uniform(-1.,+1)*hy*xi
        if i==0:
           x_V[counter]=0
        if i==nnx-1:
           x_V[counter]=Lx
        if j==0:
           y_V[counter]=0
        if j==nny-1:
           y_V[counter]=Ly
        counter += 1
    #end for
#end for

print("grid nodes setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter+=1
    #end for
#end for

print("connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

if bench==1:
   for i in range(0,nn_V):
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
   #end for
elif bench==2:
   for i in range(0,nn_V):
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 1. #x[i]**2*(1-x[i])**2
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
   #end for
elif bench==4:
   for i in range(0,nn_V):
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = x[i]**2*(1-x[i])**2
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
   #end for

elif bench==3:
   for i in range(0,nn_V):
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps) and x[i]<0.5:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = -1.
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          #bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V]   = True ; bc_val_V[i*ndof_V]   = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
   #end for
else:
   exit('unknown bench')
#end if

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64) # matrix of Ax=b
b_fem=np.zeros(Nfem,dtype=np.float64)          # right hand side of Ax=b
B=np.zeros((3,ndof_V*m_V),dtype=np.float64)    # gradient matrix B 
area=np.zeros(nel,dtype=np.float64)  
H=np.array([[1,1,0],[1,1,0],[0,0,0]],dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)
N_V=np.zeros(m_V,dtype=np.float64)            # shape functions
dNdx_V=np.zeros(m_V,dtype=np.float64)         # shape functions derivatives
dNdy_V=np.zeros(m_V,dtype=np.float64)         # shape functions derivatives
dNdr_V=np.zeros(m_V,dtype=np.float64)         # shape functions derivatives
dNds_V=np.zeros(m_V,dtype=np.float64)         # shape functions derivatives

for iel in range(0,nel):

    # set 2 arrays to 0 every loop
    A_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    b_el=np.zeros(m_V*ndof_V,dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq

            # calculate inverse of the jacobian matrix
            jcbi = np.linalg.inv(jcb)

            # compute coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            # compute dNdx & dNdy
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            # construct 3x8 B matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental A_el matrix
            A_el+=B.T.dot(C.dot(B))*viscosity*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                b_el[2*i  ]+=N_V[i]*JxWq*bx(xq,yq)
                b_el[2*i+1]+=N_V[i]*JxWq*by(xq,yq)
            #end for

        #end for
    #end for

    # integrate penalty term at 1 point
    rq=0.
    sq=0.
    weightq=2.*2.

    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # compute the jacobian
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate determinant of the jacobian
    JxWq=np.linalg.det(jcb)*weightq

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute gradient matrix
    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    # compute elemental matrix
    A_el+=B.T.dot(H.dot(B))*penalty*JxWq

    # apply boundary conditions
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]: 
               fixt=bc_val_V[m1]
               ikk=ndof_V*k1+i1
               aref=A_el[ikk,ikk]
               for jkk in range(0,m_V*ndof_V):
                   b_el[jkk]-=A_el[jkk,ikk]*fixt
                   A_el[ikk,jkk]=0.
                   A_el[jkk,ikk]=0.
               #end for
               A_el[ikk,ikk]=aref
               b_el[ikk]=aref*fixt
            #end if
        #end for
    #end for

    # assemble matrix A_fem and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=A_el[ikk,jkk]
                #end for
            #end for
            b_fem[m1]+=b_el[ikk]
        #end for
    #end for

#end for

print('     -> total area=',np.sum(area))

print("building FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(sps.csr_matrix(A_fem),b_fem)

print("solve time: %.3f s" % (clock.time()-start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol,(nn_V,2)).T

print("     -> u (m,M) %.5f %.5f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5f %.5f " %(np.min(v),np.max(v)))

if debug: np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

print("reshape: %.3f s" % (clock.time()-start))

###############################################################################
# retrieve elemental pressure & strain rate from velocity divergence
###############################################################################
start=clock.time()

p=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):

    rq=0.0
    sq=0.0
    weightq=2.0*2.0

    # calculate shape functions
    N_V[0]=0.25*(1.-rq)*(1.-sq)
    N_V[1]=0.25*(1.+rq)*(1.-sq)
    N_V[2]=0.25*(1.+rq)*(1.+sq)
    N_V[3]=0.25*(1.-rq)*(1.+sq)

    # calculate shape function derivatives
    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # calculate jacobian matrix
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate inverse of the jacobian matrix
    jcbi = np.linalg.inv(jcb)

    # compute dNdx & dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute coords of center 
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    p[iel]=-penalty*(exx[iel]+eyy[iel])

#end for iel

print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))

if bench==1 or bench==2 or bench==4:
   int_p=area.dot(p)
   print("     -> avrg pressure=",int_p/Lx/Ly)
   p[:]=p[:]-int_p
else:
   int_p=0.
   
print("     -> rawp (m,M,avrg) %.5f %.5f %.5f %d" %(np.min(p),np.max(p),int_p,nel))

np.savetxt('p_top.ascii',np.array([x_e[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,p')

if debug:
   np.savetxt('pressure.ascii',np.array([x_e,y_e,p]).T,header='# x,y,p')
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute p,exx,eyy,exy: %.3f s" % (clock.time()-start))

###############################################################################
# +1/-1 checkerboard automatic removal 
###############################################################################
start=clock.time()

pcb  = np.zeros(nel,dtype=np.float64)  
for iely in range(0,nely):
    for ielx in range(0,nelx):
        pcb[iely*nelx+ielx]=(-1)**ielx*(-1)**iely

if use_filter:

   int_p=0
   for iel in range(0,nel):
       int_p+=p[iel]*pcb[iel]*area[iel]
   print('     -> amplitude of checkerboard=',int_p)

   #missing 1/V term ?

   for iel in range(0,nel):
       p[iel]-=int_p*pcb[iel]

   print("     -> raw2 (m,M,avrg) %.5f %.5f %.5f %d" %(np.min(p),np.max(p),int_p,nel))

   #np.savetxt('pressure2.ascii',np.array([xc,yc,p]).T,header='# xc,yc,p')

   np.savetxt('p_top_filtered.ascii',np.array([x_e[nel-nelx:nel],p[nel-nelx:nel]]).T,header='# x,p')

#end if

print("filter p with pcb: %.3f s" % (clock.time()-start))

###############################################################################
###############################################################################

print("******************************")
print("PROJECTING PRESSURE ONTO NODES")
print("******************************")

###############################################################################
# Scheme 1
###############################################################################
start=clock.time()

q1=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q1[icon_V[0,iel]]+=p[iel]
    q1[icon_V[1,iel]]+=p[iel]
    q1[icon_V[2,iel]]+=p[iel]
    q1[icon_V[3,iel]]+=p[iel]
    count[icon_V[0,iel]]+=1
    count[icon_V[1,iel]]+=1
    count[icon_V[2,iel]]+=1
    count[icon_V[3,iel]]+=1
#end for

q1/=count

np.savetxt('q1.ascii',np.array([x_V,y_V,q1]).T,header='# x,y,q1')
np.savetxt('q1_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q1[nn_V-nnx:nn_V]]).T,header='# x,q1')

print("     -> q1 (m,M) %.5f %.5f " %(np.min(q1),np.max(q1)))

print("compute q1: %.3f s" % (clock.time()-start))

###############################################################################
# Scheme 2
###############################################################################
start=clock.time()

q2=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q2[icon_V[0,iel]]+=p[iel]*area[iel]
    q2[icon_V[1,iel]]+=p[iel]*area[iel]
    q2[icon_V[2,iel]]+=p[iel]*area[iel]
    q2[icon_V[3,iel]]+=p[iel]*area[iel]
    count[icon_V[0,iel]]+=area[iel]
    count[icon_V[1,iel]]+=area[iel]
    count[icon_V[2,iel]]+=area[iel]
    count[icon_V[3,iel]]+=area[iel]
#end for

q2/=count

np.savetxt('q2.ascii',np.array([x_V,y_V,q2]).T,header='# x,y,q2')
np.savetxt('q2_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q2[nn_V-nnx:nn_V]]).T,header='# x,q2')

print("     -> q2 (m,M) %.5f %.5f " %(np.min(q2),np.max(q2)))

print("compute q2: %.3f s" % (clock.time()-start))

###############################################################################
# Scheme 3 
# AtrX is the area of the triangle attached to node X
# 3-----2
# |     |
# |     |
# 0-----1
###############################################################################
start=clock.time()

q3=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    i0=icon_V[0,iel]
    i1=icon_V[1,iel]
    i2=icon_V[2,iel]
    i3=icon_V[3,iel]
    Atr0=area_triangle(x_V[i3],y_V[i3],0,x_V[i0],y_V[i0],0,x_V[i1],y_V[i1],0)
    Atr1=area_triangle(x_V[i0],y_V[i0],0,x_V[i1],y_V[i1],0,x_V[i2],y_V[i2],0)
    Atr2=area_triangle(x_V[i1],y_V[i1],0,x_V[i2],y_V[i2],0,x_V[i3],y_V[i3],0)
    Atr3=area_triangle(x_V[i2],y_V[i2],0,x_V[i3],y_V[i3],0,x_V[i0],y_V[i0],0)
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

np.savetxt('q3.ascii',np.array([x_V,y_V,q3]).T,header='# x,y,q3')
np.savetxt('q3_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q3[nn_V-nnx:nn_V]]).T,header='# x,q3')

print("     -> q3 (m,M) %.5f %.5f " %(np.min(q3),np.max(q3)))

print("compute q3: %.3f s" % (clock.time()-start))

###############################################################################
# Scheme 4
###############################################################################
start=clock.time()

q4=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q4[icon_V[0,iel]]+=p[iel]/area[iel]
    q4[icon_V[1,iel]]+=p[iel]/area[iel]
    q4[icon_V[2,iel]]+=p[iel]/area[iel]
    q4[icon_V[3,iel]]+=p[iel]/area[iel]
    count[icon_V[0,iel]]+=1./area[iel]
    count[icon_V[1,iel]]+=1./area[iel]
    count[icon_V[2,iel]]+=1./area[iel]
    count[icon_V[3,iel]]+=1./area[iel]
#end for

q4/=count

np.savetxt('q4.ascii',np.array([x_V,y_V,q4]).T,header='# x,y,q4')
np.savetxt('q4_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q4[nn_V-nnx:nn_V]]).T,header='# x,q4')

print("     -> q4 (m,M) %.5f %.5f " %(np.min(q4),np.max(q4)))

print("compute q4: %.3f s" % (clock.time()-start))

###############################################################################
# Scheme 6: consistent recovery 
###############################################################################
start=clock.time()

q6=np.zeros(nn_V,dtype=np.float64)  
q7=np.zeros(nn_V,dtype=np.float64)  
a_mat6=np.zeros((nn_V,nn_V),dtype=np.float64)  
a_mat7=np.zeros((nn_V,nn_V),dtype=np.float64)  
rhs=np.zeros(nn_V,dtype=np.float64)        
velvect=np.zeros(m_V*ndof_V,dtype=np.float64)  
Nmat=np.zeros((m_V,3),dtype=np.float64)  

for iel in range(0,nel):

    velvect[0]=u[icon_V[0,iel]]
    velvect[1]=v[icon_V[0,iel]]
    velvect[2]=u[icon_V[1,iel]]
    velvect[3]=v[icon_V[1,iel]]
    velvect[4]=u[icon_V[2,iel]]
    velvect[5]=v[icon_V[2,iel]]
    velvect[6]=u[icon_V[3,iel]]
    velvect[7]=v[icon_V[3,iel]]

    M_el=np.zeros((m_V,m_V),dtype=np.float64)  
    b_el=np.zeros(m_V,dtype=np.float64)  

    for iq in [-1, 1]:
        for jq in [-1, 1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq

            for i in range(0,m_V):
                for j in range(0,m_V):
                    M_el[i,j]+=N_V[i]*N_V[j]*JxWq
                #end for
            #end for

        #end for jq
    #end for iq

    rq=0.
    sq=0.
    weightq=2.*2.

    # calculate shape functions
    N_V[0]=0.25*(1.-rq)*(1.-sq)
    N_V[1]=0.25*(1.+rq)*(1.-sq)
    N_V[2]=0.25*(1.+rq)*(1.+sq)
    N_V[3]=0.25*(1.-rq)*(1.+sq)

    # calculate shape function derivatives
    dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
    dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
    dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
    dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

    # calculate jacobian matrix
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

    # calculate the determinant of the jacobian
    JxWq=np.linalg.det(jcb)*weightq

    # calculate the inverse of the jacobian
    jcbi=np.linalg.inv(jcb)

    # compute dNdx and dNdy
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

    # compute gradient matrix
    for i in range(0,m_V):
        B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.       ],
                          [0.       ,dNdy_V[i]],
                          [dNdy_V[i],dNdx_V[i]]]

    for k in range(0,m_V):
        Nmat[k,0]=N_V[k]
        Nmat[k,1]=N_V[k]
        Nmat[k,2]=0.
    #end for

    # compute elemental matrices
    Del= Nmat.dot(B)*JxWq
    b_el=-Del.dot(velvect)*penalty    # no JxWq ??! still in use ?!

    #I here decide to use p directly so that 
    #the filter that has been applied does 
    #alter the q6 and q7 fields.
    b_el[0]=N_V[0]*p[iel]*JxWq
    b_el[1]=N_V[1]*p[iel]*JxWq
    b_el[2]=N_V[2]*p[iel]*JxWq
    b_el[3]=N_V[3]*p[iel]*JxWq

    # assemble matrix a_mat and right hand side rhs
    for k1 in range(0,m_V):
        ik=icon_V[k1,iel]
        for k2 in range(0,m_V):
            jk=icon_V[k2,iel]
            a_mat6[ik,jk]+=M_el[k1,k2]
            a_mat7[ik,ik]+=M_el[k1,k2] # lumping while assembling
        #end for
        rhs[ik]+=b_el[k1]
    #end for

#end for

q6=sps.linalg.spsolve(sps.csr_matrix(a_mat6),rhs)
q7=sps.linalg.spsolve(sps.csr_matrix(a_mat7),rhs)

np.savetxt('q6.ascii',np.array([x_V,y_V,q6]).T,header='# x,y,q6')
np.savetxt('q7.ascii',np.array([x_V,y_V,q7]).T,header='# x,y,q7')

np.savetxt('q6_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q6[nn_V-nnx:nn_V]]).T,header='# x,q6')
np.savetxt('q7_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q7[nn_V-nnx:nn_V]]).T,header='# x,q7')

print("     -> q6 (m,M) %.5f %.5f " %(np.min(q6),np.max(q6)))
print("     -> q7 (m,M) %.5f %.5f " %(np.min(q7),np.max(q7)))

print("compute q6,q7: %.3f s" % (clock.time()-start))

###############################################################################
# Scheme 8
###############################################################################
start=clock.time()

q8=np.zeros(nn_V,dtype=np.float64)  
mat=np.zeros((m_V,m_V),dtype=np.float64)  
rhs=np.zeros(m_V,dtype=np.float64)  

for iely in range(0,nely-1):
    for ielx in range(0,nelx-1):
        iel1=iely*nelx+ielx
        iel2=iel1+1
        iel3=iel1+nelx
        iel4=iel1+nelx+1
        mat[0,0]=1 ; mat[0,1]=x_e[iel1] ; mat[0,2]=y_e[iel1] ; mat[0,3]=x_e[iel1]*y_e[iel1]
        mat[1,0]=1 ; mat[1,1]=x_e[iel2] ; mat[1,2]=y_e[iel2] ; mat[1,3]=x_e[iel2]*y_e[iel2]
        mat[2,0]=1 ; mat[2,1]=x_e[iel3] ; mat[2,2]=y_e[iel3] ; mat[2,3]=x_e[iel3]*y_e[iel3]
        mat[3,0]=1 ; mat[3,1]=x_e[iel4] ; mat[3,2]=y_e[iel4] ; mat[3,3]=x_e[iel4]*y_e[iel4]
        rhs[0]=p[iel1]
        rhs[1]=p[iel2]
        rhs[2]=p[iel3]
        rhs[3]=p[iel4]
        sol=sps.linalg.spsolve(sps.csr_matrix(mat),rhs)
        inode=icon_V[2,iel1] 

        q8[inode]=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]

        if ielx==0 and iely==0: #lower left corner
           inode=icon_V[0,iel1] 
           q8[inode]=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if ielx==nelx-2 and iely==0: #lower right corner
           inode=icon_V[1,iel2] 
           q8[inode]=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if ielx==0 and iely==nely-2: #upper left corner
           inode=icon_V[3,iel3] 
           q8[inode]=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if ielx==nelx-2 and iely==nely-2: #upper right corner
           inode=icon_V[2,iel4] 
           q8[inode]=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]

        if ielx==0: # left side without corners
           inode=icon_V[3,iel1] 
           q8[inode]+=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if ielx==nelx-2: # right side without corners
           inode=icon_V[2,iel2] 
           q8[inode]+=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if iely==0: # left side without corners
           inode=icon_V[1,iel1] 
           q8[inode]+=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]
        if iely==nely-2: # left side without corners
           inode=icon_V[2,iel3] 
           q8[inode]+=sol[0]+sol[1]*x_V[inode]+sol[2]*y_V[inode]+sol[3]*x_V[inode]*y_V[inode]

    #end for
#end for

np.savetxt('q8.ascii',np.array([x_V,y_V,q8]).T,header='# x,y,q8')
np.savetxt('q8_top.ascii',np.array([x_V[nn_V-nnx:nn_V],q8[nn_V-nnx:nn_V]]).T,header='# x,q8')

print("     -> q8 (m,M) %.5f %.5f " %(np.min(q8),np.max(q8)))

###############################################################################
# normalise pressure fields 
###############################################################################
start=clock.time()

avrg_p=0
avrg_q1=0
avrg_q2=0
avrg_q3=0
avrg_q4=0
avrg_q6=0

for iel in range(0,nel):
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq
      
            avrg_p+=p[iel]*JxWq

            q1_q=np.dot(N_V,q1[icon_V[:,iel]])
            q2_q=np.dot(N_V,q2[icon_V[:,iel]])
            q3_q=np.dot(N_V,q3[icon_V[:,iel]])
            q4_q=np.dot(N_V,q4[icon_V[:,iel]])
            q6_q=np.dot(N_V,q6[icon_V[:,iel]])

            avrg_q1+=q1_q*JxWq
            avrg_q2+=q2_q*JxWq
            avrg_q3+=q3_q*JxWq
            avrg_q4+=q4_q*JxWq
            avrg_q6+=q6_q*JxWq

        #end for jq
    #end for iq
#end for iel

print ("     -> avrg(p )= %.5e" % avrg_p)
print ("     -> avrg(q1)= %.5e" % avrg_q1)
print ("     -> avrg(q2)= %.5e" % avrg_q2)
print ("     -> avrg(q3)= %.5e" % avrg_q3)
print ("     -> avrg(q4)= %.5e" % avrg_q4)
print ("     -> avrg(q6)= %.5e" % avrg_q6)

#p-=avrg_p
#q1-=avrg_q1
#q2-=avrg_q2
#q3-=avrg_q3
#q4-=avrg_q4
#q6-=avrg_q6

print("normalise p fields: %.3fs" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

error_p=np.zeros(nel,dtype=np.float64)
error_u=np.zeros(nn_V,dtype=np.float64)
error_v=np.zeros(nn_V,dtype=np.float64)
error_q1=np.zeros(nn_V,dtype=np.float64)
error_q2=np.zeros(nn_V,dtype=np.float64)
error_q3=np.zeros(nn_V,dtype=np.float64)
error_q4=np.zeros(nn_V,dtype=np.float64)
error_q6=np.zeros(nn_V,dtype=np.float64)
error_q7=np.zeros(nn_V,dtype=np.float64)
error_q8=np.zeros(nn_V,dtype=np.float64)

for i in range(0,nn_V): 
    error_u[i]=u[i]-velocity_x(x_V[i],y_V[i])
    error_v[i]=v[i]-velocity_y(x_V[i],y_V[i])
    error_q1[i]=q1[i]-pressure(x_V[i],y_V[i])
    error_q2[i]=q2[i]-pressure(x_V[i],y_V[i])
    error_q3[i]=q3[i]-pressure(x_V[i],y_V[i])
    error_q4[i]=q4[i]-pressure(x_V[i],y_V[i])
    error_q6[i]=q6[i]-pressure(x_V[i],y_V[i])
    error_q7[i]=q7[i]-pressure(x_V[i],y_V[i])
    error_q8[i]=q8[i]-pressure(x_V[i],y_V[i])
#end for

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(x_e[i],y_e[i])
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

            # calculate shape functions
            N_V[0]=0.25*(1.-rq)*(1.-sq)
            N_V[1]=0.25*(1.+rq)*(1.-sq)
            N_V[2]=0.25*(1.+rq)*(1.+sq)
            N_V[3]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr_V[0]=-0.25*(1.-sq) ; dNds_V[0]=-0.25*(1.-rq)
            dNdr_V[1]=+0.25*(1.-sq) ; dNds_V[1]=-0.25*(1.+rq)
            dNdr_V[2]=+0.25*(1.+sq) ; dNds_V[2]=+0.25*(1.+rq)
            dNdr_V[3]=-0.25*(1.+sq) ; dNds_V[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])

            # calculate the determinant of the jacobian
            JxWq=np.linalg.det(jcb)*weightq

            # compute coords of quad point
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])

            q1_q=np.dot(N_V,q1[icon_V[:,iel]])
            q2_q=np.dot(N_V,q2[icon_V[:,iel]])
            q3_q=np.dot(N_V,q3[icon_V[:,iel]])
            q4_q=np.dot(N_V,q4[icon_V[:,iel]])
            q6_q=np.dot(N_V,q6[icon_V[:,iel]])
            q7_q=np.dot(N_V,q7[icon_V[:,iel]])
            q8_q=np.dot(N_V,q8[icon_V[:,iel]])

            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*JxWq
            errp+=(p[iel]-pressure(xq,yq))**2*JxWq
            errq1+=(q1_q-pressure(xq,yq))**2*JxWq
            errq2+=(q2_q-pressure(xq,yq))**2*JxWq
            errq3+=(q3_q-pressure(xq,yq))**2*JxWq
            errq4+=(q4_q-pressure(xq,yq))**2*JxWq
            errq6+=(q6_q-pressure(xq,yq))**2*JxWq
            errq7+=(q7_q-pressure(xq,yq))**2*JxWq
            errq8+=(q8_q-pressure(xq,yq))**2*JxWq
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

print("compute discr. errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution export to vtu format
###############################################################################
start=clock.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%.4e %.4e %.1e \n" %(x_V[i],y_V[i],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   p.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error_p' Format='ascii'> \n")
   error_p.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='pcb' Format='ascii'> \n")
   pcb.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='area' Format='ascii'> \n")
   area.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q1' Format='ascii'> \n")
   q1.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q2' Format='ascii'> \n")
   q2.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q3' Format='ascii'> \n")
   q3.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q4' Format='ascii'> \n")
   q4.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q6' Format='ascii'> \n")
   q6.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q7' Format='ascii'> \n")
   q7.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q8' Format='ascii'> \n")
   q8.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q1' Format='ascii'> \n")
   error_q1.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q2' Format='ascii'> \n")
   error_q2.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q3' Format='ascii'> \n")
   error_q3.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q4' Format='ascii'> \n")
   error_q4.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q6' Format='ascii'> \n")
   error_q6.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q7' Format='ascii'> \n")
   error_q7.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='error q8' Format='ascii'> \n")
   error_q8.tofile(vtufile,sep=' ',format='%.4e')
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel]))
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
   print("export to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
