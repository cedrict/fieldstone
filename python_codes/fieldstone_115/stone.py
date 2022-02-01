import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix, lil_matrix
import time as time
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def bx(x, y):
    if experiment==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if experiment==2:
       val=0.
    if experiment==3:
       val=0
    if experiment==4:
       val=0
    return val

def by(x, y):
    if experiment==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if experiment==2:
       val=-1
    if experiment==3:
       val=0
    if experiment==4:
       val=0
    return val

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if experiment==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if experiment==2:
       val=0
    if experiment==3:
       val=0
    if experiment==4:
       val=20*x*y**3
    return val

def velocity_y(x,y):
    if experiment==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if experiment==2:
       val=0
    if experiment==3:
       val=0
    if experiment==4:
       val=5*x**4-5*y**4
    return val

def pressure(x,y):
    if experiment==1:
       val=x*(1.-x)-1./6.
    if experiment==2:
       val=0.5-y
    if experiment==3:
       val=0
    if experiment==4:
       val=60*x**2*y-20*y**3-5
    return val

#------------------------------------------------------------------------------
# 1: donea & huerta
# 2: aquarium 
# 3: lid driven cavity
# 4: buha06

experiment=1

#------------------------------------------------------------------------------

eps=1.e-10
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")
print("variable declaration")

mV=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   epsi = float(sys.argv[4])
else:
   nelx = 16
   nely = 16
   visu = 1
   epsi = 1e-1
    
nnx=nelx+1       # number of elements, x direction
nny=nely+1       # number of elements, y direction
NV=nnx*nny       # number of nodes
nel=nelx*nely    # number of elements, total
NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs
hx=Lx/nelx
hy=Ly/nely

eta=1.  # dynamic viscosity 

pnormalise=True

Gscaling=1 #eta/(Ly/nely)

#0: no stab
#1: penalty
#2: global 
#3: local
#4: macro-element

stabilisation=0

if stabilisation==3 and nelx%2==1: exit()
if stabilisation==3 and nely%2==1: exit()
if stabilisation==4 and nelx%2==1: exit()
if stabilisation==4 and nely%2==1: exit()

#################################################################

print('nelx,nely    =',nelx,nely)
print('epsi         =',epsi)
print('stabilisation=',stabilisation)

#################################################################
# grid point setup
#################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((mV,nel),dtype=np.int32)
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

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

if experiment==1:
   for i in range(0,NV):
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
   #end for
#end if
if experiment==2:
   for i in range(0,NV):
       if x[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if y[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if y[i]>(Ly-eps):
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
   #end for
#end if
if experiment==3:
   for i in range(0,NV):
       if y[i]>(Ly-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 1.
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
       if x[i]<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if x[i]>(Lx-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       if y[i]<eps:
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
   #end for
#end if
if experiment==4:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i]) 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i]) 
       if x[i]/Lx>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i]) 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i]) 
       if y[i]/Ly<eps:
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i]) 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i]) 
       if y[i]/Ly>(1-eps):
          bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i]) 
          bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i]) 
   #end for
#end if

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

if pnormalise:
   A_mat = lil_matrix((Nfem+1,Nfem+1),dtype=np.float64)# matrix A 
   rhs   = np.zeros((Nfem+1),dtype=np.float64)         # right hand side 
   A_mat[Nfem,NfemV:Nfem]=1
   A_mat[NfemV:Nfem,Nfem]=1
else:
   A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side 

b_mat = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
N     = np.zeros(mV,dtype=np.float64)            # shape functions
dNdx  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdy  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNdr  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNds  = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
u     = np.zeros(NV,dtype=np.float64)            # x-component velocity
v     = np.zeros(NV,dtype=np.float64)            # y-component velocity
p     = np.zeros(nel,dtype=np.float64)           # pressure field 
S     = np.zeros(nel,dtype=np.float64)           # pressure field 
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) # a
#c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64)  #b

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    K_L  =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)
    C_el=np.zeros((1,1),dtype=np.float64)

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
            jcb = np.zeros((2,2),dtype=np.float64)
            for k in range(0,mV):
                jcb[0, 0] += dNdr[k]*x[icon[k,iel]]
                jcb[0, 1] += dNdr[k]*y[icon[k,iel]]
                jcb[1, 0] += dNds[k]*x[icon[k,iel]]
                jcb[1, 1] += dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
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
                b_mat[0:3,2*i:2*i+2] = [[dNdx[i],0.     ],
                                        [0.     ,dNdy[i]],
                                        [dNdy[i],dNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=N[i]*jcob*weightq*bx(xq,yq)
                f_el[ndofV*i+1]+=N[i]*jcob*weightq*by(xq,yq)
                G_el[ndofV*i  ,0]-=dNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNdy[i]*jcob*weightq

            C_el+=jcob*weightq*epsi

        #end for jq
    #end for iq

    G_el*=Gscaling

    for i in range(0,8):
        for j in range(0,8):
            K_L[i,i]+=abs(K_el[i,j])

    S[iel]=G_el.T.dot(K_L.dot(G_el))/hx/hy

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
                    A_mat[m1,m2]+=K_el[ikk,jkk]
            rhs[m1]+=f_el[ikk]
            A_mat[m1,NfemV+iel]+=G_el[ikk,0]
            A_mat[NfemV+iel,m1]+=G_el[ikk,0]
    #end for
    rhs[NfemV+iel]+=h_el[0]

    if stabilisation==1: #penalty 
       A_mat[NfemV+iel,NfemV+iel]+=C_el

#end for iel

if stabilisation==2: #global
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           i_left=i-1
           i_right=i+1
           j_bot=j-1
           j_top=j+1
           elt_left=counter-1
           elt_right=counter+1
           elt_bot=counter-nelx
           elt_top=counter+nelx
           if i_left!=-1:
              A_mat[NfemV+ counter, NfemV+ elt_left]+=epsi*hx*hy
              A_mat[NfemV+ counter, NfemV+ counter] -=epsi*hx*hy
           if i_right!=nelx:
              A_mat[NfemV+ counter, NfemV+ elt_right]+=epsi*hx*hy
              A_mat[NfemV+ counter, NfemV+ counter]  -=epsi*hx*hy
           if j_bot!=-1:
              A_mat[NfemV+ counter, NfemV+ elt_bot]+=epsi*hx*hy
              A_mat[NfemV+ counter, NfemV+ counter]-=epsi*hx*hy
           if j_top!=nely:
              A_mat[NfemV+ counter, NfemV+ elt_top]+=epsi*hx*hy
              A_mat[NfemV+ counter, NfemV+ counter]-=epsi*hx*hy
           counter += 1
       #end for
   #end for

if stabilisation==3: #local
   # -------
   # |UL|UR|
   # +-----+
   # |LL|LR|
   # -------
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           if j%2==0 and i%2==0:
              elt_LL=counter 
              elt_LR=counter+1
              elt_UL=counter+nelx
              elt_UR=counter+nelx+1
              #lower left element LL
              A_mat[NfemV+ elt_LL, NfemV+ elt_LL] -=epsi*hx*hy*2
              A_mat[NfemV+ elt_LL, NfemV+ elt_LR] +=epsi*hx*hy
              A_mat[NfemV+ elt_LL, NfemV+ elt_UL] +=epsi*hx*hy
              #lower right element LR
              A_mat[NfemV+ elt_LR, NfemV+ elt_LR] -=epsi*hx*hy*2
              A_mat[NfemV+ elt_LR, NfemV+ elt_LL] +=epsi*hx*hy   
              A_mat[NfemV+ elt_LR, NfemV+ elt_UR] +=epsi*hx*hy
              #upper left element LL
              A_mat[NfemV+ elt_UL, NfemV+ elt_UL] -=epsi*hx*hy*2
              A_mat[NfemV+ elt_UL, NfemV+ elt_UR] +=epsi*hx*hy
              A_mat[NfemV+ elt_UL, NfemV+ elt_LL] +=epsi*hx*hy
              #upper right element UR
              A_mat[NfemV+ elt_UR, NfemV+ elt_UR] -=epsi*hx*hy*2
              A_mat[NfemV+ elt_UR, NfemV+ elt_UL] +=epsi*hx*hy   
              A_mat[NfemV+ elt_UR, NfemV+ elt_LR] +=epsi*hx*hy
           counter += 1
       #end for
   #end for

if stabilisation==4: #macro-element
   # -------
   # |UL|UR|
   # +-----+
   # |LL|LR|
   # -------
   counter = 0
   for j in range(0, nely):
       for i in range(0, nelx):
           if j%2==0 and i%2==0:
              elt_LL=counter 
              elt_LR=counter+1
              elt_UL=counter+nelx
              elt_UR=counter+nelx+1
              #lower left element LL
              A_mat[NfemV+ elt_LL, NfemV+ elt_LL] -=epsi*hx*hy
              A_mat[NfemV+ elt_LL, NfemV+ elt_LR] +=epsi*hx*hy
              A_mat[NfemV+ elt_LL, NfemV+ elt_UL] +=epsi*hx*hy
              A_mat[NfemV+ elt_LL, NfemV+ elt_UR] -=epsi*hx*hy
              #lower right element LR
              A_mat[NfemV+ elt_LR, NfemV+ elt_LR] -=epsi*hx*hy
              A_mat[NfemV+ elt_LR, NfemV+ elt_LL] +=epsi*hx*hy   
              A_mat[NfemV+ elt_LR, NfemV+ elt_UR] +=epsi*hx*hy
              A_mat[NfemV+ elt_LR, NfemV+ elt_UL] -=epsi*hx*hy
              #upper left element LL
              A_mat[NfemV+ elt_UL, NfemV+ elt_UL] -=epsi*hx*hy
              A_mat[NfemV+ elt_UL, NfemV+ elt_UR] +=epsi*hx*hy
              A_mat[NfemV+ elt_UL, NfemV+ elt_LL] +=epsi*hx*hy
              A_mat[NfemV+ elt_UL, NfemV+ elt_LR] -=epsi*hx*hy
              #upper right element UR
              A_mat[NfemV+ elt_UR, NfemV+ elt_UR] -=epsi*hx*hy
              A_mat[NfemV+ elt_UR, NfemV+ elt_UL] +=epsi*hx*hy   
              A_mat[NfemV+ elt_UR, NfemV+ elt_LR] +=epsi*hx*hy
              A_mat[NfemV+ elt_UR, NfemV+ elt_LL] -=epsi*hx*hy
           counter += 1
       #end for
   #end for

A_mat=A_mat.tocsr()

#plt.spy(A_mat, markersize=0.25)
#plt.savefig('matrix.png', bbox_inches='tight')
#plt.clf()

print("     -> S (m,M) %.4e %.4e " %(np.min(S),np.max(S)))

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

sol=sps.linalg.spsolve(A_mat,rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:NfemV],(NV,2)).T
p=sol[NfemV:Nfem]*Gscaling

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))
print("     -> lagr. mult. %e" %sol[NfemV])


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

pavrg=0
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

    jcb=np.zeros((2,2),dtype=np.float64)
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

    pavrg+=weightq*jcob*p[iel]

#end for

pavrg/=(Lx*Ly)

p[:]-=pavrg

print("     -> avrg(p)   %e "      %(pavrg))
print("     -> p (m,M)   %.4f %.4f " %(np.min(p),np.max(p)))
print("     -> exx (m,M) %.4f %.4f " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4f %.4f " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4f %.4f " %(np.min(exy),np.max(exy)))
print("     -> nel= %6d ; min(divv)= %e ; max(divv)= %e" %(nel,np.min(exx+eyy),np.max(exx+eyy)))

np.savetxt('p.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
#np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure q following section 6 of bodg06
# nodal values are area weighed averages of the surrounding 
# constant pressure values.
######################################################################

q=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  
area=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    area[iel]=hx*hy
    q[icon[0,iel]]+=p[iel]*area[iel]/4
    q[icon[1,iel]]+=p[iel]*area[iel]/4
    q[icon[2,iel]]+=p[iel]*area[iel]/4
    q[icon[3,iel]]+=p[iel]*area[iel]/4
    count[icon[0,iel]]+=area[iel]/4
    count[icon[1,iel]]+=area[iel]/4
    count[icon[2,iel]]+=area[iel]/4
    count[icon[3,iel]]+=area[iel]/4
#end for

q=q/count

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

######################################################################
# compute error
######################################################################
start = time.time()

error_u = np.empty(NV,dtype=np.float64)
error_v = np.empty(NV,dtype=np.float64)
error_q = np.empty(NV,dtype=np.float64)
error_p = np.empty(nel,dtype=np.float64)

for i in range(0,NV): 
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
            for k in range(0,mV):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]
            jcob=np.linalg.det(jcb)
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            for k in range(0,mV):
                xq+=N[k]*x[icon[k,iel]]
                yq+=N[k]*y[icon[k,iel]]
                uq+=N[k]*u[icon[k,iel]]
                vq+=N[k]*v[icon[k,iel]]
            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
        #end jq
    #end iq
#end iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %e ; errp= %e ; epsi= %e" %(nel,errv,errp,epsi))

print("compute errors: %.3f s" % (time.time() - start))

#####################################################################

if experiment==3 or experiment==4:
   np.savetxt('psurf.ascii',np.array([xc[nel-nelx:nel],p[nel-nelx:nel]]).T)

#####################################################################
# plot of solution export to vtu format
#####################################################################
start = time.time()

if visu==1:
   vtufile=open("solution.vtu","w")
   vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
   vtufile.write("<UnstructuredGrid> \n")
   vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nel*4,nel))
   #####
   vtufile.write("<Points> \n")
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%10f %10f %10f \n" %(x[icon[i,iel]],y[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e %e %e \n" %(u[icon[i,iel]],v[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%10f %10f %10f \n" %(error_u[icon[i,iel]],error_v[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" % q[icon[i,iel]])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='S' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %S[iel])
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %error_p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %exx[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %eyy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %exy[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='strain rate' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%e \n" %e[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,mV):
           vtufile.write("%.6e \n" % (exx[iel]+eyy[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   counter=0
   for iel in range(0,nel):
       vtufile.write("%d %d %d %d \n" %(counter,counter+1,counter+2,counter+3))
       counter+=4
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
