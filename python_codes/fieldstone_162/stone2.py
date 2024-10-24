import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
import time as time
from scipy.sparse import lil_matrix
import sys as sys
import matplotlib.pyplot as plt

bench=1

#------------------------------------------------------------------------------

def NNVu(r,s):
    N_4=r*(1-r**2)*(1-s**2)
    N_0=0.25*(1.-r)*(1.-s) -0.25*N_4
    N_1=0.25*(1.+r)*(1.-s) -0.25*N_4
    N_2=0.25*(1.+r)*(1.+s) -0.25*N_4
    N_3=0.25*(1.-r)*(1.+s) -0.25*N_4
    return np.array([N_0,N_1,N_2,N_3,N_4],dtype=np.float64)

def dNNVudr(r,s):
    dNdr_4=(1-3*r**2)*(1-s**2)
    dNdr_0=-0.25*(1.-s) -0.25*dNdr_4
    dNdr_1=+0.25*(1.-s) -0.25*dNdr_4
    dNdr_2=+0.25*(1.+s) -0.25*dNdr_4
    dNdr_3=-0.25*(1.+s) -0.25*dNdr_4
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4],dtype=np.float64)

def dNNVuds(r,s):
    dNds_4=r*(1-r**2)*(-2*s)
    dNds_0=-0.25*(1.-r) -0.25*dNds_4
    dNds_1=-0.25*(1.+r) -0.25*dNds_4
    dNds_2=+0.25*(1.+r) -0.25*dNds_4
    dNds_3=+0.25*(1.-r) -0.25*dNds_4
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4],dtype=np.float64)

def NNVv(r,s):
    N_4=s*(1-r**2)*(1-s**2)
    N_0=0.25*(1.-r)*(1.-s) -0.25*N_4
    N_1=0.25*(1.+r)*(1.-s) -0.25*N_4
    N_2=0.25*(1.+r)*(1.+s) -0.25*N_4
    N_3=0.25*(1.-r)*(1.+s) -0.25*N_4
    return np.array([N_0,N_1,N_2,N_3,N_4],dtype=np.float64)

def dNNVvdr(r,s):
    dNdr_4=s*(-2*r)*(1-s**2)
    dNdr_0=-0.25*(1.-s) -0.25*dNdr_4
    dNdr_1=+0.25*(1.-s) -0.25*dNdr_4
    dNdr_2=+0.25*(1.+s) -0.25*dNdr_4
    dNdr_3=-0.25*(1.+s) -0.25*dNdr_4
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4],dtype=np.float64)

def dNNVvds(r,s):
    dNds_4=(1-r**2)*(1-3*s**2)
    dNds_0=-0.25*(1.-r) -0.25*dNds_4
    dNds_1=-0.25*(1.+r) -0.25*dNds_4
    dNds_2=+0.25*(1.+r) -0.25*dNds_4
    dNds_3=+0.25*(1.-r) -0.25*dNds_4
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4],dtype=np.float64)

#------------------------------------------------------------------------------

def bx(x, y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2:
       val=0
    return val

def by(x, y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2:
       val=0
    return val

#------------------------------------------------------------------------------

def velocity_x(x,y):
    if bench==1:
       val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2:
       val=20*x*y**3
    return val

def velocity_y(x,y):
    if bench==1:
       val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2:
       val=5*x**4-5*y**4
    return val

def pressure(x,y):
    if bench==1:
       val=x*(1.-x)-1./6.
    if bench==2:
       val=60*x**2*y-20*y**3-5
    return val

#------------------------------------------------------------------------------

eps=1.e-10
sqrt3=np.sqrt(3.)

print("-----------------------------")
print("---------- stone 162 --------")
print("-----------------------------")

mV=5     # number of nodes making up an element 4+1
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

if int(len(sys.argv) == 4):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
else:
   nelx = 128
   nely = 128
   visu = 1
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction

nel=nelx*nely  # number of elements, total

NV=nnx*nny+nel  # number of nodes

eta=1.  # dynamic viscosity 

NfemV=NV*ndofV   # number of velocity dofs
NfemP=nel*ndofP  # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

Gscaling=eta/(Ly/nely)

hx=Lx/nelx
hy=Ly/nely

###############################################################################

nqperdimK=3
nqperdimG=2

if nqperdimK==1:
   qcoordsK=[0.]
   qweightsK=[2.]

if nqperdimK==2:
   qcoordsK=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweightsK=[1.,1.]

if nqperdimK==3:
   qcoordsK=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweightsK=[5./9.,8./9.,5./9.]

if nqperdimG==1:
   qcoordsG=[0.]
   qweightsG=[2.]

if nqperdimG==2:
   qcoordsG=[-1./np.sqrt(3.),1./np.sqrt(3.)]
   qweightsG=[1.,1.]

if nqperdimG==3:
   qcoordsG=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweightsG=[5./9.,8./9.,5./9.]

nq=nqperdimK**2*nel

###############################################################################

print('NV=',NV)
print('NfemV=',NfemV)
print('NfemP=',NfemP)
print('Nfem=',Nfem)
print('nqperdimK',nqperdimK)
print('nqperdimG',nqperdimG)
print('nq',nq)

###############################################################################
# grid point setup
###############################################################################
start = time.time()

x=np.empty(NV,dtype=np.float64)  # x coordinates
y=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1

for j in range(0,nely):
    for i in range(0,nelx):
        x[counter]=(i+0.5)*hx
        y[counter]=(j+0.5)*hy
        counter += 1

#np.savetxt('mesh.ascii',np.array([x,y]).T,header='# x,y')

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        icon[4,counter] = nnx*nny +counter
        #print(icon[:,counter])
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value

for i in range(0,NV):
    if x[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if x[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if y[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
    if y[i]>(Ly-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = velocity_x(x[i],y[i])
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = velocity_y(x[i],y[i])
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

###############################################################################
# compute area of elements
###############################################################################
start = time.time()

xq    = np.zeros(nq,dtype=np.float64) 
yq    = np.zeros(nq,dtype=np.float64) 
area  = np.zeros(nel,dtype=np.float64) 
jcbi  = np.zeros((2,2),dtype=np.float64)
dNudx = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNudy = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNvdx = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNvdy = np.zeros(mV,dtype=np.float64)    # shape functions derivatives

counterq=0
for iel in range(0,nel):
    for iq in range(0,nqperdimK):
        for jq in range(0,nqperdimK):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]

            # calculate shape functions
            Nu=NNVu(rq,sq)
            Nv=NNVv(rq,sq)
            dNudr=dNNVudr(rq,sq)
            dNuds=dNNVuds(rq,sq)
            dNvdr=dNNVvdr(rq,sq)
            dNvds=dNNVvds(rq,sq)

            xq[counterq]=Nu.dot(x[icon[0:mV,iel]])
            yq[counterq]=Nv.dot(y[icon[0:mV,iel]])

            #compute jacobian matrix and determinant
            jcob=hx*hy/4
            jcbi[0,0]=2/hx ; jcbi[0,1]=0    
            jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

            # compute dNdx, dNdy
            for k in range(0,mV):
                dNudx[k]=jcbi[0,0]*dNudr[k]+jcbi[0,1]*dNuds[k]
                dNudy[k]=jcbi[1,0]*dNudr[k]+jcbi[1,1]*dNuds[k]
                dNvdx[k]=jcbi[0,0]*dNvdr[k]+jcbi[0,1]*dNvds[k]
                dNvdy[k]=jcbi[1,0]*dNvdr[k]+jcbi[1,1]*dNvds[k]

            #oneq=dNudx.dot(x[icon[0:mV,iel]])
            #print(oneq)
            #oneq=dNudx.dot(y[icon[0:mV,iel]])
            #print(oneq)
            #oneq=dNvdy.dot(x[icon[0:mV,iel]])
            #print(oneq)
            #oneq=dNvdy.dot(y[icon[0:mV,iel]])
            #print(oneq)

            area[iel]+=jcob*weightq
            counterq+=1

        #end for
    #end for
#end for

np.savetxt('gridq.ascii',np.array([xq,yq]).T,header='# x,y')
                
print("     -> vol  (m,M) %.6e %.6e " %(np.min(area),np.max(area)))
print("     -> total vol meas %.6f " %(area.sum()))
print("     -> total vol anal %.6f " %(Lx*Ly))

print("compute elements area: %.3f s" % (time.time() - start))

###############################################################################
# compute coordinates of element center
###############################################################################
start = time.time()

xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
for iel in range (0,nel):
    xc[iel]= 0.5*(x[icon[0,iel]]+x[icon[2,iel]])
    yc[iel]= 0.5*(y[icon[0,iel]]+y[icon[2,iel]])

print("compute elements center: %.3f s" % (time.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = time.time()

A_mat = lil_matrix((Nfem,Nfem),dtype=np.float64)# matrix A 
rhs   = np.zeros(Nfem,dtype=np.float64)  # right hand side 
b_mat = np.zeros((3,10),dtype=np.float64) # gradient matrix B size=3x9!
Nu    = np.zeros(mV,dtype=np.float64)    # shape functions
Nv    = np.zeros(mV,dtype=np.float64)    # shape functions
dNudx = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNudy = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNvdx = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
dNvdy = np.zeros(mV,dtype=np.float64)    # shape functions derivatives
c_mat = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) # a
#c_mat = np.array([[4/3,-2/3,0],[-2/3,4/3,0],[0,0,1]],dtype=np.float64)  #b
jcbi     = np.zeros((2,2),dtype=np.float64)

for iel in range(0,nel):

    #print('===================iel=',iel)

    # set arrays to 0 every loop
    K_el=np.zeros((10,10),dtype=np.float64)
    G_el=np.zeros((10,1),dtype=np.float64)
    f_el=np.zeros((10),dtype=np.float64)
    h_el=np.zeros((1),dtype=np.float64)

    # calculate jacobian matrix
    jcob=hx*hy/4
    jcbi[0,0]=2/hx ; jcbi[0,1]=0    
    jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

    for iq in range(0,nqperdimK):
        for jq in range(0,nqperdimK):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]

            # calculate shape functions
            Nu=NNVu(rq,sq)
            Nv=NNVv(rq,sq)
            dNudr=dNNVudr(rq,sq)
            dNuds=dNNVuds(rq,sq)
            dNvdr=dNNVvdr(rq,sq)
            dNvds=dNNVvds(rq,sq)

            xq=Nu.dot(x[icon[0:mV,iel]])
            yq=Nv.dot(y[icon[0:mV,iel]])

            for k in range(0,5):
                dNudx[k]=jcbi[0,0]*dNudr[k]+jcbi[0,1]*dNuds[k]
                dNudy[k]=jcbi[1,0]*dNudr[k]+jcbi[1,1]*dNuds[k]
                dNvdx[k]=jcbi[0,0]*dNvdr[k]+jcbi[0,1]*dNvds[k]
                dNvdy[k]=jcbi[1,0]*dNvdr[k]+jcbi[1,1]*dNvds[k]

            # construct 3x9 b_mat matrix
            b_mat[0,:]=[dNudx[0],0       ,dNudx[1],       0,dNudx[2],       0,dNudx[3],       0,dNudx[4],0]
            b_mat[1,:]=[       0,dNvdy[0],       0,dNvdy[1],       0,dNvdy[2],       0,dNvdy[3],0       ,dNvdy[4]]
            b_mat[2,:]=[dNudy[0],dNvdx[0],dNudy[1],dNvdx[1],dNudy[2],dNvdx[2],dNudy[3],dNvdx[3],dNudy[4],dNvdx[4]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta*weightq*jcob

            # compute elemental rhs vector
            f_el[0]+=Nu[0]*jcob*weightq*bx(xq,yq)
            f_el[1]+=Nv[0]*jcob*weightq*by(xq,yq)
            f_el[2]+=Nu[1]*jcob*weightq*bx(xq,yq)
            f_el[3]+=Nv[1]*jcob*weightq*by(xq,yq)
            f_el[4]+=Nu[2]*jcob*weightq*bx(xq,yq)
            f_el[5]+=Nv[2]*jcob*weightq*by(xq,yq)
            f_el[6]+=Nu[3]*jcob*weightq*bx(xq,yq)
            f_el[7]+=Nv[3]*jcob*weightq*by(xq,yq)
            f_el[8]+=Nu[4]*jcob*weightq*bx(xq,yq)
            f_el[9]+=Nv[4]*jcob*weightq*by(xq,yq)

        #end for jq
    #end for iq

    for iq in range(0,nqperdimG):
        for jq in range(0,nqperdimG):
            rq=qcoordsG[iq]
            sq=qcoordsG[jq]
            weightq=qweightsG[iq]*qweightsG[jq]

            Nu=NNVu(rq,sq)
            Nv=NNVv(rq,sq)
            dNudr=dNNVudr(rq,sq)
            dNuds=dNNVuds(rq,sq)
            dNvdr=dNNVvdr(rq,sq)
            dNvds=dNNVvds(rq,sq)

            for k in range(0,mV):
                dNudx[k]=jcbi[0,0]*dNudr[k]+jcbi[0,1]*dNuds[k]
                dNudy[k]=jcbi[1,0]*dNudr[k]+jcbi[1,1]*dNuds[k]
                dNvdx[k]=jcbi[0,0]*dNvdr[k]+jcbi[0,1]*dNvds[k]
                dNvdy[k]=jcbi[1,0]*dNvdr[k]+jcbi[1,1]*dNvds[k]

            G_el[0,0]-=dNudx[0]*jcob*weightq
            G_el[1,0]-=dNvdy[0]*jcob*weightq
            G_el[2,0]-=dNudx[1]*jcob*weightq
            G_el[3,0]-=dNvdy[1]*jcob*weightq
            G_el[4,0]-=dNudx[2]*jcob*weightq
            G_el[5,0]-=dNvdy[2]*jcob*weightq
            G_el[6,0]-=dNudx[3]*jcob*weightq
            G_el[7,0]-=dNvdy[3]*jcob*weightq
            G_el[8,0]-=dNudx[4]*jcob*weightq
            G_el[9,0]-=dNvdy[4]*jcob*weightq

        #end for jq
    #end for iq

    G_el*=Gscaling

    # impose b.c. 
    for k1 in range(0,4):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*icon[k1,iel]+i1
            if bc_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,10):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val[m1]
               h_el[0]-=G_el[ikk,0]*bc_val[m1]
               G_el[ikk,0]=0

    # assemble matrix K_mat and right hand side rhs
    for ikk in range(0,10):
        match(ikk):
            case(0): m1=ndofV*icon[0,iel]+0
            case(1): m1=ndofV*icon[0,iel]+1
            case(2): m1=ndofV*icon[1,iel]+0
            case(3): m1=ndofV*icon[1,iel]+1
            case(4): m1=ndofV*icon[2,iel]+0
            case(5): m1=ndofV*icon[2,iel]+1
            case(6): m1=ndofV*icon[3,iel]+0
            case(7): m1=ndofV*icon[3,iel]+1
            case(8): m1=ndofV*icon[4,iel]+0
            case(9): m1=ndofV*icon[4,iel]+1
            #end match
        for jkk in range(0,10):
            match(jkk):
                case(0): m2=ndofV*icon[0,iel]+0
                case(1): m2=ndofV*icon[0,iel]+1
                case(2): m2=ndofV*icon[1,iel]+0
                case(3): m2=ndofV*icon[1,iel]+1
                case(4): m2=ndofV*icon[2,iel]+0
                case(5): m2=ndofV*icon[2,iel]+1
                case(6): m2=ndofV*icon[3,iel]+0
                case(7): m2=ndofV*icon[3,iel]+1
                case(8): m2=ndofV*icon[4,iel]+0
                case(9): m2=ndofV*icon[4,iel]+1
            #end match
            #print(ikk,jkk,m1,m2)
            A_mat[m1,m2]+=K_el[ikk,jkk]
        #end for jkk 
        rhs[m1]+=f_el[ikk]
        A_mat[m1,NfemV+iel]+=G_el[ikk,0]
        A_mat[NfemV+iel,m1]+=G_el[ikk,0]
    #end for ikk 
    rhs[NfemV+iel]+=h_el[0]

#plt.spy(A_mat, markersize=0.6)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# solve system
######################################################################
start = time.time()

A_mat=A_mat.tocsr()

sol=sps.linalg.spsolve(A_mat,rhs)

print("solve time: %.3f s" % (time.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start = time.time()

u,v=np.reshape(sol[0:2*NV],(NV,2)).T
p=sol[NfemV:Nfem]*Gscaling

#uc,vc=np.reshape(sol[2*nnx*nny:NfemV],(nel,2)).T

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4f %.4f " %(np.min(p),np.max(p)))

np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')
np.savetxt('pressure.ascii',np.array([xc,yc,p]).T,header='# x,y,p')
#np.savetxt('velocity_middle.ascii',np.array([xc,yc,uc,vc]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure q
######################################################################
start = time.time()

q=np.zeros(nnx*nny,dtype=np.float64)  
count=np.zeros(nnx*nny,dtype=np.float64)  

for iel in range(0,nel):
    q[icon[0,iel]]+=p[iel]
    q[icon[1,iel]]+=p[iel]
    q[icon[2,iel]]+=p[iel]
    q[icon[3,iel]]+=p[iel]
    count[icon[0,iel]]+=1
    count[icon[1,iel]]+=1
    count[icon[2,iel]]+=1
    count[icon[3,iel]]+=1
#end for

q=q/count

#np.savetxt('q.ascii',np.array([x,y,q]).T,header='# x,y,q')

print("compute nodal pressure: %.3f s" % (time.time() - start))

######################################################################
# compute error
######################################################################
start = time.time()


error_u = np.empty(nnx*nny,dtype=np.float64)
error_v = np.empty(nnx*nny,dtype=np.float64)
error_q = np.empty(nnx*nny,dtype=np.float64)
error_p = np.empty(nnx*nny,dtype=np.float64)

for i in range(0,nnx*nny): 
    error_u[i]=u[i]-velocity_x(x[i],y[i])
    error_v[i]=v[i]-velocity_y(x[i],y[i])
    error_q[i]=q[i]-pressure(x[i],y[i])

for i in range(0,nel): 
    error_p[i]=p[i]-pressure(xc[i],yc[i])

errv=0.
errp=0.
for iel in range (0,nel):

    for iq in range(0,nqperdimK):
        for jq in range(0,nqperdimK):
            rq=qcoordsK[iq]
            sq=qcoordsK[jq]
            weightq=qweightsK[iq]*qweightsK[jq]

            # calculate shape functions
            # calculate shape functions
            Nu=NNVu(rq,sq)
            Nv=NNVv(rq,sq)

            jcob=hx*hy/4
            jcbi[0,0]=2/hx ; jcbi[0,1]=0    
            jcbi[1,0]=0    ; jcbi[1,1]=2/hy 

            xq=Nu.dot(x[icon[0:mV,iel]])
            yq=Nv.dot(y[icon[0:mV,iel]])

            uq=Nu[0]*u[icon[0,iel]]+\
               Nu[1]*u[icon[1,iel]]+\
               Nu[2]*u[icon[2,iel]]+\
               Nu[3]*u[icon[3,iel]]+\
               Nu[4]*u[icon[4,iel]]

            vq=Nv[0]*v[icon[0,iel]]+\
               Nv[1]*v[icon[1,iel]]+\
               Nv[2]*v[icon[2,iel]]+\
               Nv[3]*v[icon[3,iel]]+\
               Nv[4]*v[icon[4,iel]]

            errv+=((uq-velocity_x(xq,yq))**2+(vq-velocity_y(xq,yq))**2)*weightq*jcob
            errp+=(p[iel]-pressure(xq,yq))**2*weightq*jcob
        #end jq
    #end iq
#end iel

errv=np.sqrt(errv)
errp=np.sqrt(errp)

print("     -> nel= %6d ; errv= %.8f ; errp= %.8f" %(nel,errv,errp))

print("compute errors: %.3f s" % (time.time() - start))

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
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(x[icon[i,iel]],y[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   #vtufile.write("<CellData Scalars='scalars'>\n")
   #vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(u[icon[i,iel]],v[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%10f %10f %10f \n" %(error_u[icon[i,iel]],error_v[icon[i,iel]],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32'  Name='q' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" % q[icon[i,iel]])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" %p[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       for i in range(0,4):
           vtufile.write("%e \n" %error_p[iel])
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
