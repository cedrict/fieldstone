import numpy as np
import time as clock
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################
# 1: donea & huerta
# 2: stokes sphere FS
#-2: stokes sphere NS
# 3: block FS
#-3: block NS
# 4: Burman & Hansbo 
# 5: inclusion Yamato

bench=1

###############################################################################

def bx(x,y):
    if bench==1:
       val=((12.-24.*y)*x**4+(-24.+48.*y)*x*x*x +
            (-48.*y+72.*y*y-48.*y*y*y+12.)*x*x +
            (-2.+24.*y-72.*y*y+48.*y*y*y)*x +
            1.-4.*y+12.*y*y-8.*y*y*y)
    if bench==2 or bench==-2: val=0 
    if bench==3 or bench==-3: val=0 
    if bench==4: val=0 
    if bench==5: val=0 
    return val

def by(x,y):
    if bench==1:
       val=((8.-48.*y+48.*y*y)*x*x*x+
            (-12.+72.*y-72.*y*y)*x*x+
            (4.-24.*y+48.*y*y-48.*y*y*y+24.*y**4)*x -
            12.*y*y+24.*y*y*y-12.*y**4)
    if bench==2 or bench==-2:
       if (x-0.5)**2+(y-0.5)**2<0.123456789**2:
          val=-1.01
       else:
          val=-1.
    if bench==3 or bench==-3:
       if abs(x-.5)<0.0625 and abs(y-0.5)<0.0625:
          val=-1.01
       else:
          val=-1.
    if bench==4: val=0 
    if bench==5: val=0 
    return val

###############################################################################

def viscosity(x,y):
    if bench==1: val=1
    if bench==2 or bench==-2:
       if (x-.5)**2+(y-0.5)**2<0.123456789**2:
          val=1000.
       else:
          val=1.
    if bench==3 or bench==-3:
       if abs(x-.5)<0.0625 and abs(y-0.5)<0.0625:
          val=1000
       else:
          val=1
    if bench==4: val=1
    if bench==5:
       if x**2+y**2<0.1**2:
          val=1.
       else:
          val=1e3
    return val

###############################################################################

def uth(x,y):
    if bench==1: val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    if bench==2 or bench==-2 or bench==3 or bench==-3 or bench==5: val=0
    if bench==4: val=20*x*y**3
    return val

def vth(x,y):
    if bench==1: val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    if bench==2 or bench==-2 or bench==3 or bench==-3 or bench==5: val=0
    if bench==4: val=5*x**4-5*y**4
    return val

def pth(x,y):
    if bench==1: val=x*(1.-x)-1./6.
    if bench==2 or bench==-2 or bench==3 or bench==-3 or bench==5: val=0
    if bench==4: val=60*x**2*y-20*y**3-5
    return val

###############################################################################

def basis_functions_V(r,s):
    N_0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N_1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N_2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N_3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N_4=    (1.-r**2) * 0.5*s*(s-1.)
    N_5= 0.5*r*(r+1.) *    (1.-s**2)
    N_6=    (1.-r**2) * 0.5*s*(s+1.)
    N_7= 0.5*r*(r-1.) *    (1.-s**2)
    N_8=    (1.-r**2) *    (1.-s**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr_0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr_1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr_2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr_3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr_4=       (-2.*r) * 0.5*s*(s-1)
    dNdr_5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr_6=       (-2.*r) * 0.5*s*(s+1)
    dNdr_7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr_8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3,dNdr_4,\
                     dNdr_5,dNdr_6,dNdr_7,dNdr_8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds_0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds_1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds_2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds_3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds_4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds_5= 0.5*r*(r+1.) *       (-2.*s)
    dNds_6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds_7= 0.5*r*(r-1.) *       (-2.*s)
    dNds_8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3,dNds_4,\
                     dNds_5,dNds_6,dNds_7,dNds_8],dtype=np.float64)

def basis_functions_P(r,s):
    N_0=0.25*(1-r)*(1-s)
    N_1=0.25*(1+r)*(1-s)
    N_2=0.25*(1+r)*(1+s)
    N_3=0.25*(1-r)*(1+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 018 **********")
print("*******************************")

m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   nq_per_dim = int(sys.argv[4])
else:
   nelx = 128 
   nely = nelx 
   visu = 1
   nq_per_dim=3
    
nnx=2*nelx+1           # number of nodes, x direction
nny=2*nely+1           # number of nodes, y direction
nn_V=nnx*nny           # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes
nel=nelx*nely          # number of elements, total
Nfem_V=nn_V*ndof_V     # number of velocity dofs
Nfem_P=nn_P            # number of pressure dofs
Nfem=Nfem_V+Nfem_P     # total number of dofs

hx=Lx/nelx             # element size, x direction
hy=Ly/nely             # element size, y direction


debug=False

###############################################################################
# boundary conditions
# NS: analytical velocity prescribed on all sides
# FS: free slip on all sides
# OT: open top, free slip sides and bottom
# BO: no slip sides, free slip bottom & top
# YA: bespoke for Yamato setup

FS=False
NS=False   
OT=False
BO=False
YA=False

if bench==1:  NS=True
if bench==2:  FS=True
if bench==-2: NS=True
if bench==3:  FS=True
if bench==-3: NS=True
if bench==4:  NS=True
if bench==5:  YA=True

###############################################################################

if nq_per_dim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]

if nq_per_dim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]

if nq_per_dim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]

if nq_per_dim==6:
   qcoords=[-0.932469514203152, -0.661209386466265, -0.238619186083197,\
            +0.238619186083197, +0.661209386466265, +0.932469514203152]
   qweights=[0.171324492379170,  0.360761573048139,  0.467913934572691,\
             0.467913934572691,  0.360761573048139,  0.171324492379170]

if nq_per_dim==10:
   qcoords=[-0.973906528517172, -0.865063366688985, -0.679409568299024,\
            -0.433395394129247, -0.148874338981631,  0.148874338981631,\
             0.433395394129247,  0.679409568299024,  0.865063366688985,\
             0.973906528517172]
   qweights=[0.066671344308688, 0.149451349150581, 0.219086362515982,\
             0.269266719309996, 0.295524224714753, 0.295524224714753,\
             0.269266719309996, 0.219086362515982, 0.149451349150581,\
             0.066671344308688]

###############################################################################

if OT or BO: pnormalise=False

if NS or FS or YA: pnormalise=True

print("bench=",bench)
print("nelx=",nelx)
print("nely=",nely)
print("nel=",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("pnormalise=",pnormalise)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for

x_P=np.zeros(nn_P,dtype=np.float64)  # x coordinates
y_P=np.zeros(nn_P,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nely+1):
    for i in range(0,nelx+1):
        x_P[counter]=i*hx
        y_P[counter]=j*hy
        counter += 1
    #end for
#end for

if debug: np.savetxt('grid_V.ascii',np.array([x_V,y_V]).T,header='# x,y')
if debug: np.savetxt('grid_P.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)
icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=(i)*2+1+(j)*2*nnx -1
        icon_V[1,counter]=(i)*2+3+(j)*2*nnx -1
        icon_V[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        icon_V[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        icon_V[4,counter]=(i)*2+2+(j)*2*nnx -1
        icon_V[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        icon_V[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        icon_V[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        icon_V[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter+=1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter+=1
    #end for
#end for

if debug:
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 0",icon_V[0,iel],"at pos.",x_V[icon_V[0,iel]],y_V[icon_V[0,iel]])
       print ("node 1",icon_V[1,iel],"at pos.",x_V[icon_V[1,iel]],y_V[icon_V[1,iel]])
       print ("node 2",icon_V[2,iel],"at pos.",x_V[icon_V[2,iel]],y_V[icon_V[2,iel]])
       print ("node 3",icon_V[3,iel],"at pos.",x_V[icon_V[3,iel]],y_V[icon_V[3,iel]])
       print ("node 4",icon_V[4,iel],"at pos.",x_V[icon_V[4,iel]],y_V[icon_V[4,iel]])
       print ("node 5",icon_V[5,iel],"at pos.",x_V[icon_V[5,iel]],y_V[icon_V[5,iel]])
       print ("node 6",icon_V[6,iel],"at pos.",x_V[icon_V[6,iel]],y_V[icon_V[6,iel]])
       print ("node 7",icon_V[7,iel],"at pos.",x_V[icon_V[7,iel]],y_V[icon_V[7,iel]])
       print ("node 8",icon_V[8,iel],"at pos.",x_V[icon_V[8,iel]],y_V[icon_V[8,iel]])
   for iel in range (0,nel):
       print ("iel=",iel)
       print ("node 0",icon_P[0,iel],"at pos.",x_P[icon_P[0,iel]],y_P[icon_P[0,iel]])
       print ("node 1",icon_P[1,iel],"at pos.",x_P[icon_P[1,iel]],y_P[icon_P[1,iel]])
       print ("node 2",icon_P[2,iel],"at pos.",x_P[icon_P[2,iel]],y_P[icon_P[2,iel]])
       print ("node 3",icon_P[3,iel],"at pos.",x_P[icon_P[3,iel]],y_P[icon_P[3,iel]])

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

if NS:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = uth(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vth(x_V[i],y_V[i])
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = uth(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vth(x_V[i],y_V[i])
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = uth(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vth(x_V[i],y_V[i])
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = uth(x_V[i],y_V[i])
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = vth(x_V[i],y_V[i])
       #end if
   #end for

elif FS:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       #end if
   #end for

elif OT:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       #end if
   #end for

elif BO:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       #end if
   #end for

elif YA:
   for i in range(0,nn_V):
       if x_V[i]<eps:
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       if x_V[i]>(Lx-eps):
          bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = -1
       if y_V[i]<eps:
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
       if y_V[i]>(Ly-eps):
          bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = +1
       #end if
   #end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
b_fem=np.zeros(Nfem,dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((2,2),dtype=np.float64)
#constr=np.zeros(Nfem_P,dtype=np.float64)         # constraint matrix/vector

mass=0.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)
    #NNNNP= np.zeros(m_P,dtype=np.float64)   

    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*viscosity(xq,yq)*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*bx(xq,yq)*JxWq
                f_el[ndof_V*i+1]+=N_V[i]*by(xq,yq)*JxWq
            
            mass+=abs(by(xq,yq))*JxWq

            N_mat[0,:]=N_P[:]
            N_mat[1,:]=N_P[:]

            G_el-=B.T.dot(N_mat)*JxWq 

            #NNNNP[:]+=NNNP[:]*JxWq

        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for 
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0
            #end if 
        #end for 
    #end for 

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1          +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            for k2 in range(0,m_V):
                for i2 in range(0,ndof_V):
                    jkk=ndof_V*k2          +i2
                    m2 =ndof_V*icon_V[k2,iel]+i2
                    A_fem[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
            #end for 
            b_fem[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        b_fem[Nfem_V+m2]+=h_el[k2]
        #constr[m2]+=NNNNP[k2]
        #if pnormalise:
        #   A_fem[Nfem,Nfem_V+m2]+=constr[m2]
        #   A_fem[Nfem_V+m2,Nfem]+=constr[m2]
    #end for 

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

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

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
if debug: np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time()-start))

###############################################################################
# normalise pressure
###############################################################################
start=clock.time()

if pnormalise:

   int_p=0
   for iel in range(0,nel):
       for iq in range(0,nq_per_dim):
           for jq in range(0,nq_per_dim):
               rq=qcoords[iq]
               sq=qcoords[jq]
               weightq=qweights[iq]*qweights[jq]
               N_P=basis_functions_P(rq,sq)
               pq=np.dot(N_P,p[icon_P[:,iel]])
               jcobq=hx*hy/4 # only if rectangular elements! 
               int_p+=pq*weightq*jcobq
           #end for
       #end for
   #end for

   avrg_p=int_p/(Lx*Ly)

   print("     -> int_p %e " %(int_p))
   print("     -> avrg_p %e " %(avrg_p))

   p[:]-=avrg_p

   print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

   if debug: np.savetxt('pressure_normalised.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental strainrate, stress, viscosity
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
pc=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
sigmaxx=np.zeros(nel,dtype=np.float64)  
sigmayy=np.zeros(nel,dtype=np.float64)  
sigmaxy=np.zeros(nel,dtype=np.float64)  

rq=0.
sq=0.
weightq=2.*2.

for iel in range(0,nel):

    N_V=basis_functions_V(rq,sq)
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    JxWq=np.linalg.det(jcb)*weightq
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])

    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

    pc[iel]=np.sum(p[icon_P[:,iel]])/m_P

    eta[iel]=viscosity(x_e[iel],y_e[iel])

    sigmaxx[iel]=-pc[iel]+2*eta[iel]*exx[iel]
    sigmayy[iel]=-pc[iel]+2*eta[iel]*eyy[iel]
    sigmaxy[iel]=         2*eta[iel]*exy[iel]

#end for

print("     -> eta (m,M) %.4e %.4e " %(np.min(eta),np.max(eta)))
print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> pc  (m,M) %.4e %.4e " %(np.min(pc), np.max(pc)))
    
divv=exx+eyy

if debug:
   np.savetxt('strainrate.ascii',np.array([x_e,y_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')
   np.savetxt('sigma.ascii',np.array([x_e,y_e,sigmaxx,sigmayy,sigmaxy]).T,header='# x,y,sigmaxx,sigmayy,sigmaxy')
   np.savetxt('pressure_elemental.ascii',np.array([x_e,y_e,pc]).T,header='# x,y,p')

print("compute press & sr: %.3f s" % (clock.time()-start))

###############################################################################
# compute error
###############################################################################
start=clock.time()

vrms=0.
errv=0.
errp=0.
avrgp=0.
for iel in range (0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])

            errv+=( (uq-uth(xq,yq))**2+(vq-vth(xq,yq))**2 )*JxWq

            vrms+=(uq**2+vq**2)*JxWq

            errp+=(pq-pth(xq,yq))**2*JxWq

            avrgp+=pq*JxWq

        #end for jq
    #end for iq
#end for iel

errv=np.sqrt(errv/Lx/Ly)
errp=np.sqrt(errp/Lx/Ly)
vrms=np.sqrt(vrms/Lx/Ly)
avrgp=avrgp/Lx/Ly

print("     -> avrgp= %e" %(avrgp))
print("     -> nel= %6d ; errv= %e ; errp= %e" %(nel,errv,errp))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    q[icon_V[0,iel]]=p[icon_P[0,iel]]
    q[icon_V[1,iel]]=p[icon_P[1,iel]]
    q[icon_V[2,iel]]=p[icon_P[2,iel]]
    q[icon_V[3,iel]]=p[icon_P[3,iel]]
    q[icon_V[4,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
    q[icon_V[5,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
    q[icon_V[6,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
    q[icon_V[7,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
    q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]+\
                      p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("from p to q: %.3f s" % (clock.time()-start))

###############################################################################
# export various measurements for stokes sphere benchmark 
###############################################################################
start=clock.time()

for i in range(0,nn_V):
    if abs(x_V[i]-0.5)<eps and abs(y_V[i]-0.5)<eps:
       uc=u[i]
       vc=abs(v[i])

vel=np.sqrt(u**2+v**2)
print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      0,0,\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),
      vrms,avrgp,mass,uc,vc)

profile=open('profile.ascii',"w")
for i in range(0,nn_V):
    if abs(x_V[i]-0.5)<1e-6:
       profile.write("%e %e %e %e \n" %(y_V[i],u[i],v[i],q[i]))

print("export measurements: %.3f s" % (clock.time()-start))
    
###############################################################################
# plot of solution
###############################################################################
start=clock.time()

filename = 'solution.vtu'
vtufile=open(filename,"w")
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
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
exx.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
eyy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
exy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
divv.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eta' Format='ascii'> \n")
eta.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
pc.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='sigmaxx' Format='ascii'> \n")
sigmaxx.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='sigmayy' Format='ascii'> \n")
sigmayy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='sigmaxy' Format='ascii'> \n")
sigmaxy.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
q.tofile(vtufile,sep=' ',format='%.4e')
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='p (th)' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %pth(x_V[i],y_V[i]))
vtufile.write("</DataArray>\n")
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
                                                   icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],\
                                                   icon_V[6,iel],icon_V[7,iel],icon_V[8,iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d \n" %((iel+1)*9))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
for iel in range (0,nel):
    vtufile.write("%d \n" %28)
vtufile.write("</DataArray>\n")
#--
vtufile.write("</Cells>\n")
#####
vtufile.write("</Piece>\n")
vtufile.write("</UnstructuredGrid>\n")
vtufile.write("</VTKFile>\n")
vtufile.close()

print("export solution to vtu: %.3f s" % (clock.time()-start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
