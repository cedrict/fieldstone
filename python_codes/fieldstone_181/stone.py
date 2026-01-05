import numpy as np
import math as math
import sys as sys
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import time as clock 

###############################################################################

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

###############################################################################

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
    val=K*(rho1-rho2)/2/eta2*(Ly/2.)*gy*amplitude
    return val

###############################################################################

def basis_functions_V(r,s):
    N0= 0.5*r*(r-1.) * 0.5*s*(s-1.)
    N1= 0.5*r*(r+1.) * 0.5*s*(s-1.)
    N2= 0.5*r*(r+1.) * 0.5*s*(s+1.)
    N3= 0.5*r*(r-1.) * 0.5*s*(s+1.)
    N4=    (1.-r**2) * 0.5*s*(s-1.)
    N5= 0.5*r*(r+1.) *    (1.-s**2)
    N6=    (1.-r**2) * 0.5*s*(s+1.)
    N7= 0.5*r*(r-1.) *    (1.-s**2)
    N8=    (1.-r**2) *    (1.-s**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr0= 0.5*(2.*r-1.) * 0.5*s*(s-1)
    dNdr1= 0.5*(2.*r+1.) * 0.5*s*(s-1)
    dNdr2= 0.5*(2.*r+1.) * 0.5*s*(s+1)
    dNdr3= 0.5*(2.*r-1.) * 0.5*s*(s+1)
    dNdr4=       (-2.*r) * 0.5*s*(s-1)
    dNdr5= 0.5*(2.*r+1.) *   (1.-s**2)
    dNdr6=       (-2.*r) * 0.5*s*(s+1)
    dNdr7= 0.5*(2.*r-1.) *   (1.-s**2)
    dNdr8=       (-2.*r) *   (1.-s**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds0= 0.5*r*(r-1.) * 0.5*(2.*s-1.)
    dNds1= 0.5*r*(r+1.) * 0.5*(2.*s-1.)
    dNds2= 0.5*r*(r+1.) * 0.5*(2.*s+1.)
    dNds3= 0.5*r*(r-1.) * 0.5*(2.*s+1.)
    dNds4=    (1.-r**2) * 0.5*(2.*s-1.)
    dNds5= 0.5*r*(r+1.) *       (-2.*s)
    dNds6=    (1.-r**2) * 0.5*(2.*s+1.)
    dNds7= 0.5*r*(r-1.) *       (-2.*s)
    dNds8=    (1.-r**2) *       (-2.*s)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)

def basis_functions_P(r,s):
    N0=0.25*(1-r)*(1-s)
    N1=0.25*(1+r)*(1-s)
    N2=0.25*(1+r)*(1+s)
    N3=0.25*(1-r)*(1+s)
    return np.array([N0,N1,N2,N3],dtype=np.float64)

###############################################################################

eps=1e-8

print("*******************************")
print("********** stone 040 **********")
print("*******************************")

ndim=2 
m_V=9    # number of velocity nodes making up an element
m_P=4    # number of pressure nodes making up an element
ndof_V=2 # number of velocity degrees of freedom per node

Lx=512e3  # horizontal extent of the domain 
Ly=512e3  # vertical extent of the domain 


if int(len(sys.argv)==8):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   llambda=float(sys.argv[4])
   amplitude=float(sys.argv[5])
   eta2=float(sys.argv[6])
   assembly=int(sys.argv[7])
else:
   nelx = 300
   nely = nelx
   visu = 1
   llambda=256e3
   amplitude=4000
   eta2=1e21
   # 1: old style
   # 2: via II,JJ,VV
   # 3: row[],col[],A_fem[]
   assembly=2

nel=nelx*nely  # number of elements, total

nn_V=(2*nelx+1)*(2*nely+1)  # number of velocity nodes
nn_P=(nelx+1)*(nely+1)      # number of pressure nodes

ndof_V_el=m_V*ndof_V

Nfem_V=nn_V*ndof_V # number of velocity dofs
Nfem_P=nn_P        # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

hx=Lx/nelx
hy=Ly/nely

debug=False

eta_ref=1e21 # scaling of G blocks

###############################################################################
# material parameters 
###############################################################################

gy=10
phi1=2.*np.pi*(Ly/2.)/llambda
phi2=2.*np.pi*(Ly/2.)/llambda
eta1=1e21
rho1=3300
rho2=3000

###############################################################################
# quadrature setup

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates

counter=0
for j in range(0,2*nely+1):
    for i in range(0,2*nelx+1):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1

if debug: np.savetxt('grid.ascii',np.array([x_V,y_V]).T,header='# x,y')

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

nnx=2*nelx+1
nny=2*nely+1

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

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter+=1
    #end for
#end for

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# deform grid 
###############################################################################
# velocity    pressure
# 3---6---2   3-------2
# |       |   |       |
# 7   8   5   |       |
# |       |   |       |
# 0---4---1   0-------1
###############################################################################
start=clock.time()

# move nodes on interface
for i in range(0,nn_V):
    if abs(y_V[i]-Ly/2.)/Ly<eps:
       y_V[i]+=amplitude*np.cos(2*np.pi*x_V[i]/llambda)

# move nodes vertically on each side of interface
counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        ya=256e3+amplitude*np.cos(2*np.pi*x_V[counter]/llambda)
        if j<nny/2:
           dy=ya/(nely/2)/2
           y_V[counter]=j*dy
        else:
           dy=(Ly-ya)/(nely/2)/2
           y_V[counter]=ya+(j-2*nely/2)*dy
        counter+=1

# make sure nodes 5,7,8 are in the middle
for iel in range(0,nel):
    y_V[icon_V[7,iel]]=(y_V[icon_V[0,iel]]+y_V[icon_V[3,iel]])/2.
    y_V[icon_V[5,iel]]=(y_V[icon_V[1,iel]]+y_V[icon_V[2,iel]])/2.
    y_V[icon_V[8,iel]]=(y_V[icon_V[4,iel]]+y_V[icon_V[6,iel]])/2.
#end for

if debug: np.savetxt('grid_after.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("deform grid: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental rho,eta and element center coords
###############################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
rho=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    r=0
    s=0
    N_V=basis_functions_V(r,s)
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    rho[iel]=density(xc[iel],yc[iel])
    eta[iel]=viscosity(xc[iel],yc[iel],eta1,eta2)

print("compute elemental rho,eta: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. # free slip
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. # free slip
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. # no slip
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0. # no slip
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# sanity check
###############################################################################
start=clock.time()

jcb=np.zeros((ndim,ndim),dtype=np.float64)
area=np.zeros(nel,dtype=np.float64) 

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_V=basis_functions_V(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq
            area[iel]+=JxWq
        #end for
    #end for
#end for

print("     -> area (m,M) %e %e " %(np.min(area),np.max(area)))
print("     -> total area %e %e" %(area.sum(),Lx*Ly))

print("sanity check and area: %.3f s" % (clock.time()-start))

###############################################################################
# compute array for assembly
###############################################################################
start=clock.time()

local_to_global_V=np.zeros((ndof_V_el,nel),dtype=np.int32)

for iel in range(0,nel):
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1+i1
            local_to_global_V[ikk,iel]=ndof_V*icon_V[k1,iel]+i1
                 
print("compute local_to_global_V: %.3f s" % (clock.time() - start))

###############################################################################
# fill I,J arrays
###############################################################################
start = clock.time()

if assembly==2:

   bignb=nel*( (m_V*ndof_V)**2 + 2*(m_V*ndof_V*m_P) )
   II_V=np.zeros(bignb,dtype=np.int32)    
   JJ_V=np.zeros(bignb,dtype=np.int32)    
   VV_V=np.zeros(bignb,dtype=np.float64)    

   counter=0
   for iel in range(0,nel):
       for ikk in range(ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           for jkk in range(ndof_V_el):
               m2=local_to_global_V[jkk,iel]
               II_V[counter]=m1
               JJ_V[counter]=m2
               counter+=1
           for jkk in range(0,m_P):
               m2 =icon_P[jkk,iel]+Nfem_V
               II_V[counter]=m1
               JJ_V[counter]=m2
               counter+=1
               II_V[counter]=m2
               JJ_V[counter]=m1
               counter+=1

print("fill II_V,JJ_V arrays: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start=clock.time()

b_fem=np.zeros(Nfem,dtype=np.float64)
jcb=np.zeros((ndim,ndim),dtype=np.float64)
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) 
N_mat=np.zeros((3,m_P),dtype=np.float64) 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

if assembly==1:
   A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)

if assembly==3:
   row=[] 
   col=[]
   A_fem=[]

time_bc=0
time_ass=0
time_mat=0

counter=0
for iel in range(0,nel):

    # set arrays to 0 every loop
    K_el =np.zeros((ndof_V_el,ndof_V_el),dtype=np.float64)
    G_el=np.zeros((ndof_V_el,m_P),dtype=np.float64)
    f_el =np.zeros((ndof_V_el),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    start2=clock.time()
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
            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            K_el+=B.T.dot(C.dot(B))*eta[iel]*JxWq

            for i in range(0,m_V):
                f_el[ndof_V*i+1]-=N_V[i]*rho[iel]*gy*JxWq

            N_mat[0,:]=N_P[:]
            N_mat[1,:]=N_P[:]

            G_el-=B.T.dot(N_mat)*JxWq

        # end for
    # end for
    time_mat+=clock.time()-start2

    G_el*=eta_ref/Lx

    if assembly==1: #----------------------------------------------------------

       #impose b.c. 
       start2=clock.time()
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
       time_bc+=clock.time()-start2

       # assemble matrix and right hand side
       start2=clock.time()
       for k1 in range(0,m_V):
           for i1 in range(0,ndof_V):
               ikk=ndof_V*k1          +i1
               m1 =ndof_V*icon_V[k1,iel]+i1
               for k2 in range(0,m_V):
                   for i2 in range(0,ndof_V):
                       jkk=ndof_V*k2          +i2
                       m2 =ndof_V*icon_V[k2,iel]+i2
                       A_fem[m1,m2] += K_el[ikk,jkk]
                       #end if
                   #end for
               #end for
               for k2 in range(0,m_P):
                   jkk=k2
                   m2 =icon_P[k2,iel]
                   A_fem[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_fem[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                   #end if
               b_fem[m1]+=f_el[ikk]
           #end for
       #end for
       for k2 in range(0,m_P):
           m2=icon_P[k2,iel]
           b_fem[Nfem_V+m2]+=h_el[k2]
       #end for
       time_ass+=clock.time()-start2

    if assembly==2: #----------------------------------------------------------

       # impose b.c. 
       start2=clock.time()
       for ikk in range(0,ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           if bc_fix_V[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,ndof_V_el):
                      f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                      K_el[ikk,jkk]=0
                      K_el[jkk,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val_V[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                  G_el[ikk,:]=0
       time_bc+=clock.time()-start2

       # assemble matrix and right hand side
       start2=clock.time()
       for ikk in range(ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           for jkk in range(ndof_V_el):
               VV_V[counter]=K_el[ikk,jkk]
               counter+=1
           for jkk in range(0,m_P):
               VV_V[counter]=G_el[ikk,jkk]
               counter+=1
               VV_V[counter]=G_el[ikk,jkk]
               counter+=1
           b_fem[m1]+=f_el[ikk]
       for k2 in range(0,m_P):
           m2=icon_P[k2,iel]
           b_fem[Nfem_V+m2]+=h_el[k2]
       time_ass+=clock.time()-start2

    if assembly==3: #----------------------------------------------------------

       #impose b.c. 
       start2=clock.time()
       for ikk in range(0,ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           if bc_fix_V[m1]:
                  K_ref=K_el[ikk,ikk] 
                  for jkk in range(0,ndof_V_el):
                      f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                  K_el[ikk,:]=0
                  K_el[:,ikk]=0
                  K_el[ikk,ikk]=K_ref
                  f_el[ikk]=K_ref*bc_val_V[m1]
                  h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
                  G_el[ikk,:]=0
       time_bc+=clock.time()-start2

       # assemble matrix and right hand side
       start2=clock.time()
       for ikk in range(ndof_V_el):
           m1=local_to_global_V[ikk,iel]
           for jkk in range(ndof_V_el):
               m2=local_to_global_V[jkk,iel]
               row.append(m1)
               col.append(m2)
               A_fem.append(K_el[ikk,jkk])
           for jkk in range(0,m_P):
               m2 =icon_P[jkk,iel]+Nfem_V
               row.append(m1)
               col.append(m2)
               A_fem.append(G_el[ikk,jkk])
               row.append(m2)
               col.append(m1)
               A_fem.append(G_el[ikk,jkk])
           b_fem[m1]+=f_el[ikk]
       #end for
       for k2 in range(0,m_P):
           m2=icon_P[k2,iel]
           b_fem[Nfem_V+m2]+=h_el[k2]
       time_ass+=clock.time()-start2

#end for iel

print('     -> time bc=',time_bc,Nfem)
print('     -> time assembly=',time_ass,Nfem)
print('     -> time matrices=',time_mat,Nfem)

print("build FE matrix: %.3f s | %d" % (clock.time()-start,Nfem))

###############################################################################
# convert matrix to csr 
###############################################################################
start=clock.time()

if assembly==1: A_fem=sps.csr_matrix(A_fem)
if assembly==2: A_fem=sps.coo_matrix((VV_V,(II_V,JJ_V)),shape=(Nfem,Nfem)).tocsr()
if assembly==3: A_fem=sps.csr_matrix((A_fem,(row,col)),shape=(Nfem,Nfem))

print("convert to csr: %.3f s | %d" % (clock.time()-start,Nfem))

###############################################################################
# solve system
###############################################################################
start=clock.time()

sol=sps.linalg.spsolve(A_fem,b_fem)

print("solve time: %.3f s | %d " % (clock.time()-start,Nfem))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]*(eta_ref/Lx)

print("     -> u (m,M) %.5e %.5e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.5e %.5e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

print("     -> vy/ampl, phi1 %.5e %.5e %.5e %d" %(np.max(abs(v)),phi1,vy_th(phi1,phi2,rho1,rho2),Nfem))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time() - start))

###############################################################################
# normalise pressure
###############################################################################
start=clock.time()

int_p=0.
for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            JxWq=np.linalg.det(jcb)*weightq
            pq=np.dot(N_P,p[icon_P[:,iel]])
            int_p+=pq*JxWq
        #end for
    #end for
#end for

p-=int_p/Lx/Ly

print("     -> p (m,M) %.5e %.5e " %(np.min(p),np.max(p)))

print("normalise pressure: %.3f s" % (clock.time()-start))

###############################################################################
# compute strainrate 
###############################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    dNdr_V=basis_functions_V_dr(rq,sq)
    dNds_V=basis_functions_V_ds(rq,sq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]**2+eyy[iel]**2)+exy[iel]**2)

print("     -> exx (m,M) %.5e %.5e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.5e %.5e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.5e %.5e " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('strainrate.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# xc,yc,exx,eyy,exy')

print("compute press & sr: %.3f s" % (clock.time() - start))

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

print("project pressure on V mesh: %.3f s" % (clock.time() - start))

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
    vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div(v)' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % (e[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % (rho[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%e\n" % (eta[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='Velocity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e %e %e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='bc u' Format='ascii'> \n")
for i in range(0,nn_V):
    if bc_fix_V[i*ndof_V]:
       vtufile.write("%e \n" %1.)
    else:
       vtufile.write("%e \n" %0.)
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='bc v' Format='ascii'> \n")
for i in range(0,nn_V):
    if bc_fix_V[i*ndof_V+1]:
       vtufile.write("%e \n" %1.)
    else:
       vtufile.write("%e \n" %0.)
vtufile.write("</DataArray>\n")
#--
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
    vtufile.write("%d \n" %((iel+1)*m_V))
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

print("export to vtu: %.3f s" % (clock.time() - start))

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
