import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse import csr_matrix,lil_matrix
import time as clock

#------------------------------------------------------------------------------

def rho(x,y):
    if (x)**2+(y-Ly/2)**2<0.123456789**2:
       val=1.01
    else:
       val=1.
    return val

def eta(x,y):
    if (x)**2+(y-Ly/2)**2<0.123456789**2:
       val=1000.
    else:
       val=1.
    return val

#------------------------------------------------------------------------------

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

eps=1.e-10

print("*******************************")
print("********** stone xyz **********")
print("*******************************")

m_V=9     # number of velocity nodes making up an element
m_P=4     # number of pressure nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node
   
Lx=0.5 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx     = int(sys.argv[1])
   visu     = int(sys.argv[2])
   nqperdim = int(sys.argv[3])
   Ly       = float(sys.argv[4])
else:
   nelx = 64
   visu = 1
   nqperdim=3
   Ly=1.  

nely=int(nelx*Ly/Lx)

axisymmetric=True
    
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction

nn_V=nnx*nny           # number of V nodes
nn_P=(nelx+1)*(nely+1) # number of P nodes

nel=nelx*nely  # number of elements, total

Nfem_V=nn_V*ndof_V   # number of velocity dofs
Nfem_P=nn_P         # number of pressure dofs
Nfem=Nfem_V+Nfem_P # total number of dofs

hx=Lx/nelx
hy=Ly/nely

pnormalise=True

gx=0
gy=-1

sparse=True

debug=True

###############################################################################

if nqperdim==3:
   qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
   qweights=[5./9.,8./9.,5./9.]
if nqperdim==4:
   qc4a=np.sqrt(3./7.+2./7.*np.sqrt(6./5.))
   qc4b=np.sqrt(3./7.-2./7.*np.sqrt(6./5.))
   qw4a=(18-np.sqrt(30.))/36.
   qw4b=(18+np.sqrt(30.))/36.
   qcoords=[-qc4a,-qc4b,qc4b,qc4a]
   qweights=[qw4a,qw4b,qw4b,qw4a]
if nqperdim==5:
   qc5a=np.sqrt(5.+2.*np.sqrt(10./7.))/3.
   qc5b=np.sqrt(5.-2.*np.sqrt(10./7.))/3.
   qc5c=0.
   qw5a=(322.-13.*np.sqrt(70.))/900.
   qw5b=(322.+13.*np.sqrt(70.))/900.
   qw5c=128./225.
   qcoords=[-qc5a,-qc5b,qc5c,qc5b,qc5a]
   qweights=[qw5a,qw5b,qw5c,qw5b,qw5a]
if nqperdim==6:
   qcoords=[-0.932469514203152,\
            -0.661209386466265,\
            -0.238619186083197,\
            +0.238619186083197,\
            +0.661209386466265,\
            +0.932469514203152]
   qweights=[0.171324492379170,\
             0.360761573048139,\
             0.467913934572691,\
             0.467913934572691,\
             0.360761573048139,\
             0.171324492379170]

###############################################################################

print("Lx",Lx)
print("Ly",Ly)
print("nelx",nelx)
print("nely",nely)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nn_V=",nn_V)
print("nn_P=",nn_P)
print("axisymmetric=",axisymmetric)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx/2.
        y_V[counter]=j*hy/2.
        counter += 1
    #end for
#end for

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
        counter += 1
    #end for
#end for

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_P[0,counter]=i+j*(nelx+1)
        icon_P[1,counter]=i+1+j*(nelx+1)
        icon_P[2,counter]=i+1+(j+1)*(nelx+1)
        icon_P[3,counter]=i+(j+1)*(nelx+1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]>(Ly-eps):
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start=clock.time()

if sparse:
   A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
else:
   K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

jcb=np.zeros((2,2),dtype=np.float64)
f_rhs = np.zeros(Nfem_V,dtype=np.float64)         # right hand side f 
h_rhs = np.zeros(Nfem_P,dtype=np.float64)         # right hand side h 
constr= np.zeros(Nfem_P,dtype=np.float64)         # constraint matrix/vector

if axisymmetric:
   C   = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]],dtype=np.float64) 
   B   = np.zeros((4,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
   N_mat   = np.zeros((4,m_P),dtype=np.float64) # matrix  
else:
   C   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
   B   = np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
   N_mat   = np.zeros((3,m_P),dtype=np.float64) # matrix  

mass=0.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):

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

            if axisymmetric: #-----------------------

               for i in range(0,m_V):
                   B[0:4, 2*i:2*i+2] = [[dNdx_V[i],0.     ],
                                        [N_V[i]/xq,0.     ],
                                        [0.       ,dNdy_V[i]],
                                        [dNdy_V[i],dNdx_V[i]]]
               K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq * 2*np.pi*xq
               for i in range(0,m_P):
                   N_mat[0,i]=N_P[i]
                   N_mat[1,i]=N_P[i]
                   N_mat[2,i]=N_P[i]
                   N_mat[3,i]=0.

               G_el-=B.T.dot(N_mat)*JxWq * 2*np.pi*xq

               mass+=JxWq*rho(xq,yq) * 2*np.pi*xq

               for i in range(0,m_V):
                   f_el[ndof_V*i  ]+=N_V[i]*JxWq*gx*rho(xq,yq) * 2*np.pi*xq
                   f_el[ndof_V*i+1]+=N_V[i]*JxWq*gy*rho(xq,yq) * 2*np.pi*xq
               #end for 

            else : #----------------------------------

               for i in range(0,m_V):
                   B[0:3, 2*i:2*i+2] = [[dNdx_V[i],0.       ],
                                        [0.       ,dNdy_V[i]],
                                        [dNdy_V[i],dNdx_V[i]]]
               #end for 
               K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq
               for i in range(0,m_P):
                   N_mat[0,i]=N_P[i]
                   N_mat[1,i]=N_P[i]
                   N_mat[2,i]=0.
               G_el-=B.T.dot(N_mat)*JxWq

               mass+=JxWq*rho(xq,yq)

               for i in range(0,m_V):
                   f_el[ndof_V*i  ]+=N_V[i]*JxWq*gx*rho(xq,yq)
                   f_el[ndof_V*i+1]+=N_V[i]*JxWq*gy*rho(xq,yq)
               #end for 

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
                    if sparse:
                       A_sparse[m1,m2] += K_el[ikk,jkk]
                    else:
                       K_mat[m1,m2]+=K_el[ikk,jkk]
                    #end if
                #end for
            #end for
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                if sparse:
                   A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
                #end if
            #end for 
            f_rhs[m1]+=f_el[ikk]
        #end for 
    #end for 
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
    #end for 

#end for iel

print("build FE matrix: %.3f s" % (clock.time()-start))

######################################################################
# assemble K, G, GT, f, h into A and rhs
######################################################################
start=clock.time()

if not sparse:
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
else:
   rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

######################################################################
# solve system
######################################################################
start=clock.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (clock.time() - start))

######################################################################
# put solution into separate x,y velocity arrays
######################################################################
start=clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')

print("split vel into u,v: %.3f s" % (clock.time() - start))

######################################################################
# compute strainrate 
######################################################################
start=clock.time()

xc=np.zeros(nel,dtype=np.float64)  
yc=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.0
    sq=0.0
    weightq=2.0*2.0
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
    xc[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    yc[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V[:],u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V[:],v[icon_V[:,iel]])
    exy[iel]=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
            +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

if debug: np.savetxt('strainrate_el.ascii',np.array([xc,yc,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute elemental strainrate: %.3f s" % (clock.time()-start))

######################################################################
# compute nodal strain rate 
######################################################################
start=clock.time()

r_V=np.array([-1,1,1,-1,0,1,0,-1,0],np.float64)
s_V=np.array([-1,-1,1,1,-1,0,1,0,0],np.float64)
exx_n=np.zeros(nn_V,dtype=np.float64)  
eyy_n=np.zeros(nn_V,dtype=np.float64)  
exy_n=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    for i in range(0,m_V):
        dNdr_V=basis_functions_V_dr(r_V[i],s_V[i])
        dNds_V=basis_functions_V_ds(r_V[i],s_V[i])
        jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
        dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V
        exx_n[icon_V[i,iel]]+=np.dot(dNdx_V[:],u[icon_V[:,iel]])
        eyy_n[icon_V[i,iel]]+=np.dot(dNdy_V[:],v[icon_V[:,iel]])
        exy_n[icon_V[i,iel]]+=np.dot(dNdy_V[:],u[icon_V[:,iel]])*0.5\
                             +np.dot(dNdx_V[:],v[icon_V[:,iel]])*0.5
        count[icon_V[i,iel]]+=1

exx_n/=count
eyy_n/=count
exy_n/=count

if debug: np.savetxt('strainrate_n.ascii',np.array([x_V,y_V,exx_n,eyy_n,exy_n]).T,header='# x,y,exx,eyy,exy')

print("compute nodal strainrate: %.3f s" % (clock.time()-start))

######################################################################
# compute vrms 
######################################################################
start=clock.time()

vrms=0.
avrgp=0.
for iel in range (0,nel):
    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
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
            uq=np.dot(N_V,u[icon_V[:,iel]])
            vq=np.dot(N_V,v[icon_V[:,iel]])
            pq=np.dot(N_P,p[icon_P[:,iel]])
            if axisymmetric:
               vrms+=(uq**2+vq**2)*JxWq *2*np.pi*xq
               avrgp+=pq*JxWq *2*np.pi*xq
            else:
               vrms+=(uq**2+vq**2)*JxWq
               avrgp+=pq*JxWq
        #end for jq
    #end for iq
#end for iel

vrms=np.sqrt(vrms/(np.pi*Lx**2*Ly))
avrgp=avrgp/(np.pi*Lx**2*Ly)

if pnormalise:
   p-=avrgp

print("     -> nel= %6d ; vrms= %e " %(nel,vrms))

print("compute vrms: %.3f s" % (clock.time()-start))

#####################################################################
# interpolate pressure onto velocity grid points
#####################################################################
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
    q[icon_V[8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]]\
                     +p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.25

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("compute q pressure: %.3f s" % (clock.time()-start))

#####################################################################

if axisymmetric:

   massT=np.pi*Lx**2*Ly*1 + 4./3.*np.pi*0.123456789**3*0.01

else:

   massT=Lx*Ly*1 + 0.5*np.pi*0.123456789**2*0.01

print('mass system (anal)', massT)
print('mass system (meas)', mass)


#####################################################################
# export various measurements for stokes sphere benchmark 
#####################################################################

for i in range(0,nn_V):
    if abs(x_V[i])<eps and abs(y_V[i]-Ly/2)<eps:
       uc=u[i]
       vc=abs(v[i])

vel=np.sqrt(u**2+v**2)

vs=2./9.*0.01/1*1*0.123456789**2

fh=-2.1050*(0.123456789/Lx)  \
   +2.0865*(0.123456789/Lx)**3  \
   -1.7068*(0.123456789/Lx)**5\
   +0.72603*(0.123456789/Lx)**6

gamma_h = (1-0.75857*(0.123456789/Lx)**5)/(1+fh)

vh=vs/gamma_h

ff=-2.10444*(0.123456789/Lx)\
   +2.08877*(0.123456789/Lx)**3\
   -0.94813*(0.123456789/Lx)**5\
   -1.372*(0.123456789/Lx)**6\
   +3.87*(0.123456789/Lx)**8\
   -4.19*(0.123456789/Lx)**10

gamma_f=1./(1+ff)

vf=vs/gamma_f

print('vs=',vs)
print('vh=',vh)
print('vf=',vf)

print('bench ',Lx/nelx,nel,Nfem,\
      np.min(u),np.max(u),\
      np.min(v),np.max(v),\
      0,0,\
      np.min(vel),np.max(vel),\
      np.min(p),np.max(p),\
      vrms,avrgp,mass,uc,vc,
      vs,vh,vf,massT)

#####################################################################
# plot of solution
#####################################################################

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_V,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (eyy[iel]))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exy[iel]))
vtufile.write("</DataArray>\n")
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
for i in range(0,nn_V):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %(rho(x_V[i],y_V[i])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %(eta(x_V[i],y_V[i])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],\
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

print("*******************************")
print("********** the end ************")
print("*******************************")
