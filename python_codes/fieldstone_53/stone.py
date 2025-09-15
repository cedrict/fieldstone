import sys as sys
import numpy as np
import time as clock 
import scipy.sparse as sps
from scipy.sparse import lil_matrix

###############################################################################

def basis_functions_V(r,s):
    if serendipity==1:
       N0=(1-r)*(1-s)*(-r-s-1)*0.25
       N1=(1+r)*(1-s)*(r-s-1) *0.25
       N2=(1+r)*(1+s)*(r+s-1) *0.25
       N3=(1-r)*(1+s)*(-r+s-1)*0.25
       N4=(1-r**2)*(1-s)*0.5
       N5=(1+r)*(1-s**2)*0.5
       N6=(1-r**2)*(1+s)*0.5
       N7=(1-r)*(1-s**2)*0.5
       return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)
    else:
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
    if serendipity==1:
       dNdr0=-0.25*(s-1)*(2*r+s)
       dNdr1=-0.25*(s-1)*(2*r-s)
       dNdr2=0.25*(s+1)*(2*r+s)
       dNdr3=0.25*(s+1)*(2*r-s)
       dNdr4=r*(s-1)
       dNdr5=0.5*(1-s**2)
       dNdr6=-r*(s+1)
       dNdr7=-0.5*(1-s**2)
       return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)
    else:
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
    if serendipity==1:
       dNds0=-0.25*(r-1)*(r+2*s)
       dNds1=-0.25*(r+1)*(r-2*s)
       dNds2=0.25*(r+1)*(r+2*s)
       dNds3=0.25*(r-1)*(r-2*s)
       dNds4=-0.5*(1-r**2)
       dNds5=-(r+1)*s
       dNds6=0.5*(1-r**2)
       dNds7=(r-1)*s
       return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)
    else:
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

def eta(x,y):
    if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
       val=eta2
    else:
       val=eta1
    return val

def rho(x,y):
    if abs(x-xc_block)<d_block and abs(y-yc_block)<d_block:
       val=rho2 -rho1
    else:
       val=rho1 -rho1
    return val

###############################################################################

print("*******************************")
print("********** stone 53 ***********")
print("*******************************")

eps=1e-8
year=365.25*24*3600

ndim=2
ndof_V=2

Lx=512e3
Ly=512e3

if int(len(sys.argv) == 8):
   nelx=int(sys.argv[1])
   nely=int(sys.argv[2])
   visu=int(sys.argv[3])
   serendipity = int(sys.argv[4])
   drho=float(sys.argv[5])
   eta1=10.**(float(sys.argv[6]))
   eta2=10.**(float(sys.argv[7]))
else:
   nelx = 32
   nely = nelx
   visu = 1
   serendipity=0
   drho = 8
   eta1 = 1e21
   eta2 = 1e22

print(sys.argv[1:])

nel=nelx*nely

if serendipity==1:
   nn_V=(nelx+1)*(nely+1)+nelx*(nely+1)+ (nelx+1)*nely
   m_V=8
   m_P=4
else:
   nn_V=(2*nelx+1)*(2*nely+1)
   m_V=9
   m_P=4

nn_P=(nelx+1)*(nely+1)
Nfem_V=nn_V*ndof_V
Nfem_P=nn_P
Nfem=Nfem_V+Nfem_P

hx=Lx/nelx
hy=Ly/nely

print('nelx=',nelx)
print('nely=',nely)
print('nel=',nel)
print('nn_V=',nn_V)
print('nn_P=',nn_P)
print('Nfem_V=',Nfem_V)
print('Nfem_P=',Nfem_P)

nqperdim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

gy=-10.
rho1=3200.
rho2=rho1+drho
eta_ref=1e21      # scaling of G blocks

xc_block=256e3
yc_block=384e3
d_block=64e3

print('rho1=',rho1)
print('rho2=',rho2)
print('eta1=',eta1)
print('eta2=',eta2)

sparse=True
pnormalise=True

debug=False

if serendipity==1:
   r_V=[-1,1,1,-1,0,1,0,-1]
   s_V=[-1,-1,1,1,-1,0,1,0]
else:
   r_V=[-1,1,1,-1,0,1,0,-1,0]
   s_V=[-1,-1,1,1,-1,0,1,0,0]

print("*******************************")

###############################################################################
# grid point setup
###############################################################################
start = clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates

if serendipity==1:
   counter = 0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x_V[counter]=i*hx
           y_V[counter]=j*hy
           counter += 1

   for j in range(nely):
       for i in range(0,nelx):
           x_V[counter]=i*hx+hx/2
           y_V[counter]=j*hy
           counter+=1
       for i in range(0,nelx+1):
           x_V[counter]=i*hx
           y_V[counter]=j*hy+hy/2
           counter+=1

   for i in range(0,nelx):
       x_V[counter]=i*hx+hx/2
       y_V[counter]=nely*hy
       counter+=1

else:

   counter = 0
   for j in range(0,2*nely+1):
       for i in range(0,2*nelx+1):
           x_V[counter]=i*hx/2.
           y_V[counter]=j*hy/2.
           counter += 1

if debug: np.savetxt('gridV.ascii',np.array([x_V,y_V]).T,header='# x,y')

print("setup: grid points: %.3f s" % (clock.time() - start))

###############################################################################
# connectivity
###############################################################################
start = clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

if serendipity==1:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_V[0,counter]=i + j * (nelx + 1)
           icon_V[1,counter]=i + 1 + j * (nelx + 1)
           icon_V[2,counter]=i + 1 + (j + 1) * (nelx + 1)
           icon_V[3,counter]=i + (j + 1) * (nelx + 1)
           icon_V[4,counter]=(nelx+1)*(nely+1)+i +(2*nelx+1)*j
           icon_V[5,counter]=(nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)
           icon_V[6,counter]=(nelx+1)*(nely+1)+i +(2*nelx+1)*(j+1) 
           icon_V[7,counter]=(nelx+1)*(nely+1)+i +(2*nelx+1)*j + (nelx+1)-1
           counter += 1
else:
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
           counter += 1

###############################################################################
# build pressure grid and icon_P 
###############################################################################
start = clock.time()

x_P=np.empty(nn_P,dtype=np.float64)     # x coordinates
y_P=np.empty(nn_P,dtype=np.float64)     # y coordinates
icon_P=np.zeros((m_P,nel),dtype=np.int32)

if serendipity:
   icon_P[0:m_P,0:nel]=icon_V[0:m_P,0:nel]
   x_P[0:nn_P]=x_V[0:nn_P]
   y_P[0:nn_P]=y_V[0:nn_P]
else:
   counter=0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon_P[0,counter]=i+j*(nelx+1)
           icon_P[1,counter]=i+1+j*(nelx+1)
           icon_P[2,counter]=i+1+(j+1)*(nelx+1)
           icon_P[3,counter]=i+(j+1)*(nelx+1)
           counter += 1
   counter=0
   for j in range(0,nely+1):
       for i in range(0,nelx+1):
           x_P[counter]=i*hx
           y_P[counter]=j*hy
           counter += 1

if debug: np.savetxt('gridP.ascii',np.array([x_P,y_P]).T,header='# x,y')

print("build P grid: %.3f s" % (clock.time() - start))

###############################################################################
# define boundary conditions
###############################################################################
start = clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)       # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64) # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]/Lx<eps:
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
    if x_V[i]/Lx>(1-eps):
       bc_fix_V[i*ndof_V  ]=True ; bc_val_V[i*ndof_V  ]=0.
    if y_V[i]/Ly<eps:
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.
    if y_V[i]/Ly>(1-eps):
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=0.

print("setup: boundary conditions: %.3f s" % (clock.time() - start))

###############################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
###############################################################################
start = clock.time()

if sparse:
   A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
else:   
   K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
   G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT

N_mat=np.zeros((3,m_P),dtype=np.float64) # matrix  
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix B 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

constr  =np.zeros(Nfem_P,dtype=np.float64)         # constraint matrix/vector
f_rhs   =np.zeros(Nfem_V,dtype=np.float64)        # right hand side f 
h_rhs   =np.zeros(Nfem_P,dtype=np.float64)        # right hand side h 

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros((m_P),dtype=np.float64)
    NNNNP= np.zeros(m_P,dtype=np.float64)           # int of shape functions P

    for iq in range(0,nqperdim):
        for jq in range(0,nqperdim):
            rq=qcoords[iq]
            sq=qcoords[jq]
            weightq=qweights[iq]*qweights[jq]

            N_V=basis_functions_V(rq,sq)
            N_P=basis_functions_P(rq,sq)
            dNdr_V=basis_functions_V_dr(rq,sq)
            dNds_V=basis_functions_V_ds(rq,sq)

            # calculate jacobian matrix
            jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
            jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
            jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
            jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
            jcbi=np.linalg.inv(jcb)
            JxWq=np.linalg.det(jcb)*weightq # avoid jcob

            xq=np.dot(N_V,x_V[icon_V[:,iel]])
            yq=np.dot(N_V,y_V[icon_V[:,iel]])

            dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V
            dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V

            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental a_mat matrix
            K_el+=B.T.dot(C.dot(B))*eta(xq,yq)*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i+1]+=N_V[i]*rho(xq,yq)*gy*JxWq

            for i in range(0,m_P):
                N_mat[0,i]=N_P[i]
                N_mat[1,i]=N_P[i]
                N_mat[2,i]=0.

            G_el-=B.T.dot(N_mat)*JxWq

            NNNNP[:]+=N_P[:]*JxWq

        # end for jq
    # end for iq

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
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[:]-=G_el[ikk,:]*bc_val_V[m1]
               G_el[ikk,:]=0

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

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
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                if sparse:
                   A_sparse[m1,Nfem_V+m2]+=G_el[ikk,jkk]
                   A_sparse[Nfem_V+m2,m1]+=G_el[ikk,jkk]
                else:
                   G_mat[m1,m2]+=G_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNNP[k2]

if not sparse:
   print("     -> K_mat (m,M) %.4e %.4e " %(np.min(K_mat),np.max(K_mat)))
   print("     -> G_mat (m,M) %.4e %.4e " %(np.min(G_mat),np.max(G_mat)))

print("build FE matrix: %.3fs - %d elts" % (clock.time()-start, nel))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start = clock.time()

if not sparse:
   if pnormalise:
      a_mat=np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
      rhs=np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
      a_mat[0:Nfem_V,0:Nfem_V]=K_mat
      a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
      a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
      a_mat[Nfem,Nfem_V:Nfem]=constr
      a_mat[Nfem_V:Nfem,Nfem]=constr
   else:
      a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
      rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
      a_mat[0:Nfem_V,0:Nfem_V]=K_mat
      a_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
      a_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   #end if
else:
   rhs=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b

rhs[0:Nfem_V]=f_rhs
rhs[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time() - start))

###############################################################################
# solve system
###############################################################################
start = clock.time()

if sparse:
   sparse_matrix=A_sparse.tocsr()
else:
   sparse_matrix=sps.csr_matrix(a_mat)

sol=sps.linalg.spsolve(sparse_matrix,rhs)

print("solve time: %.3f s" % (clock.time() - start))

###############################################################################
# put solution into separate x,y velocity arrays
###############################################################################
start = clock.time()

u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

if pnormalise and not sparse:
   print("     -> Lagrange multiplier: %.4e" % sol[Nfem])

if debug:
   np.savetxt('velocity.ascii',np.array([x_V,y_V,u,v]).T,header='# x,y,u,v')
   np.savetxt('pressure.ascii',np.array([x_P,y_P,p]).T,header='# x,y,p')

print("split vel into u,v: %.3f s" % (clock.time() - start))

###############################################################################
# project pressure onto velocity grid
###############################################################################
start = clock.time()

q=np.zeros(nn_V,dtype=np.float64)
c=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    for i in range(0,m_V):
        N_P=basis_functions_P(r_V[i],s_V[i])
        q[icon_V[i,iel]]+=np.dot(N_P,p[icon_P[:,iel]])
        c[icon_V[i,iel]]+=1.
    # end for i
# end for iel

q/=c

if debug: np.savetxt('q.ascii',np.array([x_V,y_V,q]).T,header='# x,y,q')

print("project p onto Vnodes: %.3f s" % (clock.time() - start))

###############################################################################
# measure vel at center of block
###############################################################################
start = clock.time()

for i in range(0,nn_V):
    if abs(x_V[i]-xc_block)<1 and abs(y_V[i]-yc_block)<1:
       print('vblock=',eta1/eta2,np.abs(v[i])*eta1/drho,u[i]*year,v[i]*year)

for i in range(0,nn_P):
    if abs(x_P[i]-xc_block)<1 and abs(y_P[i]-yc_block)<1:
       print('pblock=',eta1/eta2,p[i]/drho/np.abs(gy)/128e3)

pline_file=open('pline.ascii',"w")
for i in range(0,nn_P):
    if abs(x_P[i]-xc_block)<1:
       pline_file.write("%10e %10e \n" %(y_P[i],p[i]))
pline_file.close()

print("carry out measurements: %.3f s" % (clock.time() - start))

#####################################################################
# plot of solution
#####################################################################
start = clock.time()

if visu==1:
    vtufile=open('solution.vtu',"w")
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
    if serendipity==1:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],\
                                                        icon_V[2,iel],icon_V[3,iel],\
                                                        icon_V[4,iel],icon_V[5,iel],\
                                                        icon_V[6,iel],icon_V[7,iel]))
    else:
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d %d \n" %(icon_V[0,iel],icon_V[1,iel],\
                                                           icon_V[2,iel],icon_V[3,iel],\
                                                           icon_V[4,iel],icon_V[5,iel],\
                                                           icon_V[6,iel],icon_V[7,iel],\
                                                           icon_V[8,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*m_V))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    if serendipity==1:
       for iel in range (0,nel):
           vtufile.write("%d \n" %23)
    else:
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
