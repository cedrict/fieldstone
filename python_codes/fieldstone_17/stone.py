import numpy as np
import sys as sys
import scipy.sparse as sps
import time as clock
from basis_functions import *

###############################################################################
beta=0

experiment=2

if experiment==1:
   from functions_experiment1 import*
if experiment==2:
   from functions_experiment2 import*

###############################################################################

eps=1.e-10

print("*******************************")
print("********** stone 017 **********")
print("*******************************")

m_V=27    # number of velocity nodes making up an element
m_P=8     # number of pressure nodes making up an element
ndof_V=3  # number of velocity degrees of freedom per node

Lx=1.  # x- extent of the domain 
Ly=1.  # y- extent of the domain 
Lz=1.  # z- extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   visu = int(sys.argv[4])
else:
   nelx = 6
   nely = nelx
   nelz = nelx
   visu=1

nnx=2*nelx+1                      # number of elements, x direction
nny=2*nely+1                      # number of elements, y direction
nnz=2*nelz+1                      # number of elements, z direction
nn_V=nnx*nny*nnz                  # number of nodes
nel=nelx*nely*nelz                # number of elements, total
Nfem_V=nn_V*ndof_V                # number of velocity dofs
Nfem_P=(nelx+1)*(nely+1)*(nelz+1) # number of pressure dofs
Nfem=Nfem_V+Nfem_P                # total number of dofs

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

pnormalise=True

debug=False

###############################################################################
# quadrature points and weights
###############################################################################

nq_per_dim=3
qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
qweights=[5./9.,8./9.,5./9.]

###############################################################################

print("nelx",nelx)
print("nely",nely)
print("nelz",nelz)
print("nel",nel)
print("nnx=",nnx)
print("nny=",nny)
print("nnz=",nnz)
print("nn_V=",nn_V)
print("Nfem_V=",Nfem_V)
print("Nfem_P=",Nfem_P)
print("Nfem=",Nfem)
print("------------------------------")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64) # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64) # y coordinates
z_V=np.zeros(nn_V,dtype=np.float64) # z coordinates

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x_V[counter]=i*hx/2
            y_V[counter]=j*hy/2
            z_V[counter]=k*hz/2
            counter += 1
        #end for
    #end for
#end for

print("grid points setup: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_V[ 0,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon_V[ 1,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon_V[ 2,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon_V[ 3,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon_V[ 4,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon_V[ 5,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon_V[ 6,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon_V[ 7,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon_V[ 8,counter]=(2*k+1)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon_V[ 9,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon_V[10,counter]=(2*k+1)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon_V[11,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon_V[12,counter]=(2*k+3)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon_V[13,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon_V[14,counter]=(2*k+3)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon_V[15,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon_V[16,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+0) -1
            icon_V[17,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+2) -1
            icon_V[18,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+2) -1
            icon_V[19,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+0) -1
            icon_V[20,counter]=(2*k+1)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            icon_V[21,counter]=(2*k+2)+ nnz*(2*j+0) + nny*nnz*(2*i+1) -1
            icon_V[22,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+2) -1
            icon_V[23,counter]=(2*k+2)+ nnz*(2*j+2) + nny*nnz*(2*i+1) -1
            icon_V[24,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+0) -1
            icon_V[25,counter]=(2*k+3)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            icon_V[26,counter]=(2*k+2)+ nnz*(2*j+1) + nny*nnz*(2*i+1) -1
            counter += 1
        #end for
    #end for
#end for

if debug:
   print ('=======icon_V=======')
   for iel in range (0,nel):
       print ("iel=",iel)
       for i in range(0,m_V):
           print ("node",i,icon_V[i,iel],"at pos.",x_V[icon_V[i,iel]],y_V[icon_V[i,iel]],z_V[icon_V[i,iel]])

icon_P=np.zeros((m_P,nel),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_P[0,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k  
            icon_P[1,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k  
            icon_P[2,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k  
            icon_P[3,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k  
            icon_P[4,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j-1+1)+k+1  
            icon_P[5,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j-1+1)+k+1  
            icon_P[6,counter]=(nely+1)*(nelz+1)*(i  +1)+(nelz+1)*(j  +1)+k+1  
            icon_P[7,counter]=(nely+1)*(nelz+1)*(i-1+1)+(nelz+1)*(j  +1)+k+1  
            counter += 1
        #end for
    #end for
#end for

if debug:
   print ('=======icon_P=======')
   for iel in range (0,nel):
       print ("iel=",iel)
       for i in range(0,8):
           print ("node",i,icon_P[i,iel])

print("build connectivity: %.3f s" % (clock.time()-start))

###############################################################################
# define boundary conditions
###############################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps or y_V[i]<eps or z_V[i]<eps or \
       x_V[i]>(Lx-eps) or y_V[i]>(Ly-eps) or z_V[i]>(Lz-eps):
       bc_fix_V[i*ndof_V+0]=True ; bc_val_V[i*ndof_V+0]=uth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+1]=True ; bc_val_V[i*ndof_V+1]=vth(x_V[i],y_V[i],z_V[i])
       bc_fix_V[i*ndof_V+2]=True ; bc_val_V[i*ndof_V+2]=wth(x_V[i],y_V[i],z_V[i])

print("define b.c.: %.3f s" % (clock.time()-start))

###############################################################################
# build FE matrix
###############################################################################
start=clock.time()

K_mat = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat = np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
f_rhs = np.zeros(Nfem_V,dtype=np.float64)          # right hand side f 
h_rhs = np.zeros(Nfem_P,dtype=np.float64)          # right hand side h 
constr= np.zeros(Nfem_P,dtype=np.float64)          # constraint matrix/vector
B=np.zeros((6,ndof_V*m_V),dtype=np.float64)        # gradient matrix B 
N_mat = np.zeros((6,m_P),dtype=np.float64)         # matrix  
jcb=np.zeros((3,3),dtype=np.float64)

C=np.zeros((6,6),dtype=np.float64) 
C[0,0]=2.; C[1,1]=2.; C[2,2]=2.
C[3,3]=1.; C[4,4]=1.; C[5,5]=1.

for iel in range(0,nel):

    # set arrays to 0 every loop
    f_el=np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el=np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,m_P),dtype=np.float64)
    h_el=np.zeros(m_P,dtype=np.float64)
    NNNP=np.zeros(m_P,dtype=np.float64) # int of shape functions P

    # integrate viscous term at 3*3*3 quadrature points
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])
                for i in range(0,m_V):
                    B[0:6,3*i:3*i+3] = [[dNdx_V[i],0.       ,0.       ],
                                        [0.       ,dNdy_V[i],0.       ],
                                        [0.       ,0.       ,dNdz_V[i]],
                                        [dNdy_V[i],dNdx_V[i],0.       ],
                                        [dNdz_V[i],0.       ,dNdx_V[i]],
                                        [0.       ,dNdz_V[i],dNdy_V[i]]]

                K_el+=B.T.dot(C.dot(B))*viscosity(xq,yq,zq,beta)*JxWq

                # compute elemental rhs vector
                for i in range(0,m_V):
                    f_el[ndof_V*i+0]-=N_V[i]*bx(xq,yq,zq,beta)*JxWq
                    f_el[ndof_V*i+1]-=N_V[i]*by(xq,yq,zq,beta)*JxWq
                    f_el[ndof_V*i+2]-=N_V[i]*bz(xq,yq,zq,beta)*JxWq

                for i in range(0,m_P):
                    N_mat[0,i]=N_P[i]
                    N_mat[1,i]=N_P[i]
                    N_mat[2,i]=N_P[i]

                G_el-=B.T.dot(N_mat)*JxWq

                NNNP[:]+=N_P[:]*JxWq

            #end for
        #end for
    #end for

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
                    K_mat[m1,m2]+=K_el[ikk,jkk]
                #end for
            #end for
            for k2 in range(0,m_P):
                jkk=k2
                m2 =icon_P[k2,iel]
                G_mat[m1,m2]+=G_el[ikk,jkk]
            #end for
            f_rhs[m1]+=f_el[ikk]
        #end for
    #end for
    for k2 in range(0,m_P):
        m2=icon_P[k2,iel]
        h_rhs[m2]+=h_el[k2]
        constr[m2]+=NNNP[k2]
    #end for

#end for iel

print("     -> K_mat (m,M) %.4f %.4f " %(np.min(K_mat),np.max(K_mat)))
print("     -> G_mat (m,M) %.4f %.4f " %(np.min(G_mat),np.max(G_mat)))
print("     -> f_mat (m,M) %.4f %.4f " %(np.min(f_rhs),np.max(f_rhs)))
print("     -> h_mat (m,M) %.4f %.4f " %(np.min(h_rhs),np.max(h_rhs)))

print("build FE matrix: %.3f s" % (clock.time()-start))

###############################################################################
# assemble K, G, GT, f, h into A and rhs
###############################################################################
start=clock.time()

if pnormalise:
   A_fem=np.zeros((Nfem+1,Nfem+1),dtype=np.float64) # matrix of Ax=b
   b_fem=np.zeros(Nfem+1,dtype=np.float64)          # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   A_fem[Nfem,Nfem_V:Nfem]=constr
   A_fem[Nfem_V:Nfem,Nfem]=constr
else:
   A_fem=np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   b_fem=np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   A_fem[0:Nfem_V,0:Nfem_V]=K_mat
   A_fem[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_fem[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
#end if

b_fem[0:Nfem_V]=f_rhs
b_fem[Nfem_V:Nfem]=h_rhs

print("assemble blocks: %.3f s" % (clock.time()-start))

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

u,v,w=np.reshape(sol[0:Nfem_V],(nn_V,3)).T
p=sol[Nfem_V:Nfem]

print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
print("     -> w (m,M) %e %e " %(np.min(w),np.max(w)))
print("     -> p (m,M) %e %e " %(np.min(p),np.max(p)))
    
if pnormalise: print("     -> Lagrange multiplier: %e" % sol[Nfem])

if debug: np.savetxt('velocity.ascii',np.array([x_V,y_V,z_V,u,v,w]).T,header='# x,y,z,u,v,w')

print("transfer solution: %.3f s" % (clock.time()-start))

###############################################################################
# compute elemental strainrate 
###############################################################################
start=clock.time()

sr=np.zeros(nel,dtype=np.float64)  
x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
z_e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
ezz=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  
exz=np.zeros(nel,dtype=np.float64)  
eyz=np.zeros(nel,dtype=np.float64)  
eta=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq=0.
    sq=0.
    tq=0.
    N_V=basis_functions_V(rq,sq,tq)
    N_P=basis_functions_P(rq,sq,tq)
    dNdr_V=basis_functions_V_dr(rq,sq,tq)
    dNds_V=basis_functions_V_ds(rq,sq,tq)
    dNdt_V=basis_functions_V_dt(rq,sq,tq)
    jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
    jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
    jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
    jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
    jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
    jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
    jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
    jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
    jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
    jcbi=np.linalg.inv(jcb)
    JxWq=np.linalg.det(jcb)*weightq
    dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
    dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
    dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
    x_e[iel]=np.dot(N_V,x_V[icon_V[:,iel]])
    y_e[iel]=np.dot(N_V,y_V[icon_V[:,iel]])
    z_e[iel]=np.dot(N_V,z_V[icon_V[:,iel]])
    exx[iel]=np.dot(dNdx_V,u[icon_V[:,iel]])
    eyy[iel]=np.dot(dNdy_V,v[icon_V[:,iel]])
    ezz[iel]=np.dot(dNdz_V,w[icon_V[:,iel]])
    exy[iel]=(np.dot(dNdy_V,u[icon_V[:,iel]])+np.dot(dNdx_V,v[icon_V[:,iel]]))*0.5
    exz[iel]=(np.dot(dNdz_V,u[icon_V[:,iel]])+np.dot(dNdx_V,w[icon_V[:,iel]]))*0.5
    eyz[iel]=(np.dot(dNdz_V,v[icon_V[:,iel]])+np.dot(dNdy_V,w[icon_V[:,iel]]))*0.5
    eta[iel]=viscosity(x_e[iel],y_e[iel],z_e[iel],beta)
    sr[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel]+ezz[iel]*ezz[iel])
                    +exy[iel]*exy[iel]+exz[iel]*exz[iel]+eyz[iel]*eyz[iel])

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> ezz (m,M) %.4e %.4e " %(np.min(ezz),np.max(ezz)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))
print("     -> exz (m,M) %.4e %.4e " %(np.min(exz),np.max(exz)))
print("     -> eyz (m,M) %.4e %.4e " %(np.min(eyz),np.max(eyz)))
print("     -> eta (m,M) %.4e %.4e " %(np.min(eta),np.max(eta)))

if debug: np.savetxt('strainrate.ascii',np.array([x_e,y_e,z_e,exx,eyy,exy]).T,header='# x,y,exx,eyy,exy')

print("compute strainrate: %.3f s" % (clock.time()-start))

###############################################################################
# interpolate pressure onto velocity grid points
###############################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)

for iel in range(0,nel):
    q[icon_V[ 0,iel]]= p[icon_P[0,iel]]
    q[icon_V[ 1,iel]]= p[icon_P[1,iel]]
    q[icon_V[ 2,iel]]= p[icon_P[2,iel]]
    q[icon_V[ 3,iel]]= p[icon_P[3,iel]]
    q[icon_V[ 4,iel]]= p[icon_P[4,iel]]
    q[icon_V[ 5,iel]]= p[icon_P[5,iel]]
    q[icon_V[ 6,iel]]= p[icon_P[6,iel]]
    q[icon_V[ 7,iel]]= p[icon_P[7,iel]]
    q[icon_V[ 8,iel]]=(p[icon_P[0,iel]]+p[icon_P[1,iel]])*0.5
    q[icon_V[ 9,iel]]=(p[icon_P[1,iel]]+p[icon_P[2,iel]])*0.5
    q[icon_V[10,iel]]=(p[icon_P[2,iel]]+p[icon_P[3,iel]])*0.5
    q[icon_V[11,iel]]=(p[icon_P[3,iel]]+p[icon_P[0,iel]])*0.5
    q[icon_V[12,iel]]=(p[icon_P[4,iel]]+p[icon_P[5,iel]])*0.5
    q[icon_V[13,iel]]=(p[icon_P[5,iel]]+p[icon_P[6,iel]])*0.5
    q[icon_V[14,iel]]=(p[icon_P[6,iel]]+p[icon_P[7,iel]])*0.5
    q[icon_V[15,iel]]=(p[icon_P[7,iel]]+p[icon_P[4,iel]])*0.5
    q[icon_V[16,iel]]=(p[icon_P[0,iel]]+p[icon_P[4,iel]])*0.5
    q[icon_V[17,iel]]=(p[icon_P[1,iel]]+p[icon_P[5,iel]])*0.5
    q[icon_V[18,iel]]=(p[icon_P[2,iel]]+p[icon_P[6,iel]])*0.5
    q[icon_V[19,iel]]=(p[icon_P[3,iel]]+p[icon_P[7,iel]])*0.5
#end for

print("     -> q (m,M) %.4f %.4f " %(np.min(q),np.max(q)))

print("compute q : %.3f s" % (clock.time()-start))

########################################################################
# compute error in L2 norm 
#################################################################
start=clock.time()

errv=0.
errp=0.
errexx=0.
erreyy=0.
errezz=0.
errexy=0.
errexz=0.
erreyz=0.

for iel in range(0,nel):
    for iq in range(0,nq_per_dim):
        for jq in range(0,nq_per_dim):
            for kq in range(0,nq_per_dim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                tq=qcoords[kq]
                weightq=qweights[iq]*qweights[jq]*qweights[kq]
                N_V=basis_functions_V(rq,sq,tq)
                N_P=basis_functions_P(rq,sq,tq)
                dNdr_V=basis_functions_V_dr(rq,sq,tq)
                dNds_V=basis_functions_V_ds(rq,sq,tq)
                dNdt_V=basis_functions_V_dt(rq,sq,tq)
                jcb[0,0]=np.dot(dNdr_V,x_V[icon_V[:,iel]])
                jcb[0,1]=np.dot(dNdr_V,y_V[icon_V[:,iel]])
                jcb[0,2]=np.dot(dNdr_V,z_V[icon_V[:,iel]])
                jcb[1,0]=np.dot(dNds_V,x_V[icon_V[:,iel]])
                jcb[1,1]=np.dot(dNds_V,y_V[icon_V[:,iel]])
                jcb[1,2]=np.dot(dNds_V,z_V[icon_V[:,iel]])
                jcb[2,0]=np.dot(dNdt_V,x_V[icon_V[:,iel]])
                jcb[2,1]=np.dot(dNdt_V,y_V[icon_V[:,iel]])
                jcb[2,2]=np.dot(dNdt_V,z_V[icon_V[:,iel]])
                jcbi=np.linalg.inv(jcb)
                JxWq=np.linalg.det(jcb)*weightq
                dNdx_V=jcbi[0,0]*dNdr_V+jcbi[0,1]*dNds_V+jcbi[0,2]*dNdt_V
                dNdy_V=jcbi[1,0]*dNdr_V+jcbi[1,1]*dNds_V+jcbi[1,2]*dNdt_V
                dNdz_V=jcbi[2,0]*dNdr_V+jcbi[2,1]*dNds_V+jcbi[2,2]*dNdt_V
                xq=np.dot(N_V,x_V[icon_V[:,iel]])
                yq=np.dot(N_V,y_V[icon_V[:,iel]])
                zq=np.dot(N_V,z_V[icon_V[:,iel]])
                uq=np.dot(N_V,u[icon_V[:,iel]])
                vq=np.dot(N_V,v[icon_V[:,iel]])
                wq=np.dot(N_V,w[icon_V[:,iel]])
                pq=np.dot(N_P,p[icon_P[:,iel]])
                exxq=np.dot(dNdx_V,u[icon_V[:,iel]])
                eyyq=np.dot(dNdy_V,v[icon_V[:,iel]])
                ezzq=np.dot(dNdz_V,w[icon_V[:,iel]])
                exyq=(np.dot(dNdy_V,u[icon_V[:,iel]])+np.dot(dNdx_V,v[icon_V[:,iel]]))*0.5
                exzq=(np.dot(dNdz_V,u[icon_V[:,iel]])+np.dot(dNdx_V,w[icon_V[:,iel]]))*0.5
                eyzq=(np.dot(dNdz_V,v[icon_V[:,iel]])+np.dot(dNdy_V,w[icon_V[:,iel]]))*0.5
                errv+=((uq-uth(xq,yq,zq))**2+\
                       (vq-vth(xq,yq,zq))**2+\
                       (wq-wth(xq,yq,zq))**2)*JxWq
                errp+=(pq-pth(xq,yq,zq))**2*JxWq
                errexx+=(exxq-exx_th(xq,yq,zq))**2*JxWq
                erreyy+=(eyyq-eyy_th(xq,yq,zq))**2*JxWq
                errezz+=(ezzq-ezz_th(xq,yq,zq))**2*JxWq
                errexy+=(exyq-exy_th(xq,yq,zq))**2*JxWq
                errexz+=(exzq-exz_th(xq,yq,zq))**2*JxWq
                erreyz+=(eyzq-eyz_th(xq,yq,zq))**2*JxWq
            #end for
        #end for
    #end for
#end for

errv=np.sqrt(errv)
errp=np.sqrt(errp)
errexx=np.sqrt(errexx)
erreyy=np.sqrt(erreyy)
errezz=np.sqrt(errezz)
errexy=np.sqrt(errexy)
errexz=np.sqrt(errexz)
erreyz=np.sqrt(erreyz)

print("     -> nel= %6d ; errv: %e ; p: %e ; exx,eyy,ezz,exy,exz,eyz= %e %e %e %e %e %e"\
       %(nel,errv,errp,errexx,erreyy,errezz,errexy,errexz,erreyz))

print("compute errors: %.3f s" % (clock.time()-start))

###############################################################################
# plot of solution
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
       vtufile.write("%e %e %e \n" %(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   vtufile.write("</Points> \n")
   #####
   vtufile.write("<CellData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='element id' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%d\n" % iel)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e\n" % eta[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='e' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%f\n" % sr[iel])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e %e %e %e %e %e\n" % (exx[iel],eyy[iel],ezz[iel],exy[iel],eyz[iel],exz[iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (analytical)' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e %e %e %e %e %e\n" % (exx_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              eyy_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              ezz_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              exy_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              eyz_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              exz_th(x_e[iel],y_e[iel],z_e[iel]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (error)' Format='ascii'> \n")
   for iel in range(0,nel):
       vtufile.write("%e %e %e %e %e %e\n" % (exx[iel]-exx_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              eyy[iel]-eyy_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              ezz[iel]-ezz_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              exy[iel]-exy_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              eyz[iel]-eyz_th(x_e[iel],y_e[iel],z_e[iel]), \
                                              exz[iel]-exz_th(x_e[iel],y_e[iel],z_e[iel]) ))
   vtufile.write("</DataArray>\n")

   #--
   vtufile.write("</CellData>\n")
   #####
   vtufile.write("<PointData Scalars='scalars'>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (analytical)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10f %10f %10f \n" %(uth(x_V[i],y_V[i],z_V[i]),\
                                           vth(x_V[i],y_V[i],z_V[i]),\
                                           wth(x_V[i],y_V[i],z_V[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (error)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e \n" %(u[i]-uth(x_V[i],y_V[i],z_V[i]),\
                                     v[i]-vth(x_V[i],y_V[i],z_V[i]),\
                                     w[i]-wth(x_V[i],y_V[i],z_V[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' NumberOfComponents='6' Name='strainrate (analytical)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e %e %e %e %e %e\n" % (exx_th(x_V[i],y_V[i],z_V[i]), \
                                              eyy_th(x_V[i],y_V[i],z_V[i]), \
                                              ezz_th(x_V[i],y_V[i],z_V[i]), \
                                              exy_th(x_V[i],y_V[i],z_V[i]), \
                                              eyz_th(x_V[i],y_V[i],z_V[i]), \
                                              exz_th(x_V[i],y_V[i],z_V[i]) ))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%10e \n" %q[i])
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='p (analytical)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e\n" % pth(x_V[i],y_V[i],z_V[i]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Float32' Name='q (error)' Format='ascii'> \n")
   for i in range(0,nn_V):
       vtufile.write("%e\n" % (q[i]-pth(x_V[i],y_V[i],z_V[i])))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d%6d\n" %(icon_V[0,iel],icon_V[1,iel],icon_V[2,iel],icon_V[3,iel],icon_V[4,iel],icon_V[5,iel],icon_V[6,iel],icon_V[7,iel],icon_V[8,iel],icon_V[9,iel],icon_V[10,iel],icon_V[11,iel],icon_V[12,iel],icon_V[13,iel],icon_V[14,iel],icon_V[15,iel],icon_V[16,iel],icon_V[17,iel],icon_V[18,iel],icon_V[19,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*20))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %25)

       #THIS SHOULD BE CHANGED TO 33!

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
