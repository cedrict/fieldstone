import numpy as np
import sys as sys
import time as clock
import matplotlib.pyplot as plt
from schur_complement_cg_solver import *
from schur_complement_cg_solver_LU import *
from scipy.sparse import csr_matrix,csc_matrix

###################################################################################################

def density(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def viscosity(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=1.e3
    else:
       val=1.
    return val

###################################################################################################

def basis_functions_V(r,s):
    N_0=0.25*(1.-r)*(1.-s)
    N_1=0.25*(1.+r)*(1.-s)
    N_2=0.25*(1.+r)*(1.+s)
    N_3=0.25*(1.-r)*(1.+s)
    return np.array([N_0,N_1,N_2,N_3],dtype=np.float64)

def basis_functions_V_dr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=+0.25*(1.+s) 
    dNdr_3=-0.25*(1.+s) 
    return np.array([dNdr_0,dNdr_1,dNdr_2,dNdr_3],dtype=np.float64)

def basis_functions_V_ds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.+r)
    dNds_3=+0.25*(1.-r)
    return np.array([dNds_0,dNds_1,dNds_2,dNds_3],dtype=np.float64)

###################################################################################################

eps=1.e-10
sqrt3=np.sqrt(3.)

print("*******************************")
print("********** stone 016 **********")
print("*******************************")

m_V=4     # number of nodes making up an element
ndof_V=2  # number of velocity degrees of freedom per node
ndim=2

Lx=1.  # horizontal extent of the domain 
Ly=1.  # vertical extent of the domain 

# allowing for argument parsing through command line
if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   visu = int(sys.argv[3])
   precond_type = int(sys.argv[4])
else:
   nelx = 3
   nely = 2 #nelx
   visu = 1
   precond_type=2
    
nnx=nelx+1          # number of nodes, x direction
nny=nely+1          # number of nodes, y direction
nn_V=nnx*nny        # number of velocity nodes
nel=nelx*nely       # number of elements, total
nn_P=nel            # number of pressure nodes
Nfem_V=nn_V*ndof_V  # number of velocity dofs
Nfem_P=nn_P         # number of pressure dofs
Nfem=Nfem_V+Nfem_P  # total number of dofs

hx=Lx/nelx
hy=Ly/nely

gx=0.
gy=-1.

use_SchurComplementApproach=True
use_preconditioner=True
niter_stokes=500
solver_tolerance=1e-8

eta_ref=10

# 1: computed on quad points
# 2: elemental viscosity
# 3: nodal+Q1 interp arithm
# 4: nodal+Q1 interp geom
# 5: nodal+Q1 interp harm

viscosity_field=2

use_LU=False

#################################################################
# grid point setup
#################################################################
start=clock.time()

x_V=np.zeros(nn_V,dtype=np.float64)  # x coordinates
y_V=np.zeros(nn_V,dtype=np.float64)  # y coordinates
eta_V=np.zeros(nn_V,dtype=np.float64) 

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x_V[counter]=i*hx
        y_V[counter]=j*hy
        eta_V[counter]=viscosity(x_V[counter],y_V[counter])
        counter+=1

print("setup: grid points: %.3f s" % (clock.time()-start))

#################################################################
# connectivity
#################################################################
start=clock.time()

icon_V=np.zeros((m_V,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon_V[0,counter]=i+j*(nelx+1)
        icon_V[1,counter]=i+1+j*(nelx+1)
        icon_V[2,counter]=i+1+(j+1)*(nelx+1)
        icon_V[3,counter]=i+(j+1)*(nelx+1)
        counter+=1

print("setup: connectivity: %.3f s" % (clock.time()-start))

#################################################################
# define boundary conditions: no slip on all sides
#################################################################
start=clock.time()

bc_fix_V=np.zeros(Nfem_V,dtype=bool)  # boundary condition, yes/no
bc_val_V=np.zeros(Nfem_V,dtype=np.float64)  # boundary condition, value

for i in range(0,nn_V):
    if x_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if x_V[i]>(Lx-eps):
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.
    if y_V[i]<eps:
       bc_fix_V[i*ndof_V  ] = True ; bc_val_V[i*ndof_V  ] = 0.
       bc_fix_V[i*ndof_V+1] = True ; bc_val_V[i*ndof_V+1] = 0.

print("setup: boundary conditions: %.3f s" % (clock.time()-start))

#################################################################
# compute element center coordinates
#################################################################
start=clock.time()

x_e=np.zeros(nel,dtype=np.float64)  
y_e=np.zeros(nel,dtype=np.float64)  
eta_e=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    x_e[iel]=(x_V[icon_V[0,iel]]+x_V[icon_V[2,iel]])*0.5 
    y_e[iel]=(y_V[icon_V[0,iel]]+y_V[icon_V[2,iel]])*0.5 
    eta_e[iel]=viscosity(x_e[iel],y_e[iel])

print("compute elt center: %.3f s" % (clock.time()-start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start=clock.time()

K_mat=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) # matrix K 
G_mat=np.zeros((Nfem_V,Nfem_P),dtype=np.float64) # matrix GT
C_mat=np.zeros((Nfem_P,Nfem_P),dtype=np.float64) # stays zero
f_rhs=np.zeros(Nfem_V,dtype=np.float64) # right hand side f 
h_rhs=np.zeros(Nfem_P,dtype=np.float64) # right hand side h 
B=np.zeros((3,ndof_V*m_V),dtype=np.float64) # gradient matrix 
C=np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 
jcb=np.zeros((ndim,ndim),dtype=np.float64)

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((m_V*ndof_V),dtype=np.float64)
    K_el =np.zeros((m_V*ndof_V,m_V*ndof_V),dtype=np.float64)
    G_el=np.zeros((m_V*ndof_V,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            N_V=basis_functions_V(rq,sq)
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

            if viscosity_field==1:    
               etaq=viscosity(xq,yq)

            elif viscosity_field==2:    
               etaq=eta_e[iel]

            elif viscosity_field==3:
               etaq=np.dot(N_V,eta_V[icon_V[:,iel]])

            elif viscosity_field==4:
               etaq=0.0
               for k in range(0,m_V):
                   etaq+=N_V[k]*np.log10(eta_V[icon_V[k,iel]])
               etaq=10.**etaq

            elif viscosity_field==5:
               etaq=0.0
               for k in range(0,m_V):
                   etaq+=N_V[k]*1./eta_V[icon_V[k,iel]]
               etaq=1./etaq

            # construct 3x8 B matrix
            for i in range(0,m_V):
                B[0:3,2*i:2*i+2]=[[dNdx_V[i],0.      ],
                                  [0.       ,dNdy_V[i]],
                                  [dNdy_V[i],dNdx_V[i]]]

            # compute elemental K_el matrix
            K_el+=B.T.dot(C.dot(B))*etaq*JxWq

            # compute elemental rhs vector
            for i in range(0,m_V):
                f_el[ndof_V*i  ]+=N_V[i]*JxWq*density(xq,yq)*gx
                f_el[ndof_V*i+1]+=N_V[i]*JxWq*density(xq,yq)*gy
                G_el[ndof_V*i  ,0]-=dNdx_V[i]*JxWq
                G_el[ndof_V*i+1,0]-=dNdy_V[i]*JxWq

        #end for jq
    #end for iq

    #GG_el=np.copy(G_el)*eta_ref/Ly tried smthg for G_mat

    # impose b.c. 
    for k1 in range(0,m_V):
        for i1 in range(0,ndof_V):
            ikk=ndof_V*k1           +i1
            m1 =ndof_V*icon_V[k1,iel]+i1
            if bc_fix_V[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,m_V*ndof_V):
                   f_el[jkk]-=K_el[jkk,ikk]*bc_val_V[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*bc_val_V[m1]
               h_el[0]-=G_el[ikk,0]*bc_val_V[m1]
               G_el[ikk,0]=0
            #end if
        #end for i1
    #end for k1

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
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for i1
    #end for k1
    h_rhs[iel]+=h_el[0,0]

print("     -> h_rhs (m,M) %.4e %.4e " %(np.min(h_rhs),np.max(h_rhs)))
print("     -> f_rhs (m,M) %.4e %.4e " %(np.min(f_rhs),np.max(f_rhs)))

print("build FE matrix: %.3f s" % (clock.time() - start))

######################################################################
# compute Schur preconditioner
######################################################################
start=clock.time()

M_mat=np.zeros((Nfem_P,Nfem_P),dtype=np.float64) # preconditioner 
   
if precond_type==0:
   for i in range(0,Nfem_P):
       M_mat[i,i]=1

if precond_type==1:
   for iel in range(0,nel):
       M_mat[iel,iel]=hx*hy/eta_e[iel]

if precond_type==2:
   Km1=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) 
   for i in range(0,Nfem_V):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))

if precond_type==3:
   Km1=np.zeros((Nfem_V,Nfem_V),dtype=np.float64) 
   for i in range(0,Nfem_V):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,Nfem_P):
       for j in range(0,Nfem_P):
           if i!=j:
              M_mat[i,j]=0.

if precond_type==4:
   Km1    = np.zeros((Nfem_V,Nfem_V),dtype=np.float64) 
   for i in range(0,Nfem_V):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,Nfem_P):
       for j in range(0,Nfem_P):
           if i!=j:
              M_mat[i,i]+=np.abs(M_mat[i,j])
              M_mat[i,j]=0.

plt.spy(M_mat)
plt.savefig('matrix.pdf', bbox_inches='tight')

print("build Schur matrix precond: %e s, nel= %d" % (clock.time() - start, nel))

######################################################################
# solve system  
######################################################################
start=clock.time()

if use_SchurComplementApproach:

   if precond_type==2 and use_LU:
      K_mat=csc_matrix(K_mat)
      G_mat=csr_matrix(G_mat)
      M_mat=csc_matrix(M_mat)
      solV,solP,niter=schur_complement_cg_solver_LU(K_mat,G_mat,C_mat,M_mat,f_rhs,h_rhs,\
                                                    Nfem_V,Nfem_P,niter_stokes,\
                                                    solver_tolerance,use_preconditioner)
   else:
      K_mat=csr_matrix(K_mat)
      G_mat=csr_matrix(G_mat)
      M_mat=csr_matrix(M_mat)
      solV,solP,niter=schur_complement_cg_solver(K_mat,G_mat,C_mat,M_mat,f_rhs,h_rhs,\
                                                 Nfem_V,Nfem_P,niter_stokes,\
                                                 solver_tolerance,use_preconditioner)
   u,v=np.reshape(solV[0:Nfem_V],(nn_V,2)).T
   p=solP[0:Nfem_P]*(eta_ref/Ly)

else:
   sol =np.zeros(Nfem,dtype=np.float64) 
   A_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  
   rhs   = np.zeros(Nfem,dtype=np.float64)        
   A_mat[0:Nfem_V,0:Nfem_V]=K_mat
   A_mat[0:Nfem_V,Nfem_V:Nfem]=G_mat
   A_mat[Nfem_V:Nfem,0:Nfem_V]=G_mat.T
   rhs[0:Nfem_V]=f_rhs
   rhs[Nfem_V:Nfem]=h_rhs
   sol=sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
   u,v=np.reshape(sol[0:Nfem_V],(nn_V,2)).T
   p=sol[Nfem_V:Nfem]*(eta_ref/Ly)

print("     -> converged after %d iterations, nel= %d " % (niter,nel))
print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("solve time: %.3f s, nel= %d" % (clock.time() - start, nel))

######################################################################
# compute strainrate 
######################################################################
start=clock.time()

e=np.zeros(nel,dtype=np.float64)  
exx=np.zeros(nel,dtype=np.float64)  
eyy=np.zeros(nel,dtype=np.float64)  
exy=np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    N_V=basis_functions_V(rq,sq)
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
    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])
#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.3f s" % (clock.time()-start))

######################################################################
# compute nodal pressure
######################################################################
start=clock.time()

q=np.zeros(nn_V,dtype=np.float64)  
count=np.zeros(nn_V,dtype=np.float64)  

for iel in range(0,nel):
    q[icon_V[0,iel]]+=p[iel]
    q[icon_V[1,iel]]+=p[iel]
    q[icon_V[2,iel]]+=p[iel]
    q[icon_V[3,iel]]+=p[iel]
    count[icon_V[0,iel]]+=1
    count[icon_V[1,iel]]+=1
    count[icon_V[2,iel]]+=1
    count[icon_V[3,iel]]+=1

q=q/count

print("compute nodal p: %.3f s" % (clock.time()-start))

#####################################################################
# export solution to vtu file
#####################################################################
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
    vtufile.write("%10e %10e %10e \n" %(x_V[i],y_V[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<CellData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' Name='div.v' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (exx[iel]+eyy[iel]))
vtufile.write("</DataArray>\n")
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
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (density(x_e[iel],y_e[iel])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (eta_e[iel]))
vtufile.write("</DataArray>\n")
#
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
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for i in range(0,nn_V):
    vtufile.write("%10e \n" %eta_V[i])
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
