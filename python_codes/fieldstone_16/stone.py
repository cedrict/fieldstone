import numpy as np
import math as math
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt
import schur_complement_cg_solver as cg

#------------------------------------------------------------------------------

def rho(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=2.
    else:
       val=1.
    return val

def eta(x,y):
    if (x-.5)**2+(y-0.5)**2<0.123**2:
       val=1.e3
    else:
       val=1.
    return val

#------------------------------------------------------------------------------

def NNV(r,s):
    N_0=0.25*(1.-r)*(1.-s)
    N_1=0.25*(1.+r)*(1.-s)
    N_2=0.25*(1.+r)*(1.+s)
    N_3=0.25*(1.-r)*(1.+s)
    return N_0,N_1,N_2,N_3

def dNNVdr(r,s):
    dNdr_0=-0.25*(1.-s) 
    dNdr_1=+0.25*(1.-s) 
    dNdr_2=+0.25*(1.+s) 
    dNdr_3=-0.25*(1.+s) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(r,s):
    dNds_0=-0.25*(1.-r)
    dNds_1=-0.25*(1.+r)
    dNds_2=+0.25*(1.+r)
    dNds_3=+0.25*(1.-r)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

mV=4     # number of nodes making up an element
ndofV=2  # number of velocity degrees of freedom per node
ndofP=1  # number of pressure degrees of freedom 
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
   nelx = 64
   nely = nelx
   visu = 1
   precond_type=2
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
NV=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NP=nel
NfemV=NV*ndofV   # number of velocity dofs
NfemP=NP*ndofP   # number of pressure dofs
Nfem=NfemV+NfemP # total number of dofs

hx=Lx/nelx
hy=Ly/nely

gx=0.
gy=-1.

use_SchurComplementApproach=True
use_preconditioner=True
niter_stokes=500
solver_tolerance=1e-8

eps=1.e-10
sqrt3=np.sqrt(3.)

eta_ref=10

#################################################################
# grid point setup
#################################################################
start = time.time()

xV=np.empty(NV,dtype=np.float64)  # x coordinates
yV=np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        xV[counter]=i*Lx/float(nelx)
        yV[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int32)
counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter] = i + j * (nelx + 1)
        iconV[1,counter] = i + 1 + j * (nelx + 1)
        iconV[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        iconV[3,counter] = i + (j + 1) * (nelx + 1)
        counter+=1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions: no slip on all sides
#################################################################
start = time.time()

bc_fix=np.zeros(NfemV,dtype=np.bool)  # boundary condition, yes/no
bc_val=np.zeros(NfemV,dtype=np.float64)  # boundary condition, value
for i in range(0,NV):
    if xV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if xV[i]>(Lx-eps):
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.
    if yV[i]<eps:
       bc_fix[i*ndofV  ] = True ; bc_val[i*ndofV  ] = 0.
       bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0.

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# build FE matrix
# [ K G ][u]=[f]
# [GT 0 ][p] [h]
#################################################################
start = time.time()

K_mat  = np.zeros((NfemV,NfemV),dtype=np.float64) # matrix K 
G_mat  = np.zeros((NfemV,NfemP),dtype=np.float64) # matrix GT
f_rhs  = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
h_rhs  = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
b_mat  = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
NNN    = np.zeros(mV,dtype=np.float64)            # shape functions
dNNNdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNdr = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
dNNNds = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
u      = np.zeros(NV,dtype=np.float64)          # x-component velocity
v      = np.zeros(NV,dtype=np.float64)          # y-component velocity
p      = np.zeros(nel,dtype=np.float64)          # y-component velocity
c_mat  = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

for iel in range(0, nel):

    # set arrays to 0 every loop
    f_el =np.zeros((mV*ndofV),dtype=np.float64)
    K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
    G_el=np.zeros((mV*ndofV,1),dtype=np.float64)
    h_el=np.zeros((1,1),dtype=np.float64)

    # integrate viscous term at 4 quadrature points
    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            NNN[0:mV]=NNV(rq,sq)
            dNNNdr[0:mV]=dNNVdr(rq,sq)
            dNNNds[0:mV]=dNNVds(rq,sq)

            # calculate jacobian matrix
            jcb = np.zeros((ndim,ndim),dtype=np.float64)
            for k in range(0,mV):
                jcb[0,0]+=dNNNdr[k]*xV[iconV[k,iel]]
                jcb[0,1]+=dNNNdr[k]*yV[iconV[k,iel]]
                jcb[1,0]+=dNNNds[k]*xV[iconV[k,iel]]
                jcb[1,1]+=dNNNds[k]*yV[iconV[k,iel]]
            jcob = np.linalg.det(jcb)
            jcbi = np.linalg.inv(jcb)

            # compute dNdx & dNdy
            xq=0.0
            yq=0.0
            for k in range(0,mV):
                xq+=NNN[k]*xV[iconV[k,iel]]
                yq+=NNN[k]*yV[iconV[k,iel]]
                dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
                dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]

            # construct 3x8 b_mat matrix
            for i in range(0,mV):
                b_mat[0:3, 2*i:2*i+2] = [[dNNNdx[i],0.     ],
                                         [0.     ,dNNNdy[i]],
                                         [dNNNdy[i],dNNNdx[i]]]

            # compute elemental a_mat matrix
            K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta(xq,yq)*weightq*jcob

            # compute elemental rhs vector
            for i in range(0,mV):
                f_el[ndofV*i  ]+=NNN[i]*jcob*weightq*rho(xq,yq)*gx
                f_el[ndofV*i+1]+=NNN[i]*jcob*weightq*rho(xq,yq)*gy
                G_el[ndofV*i  ,0]-=dNNNdx[i]*jcob*weightq
                G_el[ndofV*i+1,0]-=dNNNdy[i]*jcob*weightq

        #end for jq
    #end for iq

    # impose b.c. 
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1           +i1
            m1 =ndofV*iconV[k1,iel]+i1
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
            #end if
        #end for i1
    #end for k1

    G_el*=eta_ref/Ly
    h_el*=eta_ref/Ly

    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*iconV[k1,iel]+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*iconV[k2,iel]+i2
                    K_mat[m1,m2]+=K_el[ikk,jkk]
            f_rhs[m1]+=f_el[ikk]
            G_mat[m1,iel]+=G_el[ikk,0]
        #end for i1
    #end for k1
    h_rhs[iel]+=h_el[0]

print("     -> h_rhs (m,M) %.4e %.4e " %(np.min(h_rhs),np.max(h_rhs)))
print("     -> f_rhs (m,M) %.4e %.4e " %(np.min(f_rhs),np.max(f_rhs)))

print("build FE matrix: %.3f s" % (time.time() - start))

######################################################################
# compute Schur preconditioner
######################################################################
start = time.time()

M_mat = np.zeros((NfemP,NfemP),dtype=np.float64) 
   
xc = np.zeros(nel,dtype=np.float64)  
yc = np.zeros(nel,dtype=np.float64)  
for iel in range(0,nel):
       rq = 0.0
       sq = 0.0
       NNN[0:mV]=NNV(rq,sq)
       for k in range(0,mV):
           xc[iel] += NNN[k]*xV[iconV[k,iel]]
           yc[iel] += NNN[k]*yV[iconV[k,iel]]

if precond_type==0:
   for i in range(0,NfemP):
       M_mat[i,i]=1

if precond_type==1:
   for iel in range(0,nel):
       M_mat[iel,iel]=hx*hy/eta(xc[iel],yc[iel])

if precond_type==2:
   Km1    = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))

if precond_type==3:
   Km1    = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,NfemP):
       for j in range(0,NfemP):
           if i!=j:
              M_mat[i,j]=0.

if precond_type==4:
   Km1    = np.zeros((NfemV,NfemV),dtype=np.float64) 
   for i in range(0,NfemV):
       Km1[i,i]=1./K_mat[i,i] 
   M_mat=G_mat.T.dot(Km1.dot(G_mat))
   for i in range(0,NfemP):
       for j in range(0,NfemP):
           if i!=j:
              M_mat[i,i]+=np.abs(M_mat[i,j])
              M_mat[i,j]=0.

#plt.spy(M_mat)
#plt.savefig('matrix.pdf', bbox_inches='tight')

print("build Schur matrix precond: %.3f s" % (time.time() - start))

######################################################################
# solve system  
######################################################################
start = time.time()

if use_SchurComplementApproach:

   solV,solP,niter=cg.schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                   NfemV,NfemP,niter_stokes,solver_tolerance,use_preconditioner)

   u,v=np.reshape(solV[0:NfemV],(NV,2)).T
   p=solP[0:NfemP]*(eta_ref/Ly)

else:
   sol =np.zeros(Nfem,dtype=np.float64)  # x coordinates
   a_mat = np.zeros((Nfem,Nfem),dtype=np.float64)  # matrix of Ax=b
   rhs   = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
   a_mat[0:NfemV,0:NfemV]=K_mat
   a_mat[0:NfemV,NfemV:Nfem]=G_mat
   a_mat[NfemV:Nfem,0:NfemV]=G_mat.T
   rhs[0:NfemV]=f_rhs
   rhs[NfemV:Nfem]=h_rhs
   sol=sps.linalg.spsolve(sps.csr_matrix(a_mat),rhs)
   u,v=np.reshape(sol[0:NfemV],(NV,2)).T
   p=sol[NfemV:Nfem]*(eta_ref/Ly)

print("     -> converged after %d iterations, nel= %d " % (niter,nel))
print("     -> u (m,M) %.4e %.4e " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4e %.4e " %(np.min(v),np.max(v)))
print("     -> p (m,M) %.4e %.4e " %(np.min(p),np.max(p)))

print("solve time: %.3f s, nel= %d" % (time.time() - start, nel))

######################################################################
# compute strainrate 
######################################################################
start = time.time()

exx = np.zeros(nel,dtype=np.float64)  
eyy = np.zeros(nel,dtype=np.float64)  
exy = np.zeros(nel,dtype=np.float64)  
e   = np.zeros(nel,dtype=np.float64)  

for iel in range(0,nel):
    rq = 0.0
    sq = 0.0
    NNN[0:mV]=NNV(rq,sq)
    dNNNdr[0:mV]=dNNVdr(rq,sq)
    dNNNds[0:mV]=dNNVds(rq,sq)

    jcb=np.zeros((ndim,ndim),dtype=np.float64)
    for k in range(0,mV):
        jcb[0,0]+=dNNNdr[k]*xV[iconV[k,iel]]
        jcb[0,1]+=dNNNdr[k]*yV[iconV[k,iel]]
        jcb[1,0]+=dNNNds[k]*xV[iconV[k,iel]]
        jcb[1,1]+=dNNNds[k]*yV[iconV[k,iel]]
    jcob=np.linalg.det(jcb)
    jcbi=np.linalg.inv(jcb)

    for k in range(0,mV):
        dNNNdx[k]=jcbi[0,0]*dNNNdr[k]+jcbi[0,1]*dNNNds[k]
        dNNNdy[k]=jcbi[1,0]*dNNNdr[k]+jcbi[1,1]*dNNNds[k]

    for k in range(0,mV):
        exx[iel] += dNNNdx[k]*u[iconV[k,iel]]
        eyy[iel] += dNNNdy[k]*v[iconV[k,iel]]
        exy[iel] += 0.5*dNNNdy[k]*u[iconV[k,iel]]+\
                    0.5*dNNNdx[k]*v[iconV[k,iel]]

    e[iel]=np.sqrt(0.5*(exx[iel]*exx[iel]+eyy[iel]*eyy[iel])+exy[iel]*exy[iel])

#end for

print("     -> exx (m,M) %.4e %.4e " %(np.min(exx),np.max(exx)))
print("     -> eyy (m,M) %.4e %.4e " %(np.min(eyy),np.max(eyy)))
print("     -> exy (m,M) %.4e %.4e " %(np.min(exy),np.max(exy)))

print("compute press & sr: %.3f s" % (time.time() - start))

######################################################################
# compute nodal pressure
######################################################################

q=np.zeros(NV,dtype=np.float64)  
count=np.zeros(NV,dtype=np.float64)  

for iel in range(0,nel):
    q[iconV[0,iel]]+=p[iel]
    q[iconV[1,iel]]+=p[iel]
    q[iconV[2,iel]]+=p[iel]
    q[iconV[3,iel]]+=p[iel]
    count[iconV[0,iel]]+=1
    count[iconV[1,iel]]+=1
    count[iconV[2,iel]]+=1
    count[iconV[3,iel]]+=1

q=q/count

#####################################################################
# export solution to vtu file
#####################################################################

filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
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
    vtufile.write("%10e\n" % (rho(xc[iel],yc[iel])))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%10e\n" % (eta(xc[iel],yc[iel])))
vtufile.write("</DataArray>\n")

vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='q' Format='ascii'> \n")
for i in range(0,NV):
    vtufile.write("%10e \n" %q[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d \n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],iconV[3,iel]))
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

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
