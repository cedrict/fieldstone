import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import csr_matrix 
import time as time

#------------------------------------------------------------------------------

def NNV(rq,sq):
    N_0=0.25*(1.-rq)*(1.-sq)
    N_1=0.25*(1.+rq)*(1.-sq)
    N_2=0.25*(1.+rq)*(1.+sq)
    N_3=0.25*(1.-rq)*(1.+sq)
    return N_0,N_1,N_2,N_3

def dNNVdr(rq,sq):
    dNdr_0=-0.25*(1.-sq) 
    dNdr_1=+0.25*(1.-sq) 
    dNdr_2=+0.25*(1.+sq) 
    dNdr_3=-0.25*(1.+sq) 
    return dNdr_0,dNdr_1,dNdr_2,dNdr_3

def dNNVds(rq,sq):
    dNds_0=-0.25*(1.-rq)
    dNds_1=-0.25*(1.+rq)
    dNds_2=+0.25*(1.+rq)
    dNds_3=+0.25*(1.-rq)
    return dNds_0,dNds_1,dNds_2,dNds_3

#------------------------------------------------------------------------------

cm=0.01
year=365.25*3600*24

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2        # number of dimensions
m=4           # number of nodes making up an element
ndofV=ndim    # number of velocity degrees of freedom per node
ndofP=1       # number of pressure degrees of freedom 
ndofT=1       # number of temperature degrees of freedom 

Lx=3000e3               # horizontal extent of the domain 
Ly=3000e3               # vertical extent of the domain 
hcond=4            # thermal conductivity
hcapa=1250            # heat capacity
rho0=4000             # reference density
vel=0.01*cm/year
nelx = 100
nely = 100

#--------------------------------------

nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemT=nnp        # number of T dofs

hx=Lx/nelx
hy=Ly/nely

eps=1.e-10
sqrt3=np.sqrt(3.)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(nnp,dtype=np.float64)  # x coordinates
y = np.empty(nnp,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx) 
        y[counter]=j*Ly/float(nely) 
        counter += 1
    #end for
#end for

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# connectivity
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0,counter]=i+j*(nelx+1)
        icon[1,counter]=i+1+j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx + 1)
        icon[3,counter]=i+(j+1)*(nelx + 1)
        counter += 1
    #end for
#end for

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# define boundary conditions
#################################################################
start = time.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool) # boundary condition, yes/no
bc_valT=np.zeros(NfemT,dtype=np.float64)  # boundary condition, value

for i in range(0,nnp):
    if y[i]<eps:
       bc_fixT[i] = True ; bc_valT[i] = 4000. 
    #end if
    if y[i]/Ly>1-eps:
       bc_fixT[i] = True ; bc_valT[i] = 273. 
    #end if
#end for

print("setup: boundary conditions: %.3f s" % (time.time() - start))

#################################################################
# velocity field 
#################################################################
start = time.time()

u=np.zeros(nnp,dtype=np.float64)
v=np.zeros(nnp,dtype=np.float64)

for i in range(0,nnp):
    u[i]= np.sin(np.pi*x[i]/Lx)*np.cos(np.pi*y[i]/Ly)
    v[i]=-np.cos(np.pi*x[i]/Lx)*np.sin(np.pi*y[i]/Ly)

u*=vel
v*=vel

###############################################################################
# DO NOT MODIFY THE INDENTED CODE !!
###############################################################################

for istep in range(0,1):
    
    ######################################################################
    # build FE matrix for Temperature 
    ######################################################################
    start = time.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    N     = np.zeros(m,dtype=np.float64)             # shape functions
    dNdx  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdy  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNdr  = np.zeros(m,dtype=np.float64)             # shape functions derivatives
    dNds  = np.zeros(m,dtype=np.float64)             # shape functions derivatives

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)
        f_el=np.zeros(m*ndofT,dtype=np.float64)

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.

                # calculate shape functions
                N_mat[0:m,0]=NNV(rq,sq)
                dNdr[0:m]=dNNVdr(rq,sq)
                dNds[0:m]=dNNVds(rq,sq)

                # calculate jacobian matrix
                jcb=np.zeros((2, 2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                #end for
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)

                # compute dNdx & dNdy
                vel[0,0]=0.
                vel[0,1]=0.
                for k in range(0,m):
                    vel[0,0]+=N_mat[k,0]*u[icon[k,iel]]
                    vel[0,1]+=N_mat[k,0]*v[icon[k,iel]]
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                    B_mat[0,k]=dNdx[k]
                    B_mat[1,k]=dNdy[k]
                #end for

                # compute diffusion matrix
                Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

            # end for jq
        # end for iq

        a_el=Ka+Kd

        # apply boundary conditions

        for k1 in range(0,m):
            m1=icon[k1,iel]
            if bc_fixT[m1]:
               Aref=a_el[k1,k1]
               for k2 in range(0,m):
                   m2=icon[k2,iel]
                   b_el[k2]-=a_el[k2,k1]*bc_valT[m1]
                   a_el[k1,k2]=0
                   a_el[k2,k1]=0
               #end for
               a_el[k1,k1]=Aref
               b_el[k1]=Aref*bc_valT[m1]
            #end for
        #end for

        # assemble matrix A_mat and right hand side rhs
        for k1 in range(0,m):
            m1=icon[k1,iel]
            for k2 in range(0,m):
                m2=icon[k2,iel]
                A_mat[m1,m2]+=a_el[k1,k2]
            #end for
            rhs[m1]+=b_el[k1]
        #end for

    # end for iel

    print("build FEM matrix T: %.3f s" % (time.time() - start))

    start = time.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))
    print("solve T: %.3f s" % (time.time() - start))

#####################################################################
# generate pressure field ? 
#####################################################################

#####################################################################
# compute Nusselt number at surface 
#####################################################################




#####################################################################
# plot of solution
#####################################################################
start = time.time()

filename = 'solution_{:04d}.vtu'.format(istep) 
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='T' Format='ascii'> \n")
for i in range(0,nnp):
    vtufile.write("%10e \n" % T[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("</PointData>\n")
#####
vtufile.write("<Cells>\n")
#--
vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
for iel in range (0,nel):
    vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
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
    
print("generate vtu: %.3f s" % (time.time() - start))

#####################################################################
# compute T profile 
#####################################################################
start = time.time()

y_profile=np.zeros(nny,dtype=np.float64)
T_profile=np.zeros(nny,dtype=np.float64)
counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        T_profile[j]+=T[counter]/nnx
        y_profile[j]+=y[counter]/nnx
        counter += 1
    #end for
#end for
np.savetxt('T_profile.ascii',np.array([y_profile,T_profile]).T,header='# y,T')



print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
