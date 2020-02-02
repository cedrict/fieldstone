import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

sqrt3=np.sqrt(3.)
eps=1.e-10 

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

Lx=1.        # horizontal extent of the domain 
Ly=0.1       # vertical extent of the domain 
nelx = 10
nely =  4
hx=Lx/float(nelx)
hy=Ly/float(nely)

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofT=1      # number of degrees of freedom per node
hcapa=1.     # heat capacity
rho0=1       # reference density
f=1.
Pe=1
u0=1.
hcond=u0*hx*rho0*hcapa/2/Pe     # thermal conductivity
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
NV=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemT=NV*ndofT  # Total number of degrees of temperature freedom

#####################################################################

print("Lx=",Lx)
print("Ly=",Ly)
print("hx=",hx)
print("hy=",hy)
print("hcond=",hcond)
print("hcapa=",hcapa)
print("Pe=",Pe)

#####################################################################
# grid point setup 
#####################################################################

print("grid point setup")

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
u = np.zeros(NV,dtype=np.float64)  # x-component velocity
v = np.zeros(NV,dtype=np.float64)  # y-component velocity

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        u[counter]=u0
        v[counter]=0
        counter += 1
    #end for
#end for

#####################################################################
# connectivity
#####################################################################

print("connectivity array")

icon =np.zeros((m, nel),dtype=np.int32)
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

#####################################################################
# define temperature boundary conditions
#####################################################################

print("defining temperature boundary conditions")

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,NV):
    if x[i]/Lx<eps:
       bc_fixT[i]=True ; bc_valT[i]=0.
    if x[i]/Lx>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
#end for

#####################################################################
# compute analytical solution
#####################################################################

T_anal  = np.zeros(NV,dtype=np.float64)

for i in range(0,NV):
    T_anal[i]=(x[i]-(1-np.exp(2*Pe*x[i]/hx))/(1-np.exp(2*Pe/hx)))/u0*f

#####################################################################
# create necessary arrays 
#####################################################################

T     = np.zeros(NV,dtype=np.float64)
N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
B_mat = np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

for iel in range (0,nel):

    b_el=np.zeros(m*ndofT,dtype=np.float64)
    a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
    Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
    Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
    MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
    vel=np.zeros((1,ndim),dtype=np.float64)

    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate shape functions
            N_mat[0,0]=0.25*(1.-rq)*(1.-sq)
            N_mat[1,0]=0.25*(1.+rq)*(1.-sq)
            N_mat[2,0]=0.25*(1.+rq)*(1.+sq)
            N_mat[3,0]=0.25*(1.-rq)*(1.+sq)

            # calculate shape function derivatives
            dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
            dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
            dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
            dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)

            # calculate jacobian matrix
            jcb=np.zeros((2, 2),dtype=np.float64)
            for k in range(0,m):
                jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                jcb[1,1]+=dNds[k]*y[icon[k,iel]]

            # calculate the determinant of the jacobian
            jcob=np.linalg.det(jcb)

            # calculate inverse of the jacobian matrix
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
                b_el[k]=N_mat[k,0]*f*jcob*weightq

            # compute mass matrix
            MM=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

            # compute diffusion matrix
            Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

            # compute advection matrix
            Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

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
                   a_el[k1,k1]=Aref
                   b_el[k1]=Aref*bc_valT[m1]
                #end if
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

        #end for jq
    #end for iq
#end for iel

#################################################################
# solve system
#################################################################
start = timing.time()

T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
print("solve T time: %.3f s" % (timing.time() - start))

print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

np.savetxt('Temperature.ascii',np.array([x,y,T,T_anal]).T,header='# x,y,T')

#################################################################
# visualisation 
#################################################################

visu=0

if visu==1:
    vtufile=open('solution.vtu',"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
    for i in range(0,NV):
        vtufile.write("%10f \n" %T[i])
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

    filename = 'solution.pdf'
    fig = plt.figure ()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
    ax.set_xlabel ( 'X')
    ax.set_ylabel ( 'Y')
    ax.set_zlabel ( ' Temperature')
    plt.title('Peclet Nb: %e' %(Pe),loc='right')
    plt.grid ()
    plt.savefig(filename)
    #plt.show ()
    plt.close()
    
print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
