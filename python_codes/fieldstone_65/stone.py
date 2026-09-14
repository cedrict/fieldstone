import numpy as np
import time as clock
import scipy.sparse as sps
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

###############################################################################

sqrt3=np.sqrt(3.)
eps=1.e-10 

print("*******************************")
print("********** stone 065 **********")
print("*******************************")

ndim=2 # number of space dimensions
m=4    # number of nodes making up an element

# exp 1: 1D problem
# exp 2: 2D skew to the mesh
# exp 3: 2D setup from Li, DG book
# exp 4: 2D heat flow around cylinder

experiment=4

if experiment==1:
   Lx=1.        # horizontal extent of the domain 
   Ly=0.1       # vertical extent of the domain 
   nelx = 10
   nely =  4
   hx=Lx/float(nelx)
   hy=Ly/float(nely)
   f=1.
   Pe=5
   u0=1.
   hcapa=1.     # heat capacity
   rho0=1       # reference density
   hcond=u0*hx*rho0*hcapa/2/Pe     # thermal conductivity

if experiment==2:
   Lx=1.
   Ly=1. 
   nelx = 10
   nely = 10
   hx=Lx/float(nelx)
   hy=Ly/float(nely)
   f=0.
   Pe=1e4
   u0=1.
   hcapa=1.     # heat capacity
   rho0=1       # reference density
   hcond=u0*hx*rho0*hcapa/2/Pe     # thermal conductivity

if experiment==3:
   Lx=1.
   Ly=1. 
   nelx = 32
   nely = 32
   hx=Lx/float(nelx)
   hy=Ly/float(nely)
   f=0.
   xi=1 # 1,10 or 100
   Pe=xi*hx/2
   u0=1.
   hcapa=1.     # heat capacity
   rho0=1       # reference density
   hcond=u0*hx*rho0*hcapa/2/Pe     # thermal conductivity

if experiment==4:
   Lx=10.
   Ly=4. 
   nelx = 250
   nely = 100
   hx=Lx/float(nelx)
   hy=Ly/float(nely)
   f=0.
   Pe=1
   u0=1.
   hcapa=1.     # heat capacity
   rho0=1       # reference density
   hcond=u0*hx*rho0*hcapa/2/Pe     # thermal conductivity

use_artificial_diffusion=False

use_supg=False

kappa=hcond/rho0/hcapa

nnx=nelx+1    # number of elements, x direction
nny=nely+1    # number of elements, y direction
nn_T=nnx*nny  # number of nodes
nel=nelx*nely # number of elements, total
NfemT=nn_T    # Total number of degrees of temperature freedom

#####################################################################

if use_artificial_diffusion and use_supg:
   exit("both options together not compatible")

if experiment==1:
   if use_artificial_diffusion:
      beta=np.cosh(Pe)/np.sinh(Pe)-1./Pe # coth(Pe)-1/Pe
   else:
      beta=0
   #beta=1

if experiment==2:
   if use_artificial_diffusion:
      beta=1.                      # Pe>>1 so coth(Pe)->1 
   else:
      beta=0

if experiment==3 or experiment==4: #????
   if use_artificial_diffusion:
      beta=1.                      # Pe>>1 so coth(Pe)->1 
   else:
      beta=0

kappatilde=beta*kappa*Pe
hcondtilde=kappatilde*rho0*hcapa

print("Lx=",Lx)
print("Ly=",Ly)
print("hx=",hx)
print("hy=",hy)
print("rho0=",rho0)
print("hcond=",hcond)
print("hcapa=",hcapa)
print("kappa=",hcond/rho0/hcapa)
print("Pe=",Pe)
print("beta=",beta)
print("kappatilde=",kappatilde)
print("*******************************")

###############################################################################
# grid point setup 
###############################################################################
start=clock.time()

x=np.zeros(nn_T,dtype=np.float64)  # x coordinates
y=np.zeros(nn_T,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter += 1
    #end for
#end for

print("node coordinates: %.3f s" % (clock.time()-start))

###############################################################################
# velocity field 
###############################################################################
start=clock.time()

u=np.zeros(nn_T,dtype=np.float64)  # x-component velocity
v=np.zeros(nn_T,dtype=np.float64)  # y-component velocity

if experiment==1 or experiment==3:
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           u[counter]=u0
           v[counter]=0
           counter += 1
       #end for
   #end for

if experiment==2:
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           u[counter]=u0*np.cos(30./180.*np.pi)
           v[counter]=u0*np.sin(30./180.*np.pi)
           counter += 1
       #end for
   #end for

if experiment==4:
   counter = 0
   for j in range(0,nny):
       for i in range(0,nnx):
           xi=x[counter]-2
           yi=y[counter]-2
           if xi**2+yi**2<1:
              u[counter]=0
              v[counter]=0
           else:
              u[counter]=u0*(1-(xi**2-yi**2)/(xi**2+yi**2)**2)
              v[counter]=-2*u0*xi*yi/(xi**2+yi**2)**2
           counter += 1
       #end for
   #end for

print("prescribe vel field: %.3f s" % (clock.time()-start))

###############################################################################
# connectivity
###############################################################################
start=clock.time()

icon=np.zeros((m,nel),dtype=np.int32)

counter=0
for j in range(0,nely):
    for i in range(0,nelx):
        icon[0,counter] = i + j * (nelx + 1)
        icon[1,counter] = i + 1 + j * (nelx + 1)
        icon[2,counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3,counter] = i + (j + 1) * (nelx + 1)
        counter+=1
    #end for
#end for

print("build connectivity: %.3f s" % (clock.time()-start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start=clock.time()

bc_fixT=np.zeros(NfemT,dtype=bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

if experiment==1:
   for i in range(0,nn_T):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.

if experiment==2:
   for i in range(0,nn_T):
       if x[i]/Lx<eps and y[i]<0.2:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx<eps and y[i]>=0.2:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if y[i]<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.

if experiment==3:
   for i in range(0,nn_T):
       if x[i]/Lx<eps:
          if y[i]>0.5:
             bc_fixT[i]=True ; bc_valT[i]=0.
          else:
             bc_fixT[i]=True ; bc_valT[i]=1.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       #if y[i]/Ly>(1-eps):
       #   bc_fixT[i]=True ; bc_valT[i]=0.

if experiment==4:
   for i in range(0,nn_T):
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if (x[i]-2)**2+(y[i]-2)**2<1:
          bc_fixT[i]=True ; bc_valT[i]=1.

print("initial temperature: %.3f s" % (clock.time()-start))

###############################################################################
# compute analytical solution
###############################################################################
start=clock.time()

T_anal=np.zeros(nn_T,dtype=np.float64)

if experiment==1:
   for i in range(0,nn_T):
       T_anal[i]=(x[i]-(1-np.exp(2*Pe*x[i]/hx))/(1-np.exp(2*Pe/hx)))/u0*f

if experiment==2 or experiment==3 or experiment==4:
   T_anal[:]=0.

###############################################################################
# build linear system 
###############################################################################
start=clock.time()

N     = np.zeros(m,dtype=np.float64) # shape functions
dNdx  = np.zeros(m,dtype=np.float64) # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64) # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64) # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64) # shape functions derivatives
A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
B_mat = np.zeros((2,m),dtype=np.float64)         # gradient matrix B 
N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

for iel in range (0,nel):

    b_el=np.zeros(m,dtype=np.float64)
    A_el=np.zeros((m,m),dtype=np.float64)
    Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
    Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
    vel=np.zeros((1,ndim),dtype=np.float64)

    for iq in [-1,1]:
        for jq in [-1,1]:

            # position & weight of quad. point
            rq=iq/sqrt3
            sq=jq/sqrt3
            weightq=1.*1.

            # calculate Q1 shape functions
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
                b_el[k]+=N_mat[k,0]*f*jcob*weightq
            # end for

            # compute diffusion matrix
            Kd+=B_mat.T.dot(B_mat)*(hcond+hcondtilde)*weightq*jcob

            if use_supg:
               tau_supg=0.5*hx/u0
               N_mat+=tau_supg*np.transpose(vel.dot(B_mat))

            # compute advection matrix
            Ka+=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

        #end for jq
    #end for iq

    A_el=Ka+Kd

    # apply boundary conditions
    for k1 in range(0,m):
        m1=icon[k1,iel]
        if bc_fixT[m1]:
           Aref=A_el[k1,k1]
           for k2 in range(0,m):
               m2=icon[k2,iel]
               b_el[k2]-=A_el[k2,k1]*bc_valT[m1]
               A_el[k1,k2]=0
               A_el[k2,k1]=0
           A_el[k1,k1]=Aref
           b_el[k1]=Aref*bc_valT[m1]
        #end if
    #end for

    # assemble matrix A_mat and right hand side rhs
    for k1 in range(0,m):
        m1=icon[k1,iel]
        for k2 in range(0,m):
            m2=icon[k2,iel]
            A_mat[m1,m2]+=A_el[k1,k2]
        #end for
        rhs[m1]+=b_el[k1]
    #end for

#end for iel

print("build system: %.3f s" % (clock.time()-start))

###############################################################################
# solve system
###############################################################################
start=clock.time()

T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

np.savetxt('Temperature.ascii',np.array([x,y,T,T_anal]).T,header='# x,y,T')

print("solve T time: %.3f s" % (clock.time()-start))

###############################################################################
# visualisation 
###############################################################################
start=clock.time()

vtufile=open('solution.vtu',"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,nn_T):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,nn_T):
        vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
T.tofile(vtufile,sep=' ',format='%.4e')
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
ax = plt.axes(projection='3d')
ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
ax.set_xlabel ( 'X')
ax.set_ylabel ( 'Y')
ax.set_zlabel ( ' Temperature')
plt.title('Peclet Nb: %e' %(Pe),loc='right')
plt.grid ()
plt.savefig(filename)
plt.show ()
plt.close()

print("export to vtu: %.3f s" % (clock.time()-start))
    
print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
