import numpy as np
import sys as sys
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import time as timing

#------------------------------------------------------------------------------

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

sqrt3=np.sqrt(3.)
eps=1.e-10 

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofT=1      # number of degrees of freedom per node
Lx=1.        # horizontal extent of the domain 
Ly=1.        # vertical extent of the domain 
hcond=0.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density
CFL=1.       # CFL number 
nstep=30    # maximum number of timestep   
sigma=0.2
xc=1./6.+1./2.
yc=1./6.+1./2.
use_bdf=True
bdf_order=2

# allowing for argument parsing through command line
if int(len(sys.argv) == 3):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
else:
   nelx = 30
   nely = 30

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
nnp=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemT=nnp*ndofT  # Total number of degrees of temperature freedom

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: crank-nicolson

alphaT=0.

#####################################################################
# grid point setup 
#####################################################################

print("grid point setup")

x = np.empty(nnp,dtype=np.float64)  # x coordinates
y = np.empty(nnp,dtype=np.float64)  # y coordinates
u = np.zeros(nnp,dtype=np.float64)  # x-component velocity
v = np.zeros(nnp,dtype=np.float64)  # y-component velocity

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        u[counter]=-(y[counter]-Ly/2)
        v[counter]=+(x[counter]-Lx/2)
        counter += 1

#####################################################################
# connectivity
#####################################################################

print("connectivity array")

icon =np.zeros((m, nel),dtype=np.int16)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

#####################################################################
# define temperature boundary conditions
#####################################################################

print("defining temperature boundary conditions")

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

for i in range(0,nnp):
    if x[i]/Lx<eps:
       bc_fixT[i]=True ; bc_valT[i]=0.
    if x[i]/Lx>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.
    if y[i]/Ly<eps:
       bc_fixT[i]=True ; bc_valT[i]=0.
    if y[i]/Ly>(1-eps):
       bc_fixT[i]=True ; bc_valT[i]=0.

#####################################################################
# initial temperature
#####################################################################

T = np.zeros(nnp,dtype=np.float64)
Tm1 = np.zeros(nnp,dtype=np.float64) # temperature at timestep n-1
Tm2 = np.zeros(nnp,dtype=np.float64) # temperature at timestep n-2
Tm3 = np.zeros(nnp,dtype=np.float64) # temperature at timestep n-3
Tm4 = np.zeros(nnp,dtype=np.float64) # temperature at timestep n-4
Tm5 = np.zeros(nnp,dtype=np.float64) # temperature at timestep n-5

for i in range(0,nnp):
    if (x[i]-xc)**2+(y[i]-yc)**2<=sigma**2:
       T[i]=0.25*(1+np.cos(np.pi*(x[i]-xc)/sigma))*(1+np.cos(np.pi*(y[i]-yc)/sigma))
    else:
       T[i]=0.

Tm1[:]=T[:]
Tm2[:]=T[:]
Tm3[:]=T[:]
Tm4[:]=T[:]
Tm5[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

#####################################################################
# create necessary arrays 
#####################################################################

N     = np.zeros(m,dtype=np.float64)    # shape functions
dNdx  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdy  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNdr  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
dNds  = np.zeros(m,dtype=np.float64)    # shape functions derivatives
Tvectm1 = np.zeros(4,dtype=np.float64)   
Tvectm2 = np.zeros(4,dtype=np.float64)   
Tvectm3 = np.zeros(4,dtype=np.float64)   
Tvectm4 = np.zeros(4,dtype=np.float64)   
Tvectm5 = np.zeros(4,dtype=np.float64)   
model_time=np.zeros(nstep,dtype=np.float64) 
Tavrg=np.zeros(nstep,dtype=np.float64)
ET=np.zeros(nstep,dtype=np.float64)
Tmin=np.zeros(nstep,dtype=np.float64)
Tmax=np.zeros(nstep,dtype=np.float64)

#==============================================================================
# time stepping loop
#==============================================================================

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)

    #################################################################
    # compute timestep
    #################################################################
    # for this experiment the timestep is fixed:

    dt=2*np.pi/nstep

    if istep==0:
       model_time[istep]=dt
    else:
       model_time[istep]=model_time[istep-1]+dt

    #################################################################
    # build temperature matrix
    #################################################################

    print("-----------------------------")
    print("building temperature matrix and rhs")

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions

    for iel in range (0,nel):

        b_el=np.zeros(m*ndofT,dtype=np.float64)
        a_el=np.zeros((m*ndofT,m*ndofT),dtype=np.float64)
        Ka=np.zeros((m,m),dtype=np.float64)   # elemental advection matrix 
        Kd=np.zeros((m,m),dtype=np.float64)   # elemental diffusion matrix 
        MM=np.zeros((m,m),dtype=np.float64)   # elemental mass matrix 
        vel=np.zeros((1,ndim),dtype=np.float64)

        for k in range(0,m):
            Tvectm1[k]=Tm1[icon[k,iel]]
            Tvectm2[k]=Tm2[icon[k,iel]]
            Tvectm3[k]=Tm3[icon[k,iel]]
            Tvectm4[k]=Tm4[icon[k,iel]]
            Tvectm5[k]=Tm5[icon[k,iel]]

        for iq in [-1,1]:
            for jq in [-1,1]:

                # position & weight of quad. point
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.

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
                jcb=np.zeros((2, 2),dtype=float)
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

                # compute mass matrix
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*wq*jcob

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*wq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*wq*jcob

                if use_bdf:
                   if istep>bdf_order:
                      if bdf_order==1:
                         a_el=MM+1.*dt*(Ka+Kd)
                         b_el=MM.dot(Tvectm1)
                      if bdf_order==2:
                         a_el=MM+2./3.*dt*(Ka+Kd)
                         b_el=4./3.*MM.dot(Tvectm1)\
                             -1./3.*MM.dot(Tvectm2)
                      if bdf_order==3:
                         a_el=MM+6./11.*dt*(Ka+Kd)
                         b_el=18./11.*MM.dot(Tvectm1)\
                              -9./11.*MM.dot(Tvectm2)\
                              +2./11.*MM.dot(Tvectm3)
                      if bdf_order==4:
                         a_el=MM+12./25.*dt*(Ka+Kd)
                         b_el=48./25.*MM.dot(Tvectm1)\
                             -36./25.*MM.dot(Tvectm2)\
                             +16./25.*MM.dot(Tvectm3)\
                              -3./25.*MM.dot(Tvectm4)
                      if bdf_order==5:
                         a_el=MM+60./137.*dt*(Ka+Kd)
                         b_el=300./137.*MM.dot(Tvectm1)\
                             -300./137.*MM.dot(Tvectm2)\
                             +200./137.*MM.dot(Tvectm3)\
                              -75./137.*MM.dot(Tvectm4)\
                              +12./137.*MM.dot(Tvectm5)
                   else:
                      a_el=MM+alphaT*(Ka+Kd)*dt
                      b_el=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvectm1)
                else:
                   a_el=MM+alphaT*(Ka+Kd)*dt
                   b_el=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvectm1)

                # assemble matrix A_mat and right hand side rhs
                for k1 in range(0,m):
                    m1=icon[k1,iel]
                    for k2 in range(0,m):
                        m2=icon[k2,iel]
                        A_mat[m1,m2]+=a_el[k1,k2]
                    rhs[m1]+=b_el[k1]

    #################################################################
    # apply boundary conditions
    #################################################################

    print("imposing boundary conditions temperature")

    for i in range(0,NfemT):
        if bc_fixT[i]:
           A_matref = A_mat[i,i]
           for j in range(0,NfemT):
               rhs[j]-= A_mat[i, j] * bc_valT[i]
               A_mat[i,j]=0.
               A_mat[j,i]=0.
               A_mat[i,i] = A_matref
           rhs[i]=A_matref*bc_valT[i]

    #print("A_mat (m,M) = %.4f %.4f" %(np.min(A_mat),np.max(A_mat)))
    #print("rhs   (m,M) = %.6f %.6f" %(np.min(rhs),np.max(rhs)))

    #################################################################
    # solve system
    #################################################################

    start = timing.time()
    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)
    print("solve T time: %.3f s" % (timing.time() - start))

    print("T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))
    Tmin[istep]=np.min(T)
    Tmax[istep]=np.max(T)

    #####################################################################
    # compute average of temperature, total mass 
    #####################################################################
    start = timing.time()

    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                wq=1.*1.
                N[0]=0.25*(1.-rq)*(1.-sq)
                N[1]=0.25*(1.+rq)*(1.-sq)
                N[2]=0.25*(1.+rq)*(1.+sq)
                N[3]=0.25*(1.-rq)*(1.+sq)
                dNdr[0]=-0.25*(1.-sq) ; dNds[0]=-0.25*(1.-rq)
                dNdr[1]=+0.25*(1.-sq) ; dNds[1]=-0.25*(1.+rq)
                dNdr[2]=+0.25*(1.+sq) ; dNds[2]=+0.25*(1.+rq)
                dNdr[3]=-0.25*(1.+sq) ; dNds[3]=+0.25*(1.-rq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,m):
                    jcb[0,0]+=dNdr[k]*x[icon[k,iel]]
                    jcb[0,1]+=dNdr[k]*y[icon[k,iel]]
                    jcb[1,0]+=dNds[k]*x[icon[k,iel]]
                    jcb[1,1]+=dNds[k]*y[icon[k,iel]]
                jcob=np.linalg.det(jcb)
                jcbi=np.linalg.inv(jcb)
                for k in range(0,m):
                    dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                    dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                Tq=0.
                for k in range(0,m):
                    Tq+=N[k]*T[icon[k,iel]]
                Tavrg[istep]+=Tq*wq*jcob
                ET[istep]+=rho0*hcapa*Tq*wq*jcob

    Tavrg[istep]/=Lx*Ly

    print("     -> avrg T= %.6e" % Tavrg[istep])

    print("compute <T>,M: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation 
    #################################################################

    filename = 'solution_{:04d}.vtu'.format(istep) 
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nnp,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
    for i in range(0,nnp):
        vtufile.write("%10f %10f %10f \n" %(u[i],v[i],0.))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='temperature' Format='ascii'> \n")
    for i in range(0,nnp):
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

    Tm5=Tm4
    Tm4=Tm3
    Tm3=Tm2
    Tm2=Tm1
    Tm1=T

    np.savetxt('Tavrg.ascii',np.array([model_time[0:istep],Tavrg[0:istep]]).T,header='# t,Tavrg')
    np.savetxt('ET.ascii',np.array([model_time[0:istep],ET[0:istep]]).T,header='# t,ET')
    np.savetxt('Tmin.ascii',np.array([model_time[0:istep],Tmin[0:istep]]).T,header='# t,Tmin')
    np.savetxt('Tmax.ascii',np.array([model_time[0:istep],Tmax[0:istep]]).T,header='# t,Tmax')

#==============================================================================
# end time stepping loop
#==============================================================================
    

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
