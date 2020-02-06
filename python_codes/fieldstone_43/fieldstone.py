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
sqrt2=np.sqrt(2.)
eps=1.e-10 

print("-----------------------------")
print("----------fieldstone---------")
print("-----------------------------")

ndim=2       # number of space dimensions
m=4          # number of nodes making up an element
ndofT=1      # number of degrees of freedom per node
hcond=0.     # thermal conductivity
hcapa=1.     # heat capacity
rho0=1       # reference density
every=10
CFLnb=0.5

use_bdf=False
bdf_order=2

use_supg=True

experiment=3

if experiment==1:
   nelx = 90
   nely = 90
   Lx=1.        # horizontal extent of the domain 
   Ly=1.        # vertical extent of the domain 
   tfinal=2.*np.pi

if experiment==2:
   nelx=100
   nely=100
   Lx=1.        # horizontal extent of the domain 
   Ly=1.        # vertical extent of the domain 
   tfinal=2.*np.pi

if experiment==3:
   nelx=50
   nely=25
   Lx=1.  
   Ly=0.5  
   tfinal=0.5

hx=Lx/float(nelx)
hy=Ly/float(nely)
    
nnx=nelx+1  # number of elements, x direction
nny=nely+1  # number of elements, y direction
NV=nnx*nny  # number of nodes
nel=nelx*nely  # number of elements, total
NfemT=NV*ndofT  # Total number of degrees of temperature freedom

# alphaT=1: implicit
# alphaT=0: explicit
# alphaT=0.5: Crank-Nicolson

alphaT=0.5

#####################################################################

stats_T_file=open('stats_T.ascii',"w")
avrg_T_file=open('avrg_T.ascii',"w")
ET_file=open('ET.ascii',"w")

#####################################################################
# grid point setup 
#####################################################################
start = timing.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
u = np.zeros(NV,dtype=np.float64)  # x-component velocity
v = np.zeros(NV,dtype=np.float64)  # y-component velocity

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        if experiment==1 or experiment==2:
           u[counter]=-(y[counter]-Ly/2)
           v[counter]=+(x[counter]-Lx/2)
        if experiment==3:
           u[counter]=1
           v[counter]=0
        counter += 1
    #end for
#end for

print("mesh (%.3fs)" % (timing.time() - start))

#####################################################################
# connectivity
#####################################################################
start = timing.time()

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

print("connectivity (%.3fs)" % (timing.time() - start))

#####################################################################
# define temperature boundary conditions
#####################################################################
start = timing.time()

bc_fixT=np.zeros(NfemT,dtype=np.bool)  
bc_valT=np.zeros(NfemT,dtype=np.float64) 

if experiment==1 or experiment==2:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly<eps:
          bc_fixT[i]=True ; bc_valT[i]=0.
       if y[i]/Ly>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

if experiment==3:
   for i in range(0,NV):
       if x[i]/Lx<eps:
          bc_fixT[i]=True ; bc_valT[i]=1.
       if x[i]/Lx>(1-eps):
          bc_fixT[i]=True ; bc_valT[i]=0.
   #end for

print("boundary conditions (%.3fs)" % (timing.time() - start))

#####################################################################
# initial temperature
#####################################################################
start = timing.time()

T = np.zeros(NV,dtype=np.float64)
Tm1 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-1
Tm2 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-2
Tm3 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-3
Tm4 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-4
Tm5 = np.zeros(NV,dtype=np.float64) # temperature at timestep n-5

if experiment==1:
   xc=2./3.
   yc=2./3.
   sigma=0.2
   for i in range(0,NV):
       if (x[i]-xc)**2+(y[i]-yc)**2<=sigma**2:
          T[i]=0.25*(1+np.cos(np.pi*(x[i]-xc)/sigma))*(1+np.cos(np.pi*(y[i]-yc)/sigma))
       #end if
   #end for

if experiment==2:
   for i in range(0,NV):
       xi=x[i]
       yi=y[i]
       if np.sqrt((xi-0.5)**2+(yi-0.75)**2)<0.15 and (np.abs(xi-0.5)>=0.025 or yi>=0.85):
          T[i]=1
       #end if
       if np.sqrt((x[i]-0.5)**2+(y[i]-0.25)**2)<0.15:
          T[i]=1-np.sqrt((x[i]-0.5)**2+(y[i]-0.25)**2)/0.15
       #end if
       if np.sqrt((x[i]-0.25)**2+(y[i]-0.5)**2)<0.15:
          T[i]=0.25*(1+np.cos(np.pi*np.sqrt((xi-0.25)**2+(yi-0.5)**2)/0.15))
       #end if
   #end for

if experiment==3:
   for i in range(0,NV):
       if x[i]<0.25:
          T[i]=1
       #end if
   #end for

Tm1[:]=T[:]
Tm2[:]=T[:]
Tm3[:]=T[:]
Tm4[:]=T[:]
Tm5[:]=T[:]

#np.savetxt('temperature_init.ascii',np.array([x,y,T]).T,header='# x,y,T')

print("initial temperature (%.3fs)" % (timing.time() - start))

#####################################################################
# create necessary arrays 
#####################################################################
start = timing.time()

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
    
print("create arrays (%.3fs)" % (timing.time() - start))

#################################################################
# compute timestep
#################################################################
start = timing.time()

dt=CFLnb*hx/np.max(np.sqrt(u**2+v**2))
print('dt=',dt)
nstep=int(tfinal/dt)+1
print('nstep=',nstep)

print("compute timestep (%.3fs)" % (timing.time() - start))

#==============================================================================
# time stepping loop
#==============================================================================

model_time=0.

for istep in range(0,nstep):
    print("-----------------------------")
    print("istep= ", istep)
    print("-----------------------------")

    model_time+=dt

    #################################################################
    # build temperature matrix
    #################################################################
    start = timing.time()

    A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
    rhs   = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
    B_mat=np.zeros((2,ndofT*m),dtype=np.float64)     # gradient matrix B 
    N_mat = np.zeros((m,1),dtype=np.float64)         # shape functions
    tau_supg = np.zeros(nel*4,dtype=np.float64)    

    counterq=0
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
                MM=N_mat.dot(N_mat.T)*rho0*hcapa*weightq*jcob

                if use_supg:
                   tau_supg[counterq]=0.5*(hx*sqrt2)/np.sqrt(vel[0,0]**2+vel[0,1]**2)/(1+1./CFLnb)
                   N_mat+= tau_supg[counterq]*np.transpose(vel.dot(B_mat))
                else:
                   tau_supg[counterq]=0.

                # compute diffusion matrix
                Kd=B_mat.T.dot(B_mat)*hcond*weightq*jcob

                # compute advection matrix
                Ka=N_mat.dot(vel.dot(B_mat))*rho0*hcapa*weightq*jcob

                if use_bdf and istep>bdf_order:
                   if bdf_order==1:
                      a_el+=MM+1.*dt*(Ka+Kd)
                      b_el+=MM.dot(Tvectm1)
                   #end if
                   if bdf_order==2:
                      a_el+=MM+2./3.*dt*(Ka+Kd)
                      b_el+=4./3.*MM.dot(Tvectm1)\
                           -1./3.*MM.dot(Tvectm2)
                   #end if
                   if bdf_order==3:
                      a_el+=MM+6./11.*dt*(Ka+Kd)
                      b_el+=18./11.*MM.dot(Tvectm1)\
                           -9./11.*MM.dot(Tvectm2)\
                           +2./11.*MM.dot(Tvectm3)
                   #end if
                   if bdf_order==4:
                      a_el+=MM+12./25.*dt*(Ka+Kd)
                      b_el+=48./25.*MM.dot(Tvectm1)\
                           -36./25.*MM.dot(Tvectm2)\
                           +16./25.*MM.dot(Tvectm3)\
                           -3./25.*MM.dot(Tvectm4)
                   #end if
                   if bdf_order==5:
                      a_el+=MM+60./137.*dt*(Ka+Kd)
                      b_el+=300./137.*MM.dot(Tvectm1)\
                           -300./137.*MM.dot(Tvectm2)\
                           +200./137.*MM.dot(Tvectm3)\
                           -75./137.*MM.dot(Tvectm4)\
                           +12./137.*MM.dot(Tvectm5)
                   #end if
                else:
                   a_el+=MM+alphaT*(Ka+Kd)*dt
                   b_el+=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvectm1)
                #end if

                counterq+=1
            #end for jq
        #end for iq

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

    #end for iel
    
    print("     -> tau_supg (m,M) %.4f %.4f " %(np.min(tau_supg),np.max(tau_supg)))
    #np.savetxt('tau_supg.ascii',np.array([counterq]).T,header='# x,y,T')

    print("build FEM matrix: %.3fs" % (timing.time() - start))

    #################################################################
    # solve system
    #################################################################
    start = timing.time()

    T = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    stats_T_file.write("%e %e %e \n" %(model_time,np.min(T),np.max(T))) ; stats_T_file.flush()

    print("solve T time: %.3f s" % (timing.time() - start))

    #####################################################################
    # compute average of temperature, total mass 
    #####################################################################
    start = timing.time()

    ET=0.
    avrg_T=0.
    for iel in range (0,nel):
        for iq in [-1,1]:
            for jq in [-1,1]:
                rq=iq/sqrt3
                sq=jq/sqrt3
                weightq=1.*1.
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
                avrg_T+=Tq*weightq*jcob
                ET+=rho0*hcapa*Tq*weightq*jcob
            #end for
        #end for
    #end for
    avrg_T/=Lx*Ly

    ET_file.write("%e %e \n" %(model_time,ET))                      ; ET_file.flush()
    avrg_T_file.write("%e %e \n" %(model_time,avrg_T))              ; avrg_T_file.flush()

    print("     -> avrg T= %.6e" % avrg_T)

    print("compute <T>,M: %.3f s" % (timing.time() - start))

    #################################################################
    # visualisation 
    #################################################################

    if istep%every==0:

       filename = 'T_{:04d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([x,y,T]).T,header='# x,y,T')

       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
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

       filename = 'solution_{:04d}.pdf'.format(istep) 
       fig = plt.figure ()
       ax = fig.gca(projection='3d')
       ax.plot_surface(x.reshape ((nny,nnx)),y.reshape((nny,nnx)),T.reshape((nny,nnx)),color = 'darkseagreen')
       ax.set_xlabel ( 'X [ m ] ')
       ax.set_ylabel ( 'Y [ m ] ')
       ax.set_zlabel ( ' Temperature  [ C ] ')
       plt.title('Timestep  %.2d' %(istep),loc='right')
       plt.grid ()
       plt.savefig(filename)
       #plt.show ()
       plt.close()

    #end if

    Tm5=Tm4
    Tm4=Tm3
    Tm3=Tm2
    Tm2=Tm1
    Tm1=T
    
#end for istep

#==============================================================================
# end time stepping loop
#==============================================================================

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
