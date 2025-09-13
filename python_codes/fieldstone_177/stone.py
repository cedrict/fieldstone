import numpy as np
import time as clock 
import scipy.sparse as sps
import sys as sys
from scipy.sparse import csr_matrix,lil_matrix

###############################################################################

def basis_functions_T(r,s,t):
    N0=0.125*(1-r)*(1-s)*(1-t)
    N1=0.125*(1+r)*(1-s)*(1-t)
    N2=0.125*(1+r)*(1+s)*(1-t)
    N3=0.125*(1-r)*(1+s)*(1-t)
    N4=0.125*(1-r)*(1-s)*(1+t)
    N5=0.125*(1+r)*(1-s)*(1+t)
    N6=0.125*(1+r)*(1+s)*(1+t)
    N7=0.125*(1-r)*(1+s)*(1+t)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7],dtype=np.float64)

def basis_functions_T_dr(r,s,t):
    dNdr0=-0.125*(1-s)*(1-t)
    dNdr1=+0.125*(1-s)*(1-t)
    dNdr2=+0.125*(1+s)*(1-t)
    dNdr3=-0.125*(1+s)*(1-t)
    dNdr4=-0.125*(1-s)*(1+t)
    dNdr5=+0.125*(1-s)*(1+t)
    dNdr6=+0.125*(1+s)*(1+t)
    dNdr7=-0.125*(1+s)*(1+t)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7],dtype=np.float64)
   
def basis_functions_T_ds(r,s,t):
    dNds0=-0.125*(1-r)*(1-t)
    dNds1=-0.125*(1+r)*(1-t)
    dNds2=+0.125*(1+r)*(1-t)
    dNds3=+0.125*(1-r)*(1-t)
    dNds4=-0.125*(1-r)*(1+t)
    dNds5=-0.125*(1+r)*(1+t)
    dNds6=+0.125*(1+r)*(1+t)
    dNds7=+0.125*(1-r)*(1+t)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7],dtype=np.float64)

def basis_functions_T_dt(r,s,t):
    dNdt0=-0.125*(1-r)*(1-s)
    dNdt1=-0.125*(1+r)*(1-s)
    dNdt2=-0.125*(1+r)*(1+s)
    dNdt3=-0.125*(1-r)*(1+s)
    dNdt4=+0.125*(1-r)*(1-s)
    dNdt5=+0.125*(1+r)*(1-s)
    dNdt6=+0.125*(1+r)*(1+s)
    dNdt7=+0.125*(1-r)*(1+s)
    return np.array([dNdt0,dNdt1,dNdt2,dNdt3,dNdt4,dNdt5,dNdt6,dNdt7],dtype=np.float64)

###############################################################################

print("*******************************")
print("********** stone 177 **********")
print("*******************************")

Lx=1
Ly=1
Lz=1

if int(len(sys.argv) == 5):
   nelx = int(sys.argv[1])
   nely = int(sys.argv[2])
   nelz = int(sys.argv[3])
   visu = int(sys.argv[4])
else:
   nelx = 16
   nely = 16
   nelz = 16
   visu = 1

nel=nelx*nely*nelz

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
hmin=max(hx,hy,hz)

m_T=8

nn_T=(nelx+1)*(nely+1)*(nelz+1)
Nfem=nn_T

experiment=2

nstep=5

alphaT=.5

CFL=0.5

method='old'

###############################################################################
# exp=1: pure conduction

if experiment==1 or experiment==2:
   hcond=1
   hcapa=1
   rho=1
   Tbottom=1
   Ttop=0

###############################################################################
# local coordinates of elemental nodes
###############################################################################

r_T=np.array([-1, 1, 1,-1,-1, 1, 1,-1],np.float64)
s_T=np.array([-1,-1, 1, 1,-1,-1, 1, 1],np.float64)
t_T=np.array([-1,-1,-1,-1, 1, 1, 1, 1],np.float64)

###############################################################################
# setup quadrature points and weights
# The first 3 values are the r,s,t coordinates, the 4th one is the weight
###############################################################################

a=1/np.sqrt(3)
quadrature_points = [(-a,-a,-a ,1),
                     ( a,-a,-a ,1) , 
                     ( a, a,-a ,1) ,
                     (-a, a,-a ,1) ,
                     (-a,-a, a ,1) , 
                     ( a,-a, a ,1) , 
                     ( a, a, a ,1) ,
                     (-a, a, a ,1) ]

###############################################################################

print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('nn_T=',nn_T)
print('Nfem=',Nfem)
print('experiment=',experiment)
print("*******************************")

Tavrgfile=open('Tavrg.ascii',"w")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x_T=np.zeros(nn_T,dtype=np.float64)
y_T=np.zeros(nn_T,dtype=np.float64)
z_T=np.zeros(nn_T,dtype=np.float64)

counter=0
for i in range(0,nelx+1):
    for j in range(0,nely+1):
        for k in range(0,nelz+1):
            x_T[counter]=i*hx
            y_T[counter]=j*hy
            z_T[counter]=k*hz
            counter += 1
        #end for
    #end for
#end for

print("mesh setup: %.3f s" % (clock.time() - start))

###############################################################################
# build connectivity array (python is row major)
###############################################################################
start=clock.time()

icon_T=np.zeros((nel,m_T),dtype=np.int32)

nny=nely+1
nnz=nelz+1

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon_T[counter,0]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon_T[counter,1]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon_T[counter,2]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon_T[counter,3]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon_T[counter,4]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon_T[counter,5]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon_T[counter,6]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon_T[counter,7]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (clock.time() - start))

###############################################################################
# prescribe velocity on mesh
###############################################################################
start=clock.time()

u=np.zeros(nn_T,dtype=np.float64)
v=np.zeros(nn_T,dtype=np.float64)
w=np.zeros(nn_T,dtype=np.float64)

if experiment==1:
   print('no advection: zero velocity')

if experiment==2:
   u[:] = x_T*(1-x_T)*(1-2*y_T)*(1-2*z_T)    *10
   v[:] = (1-2*x_T)*y_T*(1-y_T)*(1-2*z_T)    *10
   w[:] = -2*(1-2*x_T)*(1-2*y_T)*z_T*(1-z_T) *10

print("prescribe velocity: %.3f s" % (clock.time()-start))

######################################################################
# define boundary conditions temperature
######################################################################
start=clock.time()

eps=1e-8

bc_fix_T=np.zeros(Nfem,dtype=bool) 
bc_val_T=np.zeros(Nfem,dtype=np.float64)

if experiment==1 or experiment==2:
   for i in range(0,nn_T):
       if z_T[i]<eps:
          bc_fix_T[i]=True ; bc_val_T[i]=Tbottom
       if z_T[i]/Lz>1-eps:
          bc_fix_T[i]=True ; bc_val_T[i]=Ttop

print("boundary conditions: %.3f s" % (clock.time()-start))

###############################################################################
# compute time step
###############################################################################

dt1=hmin**2/(hcond/rho/hcapa)

dt2=hmin/max(np.max(u),np.max(v),np.max(w))

dt=CFL*min(dt1,dt2)

print("     -> dt %e " %(dt))

#******************************************************************************
#******************************************************************************
# start time stepping
#******************************************************************************
#******************************************************************************
    
Told=np.zeros(nn_T,dtype=np.float64)
MM=np.zeros((m_T,m_T),dtype=np.float64)
Kd=np.zeros((m_T,m_T),dtype=np.float64)
Ka=np.zeros((m_T,m_T),dtype=np.float64)
A_el=np.zeros((m_T,m_T),dtype=np.float64)
b_el=np.zeros(m_T,dtype=np.float64)

for istep in range(0,nstep):

    print("--------------------------------------------")
    print("istep= ", istep)
    print("--------------------------------------------")

    time_quad=0.
    time_ass=0.

    ###############################################################################
    # build matrix
    ###############################################################################
    start=clock.time()

    A_fem=lil_matrix((Nfem,Nfem),dtype=np.float64)
    b_fem=np.zeros(Nfem,dtype=np.float64)

    jcbi=np.diag([2/hx,2/hy,2/hz])
    jcob=hx*hy*hz/8

    if method=='new':

       for e,nodes in enumerate(icon_T):
           ue,ve,we,Te=u[nodes],v[nodes],w[nodes],Told[nodes]

           MM.fill(0)
           Ka.fill(0)
           Kd.fill(0)

           start1=clock.time()
           for rq,sq,tq,weightq in quadrature_points:
   
               N_T=0.125*(1+r_T*rq)*(1+s_T*sq)*(1+t_T*tq)
               dNdr_T=0.125*r_T*(1+s_T*sq)*(1+t_T*tq)
               dNds_T=0.125*s_T*(1+r_T*rq)*(1+t_T*tq)
               dNdt_T=0.125*t_T*(1+r_T*rq)*(1+s_T*sq)

               B=(jcbi@np.vstack((dNdr_T,dNds_T,dNdt_T))).T

               MM+=rho*hcapa*np.outer(N_T,N_T)*jcob*weightq
               Kd+=B@B.T*hcond*jcob*weightq
   
               velq=np.dot(N_T,np.vstack((ue,ve,we)).T)
               advN=B@velq
               Ka+=np.outer(N_T,advN)*jcob*weightq*rho*hcapa

           #end for quad points
           time_quad+=clock.time()-start1

           A_el=MM+alphaT*(Ka+Kd)*dt
           b_el=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Te)

           #impose boundary conditions
           for k1,m1 in enumerate(nodes):
               if bc_fix_T[m1]:
                  Aref=A_el[k1,k1]
                  for k2,m2 in enumerate(nodes):
                      b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                      A_el[k2,k1]=0
                  A_el[k1,:]=0
                  A_el[k1,k1]=Aref
                  b_el[k1]=Aref*bc_val_T[m1]
               # end if
           # end for

           #assemble
           start2=clock.time()
           A_fem[np.ix_(nodes,nodes)]+=A_el
           b_fem[nodes]+=b_el
           time_ass+=clock.time()-start2

       #end for elements

    else: # old method -------------------------------------

       sqrt3=np.sqrt(3.)
       B=np.zeros((3,m_T),dtype=np.float64) 
       Tvect=np.zeros(m_T,dtype=np.float64)
       velq=np.zeros((1,3),dtype=np.float64)

       for iel in range (0,nel):

           MM[:,:]=0
           Ka[:,:]=0
           Kd[:,:]=0

           for k in range(0,m_T):
               Tvect[k]=Told[icon_T[iel,k]]

           start1=clock.time()
           for iq in [-1,1]:
               for jq in [-1,1]:
                   for kq in [-1,1]:

                       # position & weight of quad. point
                       rq=iq/sqrt3
                       sq=jq/sqrt3
                       tq=kq/sqrt3
                       weightq=1.*1.*1.

                       # calculate shape functions
                       N_T=basis_functions_T(rq,sq,tq)
                       dNdr_T=basis_functions_T_dr(rq,sq,tq)
                       dNds_T=basis_functions_T_ds(rq,sq,tq)
                       dNdt_T=basis_functions_T_dt(rq,sq,tq)

                       velq[0,0]=np.dot(N_T,u[icon_T[iel,:]])
                       velq[0,1]=np.dot(N_T,v[icon_T[iel,:]])
                       velq[0,2]=np.dot(N_T,w[icon_T[iel,:]])

                       dNdx_T=jcbi[0,0]*dNdr_T+jcbi[0,1]*dNds_T+jcbi[0,2]*dNdt_T
                       dNdy_T=jcbi[1,0]*dNdr_T+jcbi[1,1]*dNds_T+jcbi[1,2]*dNdt_T
                       dNdz_T=jcbi[2,0]*dNdr_T+jcbi[2,1]*dNds_T+jcbi[2,2]*dNdt_T

                       B[0,:]=dNdx_T[:]
                       B[1,:]=dNdy_T[:]
                       B[2,:]=dNdz_T[:]

                       MM+=np.outer(N_T,N_T)*rho*hcapa*weightq*jcob
                       Kd+=B.T.dot(B)*hcond*weightq*jcob
                       Ka+=np.outer(N_T,velq.dot(B))*rho*hcapa*weightq*jcob

                   #end for
               #end for
           #end for
           time_quad+=clock.time()-start1

           A_el=MM+alphaT*(Ka+Kd)*dt
           b_el=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvect)

           # apply boundary conditions
           for k1 in range(0,m_T):
               m1=icon_T[iel,k1]
               if bc_fix_T[m1]:
                  Aref=A_el[k1,k1]
                  for k2 in range(0,m_T):
                      m2=icon_T[iel,k2]
                      b_el[k2]-=A_el[k2,k1]*bc_val_T[m1]
                      A_el[k1,k2]=0
                      A_el[k2,k1]=0
                  # end for
                  A_el[k1,k1]=Aref
                  b_el[k1]=Aref*bc_val_T[m1]
               # end if
           # end for

           # assemble matrix and rhs
           start2=clock.time()
           for k1 in range(0,m_T):
               m1=icon_T[iel,k1]
               for k2 in range(0,m_T):
                   m2=icon_T[iel,k2]
                   A_fem[m1,m2]+=A_el[k1,k2]
               # end for
               b_fem[m1]+=b_el[k1]
           # end for
           time_ass+=clock.time()-start2

       #end for iel

    #end if method

    print('     -> time quadrature=',time_quad,Nfem)
    print('     -> time assembly=',time_ass,Nfem)

    print("Build FE matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

    ###############################################################################
    # solve system
    ###############################################################################
    start=clock.time()

    T=sps.linalg.spsolve(A_fem.tocsr(),b_fem)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("Solve linear system: %.5f s | Nfem= %d " % (clock.time()-start,Nfem))

    ###############################################################################
    # compute average T
    ###############################################################################
    start=clock.time()

    Tavrg=0.
    for e,nodes in enumerate(icon_T):
        Te=T[nodes]
        Tavrg+=np.sum(Te)*0.125 *hx*hy*hz

    Tavrg/=(Lx*Ly*Lz)

    print("     -> T (avrg) %.4f " %(Tavrg))

    Tavrgfile.write("%d %e\n" % (istep,Tavrg))
    Tavrgfile.flush()

    print("compute avrg T: %.3f s" % (clock.time()-start))

    ###############################################################################
    # export to vtu
    ###############################################################################

    if visu==1:
       filename = 'solution_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nn_T,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%.6e %.6e %.6e \n" %(x_T[i],y_T[i],z_T[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10f \n" %(T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,nn_T):
           vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon_T[iel,0],icon_T[iel,1],\
                                                       icon_T[iel,2],icon_T[iel,3],\
                                                       icon_T[iel,4],icon_T[iel,5],\
                                                       icon_T[iel,6],icon_T[iel,7]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %12)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()
       print("export to vtu: %.3f s" % (clock.time()-start))

       Told[:]=T[:]

print("*******************************")
print("********** the end ************")
print("*******************************")

###############################################################################
