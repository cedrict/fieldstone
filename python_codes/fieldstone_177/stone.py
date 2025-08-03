import numpy as np
import time as clock 
import scipy.sparse as sps
import sys as sys
from scipy.sparse import csr_matrix,lil_matrix


###############################################################################

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
    
nnx=nelx+1    # number of elements, x direction
nny=nely+1    # number of elements, y direction
NV=nnx*nny    # number of nodes

nel=nelx*nely*nelz

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz
hmin=max(hx,hy,hz)

m=8

nnx=nelx+1
nny=nely+1
nnz=nelz+1

NT=nnx*nny*nnz
Nfem=nnx*nny*nnz

experiment=2

nstep=3

alphaT=.5

CFL=0.5

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

rnodes=np.array([-1, 1, 1,-1,-1, 1, 1 ,-1],np.float64)
snodes=np.array([-1,-1, 1, 1,-1,-1, 1 , 1],np.float64)
tnodes=np.array([-1,-1,-1,-1, 1, 1, 1 , 1],np.float64)

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

print("--------------------------------------------")
print("--------- stone 177 ------------------------")
print("--------------------------------------------")
print('Lx=',Lx)
print('Ly=',Ly)
print('Lz=',Lz)
print('nelx=',nelx)
print('nely=',nely)
print('nelz=',nelz)
print('nel=',nel)
print('NT=',NT)
print('Nfem=',Nfem)
print('experiment=',experiment)
print("--------------------------------------------")

Tavrgfile=open('Tavrg.ascii',"w")

###############################################################################
# grid point setup
###############################################################################
start=clock.time()

x=np.zeros(NT,dtype=np.float64)
y=np.zeros(NT,dtype=np.float64)
z=np.zeros(NT,dtype=np.float64)

counter=0
for i in range(0,nnx):
    for j in range(0,nny):
        for k in range(0,nnz):
            x[counter]=i*hx
            y[counter]=j*hy
            z[counter]=k*hz
            counter += 1
        #end for
    #end for
#end for

print("mesh setup: %.3f s" % (clock.time() - start))

###############################################################################
# build connectivity array (python is row major)
###############################################################################
start=clock.time()

icon=np.zeros((nel,m),dtype=np.int32)

counter = 0
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[counter,0]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[counter,1]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[counter,2]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[counter,3]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[counter,4]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[counter,5]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[counter,6]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[counter,7]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("connectivity setup: %.3f s" % (clock.time() - start))

###############################################################################
# prescribe velocity on mesh
###############################################################################
start=clock.time()

u=np.zeros(NT,dtype=np.float64)
v=np.zeros(NT,dtype=np.float64)
w=np.zeros(NT,dtype=np.float64)

if experiment==1:
   print('no advection: zero velocity')

if experiment==2:
   u[:] = x*(1-x)*(1-2*y)*(1-2*z) *10
   v[:] = (1-2*x)*y*(1-y)*(1-2*z) *10
   w[:] = -2*(1-2*x)*(1-2*y)*z*(1-z) *10

print("prescribe velocity: %.3f s" % (clock.time()-start))

######################################################################
# define boundary conditions temperature
######################################################################
start=clock.time()

eps=1e-8

bc_fix=np.zeros(Nfem,dtype=bool) 
bc_val=np.zeros(Nfem,dtype=np.float64)

if experiment==1 or experiment==2:
   for i in range(0,NT):
       if z[i]<eps:
          bc_fix[i]=True ; bc_val[i]=Tbottom
       if z[i]/Lz>1-eps:
          bc_fix[i]=True ; bc_val[i]=Ttop

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
    
Told=np.zeros(NT,dtype=np.float64)
MM=np.zeros((m,m),dtype=np.float64)
Kd=np.zeros((m,m),dtype=np.float64)
Ka=np.zeros((m,m),dtype=np.float64)
Ael=np.zeros((m,m),dtype=np.float64)
bel=np.zeros(m,dtype=np.float64)

for istep in range(0,nstep):

    print("--------------------------------------------")
    print("istep= ", istep)
    print("--------------------------------------------")

    ###############################################################################
    # build matrix
    ###############################################################################
    start=clock.time()

    Amat=lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs=np.zeros(Nfem,dtype=np.float64)

    for e,nodes in enumerate(icon):
        xe,ye,ze=x[nodes],y[nodes],z[nodes]
        ue,ve,we=u[nodes],v[nodes],w[nodes]
        Te=Told[nodes]

        MM[:,:]=0
        Ka[:,:]=0
        Kd[:,:]=0

        for rq,sq,tq,weightq in quadrature_points:

            N=0.125*(1+rnodes*rq)*(1+snodes*sq)*(1+tnodes*tq)

            dNdr=0.125*rnodes*(1+snodes*sq)*(1+tnodes*tq)
            dNds=0.125*snodes*(1+rnodes*rq)*(1+tnodes*tq)
            dNdt=0.125*tnodes*(1+rnodes*rq)*(1+snodes*sq)

            invJ=np.diag([2/hx,2/hy,2/hz])
            jcob=hx*hy*hz/8

            B=(invJ@np.vstack((dNdr,dNds,dNdt))).T  # (8x3) shape

            velq=np.dot(N,np.vstack((ue,ve,we)).T) # (3,) shape
            #print(np.shape(velq))
      
            advN=B@velq # (8,) shape

            MM+=rho*hcapa*np.outer(N,N)*jcob*weightq
            Ka+=np.outer(N,advN)*jcob*weightq
            Kd+=B@B.T*hcond*jcob*weightq

        #end for quad points

        Ael=MM+alphaT*(Ka+Kd)*dt
        bel=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Te)

        #impose boundary conditions
        for k1,m1 in enumerate(nodes):
            if bc_fix[m1]:
               Aref=Ael[k1,k1]
               for k2,m2 in enumerate(nodes):
                   bel[k2]-=Ael[k2,k1]*bc_val[m1]
                   Ael[k2,k1]=0
               Ael[k1,:]=0
               Ael[k1,k1]=Aref
               bel[k1]=Aref*bc_val[m1]
            # end if
        # end for

        #assemble
        Amat[np.ix_(nodes,nodes)]+=Ael
        rhs[nodes]+=bel

    #end for elements

    print("Build FE matrix: %.5f s | Nfem= %d" % (clock.time()-start,Nfem))

    ###############################################################################
    # solve system
    ###############################################################################
    start=clock.time()

    T=sps.linalg.spsolve(sps.csr_matrix(Amat),rhs)

    print("     -> T (m,M) %.4f %.4f " %(np.min(T),np.max(T)))

    print("Solve linear system: %.5f s | Nfem= %d " % (clock.time()-start,Nfem))

    ###############################################################################
    # compute average T
    ###############################################################################
    start=clock.time()

    Tavrg=0.
    for e,nodes in enumerate(icon):
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
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NT,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%.6e %.6e %.6e \n" %(x[i],y[i],z[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='T' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10f \n" %(T[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
       for i in range(0,NT):
           vtufile.write("%10f %10f %10f \n" %(u[i],v[i],w[i]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[iel,0],icon[iel,1],\
                                                       icon[iel,2],icon[iel,3],\
                                                       icon[iel,4],icon[iel,5],\
                                                       icon[iel,6],icon[iel,7]))
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

print("--------------------------------------------")
print("------------ the end -----------------------")
print("--------------------------------------------")

