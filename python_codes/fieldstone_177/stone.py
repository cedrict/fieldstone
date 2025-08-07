import numpy as np
import time as clock 
import scipy.sparse as sps
import sys as sys
from scipy.sparse import csr_matrix,lil_matrix

def NNT(r,s,t):
    return 0.125*(1-r)*(1-s)*(1-t),\
           0.125*(1+r)*(1-s)*(1-t),\
           0.125*(1+r)*(1+s)*(1-t),\
           0.125*(1-r)*(1+s)*(1-t),\
           0.125*(1-r)*(1-s)*(1+t),\
           0.125*(1+r)*(1-s)*(1+t),\
           0.125*(1+r)*(1+s)*(1+t),\
           0.125*(1-r)*(1+s)*(1+t)

def dNNTdr(r,s,t):
    return -0.125*(1-s)*(1-t),\
           +0.125*(1-s)*(1-t),\
           +0.125*(1+s)*(1-t),\
           -0.125*(1+s)*(1-t),\
           -0.125*(1-s)*(1+t),\
           +0.125*(1-s)*(1+t),\
           +0.125*(1+s)*(1+t),\
           -0.125*(1+s)*(1+t)
   
def dNNTds(r,s,t):
    return -0.125*(1-r)*(1-t),\
           -0.125*(1+r)*(1-t),\
           +0.125*(1+r)*(1-t),\
           +0.125*(1-r)*(1-t),\
           -0.125*(1-r)*(1+t),\
           -0.125*(1+r)*(1+t),\
           +0.125*(1+r)*(1+t),\
           +0.125*(1-r)*(1+t)

def dNNTdt(r,s,t):
    return -0.125*(1-r)*(1-s),\
           -0.125*(1+r)*(1-s),\
           -0.125*(1+r)*(1+s),\
           -0.125*(1-r)*(1+s),\
           +0.125*(1-r)*(1-s),\
           +0.125*(1+r)*(1-s),\
           +0.125*(1+r)*(1+s),\
           +0.125*(1-r)*(1+s)



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

    time_quad=0.
    time_ass=0.

    ###############################################################################
    # build matrix
    ###############################################################################
    start=clock.time()

    Amat=lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs=np.zeros(Nfem,dtype=np.float64)

    jcbi=np.diag([2/hx,2/hy,2/hz])
    jcob=hx*hy*hz/8

    if method=='new':

       for e,nodes in enumerate(icon):
           ue,ve,we,Te=u[nodes],v[nodes],w[nodes],Told[nodes]

           MM[:,:]=0
           Ka[:,:]=0
           Kd[:,:]=0

           start1=clock.time()
           for rq,sq,tq,weightq in quadrature_points:
   
               N=0.125*(1+rnodes*rq)*(1+snodes*sq)*(1+tnodes*tq)
   
               dNdr=0.125*rnodes*(1+snodes*sq)*(1+tnodes*tq)
               dNds=0.125*snodes*(1+rnodes*rq)*(1+tnodes*tq)
               dNdt=0.125*tnodes*(1+rnodes*rq)*(1+snodes*sq)

               B=(jcbi@np.vstack((dNdr,dNds,dNdt))).T

               MM+=rho*hcapa*np.outer(N,N)*jcob*weightq
               Kd+=B@B.T*hcond*jcob*weightq
   
               velq=np.dot(N,np.vstack((ue,ve,we)).T)
               advN=B@velq
               Ka+=np.outer(N,advN)*jcob*weightq*rho*hcapa

           #end for quad points
           time_quad+=clock.time()-start1

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
           start2=clock.time()
           Amat[np.ix_(nodes,nodes)]+=Ael
           rhs[nodes]+=bel
           time_ass+=clock.time()-start2

       #end for elements

    else: # old method -------------------------------------

       sqrt3=np.sqrt(3.)
       B_mat=np.zeros((3,m),dtype=np.float64) 
       NNNT_mat=np.zeros((m,1),dtype=np.float64) 
       dNNNTdx=np.zeros(m,dtype=np.float64)     
       dNNNTdy=np.zeros(m,dtype=np.float64)   
       dNNNTdz=np.zeros(m,dtype=np.float64) 
       dNNNTdr=np.zeros(m,dtype=np.float64) 
       dNNNTds=np.zeros(m,dtype=np.float64) 
       dNNNTdt=np.zeros(m,dtype=np.float64) 
       Tvect=np.zeros(m,dtype=np.float64)
       vel=np.zeros((1,3),dtype=np.float64)

       for iel in range (0,nel):

           MM[:,:]=0
           Ka[:,:]=0
           Kd[:,:]=0

           for k in range(0,m):
               Tvect[k]=Told[icon[iel,k]]

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
                       NNNT_mat[0:m,0]=NNT(rq,sq,tq)
                       dNNNTdr[0:m]=dNNTdr(rq,sq,tq)
                       dNNNTds[0:m]=dNNTds(rq,sq,tq)
                       dNNNTdt[0:m]=dNNTdt(rq,sq,tq)

                       vel[0,:]=0.
                       for k in range(0,m):
                           vel[0,0]+=NNNT_mat[k,0]*u[icon[iel,k]]
                           vel[0,1]+=NNNT_mat[k,0]*v[icon[iel,k]]
                           vel[0,2]+=NNNT_mat[k,0]*w[icon[iel,k]]
                       # end for 

                       # compute dNdx, dNdy, dNdz 
                       for k in range(0,m):
                           dNNNTdx[k]=jcbi[0,0]*dNNNTdr[k]+jcbi[0,1]*dNNNTds[k]+jcbi[0,2]*dNNNTdt[k]
                           dNNNTdy[k]=jcbi[1,0]*dNNNTdr[k]+jcbi[1,1]*dNNNTds[k]+jcbi[1,2]*dNNNTdt[k]
                           dNNNTdz[k]=jcbi[2,0]*dNNNTdr[k]+jcbi[2,1]*dNNNTds[k]+jcbi[2,2]*dNNNTdt[k]
                           B_mat[0,k]=dNNNTdx[k]
                           B_mat[1,k]=dNNNTdy[k]
                           B_mat[2,k]=dNNNTdz[k]
                       # end for 

                       MM+=NNNT_mat.dot(NNNT_mat.T)*rho*hcapa*weightq*jcob
                       Kd+=B_mat.T.dot(B_mat)*hcond*weightq*jcob
                       Ka+=NNNT_mat.dot(vel.dot(B_mat))*rho*hcapa*weightq*jcob

                   #end for
               #end for
           #end for
           time_quad+=clock.time()-start1

           Ael=MM+alphaT*(Ka+Kd)*dt
           bel=(MM-(1-alphaT)*(Ka+Kd)*dt).dot(Tvect)

           # apply boundary conditions
           for k1 in range(0,m):
               m1=icon[iel,k1]
               if bc_fix[m1]:
                  Aref=Ael[k1,k1]
                  for k2 in range(0,m):
                      m2=icon[iel,k2]
                      bel[k2]-=Ael[k2,k1]*bc_val[m1]
                      Ael[k1,k2]=0
                      Ael[k2,k1]=0
                  # end for
                  Ael[k1,k1]=Aref
                  bel[k1]=Aref*bc_val[m1]
               # end if
           # end for

           # assemble matrix Amat and right hand side rhs
           start2=clock.time()
           for k1 in range(0,m):
               m1=icon[iel,k1]
               for k2 in range(0,m):
                   m2=icon[iel,k2]
                   Amat[m1,m2]+=Ael[k1,k2]
               # end for
               rhs[m1]+=bel[k1]
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

