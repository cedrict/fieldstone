import numpy as np
import time

import scipy.sparse as sps

cm=0.01
year=365.25*3600*24

###############################################################################

Lx=1500e3
Ly=1000e3

nnx=51
nny=41

gy=-10

eta0=1e21

###############################################################################

N=nnx*nny

hx=Lx/(nnx-1)
hy=Ly/(nny-1)

###############################################################################
# mesh nodes layout 
###############################################################################
start = time.time()

x=np.zeros(N,dtype=np.float64)
y=np.zeros(N,dtype=np.float64)

counter=0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx
        y[counter]=j*hy
        counter+=1

###############################################################################
# assign density to nodes
###############################################################################
start = time.time()

rho=np.zeros(N,dtype=np.float64)

for i in range(0,N):
    if x[i]<Lx/2:
       rho[i]=3200
    else:
       rho[i]=3300

###############################################################################
# build matrix
###############################################################################
start = time.time()

A=np.zeros((N,N),dtype=np.float64)
b=np.zeros(N,dtype=np.float64)

for j in range(0,nny):
    for i in range(0,nnx):

        k=j*nnx+i
        k_west=j*nnx+(i-1)
        k_east=j*nnx+(i+1)
        k_north=(j+1)*nnx+i
        k_south=(j-1)*nnx+i

        #print (k,k_west,k_east,k_south,k_north)

        if i==0 or i==nnx-1 or j==0 or j==nny-1:
           A[k,k]=1
           b[k]=0
        else:
           A[k,k]=-2/hx**2-2/hy**2
           A[k,k_west]=1/hx**2
           A[k,k_east]=1/hx**2
           A[k,k_north]=1/hy**2
           A[k,k_south]=1/hy**2
           b[k]=-gy/eta0*(rho[k_west]-rho[k_east])/(2*hx)
        #end if

    #end for
#end for


###############################################################################
# solve system
###############################################################################
start = time.time()

omega=sps.linalg.spsolve(sps.csr_matrix(A),b)

print("Solve linear system: %.5f s" % (time.time() - start))

np.savetxt('omega.ascii',np.array([x,y,omega]).T)

###############################################################################

Psi=sps.linalg.spsolve(sps.csr_matrix(A),-omega)

np.savetxt('psi.ascii',np.array([x,y,Psi]).T)

###############################################################################
# recover velocity
###############################################################################

u=np.zeros(N,dtype=np.float64)
v=np.zeros(N,dtype=np.float64)

for j in range(0,nny):
    for i in range(0,nnx):

        k=j*nnx+i
        k_west=j*nnx+(i-1)
        k_east=j*nnx+(i+1)
        k_north=(j+1)*nnx+i
        k_south=(j-1)*nnx+i

        if i==0 and j==0:
           u[k]=0
           v[k]=0
        elif i==nnx-1 and j==0:
           u[k]=0
           v[k]=0
        elif i==nnx-1 and j==nny-1:
           u[k]=0
           v[k]=0
        elif i==0 and j==nny-1:
           u[k]=0
           v[k]=0
        elif i==0 or i==nnx-1: #left,right 
           u[k]=0
           #v[k]=-(Psi[k_east] -Psi[k_west] )/(hx)
        elif j==0 or j==nny-1: #bottom, top
           #u[k]=(Psi[k_north]-Psi[k_south])/(hy)
           v[k]=0
        else:
           u[k]= (Psi[k_north]-Psi[k_south])/(2*hy)
           v[k]=-(Psi[k_east] -Psi[k_west] )/(2*hx)
        #end if

    #end for
#end for

np.savetxt('velocity.ascii',np.array([x,y,u/cm*year,v/cm*year]).T)

###############################################################################
# export fields to vtu
###############################################################################

m=4

nelx=nnx-1
nely=nny-1
nel=(nnx-1)*(nny-1)


icon =np.zeros((m,nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1


filename = 'solution.vtu'
vtufile=open(filename,"w")
vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
vtufile.write("<UnstructuredGrid> \n")
vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(N,nel))
#####
vtufile.write("<Points> \n")
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
vtufile.write("</DataArray>\n")
vtufile.write("</Points> \n")
#####
#vtufile.write("<CellData Scalars='scalars'>\n")
#--
#vtufile.write("<DataArray type='Float32' Name='p' Format='ascii'> \n")
#for iel in range (0,nel):
#    vtufile.write("%10e\n" % p[iel])
#vtufile.write("</DataArray>\n")
#--
#vtufile.write("</CellData>\n")
#####
vtufile.write("<PointData Scalars='scalars'>\n")
#--
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='omega' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %omega[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='psi' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %Psi[i])
vtufile.write("</DataArray>\n")
#--
vtufile.write("<DataArray type='Float32' Name='density' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e \n" %rho[i])
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
    vtufile.write("%d \n" %((iel+1)*m))
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

