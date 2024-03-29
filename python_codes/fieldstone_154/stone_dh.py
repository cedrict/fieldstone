import numpy as np
import time
import scipy.sparse as sps

print("-----------------------------")
print("---------- stone 153 --------")
print("-----------------------------")

###############################################################################

Lx=1
Ly=1

nnx=33
nny=33

eta0=1

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

print("Nodes setup: %.5f s" % (time.time() - start))

###############################################################################
# functions for rhs
###############################################################################

def dbxdy(x,y):
    return -24*x**4 +48*x**3 -48*x**2 +144*x**2*y -144*x**2*y**2 +24*x -144*x*y\
            +144*x*y**2  -4 + 24*y -24*y**2 

def dbydx(x,y):
    return  24*x**2 -144*x**2*y +144*x**2*y**2 -24*x + 144*x*y -144*x*y**2 + 4 \
            -24*y + 48*y**2 - 48*y**3 + 24*y**4 

###############################################################################
# build matrix
###############################################################################
start = time.time()

A=np.zeros((N,N),dtype=np.float64)
b=np.zeros(N,dtype=np.float64)

for j in range(0,nny):
    for i in range(0,nnx):

        k=j*nnx+i
        kW=j*nnx+(i-1)
        kE=j*nnx+(i+1)
        kN=(j+1)*nnx+i
        kS=(j-1)*nnx+i

        if i==0 or i==nnx-1 or j==0 or j==nny-1:
           A[k,k]=1
           b[k]=0
        else:
           A[k,k]=-2/hx**2-2/hy**2
           A[k,kW]=1/hx**2
           A[k,kE]=1/hx**2
           A[k,kN]=1/hy**2
           A[k,kS]=1/hy**2
           b[k]=-1/eta0*(-dbxdy(x[k],y[k])+dbydx(x[k],y[k]))
        #end if

    #end for
#end for

print("Build matrix: %.5f s" % (time.time() - start))

###############################################################################
# solve system for omega
###############################################################################
start = time.time()

omega=sps.linalg.spsolve(sps.csr_matrix(A),b)

print("     -> omega (m,M) %e %e " %(np.min(omega),np.max(omega)))

print("Solve linear system omega: %.5f s" % (time.time() - start))

###############################################################################
# solve system for Psi
###############################################################################
start = time.time()

Psi=sps.linalg.spsolve(sps.csr_matrix(A),-omega)

print("     -> psi (m,M) %.4f %.4f " %(np.min(Psi),np.max(Psi)))

print("Solve linear system psi: %.5f s" % (time.time() - start))

###############################################################################
# recover velocity
###############################################################################
start = time.time()

u=np.zeros(N,dtype=np.float64)
v=np.zeros(N,dtype=np.float64)

for j in range(0,nny):
    for i in range(0,nnx):

        k=j*nnx+i
        kW=j*nnx+(i-1)
        kE=j*nnx+(i+1)
        kN=(j+1)*nnx+i
        kS=(j-1)*nnx+i

        if i==0: #left
           v[k]=-(Psi[kE] -Psi[k] )/hx
        elif i==nnx-1: #right   
           v[k]=-(Psi[k] -Psi[kW] )/hx
        elif j==0: #bottom
           u[k]= (Psi[kN]-Psi[k])/hy
        elif j==nny-1: #top
           u[k]= (Psi[k]-Psi[kS])/hy
        else:
           u[k]= (Psi[kN]-Psi[kS])/(2*hy)
           v[k]=-(Psi[kE] -Psi[kW] )/(2*hx)
        #end if

    #end for
#end for

print("     -> u (m,M) %.4f %.4f " %(np.min(u),np.max(u)))
print("     -> v (m,M) %.4f %.4f " %(np.min(v),np.max(v)))


print("Compute velocity: %.5f s" % (time.time() - start))

###############################################################################
# analytical solution
###############################################################################

def velocity_x(x,y):
    val=x*x*(1.-x)**2*(2.*y-6.*y*y+4*y*y*y)
    return val

def velocity_y(x,y):
    val=-y*y*(1.-y)**2*(2.*x-6.*x*x+4*x*x*x)
    return val

def pressure(x,y):
    val=x*(1.-x)-1./6.
    return val

###############################################################################
# export fields to vtu
###############################################################################
start = time.time()

m=4
nelx=nnx-1
nely=nny-1
nel=nelx*nely

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
vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (th)' Format='ascii'> \n")
for i in range(0,N):
    vtufile.write("%10e %10e %10e \n" % (velocity_x(x[i],y[i]),velocity_y(x[i],y[i]),0.))
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

print("Export to vtu: %.5f s" % (time.time() - start))

#np.savetxt('omega.ascii',np.array([x,y,omega]).T)
#np.savetxt('psi.ascii',np.array([x,y,Psi]).T)
#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T)

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")

