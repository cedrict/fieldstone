import numpy as np
import random
import time as clock 
import numba

###############################################################################
###############################################################################
###############################################################################

Lx=1.
Ly=.5
Lz=1.

nnx=101
nny=int(nnx*Ly/Lx)
nnz=int(nnx*Lz/Lx)

hx=Lx/(nnx-1)
hy=Ly/(nny-1)
hz=Lz/(nnz-1)

dt=1e-1

Du=0.000004
Dv=0.000002
Feed=0.035
Kill=0.0575

nelx=nnx-1
nely=nny-1
nelz=nnz-1
nel=nelx*nely*nelz
NP=nnx*nny*nnz

nstep=10000

every=100

print("-----------------------------")
print('nnx=',nnx)
print('nny=',nny)
print('nnz=',nnz)
print('NP=',NP)
print('Du=',Du)
print('Dv=',Dv)
print('Feed=',Feed)
print('Kill=',Kill)
print('nstep=',nstep)
print('dt=',dt)

print(hx**2/Du,hx**2/Dv)

###############################################################################
# create mesh 
###############################################################################
start=clock.time()

x=np.zeros(NP,dtype=np.float64)
y=np.zeros(NP,dtype=np.float64)
z=np.zeros(NP,dtype=np.float64)

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
   
icon=np.zeros((8,nel),dtype=np.int32)

counter=0 
for i in range(0,nelx):
    for j in range(0,nely):
        for k in range(0,nelz):
            icon[0,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k
            icon[1,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k
            icon[2,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k
            icon[3,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k
            icon[4,counter]=nny*nnz*(i-1+1)+nnz*(j-1+1)+k+1
            icon[5,counter]=nny*nnz*(i  +1)+nnz*(j-1+1)+k+1
            icon[6,counter]=nny*nnz*(i  +1)+nnz*(j  +1)+k+1
            icon[7,counter]=nny*nnz*(i-1+1)+nnz*(j  +1)+k+1
            counter += 1
        #end for
    #end for
#end for

print("build mesh: %.3f s" % (clock.time()-start))

###############################################################################
# initial conditions 
###############################################################################
start=clock.time()

u=np.zeros(NP,dtype=np.float64)
v=np.zeros(NP,dtype=np.float64)
X=np.zeros(2*NP,dtype=np.float64)

nseed=16
seed_size=0.05

u[:]=1
v[:]=0

for iseed in range(nseed):
    
    xs=random.uniform(0+seed_size,1-seed_size)
    ys=random.uniform(0+seed_size,1-seed_size)
    zs=random.uniform(0+seed_size,1-seed_size)
    for i in range(0,NP):
        if abs(x[i]-xs)<seed_size and\
           abs(y[i]-ys)<seed_size and\
           abs(z[i]-zs)<seed_size :
           u[i]=0.75 #random.uniform(0.5,1)

    xs=random.uniform(0+seed_size,1-seed_size)
    ys=random.uniform(0+seed_size,1-seed_size)
    zs=random.uniform(0+seed_size,1-seed_size)
    for i in range(0,NP):
        if abs(x[i]-xs)<seed_size and\
           abs(y[i]-ys)<seed_size and\
           abs(z[i]-zs)<seed_size :
           v[i]=0.125 #random.uniform(0,0.25)

X[0:NP]=u[:]
X[NP:2*NP]=v[:]

print("initial conditions: %.3f s" % (clock.time()-start))

###############################################################################

@numba.njit
def compute_node_index(i,j,k):
    return nny*nnz*i+nnz*j+k

###############################################################################
# defining function that returns dX_dt at all nodes

@numba.njit
def F(Du,Dv,F,K,NP,hx,hy,hz,u,v):
    dX_dt=np.zeros(2*NP,dtype=np.float64)

    Duhx2=Du/hx**2
    Duhy2=Du/hy**2
    Duhz2=Du/hz**2
    Dvhx2=Dv/hx**2
    Dvhy2=Dv/hy**2
    Dvhz2=Dv/hz**2

    counter=0
    for i in range(0,nnx):
        for j in range(0,nny):
            for k in range(0,nnz):
                #-----------------
                if i==0:
                   front=compute_node_index(i+1,j,k)
                   back =compute_node_index(nnx-1,j,k)
                elif i==nnx-1:
                   front=compute_node_index(0,j,k)
                   back =compute_node_index(i-1,j,k)
                else:
                   front=compute_node_index(i+1,j,k)
                   back =compute_node_index(i-1,j,k)
                #-----------------
                if j==0:
                   left=compute_node_index(i,nny-1,k)
                   right=compute_node_index(i,j+1,k)
                elif j==nny-1:
                   left=compute_node_index(i,j-1,k)
                   right=compute_node_index(i,0,k)
                else:
                   left=compute_node_index(i,j-1,k)
                   right=compute_node_index(i,j+1,k)
                #-----------------
                if k==0:
                   bottom=compute_node_index(i,j,nnz-1)
                   top=compute_node_index(i,j,k+1)
                elif k==nnz-1:
                   bottom=compute_node_index(i,j,k-1)
                   top=compute_node_index(i,j,0)
                else:
                   bottom=compute_node_index(i,j,k-1)
                   top=compute_node_index(i,j,k+1)
                #-----------------
                #print(counter,'back=',back,'front=',front,'left=',left,'right=',right,bottom,top)
                #-----------------
                dX_dt[counter]=Duhx2*(u[front]-2*u[counter]+u[back])\
                              +Duhy2*(u[left] -2*u[counter]+u[right])\
                              +Duhz2*(u[top]  -2*u[counter]+u[bottom])
                              #-u[counter]*v[counter]**2+F*(1-u[counter])

                dX_dt[counter+NP]=Dvhx2*(v[front]-2*v[counter]+v[back])\
                                 +Dvhy2*(v[left] -2*v[counter]+v[right])\
                                 +Dvhz2*(v[top]  -2*v[counter]+v[bottom])
                                 #+u[counter]*v[counter]**2-(F+K)*v[counter]
                counter+=1

            #end for
        #end for
    #end for

    return dX_dt

###############################################################################
# time stepping loop
###############################################################################

t=0
for istep in range(0,nstep+1):
    start=clock.time()
    print("-----------------------------")
    print("istep= ", istep,'| t=',t)
    X[:]+=F(Du,Dv,Feed,Kill,NP,hx,hy,hz,u,v)*dt
    u[:]=X[0:NP]
    v[:]=X[NP:2*NP]
    print("     -> u (m,M) %e %e " %(np.min(u),np.max(u)))
    print("     -> v (m,M) %e %e " %(np.min(v),np.max(v)))
    t+=dt
    print("     update solution: %.3f s" % (clock.time()-start))

    if istep%every==0 or istep==nstep:
       filename = 'solution_{:05d}.vtu'.format(istep)
       #np.savetxt(filename,np.array([x,T]).T,fmt='%1.5e')

       ########################################################################
       # export solution to vtu format
       ########################################################################
       start=clock.time()

       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%10f %10f %10f \n" %(x[i],y[i],z[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%f \n" %(u[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='v' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%f \n" %(v[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %12)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("     export to vtu: %.3f s" % (clock.time()-start))
