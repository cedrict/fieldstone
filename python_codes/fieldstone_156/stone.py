import numpy as np

print("-----------------------------")
print("--------- stone 156 ---------")
print("-----------------------------")

tfinal=10
dt=1e-4

scheme=2

sigma=10
r=28
b=8/3
Lx=np.sqrt(2)
Ly=1
laambda=Lx/Ly

every=100

###############################################################################

nnx=48
nny=32
m=4
nelx=nnx-1
nely=nny-1
nel=nelx*nely
NV=nnx*nny

nstep=int(tfinal/dt)

###############################################################################

A=np.zeros(nstep,dtype=np.float64)
B=np.zeros(nstep,dtype=np.float64)
C=np.zeros(nstep,dtype=np.float64)
t=np.zeros(nstep,dtype=np.float64)

t[0]=0
A[0]=0
B[0]=0.5
C[0]=25

###############################################################################
# compute A,B,C,t fields 
###############################################################################

if scheme==1:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       A[istep]=(-sigma*A[istep-1]+sigma*B[istep-1])*dt            +A[istep-1]
       B[istep]=(-A[istep-1]*C[istep-1]+r*A[istep-1]-B[istep-1])*dt+B[istep-1]
       C[istep]=(A[istep-1]*B[istep-1]-b*C[istep-1])*dt            +C[istep-1]

elif scheme==2:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       A[istep]=(-sigma*A[istep-1]+sigma*B[istep-1])*dt        +A[istep-1]
       B[istep]=(-A[istep]*C[istep-1]+r*A[istep]-B[istep-1])*dt+B[istep-1]
       C[istep]=(A[istep]*B[istep]-b*C[istep-1])*dt            +C[istep-1]

elif scheme==3:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       Anew=(-sigma*A[istep-1]+sigma*B[istep-1])*dt            +A[istep-1]
       Bnew=(-A[istep-1]*C[istep-1]+r*A[istep-1]-B[istep-1])*dt+B[istep-1]
       Cnew=(A[istep-1]*B[istep-1]-b*C[istep-1])*dt            +C[istep-1]
       #print(istep,Anew)
       for k in range(0,3):
           Anew=(-sigma*Anew+sigma*Bnew)*dt  +A[istep-1]
           Bnew=(-Anew*Cnew+r*Anew-Bnew)*dt  +B[istep-1]
           Cnew=(Anew*Bnew-b*Cnew)*dt        +C[istep-1]
           #print(istep,Anew)
       A[istep]=Anew
       B[istep]=Bnew
       C[istep]=Cnew

np.savetxt('ABC.ascii',np.array([t,A,B,C]).T,fmt='%1.5e')

exit()

###############################################################################
# make mesh
###############################################################################

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

###############################################################################
# build connectivity array
###############################################################################

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

###############################################################################
# swarm setup
###############################################################################

nmarker=1600
swarm_x=np.empty(nmarker,dtype=np.float64)
swarm_y=np.empty(nmarker,dtype=np.float64)

ddx=0.025
ddy=0.025

counter=0
for i in range(0,40):
    for j in range(0,40):
        swarm_x[counter]=Lx/2-ddx/2+i*ddx/39
        swarm_y[counter]=Ly/2-ddy/2+j*ddy/39
        counter+=1

###############################################################################
# compute u,v,psi in time based on B,C
###############################################################################

Psi=np.zeros(NV,dtype=np.float64)
u=np.zeros(NV,dtype=np.float64)
v=np.zeros(NV,dtype=np.float64)

for istep in range(0,nstep):

    ###########################################################################
    # compute psi,u,v
    ###########################################################################

    Psi=(laambda*np.sqrt(2)/np.pi**2)*np.sin(np.pi*y)*\
        (B[istep]*np.sin(x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*x/laambda))

    u=(laambda*np.sqrt(2)/np.pi)*np.cos(np.pi*y)*\
      (B[istep]*np.sin(x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*x/laambda))

    v=-(np.sqrt(2)/np.pi)*np.sin(np.pi*y)*\
       (B[istep]*np.cos(x*np.pi/laambda)+2*(C[istep]-27)*np.cos(2*np.pi*x/laambda))

    ###########################################################################
    # interpolate velocity onto markers and advect them
    ###########################################################################

    swarm_u=(laambda*np.sqrt(2)/np.pi)*np.cos(np.pi*swarm_y)*\
            (B[istep]*np.sin(swarm_x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*swarm_x/laambda))

    swarm_v=-(np.sqrt(2)/np.pi)*np.sin(np.pi*swarm_y)*\
             (B[istep]*np.cos(swarm_x*np.pi/laambda)+2*(C[istep]-27)*np.cos(2*np.pi*swarm_x/laambda))

    swarm_x+=swarm_u*dt
    swarm_y+=swarm_v*dt

    ###########################################################################
    # export to vtu
    ###########################################################################

    if istep%every==0:
       filename = 'solution_{:05d}.vtu'.format(istep)
       vtufile=open(filename,"w")

       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(x[i],y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='psi' Format='ascii'> \n")
       for i in range(0,NV):
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


       filename = 'markers_{:05d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % 1)
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


