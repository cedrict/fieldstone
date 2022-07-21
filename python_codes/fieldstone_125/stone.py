import time as time
import numpy as np

#################################################################

m=4

nelx=24
nely=24
Lx=1
Ly=1

nnx=nelx+1
nny=nely+1
nel=nelx*nely
NV=nnx*nny

hx=Lx/nelx
hy=Ly/nely

nmarker_per_dim=10
nmarker=nel*nmarker_per_dim**2

print('nel=',nel)
print('NV =',NV)
print('nmarker=',nmarker)

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# build connectivity array
#################################################################
start = time.time()

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# swarm (=all the particles) setup
#################################################################
start = time.time()

swarm_x=np.empty(nmarker,dtype=np.float64)        # x coordinates   
swarm_y=np.empty(nmarker,dtype=np.float64)        # y coordinates 
swarm_cell=np.empty(nmarker,dtype=np.int)   

counter=0
for iel in range(0,nel):
    x1=x[icon[0,iel]] ; y1=y[icon[0,iel]]
    x2=x[icon[1,iel]] ; y2=y[icon[1,iel]]
    x3=x[icon[2,iel]] ; y3=y[icon[2,iel]]
    x4=x[icon[3,iel]] ; y4=y[icon[3,iel]]
    for j in range(0,nmarker_per_dim):
        for i in range(0,nmarker_per_dim):
            r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
            s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
            N1=0.25*(1-r)*(1-s)
            N2=0.25*(1+r)*(1-s)
            N3=0.25*(1+r)*(1+s)
            N4=0.25*(1-r)*(1+s)
            swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
            swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
            counter+=1
        #end for 
    #end for 
#end for 

print("swarm setup: %.3f s" % (time.time() - start))

#################################################################
# voronoi cell centers point setup
#################################################################
start = time.time()

nvo=7

xvo = np.empty(nvo,dtype=np.float64)  # x coordinates
yvo = np.empty(nvo,dtype=np.float64)  # y coordinates

xvo[0]=0.129
yvo[0]=0.27

xvo[1]=0.67
yvo[1]=0.33

xvo[2]=0.76
yvo[2]=0.69

xvo[3]=0.31
yvo[3]=0.71

xvo[4]=0.499
yvo[4]=0.499

xvo[5]=0.89
yvo[5]=0.08

xvo[6]=0.53
yvo[6]=0.6

print("cell centers setup: %.3f s" % (time.time() - start))

#################################################################
# assign markers a voronoi cell
# for each marker I loop over the seeds (loop over i). 
# For every seed i, I loop over all other ones j, and then 
# I test whether the marker is closer to i than j. 
# If one the marker happens to be closer to j than i, then 
# I exit the loop over j, and try a different i. 
# The Voronoi diagram is unique, so in the end I am bound
# to find a seed i so that the marker is always closer to it
# than anyother seed.
#################################################################
start = time.time()

swarm_cell[:]=-1

iin= np.empty(nvo,dtype=np.bool)  # x coordinates

for im in range(0,nmarker):
    iin[:]=True
    for i in range(0,nvo):
        for j in range(0,nvo):
            if i != j:
               xim=swarm_x[im]-xvo[i]
               yim=swarm_y[im]-yvo[i]
               xjm=swarm_x[im]-xvo[j]
               yjm=swarm_y[im]-yvo[j]
               dim=np.sqrt(xim**2+yim**2)
               djm=np.sqrt(xjm**2+yjm**2)
               if dim>djm:
                  iin[i]=False 
                  break
               #end if
            #end if
        #end for
    #end for
    where=np.where(iin)[0]
    swarm_cell[im]=where[0]

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_cell]).T)

print("build Voronoi cells: %.3f s" % (time.time() - start))

#################################################################
# export swarm to vtu 
#################################################################

if True:
    filename = 'swarm.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='cell' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %swarm_cell[i])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nmarker):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nmarker):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()
    filename = 'voronoi_centers.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nvo,nvo))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nvo):
        vtufile.write("%10e %10e %10e \n" %(xvo[i],yvo[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    vtufile.write("<Cells>\n")
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for i in range(0,nvo):
        vtufile.write("%d " % i)
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for i in range(0,nvo):
        vtufile.write("%d " % (i+1))
    vtufile.write("</DataArray>\n")
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for i in range(0,nvo):
        vtufile.write("%d " % 1)
    vtufile.write("</DataArray>\n")
    vtufile.write("</Cells>\n")
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()

