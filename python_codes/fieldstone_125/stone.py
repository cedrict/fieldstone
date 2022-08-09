import time as time
import numpy as np
import random

#################################################################

ndim=2

print("-----------------------------")
print("---------- stone 125 --------")
print("-----------------------------")

if ndim==2: m=4 # number of nodes for mackground cells
if ndim==3: m=8 # number of nodes for mackground cells

nelx=20
nely=nelx
nelz=nelx
Lx=1
Ly=1
Lz=1

nnx=nelx+1
nny=nely+1
nnz=nelz+1

if ndim==2:
   nel=nelx*nely
   NV=nnx*nny
else:
   nel=nelx*nely*nelz
   NV=nnx*nny*nnz

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

nmarker_per_dim=10
nmarker=nel*nmarker_per_dim**ndim

print('ndim=',ndim)
print('Lx=',Lx)
print('Ly=',Ly)
print('nmarker_per_dim=',nmarker_per_dim)
print('nel=',nel)
print('NV =',NV)
print('nmarker=',nmarker)

test=4

#################################################################
# grid point setup
#################################################################
start = time.time()

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates
z = np.empty(NV,dtype=np.float64)  # y coordinates

if ndim==2:
   counter = 0
   for j in range(0, nny):
       for i in range(0, nnx):
           x[counter]=i*Lx/float(nelx)
           y[counter]=j*Ly/float(nely)
           counter += 1

if ndim==3:
   counter=0
   for i in range(0,nnx):
       for j in range(0,nny):
           for k in range(0,nnz):
               x[counter]=i*Lx/float(nelx)
               y[counter]=j*Ly/float(nely)
               z[counter]=k*Lz/float(nelz)
               counter += 1
           #end for
       #end for
   #end for

print("setup: grid points: %.3f s" % (time.time() - start))

#################################################################
# build connectivity array
#################################################################
start = time.time()

icon=np.zeros((m,nel),dtype=np.int32)

if ndim==2:
   counter = 0
   for j in range(0,nely):
       for i in range(0,nelx):
           icon[0,counter]= i + j * (nelx + 1)
           icon[1,counter]= i + 1 + j * (nelx + 1)
           icon[2,counter]= i + 1 + (j + 1) * (nelx + 1)
           icon[3,counter]= i + (j + 1) * (nelx + 1)
           counter += 1

if ndim==3:
   counter = 0 
   for i in range(0, nelx):
       for j in range(0, nely):
           for k in range(0, nelz):
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

print("setup: connectivity: %.3f s" % (time.time() - start))

#################################################################
# swarm (=all the particles) setup
#################################################################
start = time.time()

swarm_x=np.zeros(nmarker,dtype=np.float64)   # x coordinates   
swarm_y=np.zeros(nmarker,dtype=np.float64)   # y coordinates 
swarm_z=np.zeros(nmarker,dtype=np.float64)   # y coordinates 
swarm_cell=np.empty(nmarker,dtype=np.int)    # associated V cell 

if ndim==2:
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

if ndim==3:
   counter=0
   for iel in range(0,nel):
       x1=x[icon[0,iel]] ; y1=y[icon[0,iel]] ; z1=z[icon[0,iel]]
       x2=x[icon[1,iel]] ; y2=y[icon[1,iel]] ; z2=z[icon[1,iel]]
       x3=x[icon[2,iel]] ; y3=y[icon[2,iel]] ; z3=z[icon[2,iel]]
       x4=x[icon[3,iel]] ; y4=y[icon[3,iel]] ; z4=z[icon[3,iel]]
       x5=x[icon[4,iel]] ; y5=y[icon[4,iel]] ; z5=z[icon[4,iel]]
       x6=x[icon[5,iel]] ; y6=y[icon[5,iel]] ; z6=z[icon[5,iel]]
       x7=x[icon[6,iel]] ; y7=y[icon[6,iel]] ; z7=z[icon[6,iel]]
       x8=x[icon[7,iel]] ; y8=y[icon[7,iel]] ; z8=z[icon[7,iel]]
       for k in range(0,nmarker_per_dim):
           for j in range(0,nmarker_per_dim):
               for i in range(0,nmarker_per_dim):
                   r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
                   s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
                   t=-1.+k*2./nmarker_per_dim + 1./nmarker_per_dim
                   N1=0.125*(1.-r)*(1.-s)*(1.-t)
                   N2=0.125*(1.+r)*(1.-s)*(1.-t)
                   N3=0.125*(1.+r)*(1.+s)*(1.-t)
                   N4=0.125*(1.-r)*(1.+s)*(1.-t)
                   N5=0.125*(1.-r)*(1.-s)*(1.+t)
                   N6=0.125*(1.+r)*(1.-s)*(1.+t)
                   N7=0.125*(1.+r)*(1.+s)*(1.+t)
                   N8=0.125*(1.-r)*(1.+s)*(1.+t)
                   swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6+N7*x7+N8*x8
                   swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4+N5*y5+N6*y6+N7*y7+N8*y8
                   swarm_z[counter]=N1*z1+N2*z2+N3*z3+N4*z4+N5*z5+N6*z6+N7*z7+N8*z8
                   counter+=1
               #end for 
           #end for 
       #end for 
   #end for 

print("swarm setup: %.3f s" % (time.time() - start))

#################################################################
# voronoi cell centers point setup
#################################################################
start = time.time()

if test==1:
   nvo=7
   xvo = np.empty(nvo,dtype=np.float64)
   yvo = np.empty(nvo,dtype=np.float64)
   zvo = np.empty(nvo,dtype=np.float64)
   xvo[0]=0.129 ; yvo[0]=0.27  ; zvo[0]=0.12
   xvo[1]=0.67  ; yvo[1]=0.33  ; zvo[1]=0.86
   xvo[2]=0.76  ; yvo[2]=0.69  ; zvo[2]=0.45
   xvo[3]=0.31  ; yvo[3]=0.71  ; zvo[3]=0.67
   xvo[4]=0.499 ; yvo[4]=0.499 ; zvo[4]=0.33
   xvo[5]=0.89  ; yvo[5]=0.08  ; zvo[5]=0.21
   xvo[6]=0.53  ; yvo[6]=0.6   ; zvo[6]=0.71

if test==2: nvo=27
if test==3: nvo=111
if test==4: nvo=213

xvo = np.empty(nvo,dtype=np.float64) 
yvo = np.empty(nvo,dtype=np.float64) 
zvo = np.empty(nvo,dtype=np.float64) 
for i in range(0,nvo):
    xvo[i]=random.uniform(0.01,0.99)
    yvo[i]=random.uniform(0.01,0.99)
    zvo[i]=random.uniform(0.01,0.99)

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

if ndim==2:
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
   #end for
#end if

if ndim==3:
   for im in range(0,nmarker):
       iin[:]=True
       for i in range(0,nvo):
           for j in range(0,nvo):
               if i != j:
                  xim=swarm_x[im]-xvo[i]
                  yim=swarm_y[im]-yvo[i]
                  zim=swarm_z[im]-zvo[i]
                  xjm=swarm_x[im]-xvo[j]
                  yjm=swarm_y[im]-yvo[j]
                  zjm=swarm_z[im]-zvo[j]
                  dim=np.sqrt(xim**2+yim**2+zim**2)
                  djm=np.sqrt(xjm**2+yjm**2+zjm**2)
                  if dim>djm:
                     iin[i]=False 
                     break
                  #end if
               #end if
           #end for
       #end for
       where=np.where(iin)[0]
       swarm_cell[im]=where[0]
       if im%1000==0: print(im,nmarker)
   #end for
#end if

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y,swarm_z,swarm_cell]).T)

print("build Voronoi cells: %.3f s" % (time.time() - start))

#################################################################
# compute cells area/volume
#################################################################
start = time.time()
   
volume = np.zeros(nvo,dtype=np.float64)

if ndim==2: marker_volume=(hx/nmarker_per_dim)*(hy/nmarker_per_dim)
if ndim==3: marker_volume=(hx/nmarker_per_dim)*(hy/nmarker_per_dim)*(hz/nmarker_per_dim)

for im in range(0,nmarker):
    volume[swarm_cell[im]]+=marker_volume

print('    -> volume (m,M) = %.4f %.4f' %(np.min(volume),np.max(volume)))
print('    -> sum(volume)=', np.sum(volume))

print("compute cells volume: %.3f s" % (time.time() - start))

#################################################################
# volume distribution
#################################################################
start = time.time()

Vmin=0
Vmax=Lx*Ly*Lz/nvo * 6
nbin=40
delta=(Vmax-Vmin)/nbin

Vdistribution=np.zeros(nbin,dtype=np.float64)

for i in range(0,nvo):
    index=int(volume[i]/delta)
    Vdistribution[index]+=1

np.savetxt('Vdistribution.ascii',Vdistribution.T)

print("compute cells volume distribution: %.3f s" % (time.time() - start))

#################################################################
# export swarm to vtu 
#################################################################
start = time.time()

if True:
    vtufile=open("swarm.vtu","w")
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
    vtufile.write("<DataArray type='Float32' Name='volume' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%3e \n" %volume[swarm_cell[i]])
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    for i in range(0,nmarker):
        vtufile.write("%e %e %e \n" %(swarm_x[i],swarm_y[i],swarm_z[i]))
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

    vtufile=open("voronoi_centers.vtu","w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nvo,nvo))
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
    if ndim==2:
       for i in range(0,nvo):
           vtufile.write("%10e %10e %10e \n" %(xvo[i],yvo[i],0.))
    if ndim==3:
       for i in range(0,nvo):
           vtufile.write("%10e %10e %10e \n" %(xvo[i],yvo[i],zvo[i]))
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

print("make vtu files: %.3f s" % (time.time() - start))
