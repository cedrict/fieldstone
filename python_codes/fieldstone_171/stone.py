import numpy as np
import sys as sys
import random
import time as clock 
import numba
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################

three_dimensions=False

#init=1: random
#init=2: three discs
#init=3: a la pearson 1993

if int(len(sys.argv))==5:
   model  = str(sys.argv[1])
   nnx    = int(sys.argv[2])
   scheme = str(sys.argv[3])
   init   = int(sys.argv[4])
   print(sys.argv)
else:
   model='zeta'
   nnx = 1025
   scheme='RK2'
   init=1

if three_dimensions:
   Lx=2.5
   Ly=0.5
   Lz=2.5
   nny=int(nnx*Ly/Lx)
   nnz=int(nnx*Lz/Lx)
   hx=Lx/(nnx-1)
   hy=Ly/(nny-1)
   hz=Lz/(nnz-1)
   nelx=nnx-1
   nely=nny-1
   nelz=nnz-1
   nel=nelx*nely*nelz
   NP=nnx*nny*nnz
   m=8
   tyype=12 # vtu
   nseed=1000

else:
   Lx=2.5
   Lz=2.5
   nny=1
   nnz=int(nnx*Lz/Lx)
   hx=Lx/(nnx-1)
   hz=Lz/(nnz-1)
   nelx=nnx-1
   nelz=nnz-1
   nel=nelx*nelz
   NP=nnx*nnz
   m=4
   tyype=9 # vtu
   nseed=500

dt=1

nstep=100000

every_ascii=100
every_vtu=1000
every_png=1000

seed_size=0.02

###########################################################

if model=='alpha':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.01
   Kill=0.047
if model=='beta':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.014 
   Kill= 0.039
if model=='gamma':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.022 
   Kill=0.051 
if model=='delta':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.030
   Kill=0.055 
if model=='epsilon':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.018
   Kill=0.055
if model=='zeta':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.022
   Kill=0.061
if model=='eta':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.034
   Kill=0.063
if model=='theta':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.03
   Kill=0.057
if model=='iota':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.046
   Kill=0.0594
if model=='kappa':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.05
   Kill=0.063
if model=='lambda':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.026
   Kill=0.061
if model=='mu':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.046
   Kill=0.065
if model=='nu':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.054
   Kill=0.067
if model=='xi':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.01
   Kill=0.041
if model=='pi':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.062
   Kill=0.061
if model=='rho':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.09
   Kill=0.059
if model=='sigma':
   Du=2.e-5 ; Dv=1e-5
   Feed=0.09
   Kill=0.057

###########################################################

print("-----------------------------")
print('model=',model)
print('nnx=',nnx)
print('nny=',nny)
print('nnz=',nnz)
print('NP=',NP)
print('Du=',Du)
print('Dv=',Dv)
print('Feed=',Feed)
print('Kill=',Kill)
print('scheme=',scheme)
print('nstep=',nstep)
print('dt=',dt)

print('diff dt:',hx**2/Du,hx**2/Dv)

###############################################################################
# create mesh 
###############################################################################
start=clock.time()

if three_dimensions:

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
   
   icon=np.zeros((m,nel),dtype=np.int32)

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

else:

   x=np.zeros(NP,dtype=np.float64)
   y=np.zeros(NP,dtype=np.float64)
   z=np.zeros(NP,dtype=np.float64)

   counter = 0
   for j in range(0,nnz):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nelx)
           z[counter]=j*Lz/float(nelz)
           counter += 1

   icon=np.zeros((m,nel),dtype=np.int32)

   counter = 0
   for j in range(0,nelz):
       for i in range(0,nelx):
           icon[0,counter]=i+j*(nelx+1)
           icon[1,counter]=i+1+j*(nelx+1)
           icon[2,counter]=i+1+(j+1)*(nelx+1)
           icon[3,counter]=i+(j+1)*(nelx+1)
           counter += 1

print("build mesh: %.3f s" % (clock.time()-start))

###############################################################################
# initial conditions for u,v,X
###############################################################################
start=clock.time()

u=np.zeros(NP,dtype=np.float64)
v=np.zeros(NP,dtype=np.float64)

if init==1:
       
   for i in range(0,NP):
       u[i]=random.uniform(0.8,1) # close to 1
       v[i]=random.uniform(0,0.2) # close to 0 

   if three_dimensions:

      for iseed in range(nseed):
          xs=random.uniform(0+seed_size,Lx-seed_size)
          ys=random.uniform(0+seed_size,Ly-seed_size)
          zs=random.uniform(0+seed_size,Lz-seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(y[i]-ys)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 u[i]=random.uniform(0.5,0.75)
          xs=random.uniform(0+seed_size,Lx-seed_size)
          ys=random.uniform(0+seed_size,Ly-seed_size)
          zs=random.uniform(0+seed_size,Lz-seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(y[i]-ys)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 v[i]=random.uniform(0.25,0.5)

   else:

      for iseed in range(nseed):
          xs=random.uniform(0+2*seed_size,Lx-2*seed_size)
          zs=random.uniform(0+2*seed_size,Lz-2*seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 u[i]=random.uniform(0.5,1)
          xs=random.uniform(0+2*seed_size,Lx-2*seed_size)
          zs=random.uniform(0+2*seed_size,Lz-2*seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 v[i]=random.uniform(0.,0.5)

if init==2:
   for i in range(0,NP):
       if (x[i]-Lx/3)**2+(z[i]-Lz/3)**2<0.2:
          u[i]=1
          v[i]=0.5
       if (x[i]-Lx*0.67)**2+(z[i]-Lz/2)**2<0.2:
          u[i]=0.5
          v[i]=0.25
       if (x[i]-Lx*0.5)**2+(z[i]-Lz*0.75)**2<0.08:
          u[i]=0.7
          v[i]=0.125

if init==3:
   u[:]=1
   v[:]=0
   for i in range(0,NP):
       if abs(x[i]-Lx/2)<0.2 and abs(z[i]-Lz/2)<0.2:
          u[i]=1/2+random.uniform(-0.01,0.01)
          v[i]=1/4+random.uniform(-0.01,0.01)

X=np.zeros(2*NP,dtype=np.float64)
X[0:NP]=u[:]
X[NP:2*NP]=v[:]

print("initial conditions: %.3f s" % (clock.time()-start))

###############################################################################
# defining function that returns dX_dt at all nodes in 2d

@numba.njit
def F_2d(Du,Dv,F,K,NP,hx,hz,X):
    dX_dt=np.zeros(2*NP,dtype=np.float64)
    u=X[0:NP]
    v=X[NP:2*NP]

    Duhx2=Du/hx**2 ; Duhz2=Du/hz**2
    Dvhx2=Dv/hx**2 ; Dvhz2=Dv/hz**2

    counter=0
    for k in range(0,nnz):
        for i in range(0,nnx):
            if i==0:
               left=(nnx-1)+k*nnx
               right=(i+1)+k*nnx
            elif i==nnx-1:
               left=(i-1)+k*nnx
               right=(0)+k*nnx
            else:
               left=(i-1)+k*nnx
               right=(i+1)+k*nnx
            #-----------------
            if k==0:
               top=i+(k+1)*nnx
               bottom=i+(nnz-1)*nnx
            elif k==nnz-1:
               top=i+(0)*nnx
               bottom=i+(k-1)*nnx
            else:
               top=i+(k+1)*nnx
               bottom=i+(k-1)*nnx
            #-----------------
            dX_dt[counter]=Duhx2*(u[left]-2*u[counter]+u[right])\
                          +Duhz2*(u[top] -2*u[counter]+u[bottom])\
                          -u[counter]*v[counter]**2+F*(1-u[counter])
            dX_dt[counter+NP]=Dvhx2*(v[left]-2*v[counter]+v[right])\
                             +Dvhz2*(v[top] -2*v[counter]+v[bottom])\
                             +u[counter]*v[counter]**2-(F+K)*v[counter]
            counter+=1
        #end for
    #end for

    return dX_dt

###############################################################################
# defining function that returns dX_dt at all nodes in 3d

@numba.njit
def compute_node_index(i,j,k):
    return nny*nnz*i+nnz*j+k

@numba.njit
def F_3d(Du,Dv,F,K,NP,hx,hy,hz,u,v):
    dX_dt=np.zeros(2*NP,dtype=np.float64)

    Duhx2=Du/hx**2 ; Duhy2=Du/hy**2 ; Duhz2=Du/hz**2
    Dvhx2=Dv/hx**2 ; Dvhy2=Dv/hy**2 ; Dvhz2=Dv/hz**2

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
                dX_dt[counter]=Duhx2*(u[front]-2*u[counter]+u[back])\
                              +Duhy2*(u[left] -2*u[counter]+u[right])\
                              +Duhz2*(u[top]  -2*u[counter]+u[bottom])\
                              -u[counter]*v[counter]**2+F*(1-u[counter])
                dX_dt[counter+NP]=Dvhx2*(v[front]-2*v[counter]+v[back])\
                                 +Dvhy2*(v[left] -2*v[counter]+v[right])\
                                 +Dvhz2*(v[top]  -2*v[counter]+v[bottom])\
                                 +u[counter]*v[counter]**2-(F+K)*v[counter]
                counter+=1
            #end for
        #end for
    #end for

    return dX_dt

###############################################################################
# time stepping loop
###############################################################################
stats_u_file=open(model+'_stats_u.ascii',"w")
stats_v_file=open(model+'_stats_v.ascii',"w")

t=0
for istep in range(0,nstep+1):
    start=clock.time()

    if scheme=='RK1':
       if three_dimensions:
          X[:]+=F_3d(Du,Dv,Feed,Kill,NP,hx,hy,hz,u,v)*dt
       else:
          X[:]+=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X)*dt

    if scheme=='Heun':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X   )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1)*dt
       X[:]+=(k1+k2)/2

    if scheme=='RK2':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X     )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/2)*dt
       X[:]+=k2

    if scheme=='RK3':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X        )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/2   )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X-k1+2*k2)*dt
       X[:]+=(k1+4*k2+k3)/6

    if scheme=='RK38':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X         )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/3    )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X-k1/3+k2 )*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1-k2+k3)*dt
       X[:]+=(k1+3*k2+3*k3+k4)/8

    if scheme=='RK4':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X     )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/2)*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k2/2)*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k3  )*dt
       X[:]+=(k1+2*k2+2*k3+k4)/6

    if scheme=='RK5':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                                    )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/3                               )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*4/25+k2*6/25                    )*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/4   -k2*3    +k3*15/4           )*dt
       k5=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*2/27+k2*10/9 -k3*50/81 +k4*8/81 )*dt
       k6=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*2/25+k2*12/25+k3*2/15  +k4*8/75 )*dt
       X[:]+=(k1*23+125*k3-81*k5+125*k6)/192

    if scheme=='RKF':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                                                                )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X +k1*1/4                                                        )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X +k1*3/32     +k2*9/32                                          )*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X +k1*1932/2197-k2*7200/2197+k3*7296/2197                        )*dt
       k5=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X +k1*439/216  -k2*8        +k3*3680/513  -k4*845/4104           )*dt
       k6=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X -k1*8/27     +k2*2        -k3*3544/2565 +k4*1859/4104 -k5*11/40)*dt
       X[:]+=(16/135*k1 +6656/12825*k3 +28561/56430*k4 -9/50*k5 +2/55*k6)


    u[:]=X[0:NP]
    v[:]=X[NP:2*NP]

    t+=dt

    ###########################################################################
    if istep%every_ascii==0 or istep==nstep: # do stats on u,v

       min_u=np.min(u) ; max_u=np.max(u) ; avrg_u=np.average(u)
       min_v=np.min(v) ; max_v=np.max(v) ; avrg_v=np.average(v)

       print("-----------------------------")
       print("istep= ", istep,'| t=',t)
       print("     -> u (m,M) %f %f " %(min_u,max_u))
       print("     -> v (m,M) %f %f " %(min_v,max_v))
       print("     update solution: %.3f s" % (clock.time()-start))

       stats_u_file.write("%e %e %e %e\n" % (t,min_u,max_u,avrg_u)) ; stats_u_file.flush()
       stats_v_file.write("%e %e %e %e\n" % (t,min_v,max_v,avrg_v)) ; stats_v_file.flush()

       u_threshold=np.zeros(NP,dtype=np.int8)
       v_threshold=np.zeros(NP,dtype=np.int8)
       for i in range(0,NP):
           if u[i]>avrg_u: u_threshold[i]=1
           if v[i]>avrg_v: v_threshold[i]=1

       filename=model+'_solution_{:07d}'.format(istep)
       if istep==nstep: filename=model+'_solution_final'

    ###########################################################################
    if istep%every_vtu==0 or istep==nstep: # export solution to vtu format

       start=clock.time()
       vtufile=open(filename+'.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%.4e %.4e %.4e \n" %(x[i],y[i],z[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<PointData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%.3e \n" %(u[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='v' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%.3e \n" %(v[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int8' Name='u (threshold)' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%d " %(u_threshold[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int8' Name='v (threshold)' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%d " %(v_threshold[i]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if three_dimensions:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],\
                                                          icon[3,iel],icon[4,iel],icon[5,iel],\
                                                          icon[6,iel],icon[7,iel]))
       else:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))

       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d " %((iel+1)*m))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d " % tyype)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()
       print("     export to vtu: %.3f s" % (clock.time()-start))

    ###########################################################################
    if istep%every_png==0 or istep==nstep:
       if not three_dimensions:
          start=clock.time()
          plt.imshow(np.reshape(u,(nnz,nnx)), interpolation='none')
          plt.savefig(filename+'_u.png', bbox_inches='tight')
          plt.imshow(np.reshape(v,(nnz,nnx)), interpolation='none')
          plt.savefig(filename+'_v.png', bbox_inches='tight')
          print("     export to png: %.3f s" % (clock.time()-start))


