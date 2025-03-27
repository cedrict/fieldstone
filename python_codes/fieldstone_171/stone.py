import numpy as np
import sys as sys
import random
import time as clock 
import numba

###############################################################################
###############################################################################
###############################################################################

if int(len(sys.argv))==3:
   model=str(sys.argv[1])
   nnx = int(sys.argv[2])
else:
   model='sigma'
   nnx = 257

use_2d_seeds=True

Lx=2.5
Ly=1.
Lz=2.5

nny=int(nnx*Ly/Lx)
nnz=int(nnx*Lz/Lx)

if use_2d_seeds:
   nny=4
   Ly=0.01

dt=1e-1

nstep=200000

every=500

###########################################################

#default (Lukas)
if model=='default':
   Du=0.000004
   Dv=0.000002
   Feed=0.035
   Kill=0.0575
if model=='alpha':
   Du=2.e-5 ; Dv=1e-5
   #Kill=0.050600858369098715 ; Feed=0.016904176904176903
   Kill=0.050 ; Feed=0.010 # gane22 OK
if model=='beta':
   Du=2.e-5 ; Dv=1e-5
   Kill=0.040529327610872676 ; Feed= 0.014938574938574938
if model=='gamma':
   Du=2.e-5 ; Dv=1e-5
   #Kill=0.05540772532188842  ; Feed= 0.02407862407862408
   Kill=0.054 ; Feed= 0.025 # gane22 OK
if model=='delta':
   Du=2.e-5 ; Dv=1e-5
   #Kill=0.055464949928469245 ; Feed= 0.029484029484029485
   Kill=0.052 ; Feed= 0.025 # gane22 OK
if model=='epsilon':
   Du=2.e-5 ; Dv=1e-5
   Kill=0.05540772532188842 ; Feed= 0.01926289926289926
if model=='zeta':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.060386266094420604 ; Feed= 0.024373464373464375 
if model=='eta':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.06296137339055795 ; Feed= 0.0343980343980344 
if model=='theta':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.05992846924177397 ; Feed= 0.03960687960687961 
if model=='iota':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.06010014306151645 ; Feed= 0.049238329238329236 
if model=='kappa':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.06290414878397711 ; Feed= 0.04687960687960688 
if model=='lambda':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.0653648068669528 ; Feed= 0.036855036855036855
if model=='mu':
   Du=2.e-5 ; Dv=1e-5 ; 
   #Kill=0.06525035765379114 ; Feed= 0.04914004914004914
   Kill=0.064 ; Feed= 0.050 # gane22 
   #Kill=0.065 ; Feed= 0.046 # muna  OK

if model=='sigma':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.056 ; Feed=0.095 # gane22  DOES NOT WORK

if model=='xi':
   Du=2.e-5 ; Dv=1e-5 ; 
   Kill=0.042 ; Feed=0.01 # gane22  DOES NOT WORK

###########################################################

hx=Lx/(nnx-1)
hy=Ly/(nny-1)
hz=Lz/(nnz-1)

nelx=nnx-1
nely=nny-1
nelz=nnz-1
nel=nelx*nely*nelz
NP=nnx*nny*nnz

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
print('nstep=',nstep)
print('dt=',dt)

print('diff dt:',hx**2/Du,hx**2/Dv)

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

nseed=100
seed_size=0.02
       
for i in range(0,NP):
    u[i]=random.uniform(0.9,1) # close to 1
    v[i]=random.uniform(0,0.1) # close to 0 

if use_2d_seeds:
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
              v[i]=random.uniform(0,0.5)

else:

   for iseed in range(nseed):
       xs=random.uniform(0+seed_size,Lx-seed_size)
       ys=random.uniform(0+seed_size,Ly-seed_size)
       zs=random.uniform(0+seed_size,Lz-seed_size)
       for i in range(0,NP):
           if abs(x[i]-xs)<seed_size and\
              abs(y[i]-ys)<seed_size and\
              abs(z[i]-zs)<seed_size :
              u[i]=random.uniform(0.5,1)
       xs=random.uniform(0+seed_size,Lx-seed_size)
       ys=random.uniform(0+seed_size,Ly-seed_size)
       zs=random.uniform(0+seed_size,Lz-seed_size)
       for i in range(0,NP):
           if abs(x[i]-xs)<seed_size and\
              abs(y[i]-ys)<seed_size and\
              abs(z[i]-zs)<seed_size :
              v[i]=random.uniform(0,0.25)

X=np.zeros(2*NP,dtype=np.float64)
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
    X[:]+=F(Du,Dv,Feed,Kill,NP,hx,hy,hz,u,v)*dt
    u[:]=X[0:NP]
    v[:]=X[NP:2*NP]
    t+=dt

    if istep%every==0 or istep==nstep:

       min_u=np.min(u)
       max_u=np.max(u)
       min_v=np.min(v)
       max_v=np.max(v)
       avrg_u=np.average(u)
       avrg_v=np.average(v)

       print("-----------------------------")
       print("istep= ", istep,'| t=',t)
       print("     -> u (m,M) %e %e " %(min_u,max_u))
       print("     -> v (m,M) %e %e " %(min_v,max_v))
       print("     update solution: %.3f s" % (clock.time()-start))

       stats_u_file.write("%e %e %e %e\n" % (t,min_u,max_u,avrg_u)) ; stats_u_file.flush()
       stats_v_file.write("%e %e %e %e\n" % (t,min_v,max_v,avrg_v)) ; stats_v_file.flush()

       u_threshold=np.zeros(NP,dtype=np.int8)
       v_threshold=np.zeros(NP,dtype=np.int8)
       for i in range(0,NP):
           if u[i]>avrg_u: u_threshold[i]=1
           if v[i]>avrg_v: v_threshold[i]=1

       ########################################################################
       # export solution to vtu format
       ########################################################################
       start=clock.time()

       filename = model+'_solution_{:05d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%.3e %.3e %.3e \n" %(x[i],y[i],z[i]))
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
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel],
                                                       icon[4,iel],icon[5,iel],icon[6,iel],icon[7,iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d " %((iel+1)*8))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d " %12)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

       print("     export to vtu: %.3f s" % (clock.time()-start))

       if abs(min_u-max_u)<1e-3 and abs(min_v-max_v)<1e-3:
          print('avrg_u=',avrg_u) 
          print('avrg_v=',avrg_v) 
          exit('stopping iterations') 

