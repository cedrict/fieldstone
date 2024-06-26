import numpy as np
import sys
import time
import random

###############################################################################

def NNV(rq,sq):
    N_0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N_1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N_2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N_3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N_4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N_5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N_6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N_7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N_8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N_0,N_1,N_2,N_3,N_4,N_5,N_6,N_7,N_8],dtype=np.float64)

def velocity(x,y):
    return (y-2),-(x-2)

###############################################################################

print('**********************')
print('***** stone 122 ******')
print('**********************')


if int(len(sys.argv) == 5):
   nelx   = int(sys.argv[1])
   nstep  = int(sys.argv[2])
   method = sys.argv[3]
   visu   = int(sys.argv[4])
else:
   nelx   = 16
   nstep  = 32 
   method = 'RK1'
   visu   = 1

Lx=4
Ly=4

nely=nelx
nnx=2*nelx+1  # number of elements, x direction
nny=2*nely+1  # number of elements, y direction
nel=nelx*nely # number of elements
NV=nnx*nny    # number of nodes

hx=Lx/nelx
hy=Ly/nely

mV=9

experiment=1

interpolate=False

print(nelx,nely,nel,nnx,nny,NV,visu)

#rVnodes=[-1,1,1,-1,0,1,0,-1,0]
#sVnodes=[-1,-1,1,1,-1,0,1,0,0]
#for i in range(0,mV):
#    print(NNV(rVnodes[i],sVnodes[i]))

###############################################################################
#Runge-Kutta-Fehlberg coefficients
###############################################################################

rkf_c2=1./4.      
rkf_c3=3./8.    
rkf_c4=12./13.   
rkf_c5=1.    
rkf_c6=1./2.    

rkf_a21=1./4.    

rkf_a31=3./32.   
rkf_a32=9./32.                                       

rkf_a41= 1932./2197.                                  
rkf_a42=-7200./2197.         
rkf_a43= 7296./2197.        

rkf_a51= 439./216.         
rkf_a52=  -8.             
rkf_a53=3680./513.       
rkf_a54=-845./4104.     

rkf_a61=   -8./27.     
rkf_a62=    2.        
rkf_a63=-3544./2565. 
rkf_a64= 1859./4104.
rkf_a65=  -11./40.  

rkf_b1=16./135.    
rkf_b3= 6656./12825.  
rkf_b4=28561./56430. 
rkf_b5=   -9./50.   
rkf_b6=    2./55.  

###############################################################################
# ODE87 coefficients
###############################################################################
   
AA=np.array(\
  [[1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
  [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
  [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
  [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
  [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0], \
  [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],\
  [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229,\
  -180193667/1043307555, 0, 0, 0, 0, 0, 0],\
  [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059,\
  790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],\
  [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935,\
  6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],\
  [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382,\
  -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],\
  [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527,\
  5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],\
  [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556,\
  -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0] ],dtype=np.float64)

###############################################################################
# grid point setup
###############################################################################
start = time.time()

x=np.zeros(NV,dtype=np.float64)  # x coordinates
y=np.zeros(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0,nny):
    for i in range(0,nnx):
        x[counter]=i*hx/2.
        y[counter]=j*hy/2.
        counter += 1
    #end for
#end for

#np.savetxt('grid.ascii',np.array([x,y]).T,header='# x,y')

###############################################################################
# connectivity
###############################################################################
# velocity  
# 3---6---2 
# |       | 
# 7   8   5 
# |       | 
# 0---4---1 
###############################################################################
start = time.time()

iconV=np.zeros((mV,nel),dtype=np.int32)

#nelx=3 ; nnx=7
#nely=2 ; nny=5

counter = 0
for j in range(0,nely):
    for i in range(0,nelx):
        iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
        iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
        iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
        iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
        iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
        iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
        iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
        iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
        iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
        counter += 1
    #end for
#end for

###############################################################################
# generate 100 particles
###############################################################################

nmarker=100

swarm_x=np.zeros(nmarker,dtype=np.float64)  
swarm_y=np.zeros(nmarker,dtype=np.float64) 
swarm_r=np.zeros(nmarker,dtype=np.float64) 
swarm_s=np.zeros(nmarker,dtype=np.float64) 
swarm_rad=np.zeros(nmarker,dtype=np.float64) 
swarm_x0=np.zeros(nmarker,dtype=np.float64)  
swarm_y0=np.zeros(nmarker,dtype=np.float64) 
swarm_iel=np.zeros(nmarker,dtype=np.int32) 
swarm_u=np.zeros(nmarker,dtype=np.float64)  
swarm_v=np.zeros(nmarker,dtype=np.float64) 

if experiment==1:

   theta=np.linspace(0,2*np.pi,num=nmarker,endpoint=False)
   for im in range(0,nmarker):
       #swarm_x[im]=2+np.cos(theta[im])#+np.pi/123)
       #swarm_y[im]=2+np.sin(theta[im])#+np.pi/123)
       swarm_x[im]=2+np.cos(theta[im]+np.pi/123)
       swarm_y[im]=2+np.sin(theta[im]+np.pi/123)

   swarm_rad[:]=1

if experiment==2:

   x1=0.5-0.25
   x2=0.5+0.25
   x3=0.5+0.25
   x4=0.5-0.25

   y1=0.5-0.25
   y2=0.5-0.25
   y3=0.5+0.25
   y4=0.5+0.25

   for im in range(0,nmarker):
       r=random.uniform(-1.,+1)
       s=random.uniform(-1.,+1)
       N1=0.25*(1-r)*(1-s)
       N2=0.25*(1+r)*(1-s)
       N3=0.25*(1+r)*(1+s)
       N4=0.25*(1-r)*(1+s)
       swarm_x[im]=N1*x1+N2*x2+N3*x3+N4*x4
       swarm_y[im]=N1*y1+N2*y2+N3*y3+N4*y4

swarm_x0[:]=swarm_x[:]
swarm_y0[:]=swarm_y[:]

#np.savetxt('swarm.ascii',np.array([swarm_x,swarm_y]).T,header='# x,y')

###############################################################################
# compute time step dt 
###############################################################################

vel_max=1
dt=(2*np.pi/vel_max)/nstep
CFL=dt*vel_max/hx
tfinal=nstep*dt

print('vel_max=',vel_max)
print('dt=',dt)
print('CFL=',CFL)
print('nstep=',nstep)

###############################################################################
# prescribe velocity field 
###############################################################################

u=np.zeros(NV,dtype=np.float64)  # x coordinates
v=np.zeros(NV,dtype=np.float64)  # y coordinates

for i in range(0,NV):
    u[i],v[i]=velocity(x[i],y[i])

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

#########################

#vmax=max(np.sqrt(u**2+v**2))
#print(dt*vel_max/hx)
#exit()

###############################################################################
# advection
###############################################################################

if method=='RK1':
   markerRK1file=open('marker_RK1_'+str(nstep)+'.ascii',"w")
   markerRK1file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):
       for im in range(nmarker):
           if interpolate:
              ielx=int(swarm_x[im]/hx)
              iely=int(swarm_y[im]/hy)
              if (ielx<0 or ielx>nelx-1): exit()
              if (iely<0 or iely>nely-1): exit()
              iel=iely*nelx+ielx

              if swarm_x[im]<x[iconV[0,iel]] or\
                 swarm_x[im]>x[iconV[2,iel]] or\
                 swarm_y[im]<y[iconV[0,iel]] or\
                 swarm_y[im]>y[iconV[2,iel]]:
                 exit('big pb')


              rm=(swarm_x[im]-x[iconV[0,iel]])/hx-0.5
              sm=(swarm_y[im]-y[iconV[0,iel]])/hy-0.5
              if (rm>1 or rm<-1): exit()
              if (sm>1 or sm<-1): exit()
              swarm_r[im]=rm
              swarm_s[im]=sm
              swarm_iel[im]=iel
              NNNV=NNV(rm,sm)
              um=NNNV.dot(u[iconV[:,iel]])
              vm=NNNV.dot(v[iconV[:,iel]])
           else:
              um,vm=velocity(swarm_x[im],swarm_y[im])
           swarm_u[im]=um
           swarm_v[im]=vm
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK1file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

if method=='RK2':
   markerRK2file=open('marker_RK2_'+str(nstep)+'.ascii',"w")
   markerRK2file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              if xA<x[iconV[0,iel]] or\
                 xA>x[iconV[2,iel]] or\
                 yA<y[iconV[0,iel]] or\
                 yA>y[iconV[2,iel]]:
                 exit('big pb')
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              if (rm>1 or rm<-1): exit()
              if (sm>1 or sm<-1): exit()
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              if xB<x[iconV[0,iel]] or\
                 xB>x[iconV[2,iel]] or\
                 yB<y[iconV[0,iel]] or\
                 yB>y[iconV[2,iel]]:
                 exit('big pb')
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              if (rm>1 or rm<-1): exit()
              if (sm>1 or sm<-1): exit()
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           swarm_x[im]=xA+uB*dt
           swarm_y[im]=yA+vB*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK2file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

if method=='RK3':
   markerRK3file=open('marker_RK3_'+str(nstep)+'.ascii',"w")
   markerRK3file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(-uA+2*uB)*dt 
           yC=yA+(-vA+2*vB)*dt 
           if interpolate:
              ielx=int(xC/hx)
              iely=int(yC/hy)
              iel=iely*nelx+ielx
              rm=(xC-x[iconV[0,iel]])/hx-0.5
              sm=(yC-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uC=NNNV.dot(u[iconV[:,iel]])
              vC=NNNV.dot(v[iconV[:,iel]])
           else:
              uC,vC=velocity(xC,yC)
           #--------------
           swarm_x[im]=xA+(uA+4*uB+uC)*dt/6.
           swarm_y[im]=yA+(vA+4*vB+vC)*dt/6.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK3file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

if method=='RK4':
   markerRK4file=open('marker_RK4_'+str(nstep)+'.ascii',"w")
   markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):
       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+uB*dt/2.
           yC=yA+vB*dt/2.
           if interpolate:
              ielx=int(xC/hx)
              iely=int(yC/hy)
              iel=iely*nelx+ielx
              rm=(xC-x[iconV[0,iel]])/hx-0.5
              sm=(yC-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uC=NNNV.dot(u[iconV[:,iel]])
              vC=NNNV.dot(v[iconV[:,iel]])
           else:
              uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+uC*dt
           yD=yA+vC*dt
           if interpolate:
              ielx=int(xD/hx)
              iely=int(yD/hy)
              iel=iely*nelx+ielx
              rm=(xD-x[iconV[0,iel]])/hx-0.5
              sm=(yD-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uD=NNNV.dot(u[iconV[:,iel]])
              vD=NNNV.dot(v[iconV[:,iel]])
           else:
              uD,vD=velocity(xD,yD)
           #--------------
           swarm_x[im]=xA+(uA+2*uB+2*uC+uD)*dt/6.
           swarm_y[im]=yA+(vA+2*vB+2*vC+vD)*dt/6.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

if method=='RK4_38':
   markerRK4file=open('marker_RK4_38_'+str(nstep)+'.ascii',"w")
   markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):
       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/3.
           yB=yA+vA*dt/3.
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(-uA/3+uB)*dt
           yC=yA+(-vA/3+vB)*dt
           if interpolate:
              ielx=int(xC/hx)
              iely=int(yC/hy)
              iel=iely*nelx+ielx
              rm=(xC-x[iconV[0,iel]])/hx-0.5
              sm=(yC-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uC=NNNV.dot(u[iconV[:,iel]])
              vC=NNNV.dot(v[iconV[:,iel]])
           else:
              uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+(uA-uB+uC)*dt
           yD=yA+(vA-vB+vC)*dt
           if interpolate:
              ielx=int(xD/hx)
              iely=int(yD/hy)
              iel=iely*nelx+ielx
              rm=(xD-x[iconV[0,iel]])/hx-0.5
              sm=(yD-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uD=NNNV.dot(u[iconV[:,iel]])
              vD=NNNV.dot(v[iconV[:,iel]])
           else:
              uD,vD=velocity(xD,yD)
           #--------------
           swarm_x[im]=xA+(uA+3*uB+3*uC+uD)*dt/8.
           swarm_y[im]=yA+(vA+3*vB+3*vC+vD)*dt/8.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

if method=='RKF':# Runge-Kutta Fehlberg method
   markerRK5file=open('marker_RKF_'+str(nstep)+'.ascii',"w")
   markerRK5file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+(uA*rkf_a21)*dt
           yB=yA+(vA*rkf_a21)*dt
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(uA*rkf_a31+uB*rkf_a32)*dt
           yC=yA+(vA*rkf_a31+vB*rkf_a32)*dt
           if interpolate:
              ielx=int(xC/hx)
              iely=int(yC/hy)
              iel=iely*nelx+ielx
              rm=(xC-x[iconV[0,iel]])/hx-0.5
              sm=(yC-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uC=NNNV.dot(u[iconV[:,iel]])
              vC=NNNV.dot(v[iconV[:,iel]])
           else:
              uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+(uA*rkf_a41+uB*rkf_a42+uC*rkf_a43)*dt
           yD=yA+(vA*rkf_a41+vB*rkf_a42+vC*rkf_a43)*dt
           if interpolate:
              ielx=int(xD/hx)
              iely=int(yD/hy)
              iel=iely*nelx+ielx
              rm=(xD-x[iconV[0,iel]])/hx-0.5
              sm=(yD-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uD=NNNV.dot(u[iconV[:,iel]])
              vD=NNNV.dot(v[iconV[:,iel]])
           else:
              uD,vD=velocity(xD,yD)
           #--------------
           xE=xA+(uA*rkf_a51+uB*rkf_a52+uC*rkf_a53+uD*rkf_a54)*dt
           yE=yA+(vA*rkf_a51+vB*rkf_a52+vC*rkf_a53+vD*rkf_a54)*dt
           if interpolate:
              ielx=int(xE/hx)
              iely=int(yE/hy)
              iel=iely*nelx+ielx
              rm=(xE-x[iconV[0,iel]])/hx-0.5
              sm=(yE-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uE=NNNV.dot(u[iconV[:,iel]])
              vE=NNNV.dot(v[iconV[:,iel]])
           else:
              uE,vE=velocity(xE,yE)
           #--------------
           xF=xA+(uA*rkf_a61+uB*rkf_a62+uC*rkf_a63+uD*rkf_a64+uE*rkf_a65)*dt
           yF=yA+(vA*rkf_a61+vB*rkf_a62+vC*rkf_a63+vD*rkf_a64+vE*rkf_a65)*dt
           if interpolate:
              ielx=int(xF/hx)
              iely=int(yF/hy)
              iel=iely*nelx+ielx
              rm=(xF-x[iconV[0,iel]])/hx-0.5
              sm=(yF-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uF=NNNV.dot(u[iconV[:,iel]])
              vF=NNNV.dot(v[iconV[:,iel]])
           else:
              uF,vF=velocity(xF,yF)
           #--------------
           swarm_x[im]=xA+(uA*rkf_b1+uC*rkf_b3+uD*rkf_b4+uE*rkf_b5+uF*rkf_b6)*dt
           swarm_y[im]=yA+(vA*rkf_b1+vC*rkf_b3+vD*rkf_b4+vE*rkf_b5+vF*rkf_b6)*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK5file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))


if method=='ODE87':

   markerODE87file=open('marker_ODE87_'+str(nstep)+'.ascii',"w")
   markerODE87file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,0))
   for istep in range(0,nstep):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           if interpolate:
              ielx=int(xA/hx)
              iely=int(yA/hy)
              iel=iely*nelx+ielx
              rm=(xA-x[iconV[0,iel]])/hx-0.5
              sm=(yA-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uA=NNNV.dot(u[iconV[:,iel]])
              vA=NNNV.dot(v[iconV[:,iel]])
           else:
              uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+(uA*AA[0,0])*dt
           yB=yA+(vA*AA[0,0])*dt
           if interpolate:
              ielx=int(xB/hx)
              iely=int(yB/hy)
              iel=iely*nelx+ielx
              rm=(xB-x[iconV[0,iel]])/hx-0.5
              sm=(yB-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uB=NNNV.dot(u[iconV[:,iel]])
              vB=NNNV.dot(v[iconV[:,iel]])
           else:
              uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(uA*AA[1,0]+uB*AA[1,1])*dt
           yC=yA+(vA*AA[1,0]+vB*AA[1,1])*dt
           if interpolate:
              ielx=int(xC/hx)
              iely=int(yC/hy)
              iel=iely*nelx+ielx
              rm=(xC-x[iconV[0,iel]])/hx-0.5
              sm=(yC-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uC=NNNV.dot(u[iconV[:,iel]])
              vC=NNNV.dot(v[iconV[:,iel]])
           else:
              uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+(uA*AA[2,0]+uB*AA[2,1]+AA[2,2]*uC)*dt
           yD=yA+(vA*AA[2,0]+vB*AA[2,1]+AA[2,2]*vC)*dt
           if interpolate:
              ielx=int(xD/hx)
              iely=int(yD/hy)
              iel=iely*nelx+ielx
              rm=(xD-x[iconV[0,iel]])/hx-0.5
              sm=(yD-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uD=NNNV.dot(u[iconV[:,iel]])
              vD=NNNV.dot(v[iconV[:,iel]])
           else:
              uD,vD=velocity(xD,yD)
           #--------------
           xE=xA+(uA*AA[3,0]+uB*AA[3,1]+AA[3,2]*uC+AA[3,3]*uD)*dt
           yE=yA+(vA*AA[3,0]+vB*AA[3,1]+AA[3,2]*vC+AA[3,3]*vD)*dt
           if interpolate:
              ielx=int(xE/hx)
              iely=int(yE/hy)
              iel=iely*nelx+ielx
              rm=(xE-x[iconV[0,iel]])/hx-0.5
              sm=(yE-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uE=NNNV.dot(u[iconV[:,iel]])
              vE=NNNV.dot(v[iconV[:,iel]])
           else:
              uE,vE=velocity(xE,yE)
           #--------------
           xF=xA+(uA*AA[4,0]+uB*AA[4,1]+AA[4,2]*uC+AA[4,3]*uD+AA[4,4]*uE)*dt
           yF=yA+(vA*AA[4,0]+vB*AA[4,1]+AA[4,2]*vC+AA[4,3]*vD+AA[4,4]*vE)*dt
           if interpolate:
              ielx=int(xF/hx)
              iely=int(yF/hy)
              iel=iely*nelx+ielx
              rm=(xF-x[iconV[0,iel]])/hx-0.5
              sm=(yF-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uF=NNNV.dot(u[iconV[:,iel]])
              vF=NNNV.dot(v[iconV[:,iel]])
           else:
              uF,vF=velocity(xF,yF)
           #--------------
           xG=xA+(uA*AA[5,0]+uB*AA[5,1]+AA[5,2]*uC+AA[5,3]*uD+AA[5,4]*uE+AA[5,5]*uF)*dt
           yG=yA+(vA*AA[5,0]+vB*AA[5,1]+AA[5,2]*vC+AA[5,3]*vD+AA[5,4]*vE+AA[5,5]*vF)*dt
           if interpolate:
              ielx=int(xG/hx)
              iely=int(yG/hy)
              iel=iely*nelx+ielx
              rm=(xG-x[iconV[0,iel]])/hx-0.5
              sm=(yG-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uG=NNNV.dot(u[iconV[:,iel]])
              vG=NNNV.dot(v[iconV[:,iel]])
           else:
              uG,vG=velocity(xG,yG)
           #--------------
           xH=xA+(uA*AA[6,0]+uB*AA[6,1]+AA[6,2]*uC+AA[6,3]*uD+AA[6,4]*uE+AA[6,5]*uF+AA[6,6]*uG)*dt
           yH=yA+(vA*AA[6,0]+vB*AA[6,1]+AA[6,2]*vC+AA[6,3]*vD+AA[6,4]*vE+AA[6,5]*vF+AA[6,6]*vG)*dt
           if interpolate:
              ielx=int(xH/hx)
              iely=int(yH/hy)
              iel=iely*nelx+ielx
              rm=(xH-x[iconV[0,iel]])/hx-0.5
              sm=(yH-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uH=NNNV.dot(u[iconV[:,iel]])
              vH=NNNV.dot(v[iconV[:,iel]])
           else:
              uH,vH=velocity(xH,yH)
           #--------------
           xI=xA+(uA*AA[7,0]+uB*AA[7,1]+AA[7,2]*uC+AA[7,3]*uD+AA[7,4]*uE+AA[7,5]*uF+AA[7,6]*uG+AA[7,7]*uH)*dt
           yI=yA+(vA*AA[7,0]+vB*AA[7,1]+AA[7,2]*vC+AA[7,3]*vD+AA[7,4]*vE+AA[7,5]*vF+AA[7,6]*vG+AA[7,7]*vH)*dt
           if interpolate:
              ielx=int(xI/hx)
              iely=int(yI/hy)
              iel=iely*nelx+ielx
              rm=(xI-x[iconV[0,iel]])/hx-0.5
              sm=(yI-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uI=NNNV.dot(u[iconV[:,iel]])
              vI=NNNV.dot(v[iconV[:,iel]])
           else:
              uI,vI=velocity(xI,yI)
           #--------------
           xJ=xA+(uA*AA[8,0]+uB*AA[8,1]+AA[8,2]*uC+AA[8,3]*uD+AA[8,4]*uE+AA[8,5]*uF+AA[8,6]*uG+AA[8,7]*uH+AA[8,8]*uI)*dt
           yJ=yA+(vA*AA[8,0]+vB*AA[8,1]+AA[8,2]*vC+AA[8,3]*vD+AA[8,4]*vE+AA[8,5]*vF+AA[8,6]*vG+AA[8,7]*vH+AA[8,8]*vI)*dt
           if interpolate:
              ielx=int(xJ/hx)
              iely=int(yJ/hy)
              iel=iely*nelx+ielx
              rm=(xJ-x[iconV[0,iel]])/hx-0.5
              sm=(yJ-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uJ=NNNV.dot(u[iconV[:,iel]])
              vJ=NNNV.dot(v[iconV[:,iel]])
           else:
              uJ,vJ=velocity(xJ,yJ)
           #--------------
           xK=xA+(uA*AA[9,0]+uB*AA[9,1]+AA[9,2]*uC+AA[9,3]*uD+AA[9,4]*uE+AA[9,5]*uF+AA[9,6]*uG+AA[9,7]*uH+AA[9,8]*uI+AA[9,9]*uJ)*dt
           yK=yA+(vA*AA[9,0]+vB*AA[9,1]+AA[9,2]*vC+AA[9,3]*vD+AA[9,4]*vE+AA[9,5]*vF+AA[9,6]*vG+AA[9,7]*vH+AA[9,8]*vI+AA[9,9]*vJ)*dt
           if interpolate:
              ielx=int(xK/hx)
              iely=int(yK/hy)
              iel=iely*nelx+ielx
              rm=(xK-x[iconV[0,iel]])/hx-0.5
              sm=(yK-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uK=NNNV.dot(u[iconV[:,iel]])
              vK=NNNV.dot(v[iconV[:,iel]])
           else:
              uK,vK=velocity(xK,yK)
           #--------------
           xL=xA+(uA*AA[10,0]+uB*AA[10,1]+AA[10,2]*uC+AA[10,3]*uD+AA[10,4]*uE+AA[10,5]*uF+AA[10,6]*uG+AA[10,7]*uH+AA[10,8]*uI+AA[10,9]*uJ+AA[10,10]*uK)*dt
           yL=yA+(vA*AA[10,0]+vB*AA[10,1]+AA[10,2]*vC+AA[10,3]*vD+AA[10,4]*vE+AA[10,5]*vF+AA[10,6]*vG+AA[10,7]*vH+AA[10,8]*vI+AA[10,9]*vJ+AA[10,10]*vK)*dt
           if interpolate:
              ielx=int(xL/hx)
              iely=int(yL/hy)
              iel=iely*nelx+ielx
              rm=(xL-x[iconV[0,iel]])/hx-0.5
              sm=(yL-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uL=NNNV.dot(u[iconV[:,iel]])
              vL=NNNV.dot(v[iconV[:,iel]])
           else:
              uL,vL=velocity(xL,yL)
           #--------------
           xM=xA+(uA*AA[11,0]+uB*AA[11,1]+AA[11,2]*uC+AA[11,3]*uD+AA[11,4]*uE+AA[11,5]*uF+AA[11,6]*uG+AA[11,7]*uH+AA[11,8]*uI+AA[11,9]*uJ+AA[11,10]*uK+AA[11,11]*uL)*dt
           yM=yA+(vA*AA[11,0]+vB*AA[11,1]+AA[11,2]*vC+AA[11,3]*vD+AA[11,4]*vE+AA[11,5]*vF+AA[11,6]*vG+AA[11,7]*vH+AA[11,8]*vI+AA[11,9]*vJ+AA[11,10]*vK+AA[11,11]*uL)*dt
           if interpolate:
              ielx=int(xM/hx)
              iely=int(yM/hy)
              iel=iely*nelx+ielx
              rm=(xM-x[iconV[0,iel]])/hx-0.5
              sm=(yM-y[iconV[0,iel]])/hy-0.5
              NNNV=NNV(rm,sm)
              uM=NNNV.dot(u[iconV[:,iel]])
              vM=NNNV.dot(v[iconV[:,iel]])
           else:
              uM,vM=velocity(xM,yM)
           #--------------
           swarm_x[im]=xA+dt*(14005451/335480064*uA\
                             -59238493/1068277825*uF\
                             +181606767/758867731*uG\
                             +561292985/797845732*uH\
                             -1041891430/1371343529*uI\
                             +760417239/1151165299*uJ\
                             +118820643/751138087*uK\
                             -528747749/2220607170*uL\
                             +1/4*uM)
           swarm_y[im]=yA+dt*(14005451/335480064*vA\
                             -59238493/1068277825*vF\
                             +181606767/758867731*vG\
                             +561292985/797845732*vH\
                             -1041891430/1371343529*vI\
                             +760417239/1151165299*vJ\
                             +118820643/751138087*vK\
                             -528747749/2220607170*vL\
                             +1/4*vM)
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerODE87file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep+1))

###########################################################################
# export swarm to vtu
###########################################################################

if visu==1:

       vtufile=open('markers_'+method+'.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       ####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")

       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='r' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_r[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='s' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_s[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='1' Name='iel' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e \n" % swarm_iel[im])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'>\n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_u[im],swarm_v[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")

       ####
       vtufile.write("<Cells>\n")
       #
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%d " % i)
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for i in range(0,nmarker):
           vtufile.write("%d " % (i+1))
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for i in range(0,nmarker):
           vtufile.write("%d " % 1)
       vtufile.write("</DataArray>\n")
       #
       vtufile.write("</Cells>\n")
       ####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

###############################################################################

if experiment==1:
   #print('CFL,error',CFL,np.max(np.sqrt((swarm_x-swarm_x0)**2+(swarm_y-swarm_y0)**2)),nstep)
   print('CFL,error',CFL,np.max(abs(swarm_rad-1)),np.max(np.sqrt( (swarm_x-swarm_x0)**2+(swarm_y-swarm_y0)**2) ),nstep)

###############################################################################

if visu==1:
   filename = 'solution.vtu'
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
   vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity' Format='ascii'> \n")
   for i in range(0,NV):
       vtufile.write("%10e %10e %10e \n" %(u[i],v[i],0.))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</PointData>\n")
   #####
   vtufile.write("<Cells>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                      iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                      iconV[6,iel],iconV[7,iel],iconV[8,iel]))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %((iel+1)*9))
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
   for iel in range (0,nel):
       vtufile.write("%d \n" %28)
   vtufile.write("</DataArray>\n")
   #--
   vtufile.write("</Cells>\n")
   #####
   vtufile.write("</Piece>\n")
   vtufile.write("</UnstructuredGrid>\n")
   vtufile.write("</VTKFile>\n")
   vtufile.close()

###############################################################################

