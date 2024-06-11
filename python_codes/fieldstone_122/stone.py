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
    return y-2,-(x-2) 

###############################################################################

if int(len(sys.argv) == 5):
   nelx   = int(sys.argv[1])
   CFL    = float(sys.argv[2])
   method = sys.argv[3]
   visu   = int(sys.argv[4])
else:
   nelx   = 32
   CFL    = 0.1
   method = 'RK4_38'
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

print('-----------------------------')
print(nelx,nely,nel,nnx,nny,NV,visu)

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
swarm_iel=np.zeros(nmarker,dtype=np.int16) 

if experiment==1:

   theta=random.uniform(-np.pi,+np.pi)
   theta=np.linspace(0,2*np.pi,num=nmarker)
   for im in range(0,nmarker):
       swarm_x[im]=2+np.cos(theta[im])
       swarm_y[im]=2+np.sin(theta[im])

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
# prescribe velocity field 
###############################################################################

u=np.zeros(NV,dtype=np.float64)  # x coordinates
v=np.zeros(NV,dtype=np.float64)  # y coordinates

for i in range(0,NV):
    u[i],v[i]=velocity(x[i],y[i])

#np.savetxt('velocity.ascii',np.array([x,y,u,v]).T,header='# x,y,u,v')

###############################################################################
# compute time step dt 
###############################################################################

vel_max=max(np.sqrt(u**2+v**2))

dt=CFL*hx/vel_max

tfinal=2*np.pi*1

nstep=int(tfinal/dt)

print('vel_max=',vel_max)
print('dt=',dt)
print('CFL=',CFL)

###############################################################################
# advection
###############################################################################

if method=='RK1':
   markerRK1file=open('marker_RK1_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):
       for im in range(nmarker):
           ielx=int(swarm_x[im]/hx)
           iely=int(swarm_y[im]/hy)
           iel=iely*nelx+ielx
           rm=(swarm_x[im]-x[iconV[0,iel]])/hx-0.5
           sm=(swarm_y[im]-y[iconV[0,iel]])/hy-0.5
           swarm_r[im]=rm
           swarm_s[im]=sm
           #swarm_iel[im]=iel
           #NNNV=NNV(rm,sm)
           #um=NNNV.dot(u[iconV[:,iel]])
           #vm=NNNV.dot(v[iconV[:,iel]])
           um,vm=velocity(swarm_x[im],swarm_y[im])
           swarm_x[im]+=um*dt
           swarm_y[im]+=vm*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK1file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

if method=='RK2':
   markerRK2file=open('marker_RK2_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           #ielx=int(xA/hx)
           #iely=int(yA/hy)
           #iel=iely*nelx+ielx
           #rm=(xA-x[iconV[0,iel]])/hx-0.5
           #sm=(yA-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uA=NNNV.dot(u[iconV[:,iel]])
           #vA=NNNV.dot(v[iconV[:,iel]])
           uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           #ielx=int(xB/hx)
           #iely=int(yB/hy)
           #iel=iely*nelx+ielx
           #rm=(xB-x[iconV[0,iel]])/hx-0.5
           #sm=(yB-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uB=NNNV.dot(u[iconV[:,iel]])
           #vB=NNNV.dot(v[iconV[:,iel]])
           uB,vB=velocity(xB,yB)
           #--------------
           swarm_x[im]=xA+uB*dt
           swarm_y[im]=yA+vB*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK2file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

if method=='RK3':
   markerRK3file=open('marker_RK3_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           #ielx=int(xA/hx)
           #iely=int(yA/hy)
           #iel=iely*nelx+ielx
           #rm=(xA-x[iconV[0,iel]])/hx-0.5
           #sm=(yA-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uA=NNNV.dot(u[iconV[:,iel]])
           #vA=NNNV.dot(v[iconV[:,iel]])
           uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           #ielx=int(xB/hx)
           #iely=int(yB/hy)
           #iel=iely*nelx+ielx
           #rm=(xB-x[iconV[0,iel]])/hx-0.5
           #sm=(yB-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uB=NNNV.dot(u[iconV[:,iel]])
           #vB=NNNV.dot(v[iconV[:,iel]])
           uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(-uA+2*uB)*dt 
           yC=yA+(-vA+2*vB)*dt 
           #ielx=int(xC/hx)
           #iely=int(yC/hy)
           #iel=iely*nelx+ielx
           #rm=(xC-x[iconV[0,iel]])/hx-0.5
           #sm=(yC-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uC=NNNV.dot(u[iconV[:,iel]])
           #vC=NNNV.dot(v[iconV[:,iel]])
           uC,vC=velocity(xC,yC)
           #--------------
           swarm_x[im]=xA+(uA+4*uB+uC)*dt/6.
           swarm_y[im]=yA+(vA+4*vB+vC)*dt/6.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK3file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

if method=='RK4':
   markerRK4file=open('marker_RK4_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):
       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           #ielx=int(xA/hx)
           #iely=int(yA/hy)
           #iel=iely*nelx+ielx
           #rm=(xA-x[iconV[0,iel]])/hx-0.5
           #sm=(yA-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uA=NNNV.dot(u[iconV[:,iel]])
           #vA=NNNV.dot(v[iconV[:,iel]])
           uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/2.
           yB=yA+vA*dt/2.
           #ielx=int(xB/hx)
           #iely=int(yB/hy)
           #iel=iely*nelx+ielx
           #rm=(xB-x[iconV[0,iel]])/hx-0.5
           #sm=(yB-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uB=NNNV.dot(u[iconV[:,iel]])
           #vB=NNNV.dot(v[iconV[:,iel]])
           uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+uB*dt/2.
           yC=yA+vB*dt/2.
           #ielx=int(xC/hx)
           #iely=int(yC/hy)
           #iel=iely*nelx+ielx
           #rm=(xC-x[iconV[0,iel]])/hx-0.5
           #sm=(yC-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uC=NNNV.dot(u[iconV[:,iel]])
           #vC=NNNV.dot(v[iconV[:,iel]])
           uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+uC*dt
           yD=yA+vC*dt
           #ielx=int(xD/hx)
           #iely=int(yD/hy)
           #iel=iely*nelx+ielx
           #rm=(xD-x[iconV[0,iel]])/hx-0.5
           #sm=(yD-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uD=NNNV.dot(u[iconV[:,iel]])
           #vD=NNNV.dot(v[iconV[:,iel]])
           uD,vD=velocity(xD,yD)
           #--------------
           swarm_x[im]=xA+(uA+2*uB+2*uC+uD)*dt/6.
           swarm_y[im]=yA+(vA+2*vB+2*vC+vD)*dt/6.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

if method=='RK4_38':
   markerRK4file=open('marker_RK4_38_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):
       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           #ielx=int(xA/hx)
           #iely=int(yA/hy)
           #iel=iely*nelx+ielx
           #rm=(xA-x[iconV[0,iel]])/hx-0.5
           #sm=(yA-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uA=NNNV.dot(u[iconV[:,iel]])
           #vA=NNNV.dot(v[iconV[:,iel]])
           uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+uA*dt/3.
           yB=yA+vA*dt/3.
           #ielx=int(xB/hx)
           #iely=int(yB/hy)
           #iel=iely*nelx+ielx
           #rm=(xB-x[iconV[0,iel]])/hx-0.5
           #sm=(yB-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uB=NNNV.dot(u[iconV[:,iel]])
           #vB=NNNV.dot(v[iconV[:,iel]])
           uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(-uA/3+uB)*dt
           yC=yA+(-vA/3+vB)*dt
           #ielx=int(xC/hx)
           #iely=int(yC/hy)
           #iel=iely*nelx+ielx
           #rm=(xC-x[iconV[0,iel]])/hx-0.5
           #sm=(yC-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uC=NNNV.dot(u[iconV[:,iel]])
           #vC=NNNV.dot(v[iconV[:,iel]])
           uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+(uA-uB+uC)*dt
           yD=yA+(vA-vB+vC)*dt
           #ielx=int(xD/hx)
           #iely=int(yD/hy)
           #iel=iely*nelx+ielx
           #rm=(xD-x[iconV[0,iel]])/hx-0.5
           #sm=(yD-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uD=NNNV.dot(u[iconV[:,iel]])
           #vD=NNNV.dot(v[iconV[:,iel]])
           uD,vD=velocity(xD,yD)
           #--------------
           swarm_x[im]=xA+(uA+3*uB+3*uC+uD)*dt/8.
           swarm_y[im]=yA+(vA+3*vB+3*vC+vD)*dt/8.
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK4file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

if method=='RKF':# Runge-Kutta Fehlberg method
   markerRK5file=open('marker_RKF_'+str(CFL)+'.ascii',"w")
   for istep in range(0,nstep+1):

       for im in range(0,nmarker):
           #--------------
           xA=swarm_x[im]
           yA=swarm_y[im]
           #ielx=int(xA/hx)
           #iely=int(yA/hy)
           #iel=iely*nelx+ielx
           #rm=(xA-x[iconV[0,iel]])/hx-0.5
           #sm=(yA-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uA=NNNV.dot(u[iconV[:,iel]])
           #vA=NNNV.dot(v[iconV[:,iel]])
           uA,vA=velocity(xA,yA)
           #--------------
           xB=xA+(uA*rkf_a21)*dt
           yB=yA+(vA*rkf_a21)*dt
           #ielx=int(xB/hx)
           #iely=int(yB/hy)
           #iel=iely*nelx+ielx
           #rm=(xB-x[iconV[0,iel]])/hx-0.5
           #sm=(yB-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uB=NNNV.dot(u[iconV[:,iel]])
           #vB=NNNV.dot(v[iconV[:,iel]])
           uB,vB=velocity(xB,yB)
           #--------------
           xC=xA+(uA*rkf_a31+uB*rkf_a32)*dt
           yC=yA+(vA*rkf_a31+vB*rkf_a32)*dt
           #ielx=int(xC/hx)
           #iely=int(yC/hy)
           #iel=iely*nelx+ielx
           #rm=(xC-x[iconV[0,iel]])/hx-0.5
           #sm=(yC-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uC=NNNV.dot(u[iconV[:,iel]])
           #vC=NNNV.dot(v[iconV[:,iel]])
           uC,vC=velocity(xC,yC)
           #--------------
           xD=xA+(uA*rkf_a41+uB*rkf_a42+uC*rkf_a43)*dt
           yD=yA+(vA*rkf_a41+vB*rkf_a42+vC*rkf_a43)*dt
           #ielx=int(xD/hx)
           #iely=int(yD/hy)
           #iel=iely*nelx+ielx
           #rm=(xD-x[iconV[0,iel]])/hx-0.5
           #sm=(yD-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uD=NNNV.dot(u[iconV[:,iel]])
           #vD=NNNV.dot(v[iconV[:,iel]])
           uD,vD=velocity(xD,yD)
           #--------------
           xE=xA+(uA*rkf_a51+uB*rkf_a52+uC*rkf_a53+uD*rkf_a54)*dt
           yE=yA+(vA*rkf_a51+vB*rkf_a52+vC*rkf_a53+vD*rkf_a54)*dt
           #ielx=int(xE/hx)
           #iely=int(yE/hy)
           #iel=iely*nelx+ielx
           #rm=(xE-x[iconV[0,iel]])/hx-0.5
           #sm=(yE-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uE=NNNV.dot(u[iconV[:,iel]])
           #vE=NNNV.dot(v[iconV[:,iel]])
           uE,vE=velocity(xE,yE)
           #--------------
           xF=xA+(uA*rkf_a61+uB*rkf_a62+uC*rkf_a63+uD*rkf_a64+uE*rkf_a65)*dt
           yF=yA+(vA*rkf_a61+vB*rkf_a62+vC*rkf_a63+vD*rkf_a64+vE*rkf_a65)*dt
           #ielx=int(xF/hx)
           #iely=int(yF/hy)
           #iel=iely*nelx+ielx
           #rm=(xF-x[iconV[0,iel]])/hx-0.5
           #sm=(yF-y[iconV[0,iel]])/hy-0.5
           #NNNV=NNV(rm,sm)
           #uF=NNNV.dot(u[iconV[:,iel]])
           #vF=NNNV.dot(v[iconV[:,iel]])
           uF,vF=velocity(xF,yF)
           #--------------
           swarm_x[im]=xA+(uA*rkf_b1+uC*rkf_b3+uD*rkf_b4+uE*rkf_b5+uF*rkf_b6)*dt
           swarm_y[im]=yA+(vA*rkf_b1+vC*rkf_b3+vD*rkf_b4+vE*rkf_b5+vF*rkf_b6)*dt
           swarm_rad[im]=np.sqrt((swarm_x[im]-2)**2+(swarm_y[im]-2)**2)
       # end for im
       markerRK5file.write("%e %e %e %d \n" %(swarm_x[0],swarm_y[0],swarm_rad[0]-1,istep))

###########################################################################
# export swarm to vtu
###########################################################################

if visu==1:

       vtufile=open('markers'+str(istep)+'.vtu',"w")
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
   print('CFL,error',CFL,np.max(abs(swarm_rad-1)),nstep)

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

