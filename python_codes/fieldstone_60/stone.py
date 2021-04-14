import numpy as np
import matplotlib.pyplot as plt
import time as time

#------------------------------------------------------------------------------
# Declare variables

experiment=2

if experiment==1:
   Lx=1.          # Horizontal extent of the domain
   T_left=0
   T_right=0
   nstep=80000+1  # total number of timesteps
   u=0.1          # Velocity
   C=2.e-3        # CFL-number
   nel=200        # Number of elements
   every=1000

if experiment==2:
   Lx=1.          # Horizontal extent of the domain
   T_left=1
   T_right=0
   nstep=250+1  # total number of timesteps
   u=1            # Velocity
   C=0.1         # CFL-number
   nel=50        # Number of elements
   every=25

nnx=nel+1      # Number of nodes
hx=Lx/nel      # Size of element  
dt=C*hx/u      # Timestep with implied CFL condition

print (dt,nstep*dt)

#------------------------------------------------------------------------------
x=np.linspace(0,Lx,nnx) 

T_minus=np.zeros(nnx,dtype=np.float64)      # Temperature at negative side of node
T_minus_old=np.zeros(nnx,dtype=np.float64)  # Temperature at negative side of node
T_plus=np.zeros(nnx,dtype=np.float64)       # Temperature at positive side of node
T_plus_old=np.zeros(nnx,dtype=np.float64)   # Temperature at positive side of node

#------------------------------------------------------------------------------
# initial conditions

if experiment==1:
   for i in range(0,nnx):
       if x[i]<0.1:
          T_plus[i]=np.sin(10*np.pi*x[i])
          T_minus[i]=np.sin(10*np.pi*x[i])
       else: 
          T_plus[i]=0
          T_minus[i]=0
       #end if
   #end for


if experiment==2:
   for i in range(0,nnx):
       if x[i]<(Lx*0.25):
          T_plus[i]=T_left
          T_minus[i]=T_left
       else: 
          T_plus[i]=T_right
          T_minus[i]=T_right
       #end if
   #end for

T_minus_old[:]=T_minus[:]
T_plus_old[:]=T_plus[:]

np.savetxt('initial_temperatures.ascii',np.array([x,T_minus,T_plus]).T,header='# x,T_minus')

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
start=time.time()

t=0 
for istep in range(0,nstep):   

    if istep%every==0:
       print("istep=",istep,'t=',t)

    T_minus[0]=T_left
    T_plus[0]=T_left
    T_minus[nnx-1]=T_right
    T_plus[nnx-1]=T_right
   
    for iel in range(0,nel):   

        #extract connectivity nodes
        k=iel
        kp1=iel+1
        
        # Pure Advection 
        T_plus[k]   =T_plus_old[k]   +C*(-3*T_plus_old[k]-T_minus_old[k+1]+4*T_minus[k])
        T_minus[k+1]=T_minus_old[k+1]+C*( 3*T_plus_old[k]-T_minus_old[k+1]-2*T_minus[k])

    #end for

    if istep%every==0:
       filename = 'T_minus_{:05d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([x,T_minus]).T,header='# x,T_minus')
       filename = 'T_plus_{:05d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([x,T_plus]).T,header='# x,T_plus')
    
    T_minus_old[:]=T_minus[:]
    T_plus_old[:]=T_plus[:]

    t+=dt

#end for
           
