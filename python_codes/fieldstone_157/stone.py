import numpy as np
from scipy.integrate import solve_ivp

nnx=101
Lx=1

h=Lx/(nnx-1)

T_left=1
T_right=0

tfinal=10

kappa=0.01

dt=h**2/2/kappa #*0.1

print('dt=',dt)

nstep=int(tfinal/dt)

maxdt=dt

###################################
#option
# 1-> Dirichlet left and right
# 2-> Dirichlet left, Neumann right

option=1

###############################################################################

x=np.linspace(0,Lx,nnx)

###############################################################################
#initial temperature
###############################################################################

T=np.zeros(nnx,dtype=np.float64)
T[:]=0.123
T[0]=T_left
T[nnx-1]=T_right
T0=np.zeros(nnx,dtype=np.float64)
T0[:]=T[:]  
 
###############################################################################

if option==1:

   # defining function that returns dTdt at all nodes
   # assuming Dirichlet b.c. on both ends

   def F(t,T,kappa):
       dT_dt=np.zeros(nnx,dtype=np.float64)
       for i in range(0,nnx):
           if i==0:
              dT_dt[i]=0
           elif i==nnx-1: 
              dT_dt[i]=0
           else:
              dT_dt[i]=kappa*(T[i+1]-2*T[i]+T[i-1])/h**2
       return dT_dt

elif option==2:

   # defining function that returns dTdt at all nodes
   # assuming Dirichlet b.c. at x=0 and dTdx=0 at x=Lx

   def F(t,T,kappa):
       dT_dt=np.zeros(nnx,dtype=np.float64)
       for i in range(0,nnx):
           if i==0:
              dT_dt[i]=0
           elif i==nnx-1: 
              dT_dt[i]=kappa*(-T[i-1]+T[i-2])/h**2
           else:
              dT_dt[i]=kappa*(T[i+1]-2*T[i]+T[i-1])/h**2
       return dT_dt

###############################################################################
# time stepping loop
###############################################################################

print('1st order in time')

t=0
for istep in range(0,nstep+1):
    T[:]=F(t,T[:],kappa)*dt+T[:]
    t+=dt

    if istep%200==0 or istep==nstep:
       filename = 'T_{:05d}.ascii'.format(istep)
       np.savetxt(filename,np.array([x,T]).T,fmt='%1.5e')

###############################################################################

RKmethod='RK23'
#RKmethod='RK45'
#RKmethod='DOP853'
#RKmethod='BDF'
#RKmethod='Radau'

print('RK method=',RKmethod)

soln = solve_ivp(F, (0,tfinal), T0, args=(kappa,), method=RKmethod ,max_step=maxdt)

print('# timesteps',len(soln.t))

nstep=len(soln.t)

np.savetxt('T_'+RKmethod+'.ascii',np.array([x,soln.y[:,nstep-1]]).T,fmt='%1.5e')


