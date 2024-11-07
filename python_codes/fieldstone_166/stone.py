import numpy as np

experiment=1

# experiment 1: u(x,t)=sin(pi*x)*cos(pi*c*t)


#==============================================================================

def uth(x,t,c):
    return np.sin(np.pi*x)*np.cos(np.pi*c*t)

def f(x):
    return np.sin(np.pi*x)

def g(x):
    return 0

#==============================================================================

nnx=101
Lx=1
h=Lx/(nnx-1)
c=1
CFL=0.5
dt=CFL*h/c
T=25
nstep=int(T/dt)

###############################################################################
# node layout
###############################################################################

x=np.linspace(0,Lx,nnx)

###############################################################################
# initial field value
# Set initial condition u(x,0) = f(x) (uprevprev)
# Apply special formula for first step, incorporating du/dt=0 (uprev)
###############################################################################

u=np.zeros(nnx,dtype=np.float64)  
u_analytical=np.zeros(nnx,dtype=np.float64)  
uprev=np.zeros(nnx,dtype=np.float64)  
uprevprev=np.zeros(nnx,dtype=np.float64) 

for i in range(0,nnx):
    uprevprev[i]=f(x[i])

uprevprev[0] = 0
uprevprev[nnx-1] = 0

for i in range(1,nnx-1):
    uprev[i] = dt*g(x[i])+(1-CFL**2)*uprevprev[i] + 0.5*CFL**2*(uprevprev[i+1] + uprevprev[i-1])

uprev[0] = 0
uprev[nnx-1] = 0
    
print('istep=',0,'t=',0,'u (m/M)=',np.min(uprevprev),np.max(uprevprev))
print('istep=',1,'t=',dt,'u (m/M)=',np.min(uprev),np.max(uprev))
       
np.savetxt('u_0000.ascii',np.array([x,uprevprev]).T)

###############################################################################
# time loop
###############################################################################

statsfile=open('u_stats_'+str(nnx)+'.ascii',"w")

t=2*dt
for istep in range(2,nstep):

    # compute new u field (i.e. u(t+dt))
    for i in range(1,nnx-1):
        u[i]=-uprevprev[i]+2*(1-CFL**2)*uprev[i]+CFL**2*(uprev[i+1]+uprev[i-1])
    u[0] = 0
    u[nnx-1] = 0
    
    t+=dt

    for i in range(0,nnx):
        u_analytical[i]=uth(x[i],t,c)

    statsfile.write("%e %e %e \n" %(t,np.min(u),np.max(u)))

    if istep%100==0:
       print('istep=',istep,'t=',t,'u (m/M)=',np.min(u),np.max(u))
       filename = 'u_{:04d}.ascii'.format(istep) 
       np.savetxt(filename,np.array([x,u,u_analytical]).T)

    uprevprev[:]=uprev[:]
    uprev[:]=u[:]

#end for

