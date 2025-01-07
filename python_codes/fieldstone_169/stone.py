import numpy as np
from scipy.special import erf

year=365.25*24*3600

###########################################################

def analytical_solution(x,t):
    return Tb+(Ti-Tb)/2*\
           (\
           erf((d/2+x)/np.sqrt(4*kappa*t)) +\
           erf((d/2-x)/np.sqrt(4*kappa*t)) \
           )

###########################################################

nnx=501

nstep=10000

Lx=400e3
rho=2800
Cp=1100
k=2.5
kappa=k/rho/Cp
L=350e3

Tb=400
Ti=800
Ts=600
Tl=1200
d=50e3

CFL_nb=0.1

tfinal=5e6*year

h=Lx/(nnx-1)

#compute dt
dt=CFL_nb*h**2/2/kappa
print('dt=',dt)

#boundary conditions:
Tleft=Tb
Tright=Tb

#define location of nodes
x=np.empty(nnx,dtype=np.float64)
for i in range(0,nnx):
    x[i]=i*h-Lx/2

#initial temperature
Told=np.zeros(nnx,dtype=np.float64)
Told[:]=Tb
for i in range(0,nnx):
    if abs(x[i])<d/2: Told[i]=Ti

Tnew=np.empty(nnx,dtype=np.float64)
Ta=np.empty(nnx,dtype=np.float64)

#time stepping loop

Time=0.
for n in range(0,nstep):
    Time+=dt
    if Time>tfinal:
       print('*** tfinal reached***')
       break
    print("timestep #",n,'Time=',Time/year,'yr')
    for i in range (0,nnx):
        if i==0:
           Tnew[i]=Tleft
        elif i==nnx-1:
           Tnew[i]=Tright
        else:
           Tnew[i]=Told[i]+dt*kappa/h**2*(Told[i+1]-2*Told[i]+Told[i-1])
        Ta[i]=analytical_solution(x[i],Time)
    #end for
    np.savetxt('T_'+str(n)+'.ascii',np.array([x,Tnew,Ta]).T)
    Told[:]=Tnew[:]
#end for





