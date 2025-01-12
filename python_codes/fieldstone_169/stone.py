import numpy as np
from scipy.special import erf

TKelvin=273.15
year=365.25*24*3600

###########################################################

def analytical_solution(x,t,kappa):
    return Tb+(Ti-Tb)/2*\
           (\
           erf((d/2+x)/np.sqrt(4*kappa*t)) +\
           erf((d/2-x)/np.sqrt(4*kappa*t)) \
           )

###########################################################

nnx=601

nstep=1000000

Lx=600e3
rho=2800
Cp=1100
k=2.5
kappa0=k/rho/Cp 
L=320e3

Tb=400+TKelvin
Ti=1000+TKelvin
Ts=600+TKelvin
Tl=1200+TKelvin
d=50e3

linear=True
alpha=0.01

CFL_nb=0.1

tfinal=150e6*year

h=Lx/(nnx-1)

A=alpha/(np.exp(alpha*Tl)-np.exp(alpha*Ts))

#compute dt
dt=CFL_nb*h**2/2/kappa0
print('dt=',dt/year,'yr')

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

Tnew=np.zeros(nnx,dtype=np.float64)
Ta=np.zeros(nnx,dtype=np.float64)
kappa=np.zeros(nnx,dtype=np.float64)
melt=np.zeros(nnx,dtype=np.float64)

###########################################################
#time stepping loop
# in what follows we implicitely assume T cannot > Tl
###########################################################

mid_file=open('midT.ascii',"w")

Time=0.
for istep in range(0,nstep):
    Time+=dt
    if Time>tfinal:
       print('*** tfinal reached***')
       break
    if istep%100==0: print("timestep #",istep,'Time=',Time/year,'yr')

    for i in range (0,nnx):
        #------------------
        if linear:
           kappa[i]=kappa0
           if i==0:
              Tnew[i]=Tleft
           elif i==nnx-1:
              Tnew[i]=Tright
           else:
              Tnew[i]=Told[i]+dt*kappa[i]/h**2*(Told[i+1]-2*Told[i]+Told[i-1])
           Ta[i]=analytical_solution(x[i],Time,kappa[i])
        #------------------
        else:
           if Told[i]>Ts and Told[i]<Tl:
                 kappa[i]=k/rho/(Cp+L*A*np.exp(alpha*Told[i]))
           else:
                 kappa[i]=k/rho/(Cp+L*A*np.exp(alpha*Ts))
           if i==0:
              Tnew[i]=Tleft
           elif i==nnx-1:
              Tnew[i]=Tright
           else:
              Tnew[i]=Told[i]+dt*kappa[i]/h**2*(Told[i+1]-2*Told[i]+Told[i-1])
        #------------------
        if Tnew[i]>Ts:
           melt[i]=(np.exp(alpha*Tnew[i])-np.exp(alpha*Ts))/(np.exp(alpha*Tl)-np.exp(alpha*Ts))
        else:
           melt[i]=0 
    #end for
    if istep%10==0: 
       mid_file.write("%e %e \n" %(Time/year,Tnew[x==0][0]-TKelvin))
       np.savetxt('T_'+str(istep)+'.ascii',np.array([x,Tnew,Ta,kappa,melt]).T)
    Told[:]=Tnew[:]
#end for

###########################################################
###########################################################
