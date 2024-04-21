import numpy as np


tfinal=10
dt=5e-7

sigma=10
r=28
b=8/3

nstep=int(tfinal/dt)

A=np.zeros(nstep,dtype=np.float64)
B=np.zeros(nstep,dtype=np.float64)
C=np.zeros(nstep,dtype=np.float64)
t=np.zeros(nstep,dtype=np.float64)

t[0]=0
A[0]=0
B[0]=0.5
C[0]=25

for istep in range(1,nstep):

    A[istep]=(-sigma*A[istep-1]+sigma*B[istep-1])*dt            +A[istep-1]

    B[istep]=(-A[istep-1]*C[istep-1]+r*A[istep-1]-B[istep-1])*dt+B[istep-1]

    C[istep]=(A[istep-1]*B[istep-1]-b*C[istep-1])*dt            +C[istep-1]

    t[istep]=istep*dt

    #print('istep=',istep,'| A,B,C=',A[istep],B[istep],C[istep])

np.savetxt('ABC.ascii',np.array([t,A,B,C]).T,fmt='%1.3e')
