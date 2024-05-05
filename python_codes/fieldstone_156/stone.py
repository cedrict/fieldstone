import numpy as np

###############################################################################

def fA(A,B):
    return -sigma*A+sigma*B

def fB(A,B,C):
    return -A*C+r*A-B

def fC(A,B,C):
    return A*B-b*C

###############################################################################

print("-----------------------------")
print("--------- stone 156 ---------")
print("-----------------------------")

tfinal=50

dt=1e-6
dtmax=1e-6

scheme=7

sigma=10
r=28
b=8/3
Lx=np.sqrt(2)
Ly=1
laambda=Lx/Ly

every=1000

tol=1e-12

###############################################################################

nnx=48
nny=32
m=4
nelx=nnx-1
nely=nny-1
nel=nelx*nely
NV=nnx*nny

nstep=int(tfinal/dt)

print('nstep=',nstep)
print('scheme=',scheme)

###############################################################################

A=np.zeros(nstep,dtype=np.float64)
B=np.zeros(nstep,dtype=np.float64)
C=np.zeros(nstep,dtype=np.float64)
t=np.zeros(nstep,dtype=np.float64)

t[0]=0
A[0]=0
B[0]=0.5
C[0]=25

###############################################################################
# compute A,B,C,t fields 
###############################################################################

if scheme==1:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       A[istep]=fA(A[istep-1],B[istep-1])*dt            +A[istep-1]
       B[istep]=fB(A[istep-1],B[istep-1],C[istep-1])*dt +B[istep-1]
       C[istep]=fC(A[istep-1],B[istep-1],C[istep-1])*dt +C[istep-1]

elif scheme==2:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       A[istep]=fA(A[istep-1],B[istep-1])*dt            +A[istep-1]
       B[istep]=fB(A[istep]  ,B[istep-1],C[istep-1])*dt +B[istep-1]
       C[istep]=fC(A[istep]  ,B[istep]  ,C[istep-1])*dt +C[istep-1]

elif scheme==3:
   for istep in range(1,nstep):
       t[istep]=istep*dt
       Anew=fA(A[istep-1],B[istep-1])*dt            +A[istep-1]
       Bnew=fB(A[istep-1],B[istep-1],C[istep-1])*dt +B[istep-1]
       Cnew=fC(A[istep-1],B[istep-1],C[istep-1])*dt +C[istep-1]
       for k in range(0,3):
           Anew=fA(Anew,Bnew)*dt      +A[istep-1]
           Bnew=fB(Anew,Bnew,Cnew)*dt +B[istep-1]
           Cnew=fC(Anew,Bnew,Cnew)*dt +C[istep-1]
       A[istep]=Anew
       B[istep]=Bnew
       C[istep]=Cnew

elif scheme==4: #RK4
   for istep in range(1,nstep):
       t[istep]=istep*dt

       A0=A[istep-1]
       B0=B[istep-1]
       C0=C[istep-1]

       A1=fA(A0,B0)
       B1=fB(A0,B0,C0)
       C1=fC(A0,B0,C0)

       A2=fA(A0+dt*A1/2,\
             B0+dt*B1/2)
       B2=fB(A0+dt*A1/2,\
             B0+dt*B1/2,\
             C0+dt*C1/2)
       C2=fC(A0+dt*A1/2,\
             B0+dt*B1/2,\
             C0+dt*C1/2)

       A3=fA(A0+dt*A2/2,\
             B0+dt*B2/2)
       B3=fB(A0+dt*A2/2,\
             B0+dt*B2/2,\
             C0+dt*C2/2)
       C3=fC(A0+dt*A2/2,\
             B0+dt*B2/2,\
             C0+dt*C2/2)

       A4=fA(A0+dt*A3,\
             B0+dt*B3)
       B4=fB(A0+dt*A3,\
             B0+dt*B3,\
             C0+dt*C3)
       C4=fC(A0+dt*A3,\
             B0+dt*B3,\
             C0+dt*C3)

       A[istep]=A[istep-1]+dt/6*(A1+2*A2+2*A3+A4)
       B[istep]=B[istep-1]+dt/6*(B1+2*B2+2*B3+B4)
       C[istep]=C[istep-1]+dt/6*(C1+2*C2+2*C3+C4)

elif scheme==5: #RK4 3/8
   for istep in range(1,nstep):
       t[istep]=istep*dt

       A0=A[istep-1]
       B0=B[istep-1]
       C0=C[istep-1]

       A1=fA(A0,B0)
       B1=fB(A0,B0,C0)
       C1=fC(A0,B0,C0)

       A2=fA(A0+dt*A1/3,\
             B0+dt*B1/3)
       B2=fB(A0+dt*A1/3,\
             B0+dt*B1/3,\
             C0+dt*C1/3)
       C2=fC(A0+dt*A1/3,\
             B0+dt*B1/3,\
             C0+dt*C1/3)

       A3=fA(A0+dt*(-A1/3+A2),\
             B0+dt*(-B1/3+B2))
       B3=fB(A0+dt*(-A1/3+A2),\
             B0+dt*(-B1/3+B2),\
             C0+dt*(-C1/3+C2))
       C3=fC(A0+dt*(-A1/3+A2),\
             B0+dt*(-B1/3+B2),\
             C0+dt*(-C1/3+C2))

       A4=fA(A0+dt*(A1-A2+A3),\
             B0+dt*(B1-B2+B3))
       B4=fB(A0+dt*(A1-A2+A3),\
             B0+dt*(B1-B2+B3),\
             C0+dt*(C1-C2+C3))
       C4=fC(A0+dt*(A1-A2+A3),\
             B0+dt*(B1-B2+B3),\
             C0+dt*(C1-C2+C3))

       A[istep]=A[istep-1]+dt/8*(A1+3*A2+3*A3+A4)
       B[istep]=B[istep-1]+dt/8*(B1+3*B2+3*B3+B4)
       C[istep]=C[istep-1]+dt/8*(C1+3*C2+3*C3+C4)

elif scheme==6: #RKF45

   dtfile=open('dt.ascii',"w")

   for istep in range(1,nstep):

       t[istep]=t[istep-1]+dt

       A0=A[istep-1]
       B0=B[istep-1]
       C0=C[istep-1]

       A1=fA(A0,B0)
       B1=fB(A0,B0,C0)
       C1=fC(A0,B0,C0)

       A2=fA(A0+dt*A1/4,\
             B0+dt*B1/4)
       B2=fB(A0+dt*A1/4,\
             B0+dt*B1/4,\
             C0+dt*C1/4)
       C2=fC(A0+dt*A1/4,\
             B0+dt*B1/4,\
             C0+dt*C1/4)

       A3=fA(A0+dt*(3*A1/32+9*A2/32),
             B0+dt*(3*B1/32+9*B2/32))
       B3=fB(A0+dt*(3*A1/32+9*A2/32),
             B0+dt*(3*B1/32+9*B2/32),
             C0+dt*(3*C1/32+9*C2/32))
       C3=fC(A0+dt*(3*A1/32+9*A2/32),
             B0+dt*(3*B1/32+9*B2/32),
             C0+dt*(3*C1/32+9*C2/32))

       A4=fA(A0+dt*(1932*A1/2197-7200*A2/2197+7296*A3/2197),\
             B0+dt*(1932*B1/2197-7200*B2/2197+7296*B3/2197))
       B4=fB(A0+dt*(1932*A1/2197-7200*A2/2197+7296*A3/2197),\
             B0+dt*(1932*B1/2197-7200*B2/2197+7296*B3/2197),\
             C0+dt*(1932*C1/2197-7200*C2/2197+7296*C3/2197))
       C4=fC(A0+dt*(1932*A1/2197-7200*A2/2197+7296*A3/2197),\
             B0+dt*(1932*B1/2197-7200*B2/2197+7296*B3/2197),\
             C0+dt*(1932*C1/2197-7200*C2/2197+7296*C3/2197))

       A5=fA(A0+dt*(439/216*A1-8*A2+3680/513*A3-845/4104*A4),\
             B0+dt*(439/216*B1-8*B2+3680/513*B3-845/4104*B4))
       B5=fB(A0+dt*(439/216*A1-8*A2+3680/513*A3-845/4104*A4),\
             B0+dt*(439/216*B1-8*B2+3680/513*B3-845/4104*B4),\
             C0+dt*(439/216*C1-8*C2+3680/513*C3-845/4104*C4))
       C5=fC(A0+dt*(439/216*A1-8*A2+3680/513*A3-845/4104*A4),\
             B0+dt*(439/216*B1-8*B2+3680/513*B3-845/4104*B4),\
             C0+dt*(439/216*C1-8*C2+3680/513*C3-845/4104*C4))

       A6=fA(A0+dt*(-8/27*A1+2*A2-3544/2565*A3+1859/4104*A4-11/40*A5),\
             B0+dt*(-8/27*B1+2*B2-3544/2565*B3+1859/4104*B4-11/40*B5))
       B6=fB(A0+dt*(-8/27*A1+2*A2-3544/2565*A3+1859/4104*A4-11/40*A5),\
             B0+dt*(-8/27*B1+2*B2-3544/2565*B3+1859/4104*B4-11/40*B5),\
             C0+dt*(-8/27*C1+2*C2-3544/2565*C3+1859/4104*C4-11/40*C5))
       C6=fC(A0+dt*(-8/27*A1+2*A2-3544/2565*A3+1859/4104*A4-11/40*A5),\
             B0+dt*(-8/27*B1+2*B2-3544/2565*B3+1859/4104*B4-11/40*B5),\
             C0+dt*(-8/27*C1+2*C2-3544/2565*C3+1859/4104*C4-11/40*C5))

       A[istep]=A0+dt*(16/135*A1+6656/12825*A3+28561/56430*A4-9/50*A5+2/55*A6)
       B[istep]=B0+dt*(16/135*B1+6656/12825*B3+28561/56430*B4-9/50*B5+2/55*B6)
       C[istep]=C0+dt*(16/135*C1+6656/12825*C3+28561/56430*C4-9/50*C5+2/55*C6)

       AA=A0+dt*(25/216*A1+1408/2565*A3+2197/4104*A4-A5/5)
       BB=B0+dt*(25/216*B1+1408/2565*B3+2197/4104*B4-B5/5)
       CC=C0+dt*(25/216*C1+1408/2565*C3+2197/4104*C4-C5/5)

       RA=max(abs(AA-A[istep])/dt,1e-25)
       RB=max(abs(BB-B[istep])/dt,1e-25)
       RC=max(abs(CC-C[istep])/dt,1e-25)

       sA=(0.5*tol/RA)**(1/5)
       sB=(0.5*tol/RB)**(1/5)
       sC=(0.5*tol/RC)**(1/5)

       dtfile.write("%e %e %e %e %e\n" %(dt,sA,sB,sC,min(sA,sB,sC)))

       dt=min(sA,sB,sC)*dt

       if t[istep]>tfinal: 
          break

   #end for

   np.savetxt('ABC.ascii',np.array([t[0:istep],A[0:istep],B[0:istep],C[0:istep]]).T,fmt='%1.5e')

elif scheme==7:

   # from https://nl.mathworks.com/matlabcentral/fileexchange/3616-ode87-integrator
   # The coefficients of method originate in prdo81

   aij=np.array(\
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

   dtfile=open('dt.ascii',"w")

   for istep in range(1,nstep):

       t[istep]=t[istep-1]+dt

       if istep%1234==0:
          print(istep,dt,t[istep])

       A0=A[istep-1]
       B0=B[istep-1]
       C0=C[istep-1]

       A1=fA(A0,B0)
       B1=fB(A0,B0,C0)
       C1=fC(A0,B0,C0)

       A2=fA(A0+dt*A1*aij[0,0],\
             B0+dt*B1*aij[0,0])
       B2=fB(A0+dt*A1*aij[0,0],\
             B0+dt*B1*aij[0,0],\
             C0+dt*C1*aij[0,0])
       C2=fC(A0+dt*A1*aij[0,0],\
             B0+dt*B1*aij[0,0],\
             C0+dt*C1*aij[0,0])

       A3=fA(A0+dt*(A1*aij[1,0]+A2*aij[1,1]),\
             B0+dt*(B1*aij[1,0]+B2*aij[1,1]))
       B3=fB(A0+dt*(A1*aij[1,0]+A2*aij[1,1]),\
             B0+dt*(B1*aij[1,0]+B2*aij[1,1]),\
             C0+dt*(C1*aij[1,0]+C2*aij[1,1]))
       C3=fC(A0+dt*(A1*aij[1,0]+A2*aij[1,1]),\
             B0+dt*(B1*aij[1,0]+B2*aij[1,1]),\
             C0+dt*(C1*aij[1,0]+C2*aij[1,1]))

       A4=fA(A0+dt*(A1*aij[2,0]+A2*aij[2,1]+A3*aij[2,2]),\
             B0+dt*(B1*aij[2,0]+B2*aij[2,1]+B3*aij[2,2]))
       B4=fB(A0+dt*(A1*aij[2,0]+A2*aij[2,1]+A3*aij[2,2]),\
             B0+dt*(B1*aij[2,0]+B2*aij[2,1]+B3*aij[2,2]),\
             C0+dt*(C1*aij[2,0]+C2*aij[2,1]+C3*aij[2,2]))
       C4=fC(A0+dt*(A1*aij[2,0]+A2*aij[2,1]+A3*aij[2,2]),\
             B0+dt*(B1*aij[2,0]+B2*aij[2,1]+B3*aij[2,2]),\
             C0+dt*(C1*aij[2,0]+C2*aij[2,1]+C3*aij[2,2]))

       A5=fA(A0+dt*(A1*aij[3,0]+A2*aij[3,1]+A3*aij[3,2]+A4*aij[3,3] ),\
             B0+dt*(B1*aij[3,0]+B2*aij[3,1]+B3*aij[3,2]+B4*aij[3,3] ))
       B5=fB(A0+dt*(A1*aij[3,0]+A2*aij[3,1]+A3*aij[3,2]+A4*aij[3,3] ),\
             B0+dt*(B1*aij[3,0]+B2*aij[3,1]+B3*aij[3,2]+B4*aij[3,3] ),\
             C0+dt*(C1*aij[3,0]+C2*aij[3,1]+C3*aij[3,2]+C4*aij[3,3] ))
       C5=fC(A0+dt*(A1*aij[3,0]+A2*aij[3,1]+A3*aij[3,2]+A4*aij[3,3] ),\
             B0+dt*(B1*aij[3,0]+B2*aij[3,1]+B3*aij[3,2]+B4*aij[3,3] ),\
             C0+dt*(C1*aij[3,0]+C2*aij[3,1]+C3*aij[3,2]+C4*aij[3,3] ))

       A6=fA(A0+dt*(A1*aij[4,0]+A2*aij[4,1]+A3*aij[4,2]+A4*aij[4,3]+A5*aij[4,4] ),\
             B0+dt*(B1*aij[4,0]+B2*aij[4,1]+B3*aij[4,2]+B4*aij[4,3]+B5*aij[4,4] ))
       B6=fB(A0+dt*(A1*aij[4,0]+A2*aij[4,1]+A3*aij[4,2]+A4*aij[4,3]+A5*aij[4,4] ),\
             B0+dt*(B1*aij[4,0]+B2*aij[4,1]+B3*aij[4,2]+B4*aij[4,3]+B5*aij[4,4] ),\
             C0+dt*(C1*aij[4,0]+C2*aij[4,1]+C3*aij[4,2]+C4*aij[4,3]+C5*aij[4,4] ))
       C6=fC(A0+dt*(A1*aij[4,0]+A2*aij[4,1]+A3*aij[4,2]+A4*aij[4,3]+A5*aij[4,4] ),\
             B0+dt*(B1*aij[4,0]+B2*aij[4,1]+B3*aij[4,2]+B4*aij[4,3]+B5*aij[4,4] ),\
             C0+dt*(C1*aij[4,0]+C2*aij[4,1]+C3*aij[4,2]+C4*aij[4,3]+C5*aij[4,4] ))

       A7=fA(A0+dt*(A1*aij[5,0]+A2*aij[5,1]+A3*aij[5,2]+A4*aij[5,3]+A5*aij[5,4]+A6*aij[5,5] ),\
             B0+dt*(B1*aij[5,0]+B2*aij[5,1]+B3*aij[5,2]+B4*aij[5,3]+B5*aij[5,4]+B6*aij[5,5] ))
       B7=fB(A0+dt*(A1*aij[5,0]+A2*aij[5,1]+A3*aij[5,2]+A4*aij[5,3]+A5*aij[5,4]+A6*aij[5,5] ),\
             B0+dt*(B1*aij[5,0]+B2*aij[5,1]+B3*aij[5,2]+B4*aij[5,3]+B5*aij[5,4]+B6*aij[5,5] ),\
             C0+dt*(C1*aij[5,0]+C2*aij[5,1]+C3*aij[5,2]+C4*aij[5,3]+C5*aij[5,4]+C6*aij[5,5] ))
       C7=fC(A0+dt*(A1*aij[5,0]+A2*aij[5,1]+A3*aij[5,2]+A4*aij[5,3]+A5*aij[5,4]+A6*aij[5,5] ),\
             B0+dt*(B1*aij[5,0]+B2*aij[5,1]+B3*aij[5,2]+B4*aij[5,3]+B5*aij[5,4]+B6*aij[5,5] ),\
             C0+dt*(C1*aij[5,0]+C2*aij[5,1]+C3*aij[5,2]+C4*aij[5,3]+C5*aij[5,4]+C6*aij[5,5] ))

       A8=fA(A0+dt*(A1*aij[6,0]+A2*aij[6,1]+A3*aij[6,2]+A4*aij[6,3]+A5*aij[6,4]+A6*aij[6,5]+A7*aij[6,6] ),\
             B0+dt*(B1*aij[6,0]+B2*aij[6,1]+B3*aij[6,2]+B4*aij[6,3]+B5*aij[6,4]+B6*aij[6,5]+B7*aij[6,6] ))
       B8=fB(A0+dt*(A1*aij[6,0]+A2*aij[6,1]+A3*aij[6,2]+A4*aij[6,3]+A5*aij[6,4]+A6*aij[6,5]+A7*aij[6,6] ),\
             B0+dt*(B1*aij[6,0]+B2*aij[6,1]+B3*aij[6,2]+B4*aij[6,3]+B5*aij[6,4]+B6*aij[6,5]+B7*aij[6,6] ),\
             C0+dt*(C1*aij[6,0]+C2*aij[6,1]+C3*aij[6,2]+C4*aij[6,3]+C5*aij[6,4]+C6*aij[6,5]+C7*aij[6,6] ))
       C8=fC(A0+dt*(A1*aij[6,0]+A2*aij[6,1]+A3*aij[6,2]+A4*aij[6,3]+A5*aij[6,4]+A6*aij[6,5]+A7*aij[6,6] ),\
             B0+dt*(B1*aij[6,0]+B2*aij[6,1]+B3*aij[6,2]+B4*aij[6,3]+B5*aij[6,4]+B6*aij[6,5]+B7*aij[6,6] ),\
             C0+dt*(C1*aij[6,0]+C2*aij[6,1]+C3*aij[6,2]+C4*aij[6,3]+C5*aij[6,4]+C6*aij[6,5]+C7*aij[6,6] ))

       A9=fA(A0+dt*(A1*aij[7,0]+A2*aij[7,1]+A3*aij[7,2]+A4*aij[7,3]+A5*aij[7,4]+A6*aij[7,5]+A7*aij[7,6]+A8*aij[7,7] ),\
             B0+dt*(B1*aij[7,0]+B2*aij[7,1]+B3*aij[7,2]+B4*aij[7,3]+B5*aij[7,4]+B6*aij[7,5]+B7*aij[7,6]+B8*aij[7,7] ))
       B9=fB(A0+dt*(A1*aij[7,0]+A2*aij[7,1]+A3*aij[7,2]+A4*aij[7,3]+A5*aij[7,4]+A6*aij[7,5]+A7*aij[7,6]+A8*aij[7,7] ),\
             B0+dt*(B1*aij[7,0]+B2*aij[7,1]+B3*aij[7,2]+B4*aij[7,3]+B5*aij[7,4]+B6*aij[7,5]+B7*aij[7,6]+B8*aij[7,7] ),\
             C0+dt*(C1*aij[7,0]+C2*aij[7,1]+C3*aij[7,2]+C4*aij[7,3]+C5*aij[7,4]+C6*aij[7,5]+C7*aij[7,6]+C8*aij[7,7] ))
       C9=fC(A0+dt*(A1*aij[7,0]+A2*aij[7,1]+A3*aij[7,2]+A4*aij[7,3]+A5*aij[7,4]+A6*aij[7,5]+A7*aij[7,6]+A8*aij[7,7] ),\
             B0+dt*(B1*aij[7,0]+B2*aij[7,1]+B3*aij[7,2]+B4*aij[7,3]+B5*aij[7,4]+B6*aij[7,5]+B7*aij[7,6]+B8*aij[7,7] ),\
             C0+dt*(C1*aij[7,0]+C2*aij[7,1]+C3*aij[7,2]+C4*aij[7,3]+C5*aij[7,4]+C6*aij[7,5]+C7*aij[7,6]+C8*aij[7,7] ))

       A10=fA(A0+dt*(A1*aij[8,0]+A2*aij[8,1]+A3*aij[8,2]+A4*aij[8,3]+A5*aij[8,4]+A6*aij[8,5]+A7*aij[8,6]+A8*aij[8,7]+A9*aij[8,8] ),\
              B0+dt*(B1*aij[8,0]+B2*aij[8,1]+B3*aij[8,2]+B4*aij[8,3]+B5*aij[8,4]+B6*aij[8,5]+B7*aij[8,6]+B8*aij[8,7]+B9*aij[8,8] ))
       B10=fB(A0+dt*(A1*aij[8,0]+A2*aij[8,1]+A3*aij[8,2]+A4*aij[8,3]+A5*aij[8,4]+A6*aij[8,5]+A7*aij[8,6]+A8*aij[8,7]+A9*aij[8,8] ),\
              B0+dt*(B1*aij[8,0]+B2*aij[8,1]+B3*aij[8,2]+B4*aij[8,3]+B5*aij[8,4]+B6*aij[8,5]+B7*aij[8,6]+B8*aij[8,7]+B9*aij[8,8] ),\
              C0+dt*(C1*aij[8,0]+C2*aij[8,1]+C3*aij[8,2]+C4*aij[8,3]+C5*aij[8,4]+C6*aij[8,5]+C7*aij[8,6]+C8*aij[8,7]+C9*aij[8,8] ))
       C10=fC(A0+dt*(A1*aij[8,0]+A2*aij[8,1]+A3*aij[8,2]+A4*aij[8,3]+A5*aij[8,4]+A6*aij[8,5]+A7*aij[8,6]+A8*aij[8,7]+A9*aij[8,8] ),\
              B0+dt*(B1*aij[8,0]+B2*aij[8,1]+B3*aij[8,2]+B4*aij[8,3]+B5*aij[8,4]+B6*aij[8,5]+B7*aij[8,6]+B8*aij[8,7]+B9*aij[8,8] ),\
              C0+dt*(C1*aij[8,0]+C2*aij[8,1]+C3*aij[8,2]+C4*aij[8,3]+C5*aij[8,4]+C6*aij[8,5]+C7*aij[8,6]+C8*aij[8,7]+C9*aij[8,8] ))

       A11=fA(A0+dt*(A1*aij[9,0]+A2*aij[9,1]+A3*aij[9,2]+A4*aij[9,3]+A5*aij[9,4]+A6*aij[9,5]+A7*aij[9,6]+A8*aij[9,7]+A9*aij[9,8]+A10*aij[9,9] ),\
              B0+dt*(B1*aij[9,0]+B2*aij[9,1]+B3*aij[9,2]+B4*aij[9,3]+B5*aij[9,4]+B6*aij[9,5]+B7*aij[9,6]+B8*aij[9,7]+B9*aij[9,8]+B10*aij[9,9] ))
       B11=fB(A0+dt*(A1*aij[9,0]+A2*aij[9,1]+A3*aij[9,2]+A4*aij[9,3]+A5*aij[9,4]+A6*aij[9,5]+A7*aij[9,6]+A8*aij[9,7]+A9*aij[9,8]+A10*aij[9,9] ),\
              B0+dt*(B1*aij[9,0]+B2*aij[9,1]+B3*aij[9,2]+B4*aij[9,3]+B5*aij[9,4]+B6*aij[9,5]+B7*aij[9,6]+B8*aij[9,7]+B9*aij[9,8]+B10*aij[9,9] ),\
              C0+dt*(C1*aij[9,0]+C2*aij[9,1]+C3*aij[9,2]+C4*aij[9,3]+C5*aij[9,4]+C6*aij[9,5]+C7*aij[9,6]+C8*aij[9,7]+C9*aij[9,8]+C10*aij[9,9] ))
       C11=fC(A0+dt*(A1*aij[9,0]+A2*aij[9,1]+A3*aij[9,2]+A4*aij[9,3]+A5*aij[9,4]+A6*aij[9,5]+A7*aij[9,6]+A8*aij[9,7]+A9*aij[9,8]+A10*aij[9,9] ),\
              B0+dt*(B1*aij[9,0]+B2*aij[9,1]+B3*aij[9,2]+B4*aij[9,3]+B5*aij[9,4]+B6*aij[9,5]+B7*aij[9,6]+B8*aij[9,7]+B9*aij[9,8]+B10*aij[9,9] ),\
              C0+dt*(C1*aij[9,0]+C2*aij[9,1]+C3*aij[9,2]+C4*aij[9,3]+C5*aij[9,4]+C6*aij[9,5]+C7*aij[9,6]+C8*aij[9,7]+C9*aij[9,8]+C10*aij[9,9] ))

       A12=fA(A0+dt*(A1*aij[10,0]+A2*aij[10,1]+A3*aij[10,2]+A4*aij[10,3]+A5*aij[10,4]+A6*aij[10,5]+A7*aij[10,6]+A8*aij[10,7]+A9*aij[10,8]+A10*aij[10,9]+A11*aij[10,10] ),\
              B0+dt*(B1*aij[10,0]+B2*aij[10,1]+B3*aij[10,2]+B4*aij[10,3]+B5*aij[10,4]+B6*aij[10,5]+B7*aij[10,6]+B8*aij[10,7]+B9*aij[10,8]+B10*aij[10,9]+B11*aij[10,10] ))
       B12=fB(A0+dt*(A1*aij[10,0]+A2*aij[10,1]+A3*aij[10,2]+A4*aij[10,3]+A5*aij[10,4]+A6*aij[10,5]+A7*aij[10,6]+A8*aij[10,7]+A9*aij[10,8]+A10*aij[10,9]+A11*aij[10,10] ),\
              B0+dt*(B1*aij[10,0]+B2*aij[10,1]+B3*aij[10,2]+B4*aij[10,3]+B5*aij[10,4]+B6*aij[10,5]+B7*aij[10,6]+B8*aij[10,7]+B9*aij[10,8]+B10*aij[10,9]+B11*aij[10,10] ),\
              C0+dt*(C1*aij[10,0]+C2*aij[10,1]+C3*aij[10,2]+C4*aij[10,3]+C5*aij[10,4]+C6*aij[10,5]+C7*aij[10,6]+C8*aij[10,7]+C9*aij[10,8]+C10*aij[10,9]+C11*aij[10,10] ))
       C12=fC(A0+dt*(A1*aij[10,0]+A2*aij[10,1]+A3*aij[10,2]+A4*aij[10,3]+A5*aij[10,4]+A6*aij[10,5]+A7*aij[10,6]+A8*aij[10,7]+A9*aij[10,8]+A10*aij[10,9]+A11*aij[10,10] ),\
              B0+dt*(B1*aij[10,0]+B2*aij[10,1]+B3*aij[10,2]+B4*aij[10,3]+B5*aij[10,4]+B6*aij[10,5]+B7*aij[10,6]+B8*aij[10,7]+B9*aij[10,8]+B10*aij[10,9]+B11*aij[10,10] ),\
              C0+dt*(C1*aij[10,0]+C2*aij[10,1]+C3*aij[10,2]+C4*aij[10,3]+C5*aij[10,4]+C6*aij[10,5]+C7*aij[10,6]+C8*aij[10,7]+C9*aij[10,8]+C10*aij[10,9]+C11*aij[10,10] ))

       A13=fA(A0+dt*(A1*aij[11,0]+A2*aij[11,1]+A3*aij[11,2]+A4*aij[11,3]+A5*aij[11,4]+A6*aij[11,5]+A7*aij[11,6]+A8*aij[11,7]+A9*aij[11,8]+A10*aij[11,9]+A11*aij[11,10]+A12*aij[11,11] ),\
              B0+dt*(B1*aij[11,0]+B2*aij[11,1]+B3*aij[11,2]+B4*aij[11,3]+B5*aij[11,4]+B6*aij[11,5]+B7*aij[11,6]+B8*aij[11,7]+B9*aij[11,8]+B10*aij[11,9]+B11*aij[11,10]+B12*aij[11,11] ))
       B13=fB(A0+dt*(A1*aij[11,0]+A2*aij[11,1]+A3*aij[11,2]+A4*aij[11,3]+A5*aij[11,4]+A6*aij[11,5]+A7*aij[11,6]+A8*aij[11,7]+A9*aij[11,8]+A10*aij[11,9]+A11*aij[11,10]+A12*aij[11,11] ),\
              B0+dt*(B1*aij[11,0]+B2*aij[11,1]+B3*aij[11,2]+B4*aij[11,3]+B5*aij[11,4]+B6*aij[11,5]+B7*aij[11,6]+B8*aij[11,7]+B9*aij[11,8]+B10*aij[11,9]+B11*aij[11,10]+B12*aij[11,11] ),\
              C0+dt*(C1*aij[11,0]+C2*aij[11,1]+C3*aij[11,2]+C4*aij[11,3]+C5*aij[11,4]+C6*aij[11,5]+C7*aij[11,6]+C8*aij[11,7]+C9*aij[11,8]+C10*aij[11,9]+C11*aij[11,10]+C12*aij[11,11] ))
       C13=fC(A0+dt*(A1*aij[11,0]+A2*aij[11,1]+A3*aij[11,2]+A4*aij[11,3]+A5*aij[11,4]+A6*aij[11,5]+A7*aij[11,6]+A8*aij[11,7]+A9*aij[11,8]+A10*aij[11,9]+A11*aij[11,10]+A12*aij[11,11] ),\
              B0+dt*(B1*aij[11,0]+B2*aij[11,1]+B3*aij[11,2]+B4*aij[11,3]+B5*aij[11,4]+B6*aij[11,5]+B7*aij[11,6]+B8*aij[11,7]+B9*aij[11,8]+B10*aij[11,9]+B11*aij[11,10]+B12*aij[11,11] ),\
              C0+dt*(C1*aij[11,0]+C2*aij[11,1]+C3*aij[11,2]+C4*aij[11,3]+C5*aij[11,4]+C6*aij[11,5]+C7*aij[11,6]+C8*aij[11,7]+C9*aij[11,8]+C10*aij[11,9]+C11*aij[11,10]+C12*aij[11,11] ))


       A[istep]=A0+dt*(14005451/335480064*A1\
                      -59238493/1068277825*A6\
                      +181606767/758867731*A7\
                      +561292985/797845732*A8\
                      -1041891430/1371343529*A9\
                      +760417239/1151165299*A10\
                      +118820643/751138087*A11\
                      -528747749/2220607170*A12\
                      +1/4*A13)

       B[istep]=B0+dt*(14005451/335480064*B1\
                      -59238493/1068277825*B6\
                      +181606767/758867731*B7\
                      +561292985/797845732*B8\
                      -1041891430/1371343529*B9\
                      +760417239/1151165299*B10\
                      +118820643/751138087*B11\
                      -528747749/2220607170*B12\
                      +1/4*B13)

       C[istep]=C0+dt*(14005451/335480064*C1\
                      -59238493/1068277825*C6\
                      +181606767/758867731*C7\
                      +561292985/797845732*C8\
                      -1041891430/1371343529*C9\
                      +760417239/1151165299*C10\
                      +118820643/751138087*C11\
                      -528747749/2220607170*C12\
                      +1/4*C13)

       AA=A0+dt*(13451932/455176623*A1\
                -808719846/976000145*A6\
                +1757004468/5645159321*A7\
                +656045339/265891186*A8\
                -3867574721/1518517206*A9\
                +465885868/322736535*A10\
                +53011238/667516719*A11\
                +2/45*A12)

       BB=B0+dt*(13451932/455176623*B1\
                -808719846/976000145*B6\
                +1757004468/5645159321*B7\
                +656045339/265891186*B8\
                -3867574721/1518517206*B9\
                +465885868/322736535*B10\
                +53011238/667516719*B11\
                +2/45*B12)

       CC=C0+dt*(13451932/455176623*C1\
                -808719846/976000145*C6\
                +1757004468/5645159321*C7\
                +656045339/265891186*C8\
                -3867574721/1518517206*C9\
                +465885868/322736535*C10\
                +53011238/667516719*C11\
                +2/45*C12)

       RA=max(abs(AA-A[istep])/dt,1e-25)
       RB=max(abs(BB-B[istep])/dt,1e-25)
       RC=max(abs(CC-C[istep])/dt,1e-25)

       sA=(0.5*tol/RA)**(1/9)
       sB=(0.5*tol/RB)**(1/9)
       sC=(0.5*tol/RC)**(1/9)

       dtfile.write("%e %e %e %e %e\n" %(dt,sA,sB,sC,min(sA,sB,sC)))

       dt=min(min(sA,sB,sC)*dt,dtmax)

       if t[istep]>tfinal: 
          break

   #end for

   #np.savetxt('ABC.ascii',np.array([t[0:istep],A[0:istep],B[0:istep],C[0:istep]]).T,fmt='%1.5e')


#end if

#if scheme != 6 and scheme!=7:

if dt>1e-4:
   np.savetxt('ABC.ascii',np.array([t[::10],A[::10],B[::10],C[::10]]).T,fmt='%1.5e')
elif dt>1e-5:
   np.savetxt('ABC.ascii',np.array([t[::100],A[::100],B[::100],C[::100]]).T,fmt='%1.5e')
elif dt>1e-6:
   np.savetxt('ABC.ascii',np.array([t[::1000],A[::1000],B[::1000],C[::1000]]).T,fmt='%1.5e')
elif dt>1e-7:
   np.savetxt('ABC.ascii',np.array([t[::10000],A[::10000],B[::10000],C[::10000]]).T,fmt='%1.5e')
elif dt>1e-8:
   np.savetxt('ABC.ascii',np.array([t[::100000],A[::100000],B[::100000],C[::100000]]).T,fmt='%1.5e')

exit()

###############################################################################
# make mesh
###############################################################################

x = np.empty(NV,dtype=np.float64)  # x coordinates
y = np.empty(NV,dtype=np.float64)  # y coordinates

counter = 0
for j in range(0, nny):
    for i in range(0, nnx):
        x[counter]=i*Lx/float(nelx)
        y[counter]=j*Ly/float(nely)
        counter += 1

###############################################################################
# build connectivity array
###############################################################################

icon =np.zeros((m, nel),dtype=np.int32)
counter = 0
for j in range(0, nely):
    for i in range(0, nelx):
        icon[0, counter] = i + j * (nelx + 1)
        icon[1, counter] = i + 1 + j * (nelx + 1)
        icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
        icon[3, counter] = i + (j + 1) * (nelx + 1)
        counter += 1

###############################################################################
# swarm setup
###############################################################################

nmarker=1600
swarm_x=np.empty(nmarker,dtype=np.float64)
swarm_y=np.empty(nmarker,dtype=np.float64)

ddx=0.025
ddy=0.025

counter=0
for i in range(0,40):
    for j in range(0,40):
        swarm_x[counter]=Lx/2-ddx/2+i*ddx/39
        swarm_y[counter]=Ly/2-ddy/2+j*ddy/39
        counter+=1

###############################################################################
# compute u,v,psi in time based on B,C
###############################################################################

Psi=np.zeros(NV,dtype=np.float64)
u=np.zeros(NV,dtype=np.float64)
v=np.zeros(NV,dtype=np.float64)

for istep in range(0,nstep):

    ###########################################################################
    # compute psi,u,v
    ###########################################################################

    Psi=(laambda*np.sqrt(2)/np.pi**2)*np.sin(np.pi*y)*\
        (B[istep]*np.sin(x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*x/laambda))

    u=(laambda*np.sqrt(2)/np.pi)*np.cos(np.pi*y)*\
      (B[istep]*np.sin(x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*x/laambda))

    v=-(np.sqrt(2)/np.pi)*np.sin(np.pi*y)*\
       (B[istep]*np.cos(x*np.pi/laambda)+2*(C[istep]-27)*np.cos(2*np.pi*x/laambda))

    ###########################################################################
    # interpolate velocity onto markers and advect them
    ###########################################################################

    swarm_u=(laambda*np.sqrt(2)/np.pi)*np.cos(np.pi*swarm_y)*\
            (B[istep]*np.sin(swarm_x*np.pi/laambda)+(C[istep]-27)*np.sin(2*np.pi*swarm_x/laambda))

    swarm_v=-(np.sqrt(2)/np.pi)*np.sin(np.pi*swarm_y)*\
             (B[istep]*np.cos(swarm_x*np.pi/laambda)+2*(C[istep]-27)*np.cos(2*np.pi*swarm_x/laambda))

    swarm_x+=swarm_u*dt
    swarm_y+=swarm_v*dt

    ###########################################################################
    # export to vtu
    ###########################################################################

    if istep%every==0:
       filename = 'solution_{:05d}.vtu'.format(istep)
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
       vtufile.write("<DataArray type='Float32' Name='psi' Format='ascii'> \n")
       for i in range(0,NV):
           vtufile.write("%10e \n" %Psi[i])
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*m))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %9)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()


       filename = 'markers_{:05d}.vtu'.format(istep)
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
       #####
       vtufile.write("<Points> \n")
       #--
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for im in range(0,nmarker):
           vtufile.write("%10e %10e %10e \n" %(swarm_x[im],swarm_y[im],0.))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Points> \n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d\n" % im )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % (im+1) )
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for im in range (0,nmarker):
           vtufile.write("%d \n" % 1)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

print("-----------------------------")
print("------------the end----------")
print("-----------------------------")
