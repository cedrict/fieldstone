import numpy as np
import sys as sys
import random
import time as clock 
import numba
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

three_dimensions=False

#init=1: random
#init=2: three discs
#init=3: a la pearson 1993
#init=4: two gaussians

if int(len(sys.argv))==7:
   model  = str(sys.argv[1])
   nnx    = int(sys.argv[2])
   scheme = str(sys.argv[3])
   init   = int(sys.argv[4])
   nstep  = int(sys.argv[5])
   dt     = float(sys.argv[6])
   print(sys.argv)
else:
   model='alpha1'
   nnx = 257
   scheme='RK4'
   init=4
   nstep=100000
   dt=0.1

if three_dimensions:
   Lx=2.5
   Ly=0.5
   Lz=2.5
   nny=int(nnx*Ly/Lx)
   nnz=int(nnx*Lz/Lx)
   hx=Lx/(nnx-1)
   hy=Ly/(nny-1)
   hz=Lz/(nnz-1)
   nelx=nnx-1
   nely=nny-1
   nelz=nnz-1
   nel=nelx*nely*nelz
   NP=nnx*nny*nnz
   m=8
   tyype=12 # vtu
   nseed=100

else:
   Lx=2.5
   Lz=2.5
   nny=1
   nnz=int(nnx*Lz/Lx)
   hx=Lx/(nnx-1)
   hz=Lz/(nnz-1)
   nelx=nnx-1
   nelz=nnz-1
   nel=nelx*nelz
   NP=nnx*nnz
   m=4
   tyype=9 # vtu
   nseed=500

every_ascii=100
every_vtu=5000
every_png=5000

###########################################################

if model=='alpha1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.010 ; Kill=0.047
if model=='alpha2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.014 ; Kill=0.053

if model=='beta1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.014 ; Kill=0.039
if model=='beta2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.026 ; Kill=0.051

if model=='gamma1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.022 ; Kill=0.051
if model=='gamma2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.026 ; Kill=0.055
 
if model=='delta1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.030 ; Kill=0.055 
if model=='delta2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.042 ; Kill=0.059

if model=='epsilon1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.018 ; Kill=0.055
if model=='epsilon2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.022 ; Kill=0.059

if model=='zeta1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.022 ; Kill=0.061
if model=='zeta2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.026 ; Kill=0.059

if model=='eta':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.034 ; Kill=0.063

if model=='theta1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.030 ; Kill=0.057
if model=='theta2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.038 ; Kill=0.061

if model=='iota':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.046 ; Kill=0.0594

if model=='kappa1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.050 ; Kill=0.063
if model=='kappa2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.058 ; Kill=0.063

if model=='lambda1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.026 ; Kill=0.061
if model=='lambda2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.034 ; Kill=0.065

if model=='mu1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.046 ; Kill=0.065
if model=='mu2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.058 ; Kill=0.065

if model=='nu1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.054 ; Kill=0.067
if model=='nu2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.082 ; Kill=0.063

if model=='xi1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.010 ; Kill=0.041
if model=='xi2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.014 ; Kill=0.047

if model=='pi':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.062 ; Kill=0.061

if model=='rho1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.090 ; Kill=0.059
if model=='rho2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.102 ; Kill=0.055

if model=='sigma1':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.090 ; Kill=0.057
if model=='sigma2':
   Du=2.e-5 ; Dv=1e-5 ; Feed=0.110 ; Kill=0.0523


if model=='lukas':
   Du=4.e-6 ; Dv=2e-6 ; Feed=0.035 ; Kill=0.0575

###############################################################################

if scheme=='RK3': #----------------------------------------
   # Kutta's third-order method 
   # https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods

   a=np.array(\
     [[1/2, 0],\
      [ -1, 2]],\
      dtype=np.float64)

   b=np.array([1/6,4/6,1/6],dtype=np.float64)

elif scheme=='RK4': #----------------------------------------
   #https://en.wikipedia.org/wiki/Runge-Kutta_methods

   a=np.array(\
     [[1/2,   0, 0],\
      [  0, 1/2, 0],\
      [  0,   0, 1]],\
      dtype=np.float64)

   b=np.array([1/6,1/3,1/3,1/6],dtype=np.float64)

elif scheme=='RK38': #---------------------------------------
   #https://en.wikipedia.org/wiki/Runge-Kutta_methods

   a=np.array(\
     [[ 1/3,  0, 0],\
      [-1/3,  1, 0],\
      [   1, -1, 1]],\
      dtype=np.float64)

   b=np.array([1/8,3/8,3/8,1/8],dtype=np.float64)

elif scheme=='RK5': #---------------------------------------
   # Nyström's fifth-order method  
   # https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods

   a=np.array(\
     [[     1/3,     0,      0,    0, 0],\
      [    4/25,  6/25,      0,    0, 0],\
      [     1/4,    -3,   15/4,    0, 0],\
      [    2/27,  10/9, -50/81, 8/81, 0],\
      [    2/25, 12/25,   2/15, 8/75, 0]],\
      dtype=np.float64)

   b=np.array([23/192,0,125/192,0,-27/64,125/192],dtype=np.float64)

elif scheme=='RKF5': #---------------------------------------
   #https://en.wikipedia.org/wiki/Runge-Kutta_methods
   #https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods

   a=np.array(\
     [[      1/4,          0,          0,         0,      0],\
      [     3/32,       9/32,          0,         0,      0],\
      [1932/2197, -7200/2197,  7296/2197,         0,      0],\
      [  439/216,         -8,   3680/513, -845/4104,      0],\
      [    -8/27,          2, -3544/2565, 1859/4104, -11/40]],\
      dtype=np.float64)

   b=np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],dtype=np.float64)

if scheme=='RK65': #----------------------------------------
   # The coefficients originate in Table 1 of prdo81
   # not 100% clear to me which b vector I should use

   a=np.array(\
     [[         1/10,          0,                 0,                0,              0,             0, 0],\
      [        -2/81,      20/81,                 0,                0,              0,             0, 0],\
      [     615/1372,   -270/343,         1053/1372,                0,              0,             0, 0],\
      [    3243/5500,     -54/55,       50949/71500,       4998/17875,              0,             0, 0],\
      [ -26492/37125,      72/55,        2808/23375,     -24206/37125,        338/459,             0, 0],\
      [    5561/2376,     -35/11,      -24117/31603,    899983/200772,     -5225/1836,     3925/4056, 0],\
      [465467/266112, -2945/1232, -5610201/14158144, 10513573/3212352, -424325/205632, 376225/454272, 0]],\
      dtype=np.float64)

   b=np.array([61/864, 0, 98415/321776, 16807/146016, 1375/7344, 1375/5408, -37/1120, 1/10],dtype=np.float64)

if scheme=='RKF78':
   # Table 2 of bujk16
   # also https://nl.mathworks.com/matlabcentral/fileexchange/61130-runge-kutta-fehlberg-rkf78
   # not 100% clear to me which b vector I should use, but the first one yielded worse results

   a=np.array(\
     [[      2/27,    0,      0,        0,         0,       0,         0,     0,      0,     0, 0, 0, 0],\
      [      1/36, 1/12,      0,        0,         0,       0,         0,     0,      0,     0, 0, 0, 0],\
      [      1/24,    0,    1/8,        0,         0,       0,         0,     0,      0,     0, 0, 0, 0],\
      [      5/12,    0, -25/16,    25/16,         0,       0,         0,     0,      0,     0, 0, 0, 0],\
      [      1/20,    0,      0,      1/4,       1/5,       0,         0,     0,      0,     0, 0, 0, 0],\
      [   -25/108,    0,      0,  125/108,    -65/27,  125/54,         0,     0,      0,     0, 0, 0, 0],\
      [    31/300,    0,      0,        0,    61/225,    -2/9,    13/900,     0,      0,     0, 0, 0, 0],\
      [         2,    0,      0,    -53/6,    704/45,  -107/9,     67/90,     3,      0,     0, 0, 0, 0],\
      [   -91/108,    0,      0,   23/108,  -976/135,  311/54,    -19/60,  17/6,  -1/12,     0, 0, 0, 0],\
      [ 2383/4100,    0,      0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, 0, 0, 0],\
      [     3/205,    0,      0,        0,         0,   -6/41,    -3/205, -3/41,   3/41,  6/41, 0, 0, 0],\
      [-1777/4100,    0,      0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1, 0]],\
      dtype=np.float64)
   #b=np.array([0,0,0,0,0,34/105,9/35,9/35,9/280,9/280,0,41/840,41/840],dtype=np.float64)
   b=np.array([41/840,0,0,0,0,34/105,9/35,9/35,9/280,9/280,41/840,0,0],dtype=np.float64)


if scheme=='RK87': #----------------------------------------
   # The coefficients originate in Table 2 of prdo81
   # not 100% clear to me which b vector I should use

   a=np.array(\
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

   b=np.array([14005451/335480064,\
               0,0,0,0,\
               -59238493/1068277825,\
               181606767/758867731,\
               561292985/797845732,\
               -1041891430/1371343529,\
               760417239/1151165299,\
               118820643/751138087,\
               -528747749/2220607170,\
               1/4],dtype=np.float64)

###########################################################

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
print('scheme=',scheme)
print('nstep=',nstep)
print('dt=',dt)
print('diff dt:',hx**2/Du,hx**2/Dv)

###############################################################################
# create mesh 
###############################################################################
start=clock.time()

if three_dimensions:

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
   
   icon=np.zeros((m,nel),dtype=np.int32)

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

else:

   x=np.zeros(NP,dtype=np.float64)
   y=np.zeros(NP,dtype=np.float64)
   z=np.zeros(NP,dtype=np.float64)

   counter = 0
   for j in range(0,nnz):
       for i in range(0,nnx):
           x[counter]=i*Lx/float(nelx)
           z[counter]=j*Lz/float(nelz)
           counter += 1

   icon=np.zeros((m,nel),dtype=np.int32)

   counter = 0
   for j in range(0,nelz):
       for i in range(0,nelx):
           icon[0,counter]=i+j*(nelx+1)
           icon[1,counter]=i+1+j*(nelx+1)
           icon[2,counter]=i+1+(j+1)*(nelx+1)
           icon[3,counter]=i+(j+1)*(nelx+1)
           counter += 1

print("build mesh: %.3f s" % (clock.time()-start))

###############################################################################
# initial conditions for u,v,X
###############################################################################
start=clock.time()

u=np.zeros(NP,dtype=np.float64)
v=np.zeros(NP,dtype=np.float64)

if init==1: #----------------------------------------------
       
   for i in range(0,NP):
       u[i]=random.uniform(0.8,1) # close to 1
       v[i]=random.uniform(0,0.2) # close to 0 

   if three_dimensions:
      print('starting building initial conditions....')
      seed_size=0.02

      for iseed in range(nseed):
          xs=random.uniform(0+seed_size,Lx-seed_size)
          ys=random.uniform(0+seed_size,Ly-seed_size)
          zs=random.uniform(0+seed_size,Lz-seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(y[i]-ys)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 u[i]=random.uniform(0.5,0.75)
          xs=random.uniform(0+seed_size,Lx-seed_size)
          ys=random.uniform(0+seed_size,Ly-seed_size)
          zs=random.uniform(0+seed_size,Lz-seed_size)
          for i in range(0,NP):
              if abs(x[i]-xs)<seed_size and\
                 abs(y[i]-ys)<seed_size and\
                 abs(z[i]-zs)<seed_size :
                 v[i]=random.uniform(0.25,0.5)

   else:

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
                 v[i]=random.uniform(0.,0.5)

elif init==2: #----------------------------------------------
   for i in range(0,NP):
       if (x[i]-Lx/3)**2+(z[i]-Lz/3)**2<0.2:
          u[i]=1
          v[i]=0.5
       if (x[i]-Lx*0.67)**2+(z[i]-Lz/2)**2<0.2:
          u[i]=0.5
          v[i]=0.25
       if (x[i]-Lx*0.5)**2+(z[i]-Lz*0.75)**2<0.08:
          u[i]=0.7
          v[i]=0.125

elif init==3: #----------------------------------------------
   u[:]=1
   v[:]=0
   for i in range(0,NP):
       if abs(x[i]-Lx/2)<0.2 and abs(z[i]-Lz/2)<0.2:
          u[i]=1/2+random.uniform(-0.01,0.01)
          v[i]=1/4+random.uniform(-0.01,0.01)

elif init==4: #----------------------------------------------
   # original at https://github.com/cselab/gray-scott/blob/master/python/gray_scott.py
   # their domain is [-1:1]x[-1:1] so I transpose it

   for i in range(0,NP):
       xi=x[i]-Lx/2
       zi=z[i]-Lz/2
       u[i]=1-np.exp(-80*((xi+0.05)**2+(zi+0.05)**2))
       v[i]=np.exp(-80*((xi-0.05)**2+(zi-0.05)**2))

elif init==5: #----------------------------------------------
   # modified from init=4

   for i in range(0,NP):
       xi=x[i]-Lx/2 
       zi=z[i]-Lz/2 
       u[i]=1-np.exp(-80*((xi+0.05)**2+(zi+0.05)**2))
       v[i]=np.exp(-80*((xi-0.05)**2+(zi-0.05)**2))
       xi=x[i]-Lx/2 -Lx/3.3
       zi=z[i]-Lz/2 -Lz/3.3
       u[i]+=1-np.exp(-80*((xi+0.05)**2+(zi+0.05)**2))
       v[i]+=np.exp(-80*((xi-0.05)**2+(zi-0.05)**2))
       xi=x[i]-Lx/2 +Lx/3.5
       zi=z[i]-Lz/2 +Lz/10 
       u[i]+=1-np.exp(-80*((xi+0.05)**2+(zi+0.05)**2))
       v[i]+=np.exp(-80*((xi-0.05)**2+(zi-0.05)**2))
       xi=x[i]-Lx/2 -Lx/20
       zi=z[i]-Lz/2 +Lz/2.8
       u[i]+=1-np.exp(-80*((xi+0.05)**2+(zi+0.05)**2))
       v[i]+=np.exp(-80*((xi-0.05)**2+(zi-0.05)**2))





else:

   exit('unknown init parameter')

#exit()

min_u=np.min(u) ; max_u=np.max(u) ; avrg_u=np.average(u)
min_v=np.min(v) ; max_v=np.max(v) ; avrg_v=np.average(v)
print("     -> u (m,M) %f %f " %(min_u,max_u))
print("     -> v (m,M) %f %f " %(min_v,max_v))

X=np.zeros(2*NP,dtype=np.float64)
X[0:NP]=u[:]
X[NP:2*NP]=v[:]

print("initial conditions: %.3f s" % (clock.time()-start))

###############################################################################
# defining function that returns dX_dt at all nodes in 2d

@numba.njit
def F_2d(Du,Dv,F,K,NP,hx,hz,X):
    dX_dt=np.zeros(2*NP,dtype=np.float64)
    u=X[0:NP]
    v=X[NP:2*NP]

    Duhx2=Du/hx**2 ; Duhz2=Du/hz**2
    Dvhx2=Dv/hx**2 ; Dvhz2=Dv/hz**2

    counter=0
    for k in range(0,nnz):
        for i in range(0,nnx):
            if i==0:
               left=(nnx-1)+k*nnx
               right=(i+1)+k*nnx
            elif i==nnx-1:
               left=(i-1)+k*nnx
               right=(0)+k*nnx
            else:
               left=(i-1)+k*nnx
               right=(i+1)+k*nnx
            #-----------------
            if k==0:
               top=i+(k+1)*nnx
               bottom=i+(nnz-1)*nnx
            elif k==nnz-1:
               top=i+(0)*nnx
               bottom=i+(k-1)*nnx
            else:
               top=i+(k+1)*nnx
               bottom=i+(k-1)*nnx
            #-----------------
            dX_dt[counter]=Duhx2*(u[left]-2*u[counter]+u[right])\
                          +Duhz2*(u[top] -2*u[counter]+u[bottom])\
                          -u[counter]*v[counter]**2+F*(1-u[counter])
            dX_dt[counter+NP]=Dvhx2*(v[left]-2*v[counter]+v[right])\
                             +Dvhz2*(v[top] -2*v[counter]+v[bottom])\
                             +u[counter]*v[counter]**2-(F+K)*v[counter]
            counter+=1
        #end for
    #end for

    return dX_dt

###############################################################################
# defining function that returns dX_dt at all nodes in 3d

@numba.njit
def compute_node_index(i,j,k):
    return nny*nnz*i+nnz*j+k

@numba.njit
def F_3d(Du,Dv,F,K,NP,hx,hy,hz,X):
    dX_dt=np.zeros(2*NP,dtype=np.float64)
    u=X[0:NP]
    v=X[NP:2*NP]

    Duhx2=Du/hx**2 ; Duhy2=Du/hy**2 ; Duhz2=Du/hz**2
    Dvhx2=Dv/hx**2 ; Dvhy2=Dv/hy**2 ; Dvhz2=Dv/hz**2

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

    if scheme=='RK1':
       if three_dimensions:
          X[:]+=F_3d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X)*dt
       else:
          X[:]+=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X)*dt

    elif scheme=='Heun':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X   )*dt
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1)*dt
       X[:]+=(k1+k2)/2

    elif scheme=='RK2':
       if three_dimensions:
          k1=F_3d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X     )*dt
          k2=F_3d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1/2)*dt
       else:
          k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X     )*dt
          k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1/2)*dt
       X[:]+=k2

    elif scheme=='RK3':
       if three_dimensions:
          k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X                    )*dt 
          k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1*a[0,0]          )*dt
          k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1*a[1,0]+k2*a[1,1])*dt
       else:
          k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                    )*dt 
          k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[0,0]          )*dt
          k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[1,0]+k2*a[1,1])*dt
       X[:]+=(b[0]*k1+b[1]*k2+b[2]*k3)

    elif scheme=='RK4' or scheme=='RK38':
       if three_dimensions:
          k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X                              )*dt 
          k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1*a[0,0]                    )*dt
          k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1*a[1,0]+k2*a[1,1]          )*dt
          k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hy,hz,X+k1*a[2,0]+k2*a[2,1]+k3*a[2,2])*dt
       else:
          k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                              )*dt 
          k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[0,0]                    )*dt
          k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[1,0]+k2*a[1,1]          )*dt
          k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[2,0]+k2*a[2,1]+k3*a[2,2])*dt
       X[:]+=(b[0]*k1+b[1]*k2+b[2]*k3+b[3]*k4)

    elif scheme=='RKF5' or scheme=='RK5':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                                                  )*dt 
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[0,0]                                        )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[1,0]+k2*a[1,1]                              )*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[2,0]+k2*a[2,1]+k3*a[2,2]                    )*dt
       k5=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[3,0]+k2*a[3,1]+k3*a[3,2]+k4*a[3,3]          )*dt
       k6=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[4,0]+k2*a[4,1]+k3*a[4,2]+k4*a[4,3]+k5*a[4,4])*dt
       X[:]+=(b[0]*k1+b[1]*k2+b[2]*k3+b[3]*k4+b[4]*k5+b[5]*k6)

    elif scheme=='RK65':
       k1=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X                                                                      )*dt 
       k2=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[0,0]                                                            )*dt
       k3=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[1,0]+k2*a[1,1]                                                  )*dt
       k4=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[2,0]+k2*a[2,1]+k3*a[2,2]                                        )*dt
       k5=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[3,0]+k2*a[3,1]+k3*a[3,2]+k4*a[3,3]                              )*dt
       k6=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[4,0]+k2*a[4,1]+k3*a[4,2]+k4*a[4,3]+k5*a[4,4]                    )*dt
       k7=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[5,0]+k2*a[5,1]+k3*a[5,2]+k4*a[5,3]+k5*a[5,4]+k6*a[5,5]          )*dt
       k8=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[6,0]+k2*a[6,1]+k3*a[6,2]+k4*a[6,3]+k5*a[6,4]+k6*a[6,5]+k7*a[6,6])*dt
       X[:]+=(b[0]*k1+b[1]*k2+b[2]*k3+b[3]*k4+b[4]*k5+b[5]*k6+b[6]*k7+b[7]*k8)

    elif scheme=='RK87' or scheme=='RKF78':
       k1 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X)*dt 
       k2 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[0,0])*dt
       k3 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[1,0] +k2*a[1,1])*dt
       k4 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[2,0] +k2*a[2,1] +k3*a[2,2])*dt
       k5 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[3,0] +k2*a[3,1] +k3*a[3,2] +k4*a[3,3])*dt
       k6 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[4,0] +k2*a[4,1] +k3*a[4,2] +k4*a[4,3] +k5*a[4,4])*dt
       k7 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[5,0] +k2*a[5,1] +k3*a[5,2] +k4*a[5,3] +k5*a[5,4] +k6*a[5,5])*dt
       k8 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[6,0] +k2*a[6,1] +k3*a[6,2] +k4*a[6,3] +k5*a[6,4] +k6*a[6,5] +k7*a[6,6])*dt
       k9 =F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[7,0] +k2*a[7,1] +k3*a[7,2] +k4*a[7,3] +k5*a[7,4] +k6*a[7,5] +k7*a[7,6] +k8*a[7,7])*dt
       k10=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[8,0] +k2*a[8,1] +k3*a[8,2] +k4*a[8,3] +k5*a[8,4] +k6*a[8,5] +k7*a[8,6] +k8*a[8,7] +k9*a[8,8])*dt
       k11=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[9,0] +k2*a[9,1] +k3*a[9,2] +k4*a[9,3] +k5*a[9,4] +k6*a[9,5] +k7*a[9,6] +k8*a[9,7] +k9*a[9,8] +k10*a[9,9])*dt
       k12=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[10,0]+k2*a[10,1]+k3*a[10,2]+k4*a[10,3]+k5*a[10,4]+k6*a[10,5]+k7*a[10,6]+k8*a[10,7]+k9*a[10,8]+k10*a[10,9]+k11*a[10,10])*dt 
       k13=F_2d(Du,Dv,Feed,Kill,NP,hx,hz,X+k1*a[11,0]+k2*a[11,1]+k3*a[11,2]+k4*a[11,3]+k5*a[11,4]+k6*a[11,5]+k7*a[11,6]+k8*a[11,7]+k9*a[11,8]+k10*a[11,9]+k11*a[11,10]+k11*a[11,11])*dt 
       X[:]+=(b[0]*k1+b[1]*k2+b[2]*k3+b[3]*k4+b[4]*k5+b[5]*k6+b[6]*k7+b[7]*k8+b[8]*k9+b[9]*k10+b[10]*k11+b[11]*k12+b[12]*k13)

    else:

       exit('unknwon scheme')

    u[:]=X[0:NP]
    v[:]=X[NP:2*NP]

    t+=dt

    ###########################################################################
    if istep%every_ascii==0 or istep==nstep: # do stats on u,v

       min_u=np.min(u) ; max_u=np.max(u) ; avrg_u=np.average(u)
       min_v=np.min(v) ; max_v=np.max(v) ; avrg_v=np.average(v)

       print("-----------------------------")
       print("istep= ", istep,'| t=',t)
       print("     -> u (m,M) %f %f " %(min_u,max_u))
       print("     -> v (m,M) %f %f " %(min_v,max_v))
       print("     update solution: %.3f s" % (clock.time()-start))

       stats_u_file.write("%e %e %e %e\n" % (t,min_u,max_u,avrg_u)) ; stats_u_file.flush()
       stats_v_file.write("%e %e %e %e\n" % (t,min_v,max_v,avrg_v)) ; stats_v_file.flush()

       u_threshold=np.zeros(NP,dtype=np.int8)
       v_threshold=np.zeros(NP,dtype=np.int8)
       for i in range(0,NP):
           if u[i]>avrg_u: u_threshold[i]=1
           if v[i]>avrg_v: v_threshold[i]=1

       filename=model+'_solution_{:07d}'.format(istep)
       if istep==nstep: filename=model+'_solution_final'

    ###########################################################################
    if istep%every_vtu==0 or istep==nstep: # export solution to vtu format

       start=clock.time()
       vtufile=open(filename+'.vtu',"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NP,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for i in range(0,NP):
           vtufile.write("%.4e %.4e %.4e \n" %(x[i],y[i],z[i]))
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
       #vtufile.write("<DataArray type='Int8' Name='u (threshold)' Format='ascii'> \n")
       #for i in range(0,NP):
       #    vtufile.write("%d " %(u_threshold[i]))
       #vtufile.write("</DataArray>\n")
       #vtufile.write("<DataArray type='Int8' Name='v (threshold)' Format='ascii'> \n")
       #for i in range(0,NP):
       #    vtufile.write("%d " %(v_threshold[i]))
       #vtufile.write("</DataArray>\n")
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       if three_dimensions:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d %d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],\
                                                          icon[3,iel],icon[4,iel],icon[5,iel],\
                                                          icon[6,iel],icon[7,iel]))
       else:
          for iel in range (0,nel):
              vtufile.write("%d %d %d %d \n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))

       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d " %((iel+1)*m))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d " % tyype)
       vtufile.write("</DataArray>\n")
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()
       print("     export to vtu: %.3f s" % (clock.time()-start))

    ###########################################################################
    if istep%every_png==0 or istep==nstep:
       if not three_dimensions:
          start=clock.time()
          plt.imshow(np.reshape(u,(nnz,nnx)), interpolation='none',cmap='Spectral')
          plt.colorbar()
          plt.savefig(filename+'_u.png', bbox_inches='tight')
          plt.clf()
          plt.imshow(np.reshape(v,(nnz,nnx)), interpolation='none',cmap='RdBu')
          plt.savefig(filename+'_v.png', bbox_inches='tight')
          plt.clf()
          print("     export to png: %.3f s" % (clock.time()-start))


