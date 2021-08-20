import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import matplotlib.pyplot as plt

mm=1e-3
year=365.25*3600*24

###########################################################

Lx=20000e3        # m
nnx=4001          # must be odd for experiment 1
thickness=80e3     # m
nu=0.25           # dimensionless
E=70e9            # Pa
g=9.81            # m/s^2
drho=2850         # kg/m^3
theta=0/180*np.pi # angle (rad)
b=15e3/2          # semi height of channel
eta=2e18          # viscosity (Pa)
U=100*mm/year     # velocity
a=200e3           # radius of obstacle

P=0
###########################################################

nelx=nnx-1
hx=Lx/nelx
D=E*thickness**3/12/(1-nu**2)             
alpha=(4*D/(drho)/g)**0.25
kappa=b**2/3/eta

print ('D=',D)
print ('alpha=',alpha,'m')
print ('kappa=',kappa)

###########################################################

x = np.empty(nnx,dtype=np.float64)
for i in range(0,nnx):
    x[i]=i*hx

###########################################################

A = np.zeros((nnx,nnx),dtype=np.float64)
rhs = np.zeros((nnx),dtype=np.float64)

for i in range(0,nnx):

    A[i,i]+=(drho)*g

    if i==0: 
       # w=0 at x=0
       A[0,0]=D # improves condition number
       rhs[0]=0
    elif i==1:
       # dw/dx=0 at x=0
       A[1,0]-=D/hx
       A[1,1]+=D/hx
       rhs[1]=0
    elif i==nnx-2:
       # dw/dx=0 at x=Lx
       A[nnx-2,nnx-2]=-D/hx
       A[nnx-2,nnx-1]=+D/hx
       rhs[nnx-2]=0
    elif i==nnx-1:
       # w=0 at x=Lx
       A[nnx-1,nnx-1]=D # improves condition number
       rhs[nnx-1]=0
    else:
       #https://en.wikipedia.org/wiki/Finite_difference_coefficient
       # fourth order derivative coefficient:  1 -4 6 -4 1 (centered)
       A[i,i-2]+=1*D/hx**4
       A[i,i-1]-=4*D/hx**4
       A[i,i  ]+=6*D/hx**4
       A[i,i+1]-=4*D/hx**4
       A[i,i+2]+=1*D/hx**4
       # second order derivative: 1 -2 1 (centered)
       A[i,i-1]+=1*P/hx**2
       A[i,i  ]-=2*P/hx**2
       A[i,i+1]+=1*P/hx**2
    #end if

    if abs(x[i]-Lx/2)>a:
       rhs[i]=-U*a**2/kappa/(x[i]-Lx/2)*np.cos(theta) 
    else:
       rhs[i]=0

#end for

#export matrix nonzero structure
#plt.spy(A, markersize=2.5)
#plt.savefig('matrix.png', bbox_inches='tight')
#plt.clf()

sol = sps.linalg.spsolve(sps.csr_matrix(A),rhs)

np.savetxt('w.ascii',np.array([x/1e3,sol]).T,header='# r,w')
