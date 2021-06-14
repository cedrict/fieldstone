import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import matplotlib.pyplot as plt

###########################################################
Lx=1200e3      # m
nnx=201         # must be odd for experiment 1
thickness=10e3 # m
nu=0.25        # dimensionless
E=1e11         # Pa
P=0
g=9.81         #m/s^2
rhom=3250      #kg/m^3
rhoc=2800

#1 line load (F)
#2 periodic (h0,laambda)
#3 single cosine (h0,laambda)
experiment=2
F=-1e10
h0=150
laambda=0.5*Lx

###########################################################

nelx=nnx-1
hx=Lx/nelx
D=E*thickness**3/12/(1-nu**2)
alpha=(4*D/(rhom-rhoc)/g)**0.25

print ('D=',D)
print ('alpha=',alpha,'m')

###########################################################

x = np.empty(nnx,dtype=np.float64)
for i in range(0,nnx):
    x[i]=i*hx

###########################################################

A = np.zeros((nnx,nnx),dtype=np.float64)
rhs = np.zeros((nnx),dtype=np.float64)

for i in range(0,nnx):

    A[i,i]+=(rhom-rhoc)*g

    if i==0: 
       # w=0 at x=0
       A[0,0]=D # improves condition number
       rhs[0]=0

    elif i==1:
       if experiment==1 or experiment==3:
          # dw/dx=0 at x=0
          A[1,0]-=D/hx
          A[1,1]+=D/hx
          rhs[1]=0
       else: # pure forward - could be better
          A[i,i+0]+=1*D/hx**4
          A[i,i+1]-=4*D/hx**4
          A[i,i+2]+=6*D/hx**4
          A[i,i+3]-=4*D/hx**4
          A[i,i+4]+=1*D/hx**4


    elif i==nnx-2:
       if experiment==1 or experiment==3:
          # dw/dx=0 at x=Lx
          A[nnx-2,nnx-2]=-D/hx
          A[nnx-2,nnx-1]=+D/hx
          rhs[nnx-2]=0
       else: # pure backward - could be better
          A[i,i-4]+=1*D/hx**4
          A[i,i-3]-=4*D/hx**4
          A[i,i-2]+=6*D/hx**4
          A[i,i-1]-=4*D/hx**4
          A[i,i  ]+=1*D/hx**4

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

    if experiment==1:
       #point load in the middle
       if i==int(nnx/2):
          rhs[i]=F/hx  #force per unit surface
    if experiment==2:
       rhs[i]=rhoc*g*h0*np.sin(2*np.pi*x[i]/laambda)
    if experiment==3 and abs(x[i]-Lx/2)<laambda/2:
       rhs[i]=-rhoc*g*h0*(1+np.cos(2*np.pi*(x[i]-Lx/2)/laambda))

#end for

#export matrix nonzero structure
#plt.spy(A, markersize=2.5)
#plt.savefig('matrix.png', bbox_inches='tight')
#plt.clf()

sol = sps.linalg.spsolve(sps.csr_matrix(A),rhs)

###########################################################
w_analytical = np.zeros(nnx,dtype=np.float64)

if experiment==1:
   for i in range(0,nnx):
       w_analytical[i]=F*alpha**3/8/D*np.exp(-abs(x[i]-Lx/2)/alpha) *\
                       (np.cos((x[i]-Lx/2)/alpha)+np.sin(abs(x[i]-Lx/2)/alpha))
if experiment==2:
   for i in range(0,nnx):
       w0=h0/(rhom/rhoc-1+D/rhoc/g*(2*np.pi/laambda)**4)
       w_analytical[i]=w0*np.sin(2*np.pi*x[i]/laambda)
   

np.savetxt('w.ascii',np.array([x,sol,w_analytical,rhs]).T,header='# x,w,analytical, rhs')


