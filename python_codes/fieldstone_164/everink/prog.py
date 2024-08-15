import matplotlib.pyplot as plt;
import numpy as np;
import numpy.linalg as linalg;
import math;


# Amount of elements
N = 20;

# Amount of nodes
n = N + 1;

# Element size
h = 1.0/N;

# Constant
c = 1;

# Delta time
dt = 0.001;

# Amount of iterations
iterations = 5000 #20000;

# Stepsize
stepSize = 100;

# Time coefficient matrix
T = np.zeros((n, n));

T[0, 0] = 1; # Left boundary
T[N, N] = 1; # Right boundary

for i in range(1, N):
    for j in range(0, N+1):
        if(i==j):
            T[i, j] = (2.0/3.0)*h;
        if(abs(i-j) == 1):
            T[i, j] = (1.0/6.0)*h;

# Space coefficient matrix
S = np.zeros((n, n));

for i in range(1, N):
    for j in range(0, N+1):
        if(i==j):
            S[i, j] = (2.0/h);
        if(abs(i-j)== 1):
            S[i, j] = -(1.0/h);


# A single time step
def iteration(v, vDer):
    vNew = v + dt*vDer;
    q = -c*c*S@v;
    r = linalg.solve(T, q);
    vDerNew = vDer + dt*r;
    return (vNew, vDerNew);


# The real solution
def realU(x, t):
    return np.cos(2*np.pi*t)*np.sin(2*np.pi*x);

# The initial value of the finite element problem
u = np.zeros((n, 1));
uDer = np.zeros((n, 1));

for i in range(0, n):
    x = i*h;
    u[i] = realU(x, 0);


data = [];
data.append(u);
spacing = np.linspace(0.0, 1.0, n);
bigspacing = np.linspace(0.0, 1.0, 100);
    
for i in range(0, iterations):
    print (i)
    (uNew, uDerNew) = iteration(u, uDer);
    u = uNew;
    uDer = uDerNew;
    data.append(u);

    #print (np.shape(u))
    x=np.linspace(0,1,N+1)
    #sol=np.zeros(N+1,dtype=np.float64)
    #for i in range(0,N+1):
    #    sol[i]=u[i,0]    
    filename = 'u_{:04d}.ascii'.format(i) 
    np.savetxt(filename,np.array([x,u[:,0]]).T,header='# x,u')

def animation(t):
    plt.rcParams["figure.figsize"] = (6,3);
    axes = plt.gca();
    axes.set_ylim([-1.5,1.5]);
    axes.set_xlim([0,1]);
    plt.title("t = " + str(t*dt));
    plt.ylabel("u(x,t)");
    plt.xlabel("x");
    plt.plot(bigspacing, realU(bigspacing, t*dt), "r--");
    plt.plot(spacing, data[t], "b-");

