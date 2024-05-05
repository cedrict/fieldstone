import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
from scipy.linalg import eig

def lorenz(xyz, t=0, s=10, r=28, b=8/3):
    x,y,z = xyz
    xp = s*(y - x)
    yp = r*x - y - x*z
    zp = x*y - b*z
    return xp, yp, zp

tfinal=25
dt=1e-2
nstep=int(tfinal/dt)

ts = np.linspace(0,tfinal,nstep)
#sol = odeint(lorenz, [0,1,1.05], ts)
sol = odeint(lorenz, [0,0.5,25], ts)
xs,ys,zs = sol.T

fig = plt.figure(figsize=(16,10))
ax = fig.gca(projection='3d')
ax.plot(xs, ys, zs, linewidth=0.3)
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_zlabel("C")
#plt.show()
plt.savefig('ABC.pdf')


np.savetxt('ABC.ascii',np.array([ts,xs,ys,zs]).T,fmt    ='%1.5e')

#=======================================


