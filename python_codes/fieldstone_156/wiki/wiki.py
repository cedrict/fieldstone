###https://en.wikipedia.org/wiki/Lorenz_system

import matplotlib.pyplot as plt
import numpy as np

def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

##############################################

dt = 0.000001
tfinal=20
num_steps = int(tfinal/dt)

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
t = np.empty(num_steps + 1)  # Need one more for the initial values

#xyzs[0] = (0., 1., 1.05)  # Set initial values wiki
xyzs[0] = (0., 0.5, 25)  # Set initial values

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
t[0]=0
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    t[i+1]=t[i]+dt

##############################################
# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*xyzs.T, lw=0.6)
ax.set_xlabel("A Axis")
ax.set_ylabel("B Axis")
ax.set_zlabel("C Axis")
ax.set_title("Lorenz Attractor")
plt.savefig('ABC.pdf')
#plt.show()
##############################################

np.savetxt('ABC.ascii',np.array([t,xyzs[:,0],xyzs[:,1],xyzs[:,2]]).T,fmt    ='%1.5e')

##############################################
