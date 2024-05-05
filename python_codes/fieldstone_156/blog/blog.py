###https://raw.githubusercontent.com/scipython/scipython-maths/master/lorenz/lorenz.py

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

# Lorenz paramters 
sigma, beta, rho = 10, 8/3, 28

#initial conditions.
#u0, v0, w0 = 0, 1, 1.05
u0, v0, w0 = 0, 0.5, 25

# Maximum time point and total number of time points.
tmax, n = 50, 50000

###############################################################################
def lorenz(t, X, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

###############################################################################
# Integrate the Lorenz equations.
# The step size is not bounded and determined solely by the solver.

maxdt=1e-5 #set to large value to bypass it

print('maximum dt=',maxdt)
print('tfinal=',tmax)

#print('KR45 method')
#soln = solve_ivp(lorenz, (0,tmax), (u0,v0,w0), args=(sigma, beta, rho),
#                 dense_output=True, max_step=maxdt)

#print('DOP853 method')
#soln = solve_ivp(lorenz, (0,tmax), (u0,v0,w0), args=(sigma, beta, rho),
#                 method='DOP853', max_step=maxdt)

#print('RK23 method')
#soln = solve_ivp(lorenz, (0,tmax), (u0,v0,w0), args=(sigma, beta, rho),
#                 method='RK23', max_step=maxdt)

print('Radau method')
soln = solve_ivp(lorenz, (0,tmax), (u0,v0,w0), args=(sigma, beta, rho),
                 method='Radau', max_step=maxdt)

#print('BDF method')
#soln = solve_ivp(lorenz, (0,tmax), (u0,v0,w0), args=(sigma, beta, rho),
#                 method='BDF', max_step=maxdt)

###############################################################################
# Interpolate solution onto the time grid, t.
#t = np.linspace(0, tmax, n)
#x, y, z = soln.sol(t)
# I replaced it by:

t=soln.t
x=soln.y[0]
y=soln.y[1]
z=soln.y[2]

# Plot the Lorenz attractor using a Matplotlib 3D projection.
WIDTH, HEIGHT, DPI = 1000, 750, 100
fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
ax = fig.gca(projection='3d')
ax.set_facecolor('k')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Make the line multi-coloured by plotting it in segments of length s which
# change in colour across the whole time series.
s = 10
cmap = plt.cm.winter
for i in range(0,n-s,s):
    ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)

# Remove all the axis clutter, leaving just the curve.
ax.set_axis_off()

plt.savefig('lorenz.png', dpi=DPI)
#plt.show()

np.savetxt('ABC.ascii',np.array([t,x,y,z]).T,fmt    ='%1.5e')
