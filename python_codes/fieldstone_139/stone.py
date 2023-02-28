# Translation of the matlab code 'erosion1d.m', published in (Simpson, 2017).
# This code use FEM to compute solution for 1D erosion equations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

year = 60*60*24*365

###############################################################################
# physical parameters

Lx = 1e4        # length of spatial domain [m]
w = 1e-3 / year # uplift source term [m.s⁻¹]
c0 = 1e-8       # linear "hillslope" erosion coefficient [m².s⁻¹]
rexp = 2        # fluvial erosion exponent r (discharge term)
mexp = 1        # fluvial erosion exponent m (slope term)
c = 1e-13       # fluvial erosion coefficient [m^(2-rexp).s⁻¹]

###############################################################################
# numerical parameters

dt = 1000 * year  # time step [s]
tolerance = 1e-3  # error tolerance [-]
ntime = 5000      # number of time step
nel = 50          # number of elements
m = 2             # number of nodes per elements
N = nel+1        # total number of nodes
dx = Lx/nel       # element size
nqel = 2           # number of integration points

###############################################################################
# mesh nodes

g_coord = np.arange(0, Lx+dx, dx) # spatial domain (1D mesh)

###############################################################################
# connectivity



###############################################################################
# integration data
points = np.array([-np.sqrt(1/3),np.sqrt(1/3)]) # position of the Gauss points
wts = np.array([1,1]) # Gauss-Legendre weights

qcoords_r=np.array([-np.sqrt(1/3),np.sqrt(1/3)],dtype=np.float64)
qweights = np.array([1,1],dtype=np.float64) 


# shape function and their derivatives (both in local coordinates)
# evaluated at the Gauss integration points
fun_s = np.zeros(shape=(nqel,2))
der_s = np.zeros(shape=(nqel,2))
for k in range(0,nqel):
    fun_s[k] = [ (1-points[k]) /2 , (1+points[k]) /2] # N1 and N2
    der_s[k] = [-1/2 , 1/2] # dN1.xi dN2/xi

# Dirichlet boundary condition
# bcdof = nn-1 # will take the last row by default (index -1) TODO: rework to work with sparse matrix for speed
bcval = 0 # boundary value

# define connectivity and equation numbering
g_num = np.zeros((m,nel), dtype=int) # initialisation
g_num[0] = np.arange(0,N-1)
g_num[1] = np.arange(1,N)

#******************************************************************************
#******************************************************************************
# time loop #
#******************************************************************************
#******************************************************************************


# initialisation
b = np.zeros(N) # system rhs vector initialisation
displ = np.zeros(N) # initial solution array
displ0 = displ # solution at n time
time = 0 # initialisation of time

for i in range(1,ntime+1):
    time = time + dt # time counter
    error = 1 # error reset
    while error >= tolerance: # start of nonlinear iteration

        # matrix assembly
        ff = np.zeros(N) # system load vector
        lhs = np.zeros((N,N)) # system lhs matrix
        rhs = np.zeros((N,N)) # system rhs matrix

        for iel in range(nel): # loop over elements
            num_temp = g_num[:, iel] # retrieve equation number
            num = slice(num_temp[0], num_temp[1]+1) # storing slice for equation number
            dx = np.abs(np.diff(g_coord[num])).item() # length of element
            MM = np.zeros((m,m)) # reinitialise MM (mass matrix)
            KM = np.zeros((m,m)) # reinitialise KM (stiffness matrix)
            F = np.zeros(m) # reinitialise F (load vector)
            for k in range(nqel): # integrate element matrix
                fun = fun_s[k, :] # retrieve shape function
                der = der_s[k, :] # retrieve shape fun. der.
                detjac = dx/2 # det. of the Jacobian
                invjac = 2/dx # inv of the Jacobian
                deriv = der * invjac # shape function derivative in global coordinates
                xk = fun @ g_coord[num] # x coord. at int. pt. k
                S = np.abs(deriv @ displ[num]) # topographic slope
                kappa = c0 + c * xk**rexp * S**(mexp - 1) # diffusivity
                dwt = detjac * wts[k] # multiplier
                MM = MM + fun[:,None]*fun * dwt # mass matrix
                    # note that python will not tranpose 1d array so we this is why we add a dimension
                KM+= deriv[:,None] * kappa * deriv * dwt # stuffness matrix
                F+=  w * fun * dwt # load vector
            #end for

            #boundary conditions

            #assembly

            lhs[num, num] = lhs[num, num] + MM / dt + KM # assemble lhs
            rhs[num, num] = rhs[num, num] + MM / dt # assemble rhs
            ff[num] = ff[num] + F # assemble load vector
        #end while 
        b = rhs @ displ0 + ff # form rhs vector

        # impose boundary condition
        lhs[-1, :] = 0 # zero the relevant equations
        lhs[-1,-1] = 1 # place 1 on boundary position of diagonal
        b[-1] = bcval # set boundary value
        displ_tmp = displ.copy() # store previous solution
        displ = np.linalg.solve(lhs,b) # compute new solution of system eq.
        error = max(abs(displ-displ_tmp)) / max(abs(displ)) # compute error
        print('Time step: ', i, '| Error: ', error)
    displ0 = displ.copy() # save solution from last step

if (i % 100 == 0):
   fig = plt.figure()
   ax = fig.add_subplot(1, 1, 1)
   ax.set_title('Time = ' + str(time/year) + ' years')
   ax.set_xlabel('Distance away from drainage divide [m]')
   ax.set_ylabel('Surface height [m]')
   ax.plot(g_coord, displ, 'o-')
   plt.savefig('solution'+str(i)+'.pdf', bbox_inches='tight')
   #plt.show()

   np.savetxt('solution_'+str(i)+'.ascii',np.array([x,displ]).T)

