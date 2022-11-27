#!/usr/bin/env python3

"""
Translation of the matlab code 'erosion1d.m', published in (Simpson, 2017).

This code use FEM to compute solution for 1D erosion equations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def main():
    seconds_per_yr = 60*60*24*365

    # physical parameters
    lx = 1e4 # length of spatial domain [m]
    w = 1e-3 / seconds_per_yr # uplift source term [m.s⁻¹]
    c0 = 1e-8 # linear "hillslope" erosion coefficient [m².s⁻¹]
    rexp = 2 # fluvial erosion exponent (discharge term)
    mexp = 1 # fluvial erosion exponent (slope term)
    c = 1e-13 # fluvial erosion coefficient [m^(2-rexp).s⁻¹]

    # numerical parameters
    dt = 1000 * seconds_per_yr # time step [s]
    tolerance = 1e-3 # error tolerance [-]
    ntime = 5000 # number of time step
    nels = 50 # number of elements
    nod = 2 # number of nodes per elements
    nn = nels+1 # total number of nodes
    dx = lx/nels # element size
    nip = 2 # number of integration points
    g_coord = np.arange(0, lx+dx, dx) # spatial domain (1D mesh)

    # integration data
    points = np.array([-np.sqrt(1/3),np.sqrt(1/3)]) # position of the Gauss points
    wts = np.array([1,1]) # Gauss-Legendre weights

    # shape function and their derivatives (both in local coordinates)
    # evaluated at the Gauss integration points
    fun_s = np.zeros(shape=(nip,2))
    der_s = np.zeros(shape=(nip,2))
    for k in range(0,nip):
        fun_s[k] = [ (1-points[k]) /2 , (1+points[k]) /2] # N1 and N2
        der_s[k] = [-1/2 , 1/2] # dN1.xi dN2/xi

    # Dirichlet boundary condition
    # bcdof = nn-1 # will take the last row by default (index -1) TODO: rework to work with sparse matrix for speed
    bcval = 0 # boundary value

    # define connectivity and equation numbering
    g_num = np.zeros((nod,nels), dtype=int) # initialisation
    g_num[0] = np.arange(0,nn-1)
    g_num[1] = np.arange(1,nn)

    # --------- #
    # time loop #
    # --------- #
    # initialisation
    b = np.zeros(nn) # system rhs vector initialisation
    displ = np.zeros(nn) # initial solution array
    displ0 = displ # solution at n time
    time = 0 # initialisation of time

    for i in range(1,ntime+1):
        time = time + dt # time counter
        iters = 0 # iteration counter reset
        error = 1 # error reset
        while error >= tolerance: # start of nonlinear iteration
            iters = iters + 1 # iteration counter

            # matrix assembly
            ff = np.zeros(nn) # system load vector
            lhs = np.zeros((nn, nn)) # system lhs matrix
            rhs = np.zeros((nn, nn)) # system rhs matrix
            # rhs = sp.csr_matrix((nn, nn)) # TODO: add sparse matrix to gain computation time
            for iel in range(nels): # loop over elements
                num_temp = g_num[:, iel] # retrieve equation number
                num = slice(num_temp[0], num_temp[1]+1) # storing slice for equation number
                dx = np.abs(np.diff(g_coord[num])).item() # length of element
                MM = np.zeros((nod, nod)) # reinitialise MM (mass matrix)
                KM = np.zeros((nod, nod)) # reinitialise KM (stiffness matrix)
                F = np.zeros(nod) # reinitialise F (load vector)
                for k in range(nip): # integrate element matrix
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
                    KM = KM + deriv[:,None] * kappa * deriv * dwt # stuffness matrix
                    F = F + w * fun * dwt # load vector
                lhs[num, num] = lhs[num, num] + MM / dt + KM # assemble lhs
                rhs[num, num] = rhs[num, num] + MM / dt # assemble rhs
                ff[num] = ff[num] + F # assemble load vector
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
            ax.set_title('Time = ' + str(time/seconds_per_yr) + ' years')
            ax.set_xlabel('Distance away from drainage divide [m]')
            ax.set_ylabel('Surface height [m]')
            ax.plot(g_coord, displ, 'o-')
            plt.savefig('solution'+str(i)+'.pdf', bbox_inches='tight')
            #plt.show()


if __name__ == "__main__":
    print('Starting main script')
    print('========================\n')
    main()
    print('\n========================')
    print('Done!')
    sys.exit(0)
