import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time as time
from scipy.sparse.linalg.dsolve import linsolve
import scipy.sparse as sps

#------------------------------------------------------------------------------

nel=32       # Number of elements
nnx=nel+1    # Number of nodes
Lx=1         # Horizontal extent of the domain
niter=5000    # Maximum number of timesteps
hx=Lx/nel    # Size of element  
abstol=1.e-8 # Tolerance

# Boundary conditions
T_left=0.
T_right=1.

# Stablization constants
E=4
C=0.5

#------------------------------------------------------------------------------
# Declare arrays

x=np.linspace(0,Lx,nnx) # x coordinates

T_min=np.zeros(nnx,dtype=np.float64)        # New Temperature at negative side of node
T_plus=np.zeros(nnx,dtype=np.float64)       # New Temperature at positive side of node
q_min=np.zeros(nnx,dtype=np.float64)        # New flux at negative side of node
q_plus=np.zeros(nnx,dtype=np.float64)       # New flux at positive side of node

T_min_old=np.zeros(nnx,dtype=np.float64)    # Old Temperature at negative side of node
T_plus_old=np.zeros(nnx,dtype=np.float64)   # Old Temperature at positive side of node
q_min_old=np.zeros(nnx,dtype=np.float64)    # Old flux at negative side of node
q_plus_old=np.zeros(nnx,dtype=np.float64)   # Old flux at positive side of node

# Boundary conditions
T_min[0]=T_left
T_plus[nnx-1]=T_right 


conv_file=open('convergence.ascii',"w")

################################################################################################
################################################################################################
# TIME STEPPING
################################################################################################
################################################################################################
start=time.time()

rhs_el=np.zeros(4,dtype=np.float64)     # elemental right hand side vector  
        
K=np.array([[hx/3,hx/6,-C ,-0.5],
            [hx/6,hx/3,0.5,-C  ],
            [C   ,-0.5,E  ,0   ],
            [0.5 ,C   ,0  ,E   ]]) 

K_left=np.array([[hx/3,hx/6,-0.5,-0.5],
                 [hx/6,hx/3, 0.5,-C  ],
                 [0.5 ,-0.5, E  ,0   ],
                 [0.5 ,C   , 0  ,E   ]]) 

K_right=np.array([[hx/3,hx/6,-C ,-0.5],
                  [hx/6,hx/3,0.5,0.5 ],
                  [C   ,-0.5,E  ,0   ],
                  [0.5 ,-0.5,0  ,E   ]]) 

for it in range(0,niter):
    print("----------------------------------")
    print("iter= ", it)
    print("----------------------------------")

    #update q flux at boundaries
    q_min[0]=q_plus[0]-E*(T_min[0]-T_plus[0])                 # left boundary
    q_plus[nnx-1]=q_min[nnx-1]-E*(T_min[nnx-1]-T_plus[nnx-1]) # right boundary
  
    # Iteration sweep from left to right 
    for iel in range(0,nel):

        #extract connectivity nodes
        k=iel
        kp1=iel+1

        if iel==0:
           K_el=K_left
           rhs_el[0]=-T_min[k]
           rhs_el[1]=(1/2-C)*T_plus[k+1]
           rhs_el[2]=E*T_min[k]
           rhs_el[3]=(1/2+C)*q_plus[k+1]+E*T_plus[k+1]
        elif iel==nel-1:
           K_el=K_right
           rhs_el[0]=-(1/2+C)*T_min[k]
           rhs_el[1]=T_plus[k+1]
           rhs_el[2]=-(1/2-C)*q_min[k]+E*T_min[k]
           rhs_el[3]=E*T_plus[k+1]
        else:
           K_el=K
           rhs_el[0]=-(1/2+C)*T_min[k]
           rhs_el[1]= (1/2-C)*T_plus[k+1]
           rhs_el[2]=-(1/2-C)*q_min[k]   +E*T_min[k]
           rhs_el[3]= (1/2+C)*q_plus[k+1]+E*T_plus[k+1]
        #end if 

        sol = sps.linalg.spsolve(K_el,rhs_el)
 
        q_plus[k]  =sol[0]
        q_min [kp1]=sol[1]
        T_plus[k]  =sol[2]
        T_min [kp1]=sol[3]

    #end iel
    
    # Calculate average Temperature and flux   
    #T=(T_min+T_plus)/2
    #q=(q_min+q_plus)/2

    filename = 'T_minus_{:04d}.ascii'.format(it) 
    np.savetxt(filename,np.array([x,T_min]).T,header='# x,T_min')
    filename = 'q_minus_{:04d}.ascii'.format(it) 
    np.savetxt(filename,np.array([x,q_min]).T,header='# x,q_min')

    conv_file.write("%d %e %e %e %e\n" %(it,np.max(T_min-T_min_old),np.max(T_plus-T_plus_old),np.max(q_min-q_min_old),np.max(q_plus-q_plus_old)) ) ; conv_file.flush()

    #test convergence
    if np.max(T_min-T_min_old)<abstol and\
       np.max(T_plus-T_plus_old)<abstol and\
       np.max(q_min-q_min_old)<abstol and\
       np.max(q_plus-q_plus_old)<abstol:
       print("iterations have converged after ",it,'iterations')
       break

    #np.savetxt('T_min_diff.ascii',np.array([x,T_min-T_min_old]).T,header='# x,T_min')
    #np.savetxt('T_plus_diff.ascii',np.array([x,T_plus-T_plus_old]).T,header='# x,T_plus')
    #np.savetxt('q_min_diff.ascii',np.array([x,q_min-q_min_old]).T,header='# x,q_min')
    #np.savetxt('q_plus_diff.ascii',np.array([x,q_plus-q_plus_old]).T,header='# x,q_plus')

    # Save values of last iteration
    T_min_old[:] =T_min[:]
    T_plus_old[:]=T_plus[:]
    q_min_old[:] =q_min[:]
    q_plus_old[:]=q_plus[:]

#end for it

np.savetxt('T_minus.ascii',np.array([x,T_min]).T,header='# x,T_min')
np.savetxt('T_plus.ascii',np.array([x,T_plus]).T,header='# x,T_plus')
np.savetxt('q_minus.ascii',np.array([x,q_min]).T,header='# x,q_min')
np.savetxt('q_plus.ascii',np.array([x,q_plus]).T,header='# x,q_plus')


