import numpy as np
import time as clock
import scipy.sparse as sps
import scipy.sparse.linalg as sla

############################################################################### 
# the function implicitely assumes matrices in csr format
############################################################################### 

def uzawa1_solver_L2(K_mat,G_mat,MP_mat,f_rhs,h_rhs,NfemP,niter,tol,omega):

   print('-------------------------')

   solP=np.zeros(NfemP,dtype=np.float64) # guess pressure is zero.

   conv_file=open("solver_convergence.ascii","w")

   startu=clock.time()
   for k in range (0,niter): #--------------------------------------#
                                                                    #
       #step 1                                                      #
       solV=sps.linalg.spsolve(K_mat,f_rhs-G_mat.dot(solP))         # 
                                                                    #
       #step 2                                                      #
       rhs=MP_mat.dot(solP)+omega*(G_mat.T.dot(solV)-h_rhs)         #
       solPnew=sps.linalg.spsolve(MP_mat,rhs,use_umfpack=False)     #
                                                                    #
       xi=np.linalg.norm(solPnew-solP)                              #
       conv_file.write("%d %6e \n" %(k,xi)) ; conv_file.flush()     #
       print('iter %3d xi= %e' %(k,xi))                             #
                                                                    #
       solP[:]=solPnew[:]                                           #
                                                                    #
       if xi<tol:                                                   #
          break                                                     #
                                                                    #
   #end for k #-----------------------------------------------------#
   endu=clock.time()

   conv_file.close()

   print('time per iteration:',(endu-startu)/k,NfemP)
   print('-------------------------')
    
   return solV,solP,k

############################################################################### 
