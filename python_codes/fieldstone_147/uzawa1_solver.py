import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla

############################################################################### 
#the function implicitely assumes matrices in csr format

def uzawa1_solver(K_mat,G_mat,f_rhs,h_rhs,NfemP,niter,tol,omega):

   print('-------------------------')

   # we assume that the guess/starting pressure is zero.
   solP=np.zeros(NfemP,dtype=np.float64)  

   conv_file=open("solver_convergence.ascii","w")

   for k in range (0,niter): #--------------------------------------#
                                                                    #
       #step 1                                                      #
       solV=sps.linalg.spsolve(K_mat,f_rhs-G_mat.dot(solP))         # 
                                                                    #
       #step 2                                                      #
       solPnew=solP+omega*(G_mat.T.dot(solV)-h_rhs)                 #
                                                                    #
       xi=np.linalg.norm(solPnew-solP)                              #
       conv_file.write("%d %6e \n"  %(k,xi))                        #
       conv_file.flush()                                            #
       print('iter %3d xi= %e' %(k,xi))                             #
       if xi<tol:                                                   #
          break                                                     #
                                                                    #
       solP[:]=solPnew[:]                                           #
                                                                    #
   #end for k #-----------------------------------------------------#

   conv_file.close()

   print('-------------------------')
    
   return solV,solP,k

############################################################################### 

