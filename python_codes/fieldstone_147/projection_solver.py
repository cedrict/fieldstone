import numpy as np
import scipy.sparse as sps
#import scipy.sparse.linalg as sla

############################################################################### 
#the function implicitely assumes matrices in csr format

def projection_solver(K_mat,G_mat,GTG_mat,f_rhs,h_rhs,\
                 NfemV,NfemP,niter,tol):

   print('-------------------------')

   # we assume that the guess/starting pressure is zero.
   solP=np.zeros(NfemP,dtype=np.float64)  
   Vtilde=np.zeros(NfemV,dtype=np.float64)  

   conv_file=open("solver_convergence.ascii","w")

   for k in range (0,niter): #--------------------------------------#
                                                                    #
       #step 1                                                      #
       Vtilde=sps.linalg.spsolve(K_mat,f_rhs-G_mat.dot(solP))       # 

       #step 2                                                      #

       Q=sps.linalg.spsolve(GTG_mat,h_rhs-G_mat.T.dot(Vtilde))    

       #step 3                                                      #

       solV=Vtilde+G_mat@Q
                                                                    #
       #step 4                                                      #
       solPnew=solP+(G_mat.T.dot(Vtilde)-h_rhs)                     #
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

