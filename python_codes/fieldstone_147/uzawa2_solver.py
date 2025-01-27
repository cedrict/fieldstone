import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla

############################################################################### 
#the function implicitely assumes matrices in csr format

def uzawa2_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,\
                  NfemV,NfemP,niter,tol,use_precond,inner):

   print('-------------------------')

   # we assume that the guess/starting pressure is zero.
   solP=np.zeros(NfemP,dtype=np.float64)  

   solV=sps.linalg.spsolve(K_mat,f_rhs-G_mat.dot(solP))    

   conv_file=open("solver_convergence.ascii","w")

   for k in range (0,niter): #--------------------------------------#
                                                                    #
       qk=h_rhs-G_mat.T.dot(solV)                                   #
       pk=G_mat.dot(qk)                                             #
       Hk=sps.linalg.spsolve(K_mat,pk)                              # 
       alphak=qk.dot(qk)/pk.dot(Hk)                                 #
       solPnew=solP-alphak*qk                                       #
       solVnew=solV+alphak*Hk                                       #
                                                                    #
       xiP=np.linalg.norm(solPnew-solP)                             #
       xiV=np.linalg.norm(solVnew-solV)                             #
       conv_file.write("%d %e %e %e \n" %(k,xiP,xiV,alphak))        #
       conv_file.flush()                                            #
       print('iter %3d xiP= %e xiV= %e' %(k,xiP,xiV))               #
       if xiP<tol and xiV<tol:                                      #
          break                                                     #
                                                                    #
       solP[:]=solPnew[:]                                           #
       solV[:]=solVnew[:]                                           #
                                                                    #
   #end for k #-----------------------------------------------------#

   conv_file.close()

   print('-------------------------')
    
   return solV,solP,k

############################################################################### 

