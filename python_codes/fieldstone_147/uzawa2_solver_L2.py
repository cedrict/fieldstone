import numpy as np
import time as clock
import scipy.sparse as sps

############################################################################### 
#the function implicitely assumes matrices in csr format
############################################################################### 

def uzawa2_solver_L2(K_mat,G_mat,MP_mat,H_mat,f_rhs,h_rhs,NfemP,niter,tol):

   print('-------------------------')
   
   solP=np.zeros(NfemP,dtype=np.float64) # guess pressure is zero.

   solV=sps.linalg.spsolve(K_mat,f_rhs-G_mat.dot(solP))    

   conv_file=open("solver_convergence.ascii","w")

   startu=clock.time()
   for k in range (0,niter): #--------------------------------------#
                                                                    #
       rhs=h_rhs-G_mat.T.dot(solV)                                  #
       qk=sps.linalg.spsolve(MP_mat,rhs,use_umfpack=False)          #
                                                                    #
       pk=G_mat.dot(qk)                                             #
       Hk=sps.linalg.spsolve(K_mat,pk)                              # 
                                                                    #
       numerator=qk.dot(MP_mat.dot(qk))                             #
       denominator=qk.dot(H_mat.dot(Hk))                            #
       alphak=numerator/denominator                                 # 
                                                                    #
       solPnew=solP-alphak*qk                                       #
       solVnew=solV+alphak*Hk                                       #
                                                                    #
       xiP=np.linalg.norm(solPnew-solP)                             #
       xiV=np.linalg.norm(solVnew-solV)                             #
       conv_file.write("%d %e %e %e \n" %(k,xiP,xiV,alphak))        #
       conv_file.flush()                                            #
       print('iter %3d xiP= %e xiV= %e' %(k,xiP,xiV))               #
                                                                    #
       solP[:]=solPnew[:]                                           #
       solV[:]=solVnew[:]                                           #
                                                                    #
       if xiP<tol and xiV<tol:                                      #
          break                                                     #
                                                                    #
   #end for k #-----------------------------------------------------#
   endu=clock.time()

   conv_file.close()

   print('time per iteration:',(endu-startu)/k,NfemP)
   print('-------------------------')
    
   return solV,solP,k

############################################################################### 
