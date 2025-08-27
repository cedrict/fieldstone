import numpy as np
import time as clock
import scipy.sparse as sps

############################################################################### 
# same as schur_complement_cg_solver.py but without preconditioner business 
# the function implicitely assumes matrices in csr format
############################################################################### 

def uzawa3_solver(K_mat,G_mat,f_rhs,h_rhs,NfemP,niter,tol):

   print('-------------------------')
   
   solP=np.zeros(NfemP,dtype=np.float64)  # guess pressure is zero.

   conv_file=open("solver_convergence.ascii","w")

   solV=sps.linalg.spsolve(K_mat,f_rhs)                             # compute V_0
   rvect_k=G_mat.T.dot(solV)-h_rhs                                  # compute r_0
   pvect_k=np.copy(rvect_k)                                         # compute p_0

   startu=clock.time()
   for k in range (0,niter): #--------------------------------------#
                                                                    #
       ptildevect_k=G_mat.dot(pvect_k)                              # 
       dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)               #
       alpha=(rvect_k.dot(rvect_k))/(ptildevect_k.dot(dvect_k))     #
       solP+=alpha*pvect_k                                          #
       solV-=alpha*dvect_k                                          #
       rvect_kp1=rvect_k-alpha*(G_mat.T.dot(dvect_k))               #
       beta=(rvect_kp1.dot(rvect_kp1))/(rvect_k.dot(rvect_k))       #
       pvect_kp1=rvect_kp1+beta*pvect_k                             #
       xiP=np.linalg.norm(alpha*pvect_k)  # i.e. norm of (Pk+1-Pk)  #
       xiV=np.linalg.norm(alpha*dvect_k)  # i.e. norm of (Vk+1-Vk)  #
       conv_file.write("%d %e %e %e \n" %(k,xiP,xiV,alpha))         #
       conv_file.flush()                                            #
       print('iter %3d xiP= %e xiV= %e' %(k,xiP,xiV))               #
       if xiP<tol and xiV<tol:                                      #
          break                                                     #
                                                                    #
       rvect_k=rvect_kp1                                            #
       pvect_k=pvect_kp1                                            #
                                                                    #
   #end for k #-----------------------------------------------------#
   endu=clock.time()

   conv_file.close()

   print('time per iteration:',(endu-startu)/k,NfemP)
   print('-------------------------')
    
   return solV,solP,k

############################################################################### 
