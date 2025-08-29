import numpy as np
import time as clock
import scipy.sparse as sps
import scipy.sparse.linalg as sla

############################################################################### 
# same as schur_complement_cg_solver.py but without preconditioner business 
#the function implicitely assumes matrices in csr format
############################################################################### 

def uzawa3_solver_L2(K_mat,G_mat,MP_mat,H_mat,f_rhs,h_rhs,NfemP,niter,tol,inner):

   timeK=0.

   print('-------------------------')

   solP=np.zeros(NfemP,dtype=np.float64) #  guess pressure is zero.

   conv_file=open("solver_convergence.ascii","w")

   if inner=='direct':
      solV=sps.linalg.spsolve(K_mat,f_rhs)                             # compute V_0
   elif inner=='splu':
      LU = sla.splu(K_mat)
      solV=LU.solve(f_rhs)
   else:
      exit('unknown inner solver')

   rvect_k=sps.linalg.spsolve(MP_mat,G_mat.T.dot(solV)-h_rhs)       # compute r_0
   pvect_k=np.copy(rvect_k)                                         # compute p_0

   startu=clock.time()
   for k in range (0,niter): #--------------------------------------#
                                                                    # 
       #AAA                                                         #
       startK=clock.time()                                          #
       if inner=='direct':                                          #
          dvect_k=sps.linalg.spsolve(K_mat,G_mat.dot(pvect_k))      #
       elif inner=='splu':                                          #
          dvect_k=LU.solve(G_mat.dot(pvect_k))                      #
       timeK+=clock.time()-startK                                   #
       #BBB                                                         #
       numerator=rvect_k.dot(MP_mat.dot(rvect_k))                   #
       denominator=pvect_k.dot(H_mat.dot(dvect_k))                  #
       alpha=numerator/denominator                                  #
       #CCC                                                         #
       solP+=alpha*pvect_k                                          #
       #DDD                                                         #
       solV-=alpha*dvect_k                                          #
       #EEE                                                         #
       dr=sps.linalg.spsolve(MP_mat,-alpha*G_mat.T.dot(dvect_k))    #
       rvect_kp1=rvect_k+dr                                         #
       #FFF                                                         #
       numerator=rvect_kp1.dot(MP_mat.dot(rvect_kp1))               #
       denominator=rvect_k.dot(MP_mat.dot(rvect_k))                 #
       beta=numerator/denominator                                   #
       #GGG                                                         #
       pvect_kp1=rvect_kp1+beta*pvect_k                             #
                                                                    #
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

   print('time solving with K',timeK,NfemP)
   print('time per iteration:',(endu-startu)/k,NfemP)
   print('-------------------------')
    
   return solV,solP,k

############################################################################### 
