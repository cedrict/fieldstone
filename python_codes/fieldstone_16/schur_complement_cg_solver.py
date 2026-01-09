import numpy as np
import scipy.sparse as sps

def schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,NfemV,NfemP,niter,tol,use_precond):

   #the function implicitely assumes matrices in csr format

   # we assume that the guess pressure is zero.
   solP=np.zeros(NfemP,dtype=np.float64)  

   # declare necessary arrays
   solV=np.zeros(NfemV,dtype=np.float64)  
   rvect_k=np.zeros(NfemP,dtype=np.float64) 
   pvect_k=np.zeros(NfemP,dtype=np.float64) 
   zvect_k=np.zeros(NfemP,dtype=np.float64) 
   ptildevect_k=np.zeros(NfemV,dtype=np.float64) 
   dvect_k=np.zeros(NfemV,dtype=np.float64) 

   conv_file=open("solver_convergence.ascii","w")

   # carry out solve
   solV=sps.linalg.spsolve(K_mat,f_rhs)                         # compute V_0
   rvect_k=G_mat.T.dot(solV)-h_rhs              # compute r_0
   rvect_0=np.linalg.norm(rvect_k) # 2-norm by default
   if use_precond:
      zvect_k=sps.linalg.spsolve(M_mat,rvect_k)                 # compute z_0
   else:
      zvect_k=rvect_k
   pvect_k=zvect_k                                              #compute p_0

   for k in range (0,niter): #--------------------------------------#
                                                                    #
       ptildevect_k=G_mat.dot(pvect_k)                              # 
       #Cp=C_mat.dot(pvect_k)                                       # C . p_k
       #pCp=pvect_k.dot(Cp)                                         # p_k . C . p_k
       dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)               #
       alpha=(rvect_k.dot(zvect_k))/(ptildevect_k.dot(dvect_k))     #
       solP+=alpha*pvect_k                                          #
       solV-=alpha*dvect_k                                          #
       rvect_kp1=rvect_k-alpha*(G_mat.T.dot(dvect_k))               #
       if use_precond:                                              #
           zvect_kp1=sps.linalg.spsolve(M_mat,rvect_kp1)            #
       else:                                                        #
           zvect_kp1=rvect_kp1                                      #
       beta=(zvect_kp1.dot(rvect_kp1))/(zvect_k.dot(rvect_k))       #
       pvect_kp1=zvect_kp1+beta*pvect_k                             #
                                                                    #
       rvect_k=rvect_kp1                                            #
       pvect_k=pvect_kp1                                            #
       zvect_k=zvect_kp1                                            #
                                                                    #
       xi=np.linalg.norm(rvect_k)/rvect_0                           #
       conv_file.write("%d %6e \n"  %(k,xi))                        #
       conv_file.flush()                                            #
       print('     |iter %4d xi= %.4e' %(k,xi))                     #
       if xi<tol:                                                   #
          break                                                     #
                                                                    #
   #end for k #-----------------------------------------------------#

   conv_file.close()
    
   return solV,solP,k
 
