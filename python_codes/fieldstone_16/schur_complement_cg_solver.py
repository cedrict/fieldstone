import numpy as np
import scipy.sparse as sps

def schur_complement_cg_solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,NfemV,NfemP,niter,tol,use_precond):

   conv_file=open("solver_convergence.ascii","w")

   # convert matrices to CSR format
   G_mat=sps.csr_matrix(G_mat)
   K_mat=sps.csr_matrix(K_mat)
   M_mat=sps.csr_matrix(M_mat)

   # declare necessary arrays
   solP=np.zeros(NfemP,dtype=np.float64)  
   solV=np.zeros(NfemV,dtype=np.float64)  
   rvect_k=np.zeros(NfemP,dtype=np.float64) 
   pvect_k=np.zeros(NfemP,dtype=np.float64) 
   zvect_k=np.zeros(NfemP,dtype=np.float64) 
   ptildevect_k=np.zeros(NfemV,dtype=np.float64) 
   dvect_k=np.zeros(NfemV,dtype=np.float64) 

   # carry out solve
   solV=sps.linalg.spsolve(K_mat,f_rhs)
   rvect_k=G_mat.T.dot(solV)-h_rhs
   rvect_0=np.linalg.norm(rvect_k) # 2-norm by default
   if use_precond:
      zvect_k=sps.linalg.spsolve(M_mat,rvect_k)
   else:
      zvect_k=rvect_k
   pvect_k=zvect_k
   for k in range (0,niter):
       ptildevect_k=G_mat.dot(pvect_k)
       dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)
       alpha=(rvect_k.dot(zvect_k))/(ptildevect_k.dot(dvect_k))
       solP+=alpha*pvect_k
       solV-=alpha*dvect_k
       rvect_kp1=rvect_k-alpha*G_mat.T.dot(dvect_k)
       if use_precond:
           zvect_kp1=sps.linalg.spsolve(M_mat,rvect_kp1)
       else:
           zvect_kp1=rvect_kp1
       beta=(zvect_kp1.dot(rvect_kp1))/(zvect_k.dot(rvect_k))
       pvect_kp1=zvect_kp1+beta*pvect_k
       rvect_k=rvect_kp1
       pvect_k=pvect_kp1
       zvect_k=zvect_kp1
       xi=np.linalg.norm(rvect_k)/rvect_0
       conv_file.write("%d %6e \n"  %(k,xi))
       if xi<tol:
          break
   #end for k

   conv_file.close()
    
   return solV,solP,k
 
