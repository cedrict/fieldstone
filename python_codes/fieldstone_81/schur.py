import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

def solver(K_mat,G_mat,M_mat,f_rhs,h_rhs,NfemP,NfemV,Nfem):

    use_preconditioner=False
    solver_tolerance=1e-6
    niter_stokes=50
    ls_conv_file=open("linear_solver_convergence.ascii","w")
    ls_niter_file=open("linear_solver_niter.ascii","w")
    solP    = np.zeros(NfemP,dtype=np.float64)  
    solV    = np.zeros(NfemV,dtype=np.float64)  
    Res     = np.zeros(Nfem,dtype=np.float64)         # non-linear residual 

    # convert matrices to CSR format
    G_mat=sps.csr_matrix(G_mat)
    K_mat=sps.csr_matrix(K_mat)
    M_mat=sps.csr_matrix(M_mat)

    Res[0:NfemV]=K_mat.dot(solV)+G_mat.dot(solP)-f_rhs
    Res[NfemV:Nfem]=G_mat.T.dot(solV)-h_rhs

    # declare necessary arrays
    rvect_k=np.zeros(NfemP,dtype=np.float64) 
    pvect_k=np.zeros(NfemP,dtype=np.float64) 
    zvect_k=np.zeros(NfemP,dtype=np.float64) 
    ptildevect_k=np.zeros(NfemV,dtype=np.float64) 
    dvect_k=np.zeros(NfemV,dtype=np.float64) 
   
    # carry out solve
    solP[:]=0.
    #print("opla")
    solV=sps.linalg.spsolve(K_mat,f_rhs,use_umfpack=True)
    #solV,info=sps.linalg.cg(K_mat,f_rhs,tol=1e-14)
    #print (info)
    #print("opla")
    rvect_k=G_mat.T.dot(solV)-h_rhs
    rvect_0=np.linalg.norm(rvect_k)
    if use_preconditioner:
         zvect_k=sps.linalg.spsolve(M_mat,rvect_k)
    else:
         zvect_k=rvect_k
    pvect_k=zvect_k
    for k in range (0,niter_stokes):
          #print("iteration=",k)
          ptildevect_k=G_mat.dot(pvect_k)
          dvect_k=sps.linalg.spsolve(K_mat,ptildevect_k)
          #dvect_k,info=sps.linalg.cg(K_mat,ptildevect_k,tol=1e-14,atol=1e-14,x0=dvect_k)
          #print(np.min(dvect_k),np.max(dvect_k))
          #print(info)
          alpha=(rvect_k.dot(zvect_k))/(ptildevect_k.dot(dvect_k))
          solP+=alpha*pvect_k
          solV-=alpha*dvect_k
          rvect_kp1=rvect_k-alpha*G_mat.T.dot(dvect_k)
          if use_preconditioner:
              zvect_kp1=sps.linalg.spsolve(M_mat,rvect_kp1)
          else:
              zvect_kp1=rvect_kp1
          beta=(zvect_kp1.dot(rvect_kp1))/(zvect_k.dot(rvect_k))
          pvect_kp1=zvect_kp1+beta*pvect_k
          rvect_k=rvect_kp1
          pvect_k=pvect_kp1
          zvect_k=zvect_kp1
          xi=np.linalg.norm(rvect_k)/rvect_0
          ls_conv_file.write("%d %6e \n"  %(k,xi))
          print("lin.solver: %d %6e" % (k,xi))
          if xi<solver_tolerance:
             ls_niter_file.write("%d \n"  %(k))
             break 

    sol=np.zeros(Nfem,dtype=np.float64) 
    sol[0:NfemV]=solV
    sol[NfemV:NfemV+NfemP]=solP

    return sol
   
