!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_stokes_SIMPLE

use module_parameters, only: debug,iproc,NfemVel,NfemP
use module_arrays, only: GT_matrix,Kdiag,rhs_f,rhs_h
use module_timing

implicit none

integer :: k
integer, parameter :: niter_SIMPLE=10
real(8), parameter :: omega_V=0.5
real(8), parameter :: omega_P=0.5

real(8), dimension(NfemVel) :: dV,dstarV,resV,Vk
real(8), dimension(NfemP) :: dP,dstarP,resP,Pk

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{solve\_stokes\_SIMPLE}
!@@
!@@ \begin{enumerate}
!@@ \item compute the residuals 
!@@ \begin{eqnarray}
!@@ \vec{r}_{\cal V} &=& \vec{f} - \K \cdot \vec{\cal V}^{(k)} - \G \cdot \vec{\cal P}^{(k)} \nn\\
!@@ \vec{r}_{\cal P} &=& \vec{h} - \G^T \cdot \vec{\cal V}^{(k)} \nn
!@@ \end{eqnarray}
!@@ \item Solve $\K  \cdot \delta^\star \vec{\cal V}^k =  \vec{r}_{\cal V}^k  $
!@@ \item Solve $\hat{\SSS}\cdot\delta^\star\vec{\cal P}^k =\vec{r}_{\cal P}^k-\G^T\cdot\delta^\star V^k $
!@@ \item Compute $\delta \vec{\cal V}^k = \delta^\star \vec{\cal V}^k -{\bm D}_\K^{-1} \cdot \G \cdot \delta^\star \vec{\cal P}_k $
!@@ \item Update $\delta \vec{\cal P}^k = \delta^\star \vec{\cal P}^k$
!@@ \item Update 
!@@ \begin{eqnarray}
!@@ \vec{\cal V}^{(k+1)} &=& \vec{\cal V}^{(k)} + \omega_{\cal V} \delta \vec{\cal V}^{(k)} \nn\\
!@@ \vec{\cal P}^{(k+1)} &=& \vec{\cal P}^{(k)} + \omega_{\cal P} \delta \vec{\cal P}^{(k)} \nn
!@@ \end{eqnarray}
!@@ \end{enumerate}
!@@ where ${\bm D}_\K=diag(\K)$, $\hat{\SSS} = \G^T \cdot {\bm D}_\K^{-1} \cdot \G$ 
!@@ and the parameters $\omega_{\cal V}$ and 
!@@ $\omega_{\cal P}$ are between 0 and 1.
!@@
!@@ In the subroutine:
!@@ \begin{itemize}
!@@ \item $\vec{r}_{\cal V} \rightarrow$ {\tt resV(1:NfemVel)}
!@@ \item $\vec{r}_{\cal P} \rightarrow$ {\tt resP(1:NfemP)}
!@@ \item $\delta \vec{\cal P}^k \rightarrow$ {\tt dP(1:NfemP)}
!@@ \item $\delta^\star \vec{\cal P}^k \rightarrow$ {\tt dstarP(1:NfemP)}
!@@ \end{itemize}
 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


do k=1,niter_SIMPLE

   !-----------------------------------
   ! step 1
   !-----------------------------------

   resV=rhs_f

   resP=rhs_h

   !-----------------------------------
   ! step 2
   !-----------------------------------

   !call inner_solve(resV,guess,dstarV) 

   !-----------------------------------
   ! step 3
   !-----------------------------------

   !SOLVER with hat{S} !!!

   !-----------------------------------
   ! step 4
   !-----------------------------------

   dV = dstarV - (1d0/Kdiag)*matmul(transpose(GT_matrix),dstarP)

   !-----------------------------------
   ! step 5
   !-----------------------------------

   dP = dstarP

   !-----------------------------------
   ! step 6
   !-----------------------------------

   Vk = Vk + omega_V * dV

   Pk = Pk + omega_P * dP

end do

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'solve_stokes_SIMPLE:',elapsed,' s   |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
