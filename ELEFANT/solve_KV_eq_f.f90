!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_KV_eq_f(rhs,guess)

use global_parameters
use global_arrays
!use structures
!use constants
use timing
use matrices, only : csrK

implicit none

real(8), intent(in) :: rhs(NfemV)
real(8), intent(in) :: guess(NfemV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{solve\_KV\_eq\_f}
!@@ This subroutine solves the system $\K\cdot \vec{V} = \vec{f}$. The matrix is 
!@@ implicit passed as argument via the module but the rhs and the guess vector are 
!@@ passed as arguments.
!@@ If MUMPS is used the system is solved via MUMPS (the guess is vector
!@@ is then neglected), otherwise a call is made to !@@ the {\tt solve\_cg\_diagprec} subroutine.
!==================================================================================================!

if (use_MUMPS) then



else

   call pcg_solver_csr(csrK,guess,rhs,Kdiag)

end if

end subroutine

!==================================================================================================!
!==================================================================================================!
