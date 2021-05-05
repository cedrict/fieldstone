!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_KV_eq_f(rhs,guess)

use module_parameters
use module_arrays
use module_timing
use module_sparse, only : csrK

implicit none

real(8), intent(inout) :: rhs(NfemV)
real(8), intent(in) :: guess(NfemV)

integer, dimension(:,:), allocatable :: ha
real(8), dimension(:), allocatable :: pivot
real(8) aflag(8)
integer iflag(10), ifail,nn1,nn
real(8), dimension(:), allocatable :: mat 

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{solve\_KV\_eq\_f}
!@@ This subroutine solves the system $\K\cdot \vec{V} = \vec{f}$. The matrix is 
!@@ implicit passed as argument via the module but the rhs and the guess vector are 
!@@ passed as arguments.
!@@ If MUMPS is used the system is solved via MUMPS (the guess is vector
!@@ is then neglected). Same if y12m solver is used. 
!@@ Otherwise a call is made to the {\tt pcg\_solver\_csr} subroutine.
!==================================================================================================!

if (use_MUMPS) then



else

   aflag=0
   iflag=0
   allocate(ha(NfemV,11))
   allocate(pivot(NfemV))
   allocate(mat(15*csrK%NZ)) ; mat=0d0
   nn=size(csrK%snr)
   nn1=size(csrK%rnr)

   mat(1:csrK%NZ)=csrK%mat(1:csrK%NZ)

   call y12maf(NfemV,csrK%NZ,mat,csrK%snr,nn,csrK%rnr,nn1,pivot,ha,NfemV,aflag,iflag,rhs,ifail)

   if (ifail/=0) print *,'ifail=',ifail
   if (ifail/=0) stop 'solve_KV_eq_f: problem with y12m solver'

   solV=rhs

   deallocate(ha)
   deallocate(pivot)
   deallocate(mat)

   !call pcg_solver_csr(csrK,guess,rhs,Kdiag)

end if

end subroutine

!==================================================================================================!
!==================================================================================================!
