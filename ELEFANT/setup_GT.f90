!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_GT

use module_parameters, only: GT_storage,NfemVel,NfemP,iproc,solve_stokes_system,stokes_solve_strategy
use module_arrays, only: GT_matrix
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_GT}
!@@ This subroutine is a switch. Based on the value {\tt GT\_storage}, it will call the 
!@@ appropriate subroutine.
!@@ Note that if {\tt stokes\_solve\_strategy=='penalty'} then this subroutine is bypassed.
!==================================================================================================!

if (iproc==0) then

!==============================================================================!

if (stokes_solve_strategy=='penalty') return

if (solve_stokes_system) then 

   write(*,'(a)') shift//GT_storage

   select case(GT_storage)
   !------------------
   case('matrix_FULL')

      allocate(GT_matrix(NfemP,NfemVel)) ; GT_matrix=0.d0

   !------------------
   case('matrix_CSR')

      call setup_GT_matrix_CSR

   !-----------------
   case('blocks_CSR')

   !-----------
   case default

      stop 'setup_GT: unknown GT_storage value'

   end select

else !solve_stokes_system

   write(*,'(a)') shift//'setup_GT bypassed'

end if


!==============================================================================!

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
