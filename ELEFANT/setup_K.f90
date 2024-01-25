!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K

use module_parameters, only: K_storage,iproc,solve_stokes_system,stokes_solve_strategy
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K}
!@@ This subroutine acts as a switch. Depending on the value of {\tt K\_storage} it calls 
!@@ the appropriate subroutine. 
!==================================================================================================!

if (iproc==0) then

!==============================================================================!

write(*,'(a,a)') shift//'stokes_solve_strategy=',stokes_solve_strategy

if (solve_stokes_system) then 

   write(*,'(a,a)') shift//'K_storage=',K_storage

   select case(K_storage)

   !------------------
   case('matrix_FULL')

      call setup_K_matrix_FULL

   !------------------
   case('matrix_MUMPS')

      call setup_K_matrix_MUMPS

   !------------------
   case('matrix_CSR')

      call setup_K_matrix_CSR

   !------------------
   !case('blocks_MUMPS')

   !------------------
   case('blocks_CSR')

      call setup_K_blocks_CSR

   !------------------
   case default

      print *,K_storage
      stop 'setup_K: unknown K_storage value'

   end select

else !solve_stokes_system

   write(*,'(a)') shift//'setup_K bypassed'

end if

!==============================================================================!

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
