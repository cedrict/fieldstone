!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_GT

use module_parameters, only: GT_storage,NfemVel,NfemP,use_penalty,iproc,solve_stokes_system
use module_arrays, only: GT_matrix
use module_timing

implicit none

integer inode,k,nz,i,ii,nsees,k2,jp,ip,imod
logical, dimension(:), allocatable :: alreadyseen
real(8) t3,t4

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup_GT}
!@@
!==================================================================================================!

if (iproc==0) then

!==============================================================================!

if (use_penalty) return

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
