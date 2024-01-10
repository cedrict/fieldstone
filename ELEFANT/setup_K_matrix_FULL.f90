!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_matrix_FULL

use module_parameters, only: NfemV
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
use module_arrays, only: K_matrix
use module_timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_matrix\_FULL}
!@@ This subroutine allocates two (large) arrays which store the $\K$ and $\G^T$ matrices
!@@ in full array format (not sparse). This should not be used for medium to large resolutions.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

allocate(K_matrix(NfemV,NfemV)) ; K_matrix=0.d0

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_matrix_FULL:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
