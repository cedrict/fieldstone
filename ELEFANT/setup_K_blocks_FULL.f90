!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_blocks_FULL

use module_parameters, only: iproc,NU,NV,NW,ndim
use module_arrays, only: Kxx_matrix,Kxy_matrix,Kxz_matrix,Kyx_matrix,Kyy_matrix,Kyz_matrix,Kzx_matrix,Kzy_matrix,Kzz_matrix
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_blocks\_FULL}
!@@ This subroutine allocates (large) arrays which store the $\K_{\alpha,\beta}$ blocks
!@@ in full array format (not sparse). This should not be used for medium to large resolutions.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

allocate(Kxx_matrix(NU,NU)) ; Kxx_matrix=0.d0
allocate(Kxy_matrix(NU,NV)) ; Kxy_matrix=0.d0
allocate(Kyx_matrix(NV,NU)) ; Kyx_matrix=0.d0
allocate(Kyy_matrix(NV,NV)) ; Kyy_matrix=0.d0

if (ndim==3) then
allocate(Kxz_matrix(NU,NW)) ; Kxz_matrix=0.d0
allocate(Kyz_matrix(NV,NW)) ; Kyz_matrix=0.d0
allocate(Kzx_matrix(NW,NU)) ; Kzx_matrix=0.d0
allocate(Kzy_matrix(NW,NV)) ; Kzy_matrix=0.d0
allocate(Kzz_matrix(NW,NW)) ; Kzz_matrix=0.d0
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_blocks_FULL:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
