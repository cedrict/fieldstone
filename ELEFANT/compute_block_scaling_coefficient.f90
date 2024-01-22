!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_block_scaling_coefficient

use module_parameters, only: nel,nqel,block_scaling_coeff,iproc,debug,iel,ndim
use module_mesh 
use module_timing

implicit none

integer :: iq
real(8) :: total_volume,average_viscosity

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_block\_scaling\_coefficient}
!@@ This subroutine compute the coefficient that multiplies the $\G$ block of the Stokes matrix
!@@ that is required to ensure accurate pressure calculations, see Section~\ref{pscaling}.
!@@ At the moment it is computed as $<\eta>/{V}^{1/ndim}$ where $<\eta>$ is the (arithmetic) average 
!@@ viscosity over the domain and V is the volume of the domain.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

total_volume=0
average_viscosity=0
do iel=1,nel
   do iq=1,nqel
      average_viscosity=average_viscosity+mesh(iel)%JxWq(iq)*mesh(iel)%etaq(iq)
      total_volume=total_volume+mesh(iel)%JxWq(iq)
   end do
end do

average_viscosity=average_viscosity/total_volume

block_scaling_coeff=average_viscosity/total_volume**(1.d0/ndim)

write(*,'(a,es12.4)') shift//'average viscosity=',average_viscosity
write(*,'(a,es12.4)') shift//'total volume=',total_volume
write(*,'(a,es12.4)') shift//'block_scaling_coeff=',block_scaling_coeff

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_block_scaling_coeff:',elapsed,' s    |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
