!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine prescribe_stokes_solution

use global_parameters
use structures
use timing

implicit none

integer k

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{prescribe\_stokes\_solution.f90}
!@@ This subroutine prescribes the velocity, pressure, temperature and strain rate components
!@@ on the corners of each element via the {\sl analytical\_solution} subroutine.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

do iel=1,nel
   do k=1,ncorners
      call analytical_solution(mesh(iel)%xV(k),&
                               mesh(iel)%yV(k),&
                               mesh(iel)%zV(k),&
                               mesh(iel)%u(k),&
                               mesh(iel)%v(k),&
                               mesh(iel)%w(k),&
                               mesh(iel)%q(k),&
                               mesh(iel)%T(k),&
                               mesh(iel)%exx(k),&
                               mesh(iel)%eyy(k),&
                               mesh(iel)%ezz(k),&
                               mesh(iel)%exy(k),&
                               mesh(iel)%exz(k),&
                               mesh(iel)%eyz(k))

   end do
end do

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> prescribe_stokes_solution ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
