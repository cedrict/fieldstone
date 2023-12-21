!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine prescribe_stokes_solution

use module_parameters, only: nel,mV,iproc,iel
use module_mesh 
use module_timing

implicit none

integer k
real(8) dum 

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{prescribe\_stokes\_solution.f90}
!@@ This subroutine prescribes the velocity, pressure, temperature and strain rate components
!@@ on the nodes of each element via the {\sl experiment\_analytical\_solution} subroutine.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

do iel=1,nel
   do k=1,mV
      call experiment_analytical_solution(mesh(iel)%xV(k),&
                               mesh(iel)%yV(k),&
                               mesh(iel)%zV(k),&
                               mesh(iel)%u(k),&
                               mesh(iel)%v(k),&
                               mesh(iel)%w(k),&
                               mesh(iel)%q(k),&
                               dum,&
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

write(*,'(a,f6.2,a)') '     >> prescribe_stokes_solution ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
