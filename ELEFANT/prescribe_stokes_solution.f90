!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine prescribe_stokes_solution

use module_parameters, only: nel,mU,mV,mW,iproc,iel
use module_mesh 
use module_timing

implicit none

integer k
real(8) dum1,dum2,dum3,dum4,dum5,dum6,dum7,dumu,dumv,dumw,dump 

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

   do k=1,mU
      call experiment_analytical_solution(mesh(iel)%xU(k),&
                                          mesh(iel)%yU(k),&
                                          mesh(iel)%zU(k),&
                                          mesh(iel)%u(k),dumv,dumw,dump,& 
                                          dum1,dum2,dum3,dum4,dum5,dum6,dum7)
   end do

   do k=1,mV
      call experiment_analytical_solution(mesh(iel)%xV(k),&
                                          mesh(iel)%yV(k),&
                                          mesh(iel)%zV(k),&
                                          dumu,mesh(iel)%v(k),dumw,dump,&
                                          dum1,dum2,dum3,dum4,dum5,dum6,dum7)
   end do

   do k=1,mW
      call experiment_analytical_solution(mesh(iel)%xW(k),&
                                          mesh(iel)%yW(k),&
                                          mesh(iel)%zW(k),&
                                          dumu,dumv,mesh(iel)%w(k),dump,&
                                          dum1,dum2,dum3,dum4,dum5,dum6,dum7)
   end do

end do

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') '     >> prescribe_stokes_solution ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
