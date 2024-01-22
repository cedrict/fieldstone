!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine prescribe_stokes_solution

use module_parameters, only: nel,mU,mV,mW,iproc,iel,ndim
use module_statistics 
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

!----------------------------------------------------------

u_max=-1d30
u_min=+1d30

v_max=-1d30
v_min=+1d30

v_max=-1d30
v_min=+1d30
do iel=1,nel
   u_max=max(maxval(mesh(iel)%u(1:mU)),u_max)
   u_min=min(minval(mesh(iel)%u(1:mU)),u_min)
   v_max=max(maxval(mesh(iel)%v(1:mV)),v_max)
   v_min=min(minval(mesh(iel)%v(1:mV)),v_min)
   w_max=max(maxval(mesh(iel)%w(1:mW)),w_max)
   w_min=min(minval(mesh(iel)%w(1:mW)),w_min)
end do

             write(*,'(a,2es12.4)') shift//'u (m,M)',u_min,u_max
             write(*,'(a,2es12.4)') shift//'v (m,M)',v_min,v_max
if (ndim==3) write(*,'(a,2es12.4)') shift//'w (m,M)',w_min,w_max
             write(*,'(a,2es12.4)') shift//'p (m,M)',p_min,p_max

write(1238,'(8es12.4)') u_min,u_max,v_min,v_max,w_min,w_max,p_min,p_max
call flush(1238)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'prescribe_stokes_solution:',elapsed,' s      |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
