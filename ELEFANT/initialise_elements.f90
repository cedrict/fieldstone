!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine initialise_elements

use module_parameters
use module_mesh 
!use module_constants
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{initialise\_elements}
!@@ This subroutine 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

allocate(mesh(nel))

do iel=1,nel  

   allocate(mesh(iel)%iconV(mV)) 
   allocate(mesh(iel)%xV(mV))
   allocate(mesh(iel)%yV(mV))
   allocate(mesh(iel)%zV(mV))
   allocate(mesh(iel)%u(mV)) ; mesh(iel)%u=0.d0
   allocate(mesh(iel)%v(mV)) ; mesh(iel)%v=0.d0
   allocate(mesh(iel)%w(mV)) ; mesh(iel)%w=0.d0
   allocate(mesh(iel)%q(mV)) ; mesh(iel)%q=0.d0
   allocate(mesh(iel)%exx(mV))
   allocate(mesh(iel)%eyy(mV))
   allocate(mesh(iel)%ezz(mV))
   allocate(mesh(iel)%exy(mV))
   allocate(mesh(iel)%exz(mV))
   allocate(mesh(iel)%eyz(mV))
   allocate(mesh(iel)%bnd1_node(mV))
   allocate(mesh(iel)%bnd2_node(mV))
   allocate(mesh(iel)%bnd3_node(mV))
   allocate(mesh(iel)%bnd4_node(mV))
   allocate(mesh(iel)%bnd5_node(mV))
   allocate(mesh(iel)%bnd6_node(mV))
   allocate(mesh(iel)%fix_u(mV))
   allocate(mesh(iel)%fix_v(mV))
   allocate(mesh(iel)%fix_w(mV))


   allocate(mesh(iel)%iconP(mP)) 
   allocate(mesh(iel)%p(mP)) ; mesh(iel)%p=0.d0
   allocate(mesh(iel)%xP(mP))
   allocate(mesh(iel)%yP(mP))
   allocate(mesh(iel)%zP(mP))

   allocate(mesh(iel)%iconT(mT)) 
   allocate(mesh(iel)%xT(mT))
   allocate(mesh(iel)%yT(mT))
   allocate(mesh(iel)%zT(mT))
   allocate(mesh(iel)%T(mT)) ; mesh(iel)%T=0.d0
   allocate(mesh(iel)%qx(mT))
   allocate(mesh(iel)%qy(mT))
   allocate(mesh(iel)%qz(mT))
   allocate(mesh(iel)%fix_T(mT))


   allocate(mesh(iel)%xL(mV))
   allocate(mesh(iel)%yL(mV))
   allocate(mesh(iel)%zL(mV))
  !real(8), allocatable :: xM(:),yM(:),zM(:)   ! coordinates of mapping nodes

end do


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'initialise_elements (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
