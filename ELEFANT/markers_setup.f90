!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine markers_setup

use global_parameters
use structures
!use constants

implicit none

integer i,ii,jj,counter
real(8) chi,eta

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{markers\_setup}
!@@ REDO and rebase on element, local coords and basis fcts
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

mesh(1:nel)%nmarker=nmarker_per_dim**ndim

nmarker=nel*nmarker_per_dim**ndim

write(*,*) '     -> nmarker=',nmarker

allocate(swarm(nmarker))

if (init_marker_random) then

   counter=0
   do iel=1,nel
      do i=1,mesh(iel)%nmarker
         counter=counter+1
         call random_number(eta)
         call random_number(chi)
         swarm(counter)%x=mesh(iel)%xV(1)+mesh(iel)%hx*eta
         swarm(counter)%y=mesh(iel)%yV(1)+mesh(iel)%hy*chi
         swarm(counter)%z=0.d0
         swarm(counter)%r=(eta-0.5d0)*2d0
         swarm(counter)%s=(chi-0.5d0)*2d0
         swarm(counter)%t=0.d0
      end do
   end do

else

   counter=0
   do iel=1,nel
      do ii=1,nmarker_per_dim
      do jj=1,nmarker_per_dim
         counter=counter+1
         swarm(counter)%x=mesh(iel)%xV(1)+(ii-0.5)*mesh(iel)%hx/nmarker_per_dim
         swarm(counter)%y=mesh(iel)%yV(1)+(jj-0.5)*mesh(iel)%hy/nmarker_per_dim
         swarm(counter)%y=0.d0
         swarm(counter)%r=(ii-0.5)/nmarker_per_dim-0.5
         swarm(counter)%s=(jj-0.5)/nmarker_per_dim-0.5
         swarm(counter)%t=0.d0
      end do
      end do
   end do

end if


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> markers_setup ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
