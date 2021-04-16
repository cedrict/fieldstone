!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine swarm_setup

use global_parameters
use structures
use timing

implicit none

integer i,ii,jj,counter
real(8) chi,eta,psi,NNNT(mT)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{swarm\_setup.f90}
!@@ REDO and rebase on element, local coords and basis fcts
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

mesh(1:nel)%nmarker=nmarker_per_dim**ndim

nmarker=nel*nmarker_per_dim**ndim

write(*,'(a,i7)') '        nmarker=',nmarker
write(*,'(a,l)')  '        init_marker_random=',init_marker_random

allocate(swarm(nmarker))

if (init_marker_random) then

   counter=0
   do iel=1,nel
      do i=1,mesh(iel)%nmarker
         counter=counter+1
         call random_number(eta)
         call random_number(chi)
         call random_number(psi)
         swarm(counter)%r=(eta-0.5d0)*2d0
         swarm(counter)%s=(chi-0.5d0)*2d0
         swarm(counter)%t=(psi-0.5d0)*2d0
         call NNT(swarm(counter)%r,swarm(counter)%s,swarm(counter)%t,NNNT(1:mT),mT,ndim)
         swarm(counter)%x=sum(NNNT(1:mT)*mesh(iel)%xV(1:mT))
         swarm(counter)%y=sum(NNNT(1:mT)*mesh(iel)%yV(1:mT))
         swarm(counter)%z=sum(NNNT(1:mT)*mesh(iel)%zV(1:mT))
         mesh(iel)%list_of_markers(i)=counter
         swarm(counter)%iel=iel
      end do
   end do

else

   !REDO!
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

write(*,'(a,f4.2,a)') '     >> swarm_setup                      ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
