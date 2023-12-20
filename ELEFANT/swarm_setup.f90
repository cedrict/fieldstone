!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine swarm_setup

use module_parameters, only: nmarker,ndim,iel,iproc,use_swarm,spaceV,mV,nel,nmarker_per_dim,init_marker_random
use module_swarm 
use module_mesh 
use module_timing

implicit none

integer i,ii,jj,kk,counter,counter2
real(8) chi,eta,psi,NNNV(mV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{swarm\_setup.f90}
!@@ This subroutine generates the swarm of particles. The layout is controled 
!@@ by the {\tt init\_marker\_random} parameter.
!@@ \begin{center}
!@@ \includegraphics[width=6cm]{ELEFANT/images/swarm_reg} 
!@@ \includegraphics[width=6cm]{ELEFANT/images/swarm_rand} 
!@@ \end{center}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (use_swarm) then

mesh(1:nel)%nmarker=nmarker_per_dim**ndim

nmarker=nel*nmarker_per_dim**ndim

write(*,'(a,i9)') '        nmarker=',nmarker
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
         call NNN(swarm(counter)%r,swarm(counter)%s,swarm(counter)%t,NNNV(1:mV),mV,ndim,spaceV)
         swarm(counter)%x=sum(NNNV(1:mV)*mesh(iel)%xV(1:mV))
         swarm(counter)%y=sum(NNNV(1:mV)*mesh(iel)%yV(1:mV))
         swarm(counter)%z=sum(NNNV(1:mV)*mesh(iel)%zV(1:mV))
         mesh(iel)%list_of_markers(i)=counter
         swarm(counter)%iel=iel
      end do
   end do

else

   if (ndim==2) then
      counter=0
      do iel=1,nel
         counter2=0
         do ii=1,nmarker_per_dim
         do jj=1,nmarker_per_dim
            counter=counter+1
            counter2=counter2+1
            swarm(counter)%r=((ii-0.5)/nmarker_per_dim-0.5)*2
            swarm(counter)%s=((jj-0.5)/nmarker_per_dim-0.5)*2
            swarm(counter)%t=0d0
            call NNN(swarm(counter)%r,swarm(counter)%s,swarm(counter)%t,NNNV(1:mV),mV,ndim,spaceV)
            swarm(counter)%x=sum(NNNV(1:mV)*mesh(iel)%xV(1:mV))
            swarm(counter)%y=sum(NNNV(1:mV)*mesh(iel)%yV(1:mV))
            swarm(counter)%z=0d0
            mesh(iel)%list_of_markers(counter2)=counter
            swarm(counter)%iel=iel
         end do
         end do
      end do
   end if

   if (ndim==3) then
      counter=0
      do iel=1,nel
         counter2=0
         do ii=1,nmarker_per_dim
         do jj=1,nmarker_per_dim
         do kk=1,nmarker_per_dim
            counter=counter+1
            counter2=counter2+1
            swarm(counter)%r=((ii-0.5)/nmarker_per_dim-0.5)*2
            swarm(counter)%s=((jj-0.5)/nmarker_per_dim-0.5)*2
            swarm(counter)%t=((kk-0.5)/nmarker_per_dim-0.5)*2
            call NNN(swarm(counter)%r,swarm(counter)%s,swarm(counter)%t,NNNV(1:mV),mV,ndim,spaceV)
            swarm(counter)%x=sum(NNNV(1:mV)*mesh(iel)%xV(1:mV))
            swarm(counter)%y=sum(NNNV(1:mV)*mesh(iel)%yV(1:mV))
            swarm(counter)%z=sum(NNNV(1:mV)*mesh(iel)%zV(1:mV))
            mesh(iel)%list_of_markers(counter2)=counter
            swarm(counter)%iel=iel
         end do
         end do
         end do
      end do
   end if

end if ! init_marker_random

else
   write(*,'(a)') shift//'bypassed since use_swarm=False'

end if ! use_swarm

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'swarm_setup (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
