!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_stokes

use global_parameters
use global_measurements
use global_arrays, only: rhs_f,solV,solP
use structures
use timing

implicit none

integer inode,k
real(8) :: guess(NfemV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{solve\_stokes}
!@@ This subroutine solves the Stokes system: if it is a saddle point problem 
!@@ it does so using the preconditioned conjugate gradient (PCG) applied 
!@@ to the Schur complement $\SSS$ !@@ (see Section~\ref{ss:schurpcg}).
!@@ If the penalty method is used 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

solV=0
solP=0
guess=0

if (use_penalty) then

   call solve_KVeqf(rhs_f,solV)

   !transfer velocity onto elements
   do iel=1,nel
      do k=1,mV
         inode=mesh(iel)%iconV(k)
         mesh(iel)%u(k)=solV((inode-1)*ndofV+1)
         mesh(iel)%v(k)=solV((inode-1)*ndofV+2)
         if (ndim==3) &
         mesh(iel)%w(k)=solV((inode-1)*ndofV+3)
      end do
   end do

   call recover_pressure_penalty


   !transfer pressure onto elements
   do iel=1,nel
      do k=1,mP
         inode=mesh(iel)%iconP(k)
         mesh(iel)%p(k)=solP(inode)
      end do
   end do


else !-----------------------------------------------------








   !transfer solution onto elements
   do iel=1,nel
      do k=1,mV
         inode=mesh(iel)%iconV(k)
         mesh(iel)%u(k)=solV((inode-1)*ndofV+1)
         mesh(iel)%v(k)=solV((inode-1)*ndofV+2)
         if (ndim==3) &
         mesh(iel)%w(k)=solV((inode-1)*ndofV+3)
      end do
      do k=1,mP
         inode=mesh(iel)%iconP(k)
         mesh(iel)%p(k)=solP(inode)
      end do
   end do

end if

!----------------------------------------------------------

u_max=-1d30
u_min=+1d30
v_max=-1d30
v_min=+1d30
v_max=-1d30
v_min=+1d30
do iel=1,nel
   u_max=max(maxval(mesh(iel)%u(1:mV)),u_max)
   u_min=min(minval(mesh(iel)%u(1:mV)),u_min)
   v_max=max(maxval(mesh(iel)%v(1:mV)),v_max)
   v_min=min(minval(mesh(iel)%v(1:mV)),v_min)
   w_max=max(maxval(mesh(iel)%w(1:mV)),w_max)
   w_min=min(minval(mesh(iel)%w(1:mV)),w_min)
end do
write(*,'(a,2es12.4)') '        u (m,M)',u_min,u_max
write(*,'(a,2es12.4)') '        v (m,M)',v_min,v_max
write(*,'(a,2es12.4)') '        w (m,M)',w_min,w_max


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> solve_stokes                     ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
