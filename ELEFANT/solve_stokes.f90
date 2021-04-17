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
use global_arrays
use structures
!use constants
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

   ! solve K.V=f -> solV

   call solve_KVeqf(rhs_f,SolV)

   ! compute pressure -> solP

   ! normalise pressure

else




end if

!transfer solution to elements

do iel=1,nel

   !velocity
   do k=1,mV
      inode=mesh(iel)%iconV(k)
      mesh(iel)%u(k)=solV((inode-1)*ndofV+1)
      mesh(iel)%v(k)=solV((inode-1)*ndofV+2)
      if (ndim==3) &
      mesh(iel)%w(k)=solV((inode-1)*ndofV+3)
   end do

   !pressure
   do k=1,mP
      inode=mesh(iel)%iconP(k)
      mesh(iel)%p(k)=solP(inode)
   end do

end do


p_min=minval(solP)
p_max=maxval(solP)


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> solve_stokes                     ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
