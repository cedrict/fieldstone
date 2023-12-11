!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceP

use module_parameters, only: iproc,debug,ndim,mP,nelx,nely,nelz,NP,spaceP
use module_timing
use module_arrays, only: rP,sP,tP

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_global\_parameters\_spaceP}
!@@ This subroutine computes mP,NP and assigns rP,sP,tP
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: Q0,Q1,Q2
!@@ \item supported spaces in 3D: Q0,Q1,Q2
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   select case(spaceP)
   case('__Q0')
      mP=1
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=nelx*nely
   case('__Q1')
      mP=2**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=(nelx+1)*(nely+1)
      rP=(/-1d0,+1d0,+1d0,-1d0/)
      sP=(/-1d0,-1d0,+1d0,+1d0/)
   case('__Q2')
      mP=3**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=(2*nelx+1)*(2*nely+1)
      rP=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sP=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   case default
      stop 'spaceP not supported in set_global_parameters_spaceP'
   end select

else

   select case(spaceP)
   case('__Q0')
      mP=1
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=nelx*nely*nelz
   case('__Q1')
      mP=2**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=(nelx+1)*(nely+1)*(nelz+1)
      rP=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sP=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tP=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   case('__Q2')
      mP=3**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      !missing rP
      !missing sP
      !missing tP
   case default
      stop 'spaceP not supported in set_global_parameters_spaceP'
   end select

end if

!----------------------------------------------------------

!if (debug) then
!   print *,'*************************'
!   print *,'**********debug**********'
!   print *,'mP=',mP
!   print *,'NP=',NP
!   print *,allocated(rP)
!   print *,allocated(sP)
!   print *,allocated(tP)
!   print *,'rP=',rP
!   print *,'sP=',sP
!   print *,'tP=',tP
!   print *,'**********debug**********'
!   print *,'*************************'
!end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_parameters_spaceP (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
