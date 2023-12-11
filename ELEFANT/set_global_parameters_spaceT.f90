!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceT

use module_parameters, only: iproc,debug,ndim,mT,nelx,nely,nelz,NT,spaceT
use module_timing
use module_arrays, only: rT,sT,tT

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_global\_parameters\_spaceT}
!@@ This subroutine computes mT,NT and assigns rT,sT,tT
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: Q1,Q2
!@@ \item supported spaces in 3D: Q1,Q2
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   select case(spaceT)
   case('__Q1')
      mT=2**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      NT=(nelx+1)*(nely+1)
      rT=(/-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0/)
   case('__Q2')
      mT=3**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      NT=(2*nelx+1)*(2*nely+1)
      rT=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sT=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   case default
      stop 'spaceT not supported in set_global_parameters_spaceT'
   end select

else

   select case(spaceT)
   case('__Q1')
      mT=2**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      NT=(nelx+1)*(nely+1)*(nelz+1)
      rT=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tT=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   case('__Q2')
      mT=3**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      NT=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      !missing rT
      !missing sT
      !missing tT
   case default
      stop 'spaceT not supported in set_global_parameters_spaceT'
   end select

end if

!----------------------------------------------------------

!if (debug) then
!   print *,'*************************'
!   print *,'**********debug**********'
!   print *,'mT=',mT
!   print *,'NT=',NT
!   print *,allocated(rT)
!   print *,allocated(sT)
!   print *,allocated(tT)
!   print *,'rT=',rT
!   print *,'sT=',sT
!   print *,'tT=',tT
!   print *,'**********debug**********'
!   print *,'*************************'
!end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_parameters_spaceT (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
