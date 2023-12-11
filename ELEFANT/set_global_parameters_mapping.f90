!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_mapping

use module_parameters, only: iproc,debug,ndim,mmapping,mapping
use module_timing
use module_arrays, only: rmapping,smapping,tmapping

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_global\_parameters\_mapping}
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

   select case(mapping)
   case('__Q1')
      mmapping=2**ndim
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,+1d0,+1d0,-1d0/)
      smapping=(/-1d0,-1d0,+1d0,+1d0/)
   case('__Q2')
      mmapping=3**ndim
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      smapping=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   case default
      stop 'mapping not supported in set_global_parameters_mapping'
   end select

else

   select case(mapping)
   case('__Q1')
      mmapping=2**ndim
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      smapping=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tmapping=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   case('__Q2')
      mmapping=3**ndim
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      !missing r
      !missing s
      !missing t
   case default
      stop 'mapping not supported in set_global_parameters_mapping'
   end select

end if

!----------------------------------------------------------

!if (debug) then
!print *,'*************************'
!print *,'**********debug**********'
!print *,'mmapping=',mmapping
!print *,allocated(rmapping)
!print *,allocated(smapping)
!print *,allocated(tmapping)
!print *,'rmapping=',rmapping
!print *,'smapping=',smapping
!print *,'tmapping=',tmapping
!print *,'**********debug**********'
!print *,'*************************'
!end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_parameters_mapping (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
