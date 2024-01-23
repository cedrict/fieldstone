!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_mapping

use module_parameters, only: iproc,debug,ndim,mmapping,mapping,spaceVelocity,isoparametric_mapping
use module_timing
use module_arrays, only: rmapping,smapping,tmapping

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{set\_global\_parameters\_mapping}
!@@ This subroutine computes {\tt mmapping} and assigns {\tt rmapping,smapping,tmapping}.
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: $Q_1$, $Q_2$, $Q_3$, $P_1$, $P_2$
!@@ \item supported spaces in 3D: $Q_1$, $Q_2$
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


if (isoparametric_mapping) then 
   mapping=spaceVelocity
else
   stop 'abcdgerf'
end if

write(*,'(a,a)') shift//'mapping=',mapping

write(*,'(a,l1)') shift//'isoparametric_mapping=',isoparametric_mapping

if (ndim==2) then

   select case(mapping)
   !-----------
   case('__P1')
      mmapping=3
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/0d0,1d0,0d0/)
      smapping=(/0d0,0d0,1d0/)
   !-----------
   case('__P2')
      mmapping=6
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/0d0,1d0,0d0,0.5d0,0.5d0,0d0/)
      smapping=(/0d0,0d0,1d0,0d0,0.5d0,0.5d0/)
   !-----------
   case('__Q1')
      mmapping=4
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,+1d0,+1d0,-1d0/)
      smapping=(/-1d0,-1d0,+1d0,+1d0/)

   !-----------
   case('_Q1+')
      mmapping=5
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,+1d0,+1d0,-1d0,0d0/)
      smapping=(/-1d0,-1d0,+1d0,+1d0,0d0/)

   !-----------
   case('__Q2')
      mmapping=9
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      smapping=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   !-----------
   case('__Q3')
      mmapping=16
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0/)
      smapping=(/-1d0,-1d0,-1d0,-1d0,&
           -1d0/3d0,-1d0/3d0,-1d0/3d0,-1d0/3d0,&
           1d0/3d0,1d0/3d0,1d0/3d0,1d0/3d0,&
           1d0,1d0,1d0,1d0/)
   !-----------
   case default
      stop 'mapping not supported in set_global_parameters_mapping'
   end select

else

   select case(mapping)
   !-----------
   case('__Q1')
      mmapping=8
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      rmapping=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      smapping=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tmapping=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   !-----------
   case('__Q2')
      mmapping=27
      allocate(rmapping(mmapping)) ; rmapping=0.d0
      allocate(smapping(mmapping)) ; smapping=0.d0
      allocate(tmapping(mmapping)) ; tmapping=0.d0
      !missing r
      !missing s
      !missing t
   !-----------
   case default
      stop 'mapping not supported in set_global_parameters_mapping'
   end select

end if

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'set_global_parameters_mapping'//limit
write(2345,*) 'mmapping=',mmapping
write(2345,*) allocated(rmapping)
write(2345,*) allocated(smapping)
write(2345,*) allocated(tmapping)
write(2345,*) 'rmapping=',rmapping
write(2345,*) 'smapping=',smapping
write(2345,*) 'tmapping=',tmapping
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_parameters_mapping:',elapsed,' s  |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
