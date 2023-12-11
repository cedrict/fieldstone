!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine paint_swarm

use module_parameters, only: use_swarm,nmarker,nxstripes,nystripes,nzstripes,ndim,geometry,Lx,Ly,Lz,iproc
use module_swarm 
use module_timing

implicit none

integer im
real(8) dx,dy,dz

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{paint\_swarm}
!@@ 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (use_swarm) then

   do im=1,nmarker  
      swarm(im)%paint = 0.
   end do

   if (geometry=='cartesian') then

      if (nxstripes>1 .or. nystripes>1 .or. nzstripes>1) then
         dx=Lx/nxstripes  
         dy=Ly/nystripes  
         dz=Lz/nzstripes  
         do im=1,nmarker 
            if ( (mod(int(swarm(im)%x/dx),2)==1).neqv.(mod(int(swarm(im)%y/dy),2)==1) ) then
               swarm(im)%paint=1
            end if
            if (ndim==3) then
               if ( (swarm(im)%paint==1) .neqv. (mod(int(swarm(im)%z/dz),2)==1) ) then
                  swarm(im)%paint=1
               end if
            end if
            swarm(im)%paint=swarm(im)%paint-0.5
         end do    
      end if

      if (nxstripes<1) then
         do im=1,nmarker 
            call random_number(swarm(im)%paint)
            swarm(im)%paint=swarm(im)%paint-0.5
         end do
      end if

   end if ! cartesian

   if (geometry=='annulus') then
      stop 'geometry not supported in paint_swarm'
   end if

   if (geometry=='shell') then
      stop 'geometry not supported in paint_swarm'
   end if

else
   write(*,'(a)') shift//'bypassed since use_swarm=False'

end if ! use_markers

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'paint swarm (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
