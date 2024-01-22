!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_timestep

use module_parameters, only: spaceVelocity,CFL_nb,iproc,use_T,dt
use module_statistics 
use module_timing

implicit none

real(8) hmin,maxv,p

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_timestep}
!@@ See Section~\ref{ss:cfl}.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

maxv=max(abs(u_min),abs(u_max),abs(v_min),abs(v_max),abs(w_min),abs(w_max))

hmin=(vol_min)**(1./3.)

select case(spaceVelocity)
case('__Q1','_Q1+','Q1++') 
   p=1d0
case('__Q2','__P2') 
   p=2d0
case('__Q3') 
   p=3d0
case default
   stop 'compute_timestep: spaceVelocity unsupported'
end select

if (maxv>1d-15) then
   dt=CFL_nb*hmin/p/maxv
else
   dt=1d50
end if

if (use_T) then
   dt=min(dt,hmin**2*rhoq_min*hcapaq_min/hcondq_max)
end if


write(*,'(a,es12.4)') shift//'dt=',dt 

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_timestep:',elapsed,' s               |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
