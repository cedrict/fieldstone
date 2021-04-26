!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_rho_eta_vol

use global_parameters
use global_measurements
use structures
use timing

implicit none

integer :: iq
real(8) :: r_min,r_max,e_min,e_max

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_elemental\_rho\_eta\_vol}
!@@ This subroutine computes the elemental volume, the average density and 
!@@ viscosity using the values already stored on the quadrature points. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

r_min=+1d30
r_max=-1d30
e_min=+1d30
e_max=-1d30
vol_min=+1d30
vol_max=-1d30

do iel=1,nel
   mesh(iel)%rho_avrg=0d0
   mesh(iel)%eta_avrg=0d0
   mesh(iel)%vol     =0d0
   do iq=1,nqel
      mesh(iel)%rho_avrg=mesh(iel)%rho_avrg+mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      mesh(iel)%eta_avrg=mesh(iel)%eta_avrg+mesh(iel)%etaq(iq)*mesh(iel)%JxWq(iq)
      mesh(iel)%vol     =mesh(iel)%vol     +                   mesh(iel)%JxWq(iq)
   end do
   mesh(iel)%rho_avrg=mesh(iel)%rho_avrg/mesh(iel)%vol
   mesh(iel)%eta_avrg=mesh(iel)%eta_avrg/mesh(iel)%vol
   r_min=min(r_min,mesh(iel)%rho_avrg)
   r_max=max(r_max,mesh(iel)%rho_avrg)
   e_min=min(e_min,mesh(iel)%eta_avrg)
   e_max=max(e_max,mesh(iel)%eta_avrg)
   vol_min=min(vol_min,mesh(iel)%vol)
   vol_max=max(vol_max,mesh(iel)%vol)
end do

write(*,'(a,2es10.3)') '        rho_avrg (m/M):',r_min,r_max
write(*,'(a,2es10.3)') '        eta_avrg (m/M):',e_min,e_max

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> compute_elemental_rho_eta_vol ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
