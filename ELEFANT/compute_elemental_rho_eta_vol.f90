!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_rho_eta_vol

use module_parameters
use module_statistics 
use module_mesh 
use module_timing

implicit none

integer :: iq
real(8) :: dens_min,dens_max,visc_min,visc_max

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_elemental\_rho\_eta\_vol}
!@@ This subroutine computes the elemental volume, the average density and 
!@@ viscosity (using arithmetic averaging) using the values already stored on the quadrature points. 
!@@ \[
!@@ \langle \rho \rangle_e =\frac{1}{V_e} \int_{\Omega_e} \rho dV
!@@ \]
!@@ It also returns the min/max values of these quantities.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

dens_min=+1d30
dens_max=-1d30
visc_min=+1d30
visc_max=-1d30
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
   dens_min=min(dens_min,mesh(iel)%rho_avrg)
   dens_max=max(dens_max,mesh(iel)%rho_avrg)
   visc_min=min(visc_min,mesh(iel)%eta_avrg)
   visc_max=max(visc_max,mesh(iel)%eta_avrg)
   vol_min=min(vol_min,mesh(iel)%vol)
   vol_max=max(vol_max,mesh(iel)%vol)
end do

write(*,'(a,2es10.3)') shift//'rho_avrg (m/M):',dens_min,dens_max
write(*,'(a,2es10.3)') shift//'eta_avrg (m/M):',visc_min,visc_max

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'comp_elemental_rho_eta_vol:',elapsed,' s     |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
