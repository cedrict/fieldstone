!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_schur_complement(K_el,G_el,S_el)

use module_parameters, only: mU,mV,mW,mP,mVel
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none

real(8), intent(inout) :: K_el(mVel,mVel)
real(8), intent(in) :: G_el(mVel,mP)
real(8), intent(inout) :: S_el(mP,mP)

integer :: k1,k2

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_elemental\_schur\_complement}
!@@
!==================================================================================================!


   ! build elemental approximate Schur complement
   ! only keep diagonal of K
   ! should this happen before bc are applied?
   ! add C_el ?
   do k1=1,mV
   do k2=1,mV
      if (k1/=k2) K_el(k1,k2)=0d0
      if (k1==k2) K_el(k1,k2)=1d0/K_el(k1,k2)
   end do
   end do
   S_el=matmul(transpose(G_el),matmul(K_el,G_el))

end subroutine

!==================================================================================================!
!==================================================================================================!
