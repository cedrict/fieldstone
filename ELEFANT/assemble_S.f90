!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assemble_S(S_el)

use module_parameters, only: mP,iel
use module_sparse
use module_mesh
use module_timing

implicit none

real(8), dimension(mP,mP), intent(in) :: S_el

integer :: k1,k2,m1,m2,k

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{assemble\_S}
!@@ This subroutine takes the elemental Schur complement $\hat{\SSS}_{el}$ and assembles
!@@ it in the global $\hat{\SSS}$ matrix.
!@@ Use break/exit in the loop?
!==================================================================================================!

call cpu_time(t3)

csrS%mat=0d0

do k1=1,mP
   m1=mesh(iel)%iconP(k1) ! global coordinate of pressure dof
      do k2=1,mP
         m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
         do k=csrS%ia(m1),csrS%ia(m1+1)-1    
            if (csrS%ja(k)==m2) then  
               csrS%mat(k)=csrS%mat(k)+S_el(k1,k2)  
            end if    
      end do
   end do
end do

call cpu_time(t4) ; time_assemble_S=time_assemble_S+t4-t3

end subroutine

!==================================================================================================!
!==================================================================================================!
