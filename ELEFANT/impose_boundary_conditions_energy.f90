!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine impose_boundary_conditions_energy(Ael,bel)

use global_parameters, only: iel,mT
use structures

implicit none

real(8), dimension(mT,mT), intent(out) :: Ael
real(8), dimension(mT), intent(out) :: bel

integer i,j
real(8) fixt,Aref

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{impose\_boundary\_conditions\_energy}
!@@ This routine takes as argument the elemental matrices and vectors Ael, Bel
!@@ and applies the boundary conditions to them.
!==================================================================================================!

do i=1,mT
   if (mesh(iel)%fix_T(i)) then
      fixt=mesh(iel)%T(i)
      Aref=Ael(i,i)
      do j=1,mT
         Bel(j)=Bel(j)-Ael(j,i)*fixt
         Ael(i,j)=0d0
         Ael(j,i)=0d0
      enddo
      Ael(i,i)=Aref
      Bel(i)=Aref*fixt
   endif
enddo

end subroutine

!==================================================================================================!
!==================================================================================================!
