!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine project_V_onto_L(elt,uL,vL,wL,mL)

use module_parameters, only: mV,ndim,pair
use module_mesh, only: element

implicit none

type(element), intent(in) :: elt
integer, intent(in) :: mL
real(8), intent(out) :: uL(mL),vL(mL),wL(mL)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{template}
!@@
!==================================================================================================!

select case(pair)
case('q1p0','q1q1')
   uL=elt%u(1:mV)
   vL=elt%v(1:mV)
   wL=elt%w(1:mV)
case('q2q1','q2p1')
   if (ndim==2) then
   uL(1)=elt%u(1)
   uL(2)=elt%u(3)
   uL(3)=elt%u(9)
   uL(4)=elt%u(7)
   vL(1)=elt%v(1)
   vL(2)=elt%v(3)
   vL(3)=elt%v(9)
   vL(4)=elt%v(7)
   end if
   if (ndim==3) then
   stop 'project_V_onto_L: q2q1 in 3D not implemented'
   end if
case default
   stop 'project_V_onto_L: unknown pair'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine project_P_onto_L(elt,pL,mL)

use module_parameters, only: mP,pair
use module_mesh, only: element

implicit none

type(element), intent(in) :: elt
integer, intent(in) :: mL
real(8), intent(out) :: pL(mL)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{template}
!@@
!==================================================================================================!

select case(pair)
case('q1p0')
   pL(1:mL)=elt%p(1)
case('q1q1','q2q1')
   pL=elt%p(1:mP)
case default
   stop 'project_P_onto_L: unknown pair'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine project_Q_onto_L(elt,qL,mL)

use module_parameters, only: pair,ndim
use module_mesh, only: element

implicit none

type(element), intent(in) :: elt
integer, intent(in) :: mL
real(8), intent(out) :: qL(mL)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{template}
!@@
!==================================================================================================!

select case(pair)
case('q1p0','q1q1')
   qL(1:mL)=elt%q(1:mL)
case('q2q1')
   if (ndim==2) then
   qL(1)=elt%q(1)
   qL(2)=elt%q(3)
   qL(3)=elt%q(9)
   qL(4)=elt%q(7)
   end if
case default
   stop 'project_Q_onto_L: unknown pair'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine project_T_onto_L(elt,TL,mL)

use module_parameters, only: pair,ndim
use module_mesh, only: element

implicit none

type(element), intent(in) :: elt
integer, intent(in) :: mL
real(8), intent(out) :: TL(mL)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{template}
!@@
!==================================================================================================!

select case(pair)
case('q1p0','q1q1')
   TL(1:mL)=elt%T(1:mL)
case('q2q1')
   if (ndim==2) then
   TL(1)=elt%q(1)
   TL(2)=elt%q(3)
   TL(3)=elt%q(9)
   TL(4)=elt%q(7)
   end if
case default
   stop 'project_T_onto_L: unknown pair'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

