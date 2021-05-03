!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{basis\_functions\_L.f90}
!@@ This file contains the basis functions associated with the vertices/corners
!@@ of the element, i.e. P1 for triangles, Q1 for quads. 
!@@ I have used the 'L' letter ('Linear') because 'V' for vertices or 'C' for 
!@@ corners was problematic.
!==================================================================================================!

subroutine NNL(r,s,t,NL,mL,ndim,pair)
implicit none
integer, intent(in) :: mL,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: NL(mL)
character(len=4), intent(in) :: pair

if (ndim==2) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      NL(1)=0.25*(1-r)*(1-s)
      NL(2)=0.25*(1+r)*(1-s)
      NL(3)=0.25*(1+r)*(1+s)
      NL(4)=0.25*(1-r)*(1+s)
   case default
      stop 'pb in NNL'
   end select
end if

if (ndim==3) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      NL(1)=0.125*(1-r)*(1-s)*(1-t)
      NL(2)=0.125*(1+r)*(1-s)*(1-t)
      NL(3)=0.125*(1+r)*(1+s)*(1-t)
      NL(4)=0.125*(1-r)*(1+s)*(1-t)
      NL(5)=0.125*(1-r)*(1-s)*(1+t)
      NL(6)=0.125*(1+r)*(1-s)*(1+t)
      NL(7)=0.125*(1+r)*(1+s)*(1+t)
      NL(8)=0.125*(1-r)*(1+s)*(1+t)
   case default
      stop 'pb in NNL'
   end select
end if

end subroutine

!==========================================================

subroutine dNNLdr(r,s,t,dNLdr,mL,ndim,pair)
implicit none
integer, intent(in) :: mL,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNLdr(mL)
character(len=4), intent(in) :: pair

if (ndim==2) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      dNLdr(1)=-0.25*(1-s)
      dNLdr(2)=+0.25*(1-s)
      dNLdr(3)=+0.25*(1+s)
      dNLdr(4)=-0.25*(1+s)
   case default
      stop 'pb in dNNLdr'
   end select
end if

if (ndim==3) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      dNLdr(1)=-0.125*(1-s)*(1-t)
      dNLdr(2)=+0.125*(1-s)*(1-t)
      dNLdr(3)=+0.125*(1+s)*(1-t)
      dNLdr(4)=-0.125*(1+s)*(1-t)
      dNLdr(5)=-0.125*(1-s)*(1+t)
      dNLdr(6)=+0.125*(1-s)*(1+t)
      dNLdr(7)=+0.125*(1+s)*(1+t)
      dNLdr(8)=-0.125*(1+s)*(1+t)
   case default
      stop 'pb in dNNLdr'
   end select
end if

end subroutine

!==========================================================

subroutine dNNLds(r,s,t,dNLds,mL,ndim,pair)
implicit none
integer, intent(in) :: mL,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNLds(mL)
character(len=4), intent(in) :: pair

if (ndim==2) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      dNLds(1)=-0.25*(1-r)
      dNLds(2)=-0.25*(1+r)
      dNLds(3)=+0.25*(1+r)
      dNLds(4)=+0.25*(1-r)
   case default
      stop 'pb in dNNLds'
   end select
end if

if (ndim==3) then
   select case(pair)
   case('q1p0','q1q1','q2q1')
      dNLds(1)=-0.125*(1-r)*(1-t)
      dNLds(2)=-0.125*(1+r)*(1-t)
      dNLds(3)=+0.125*(1+r)*(1-t)
      dNLds(4)=+0.125*(1-r)*(1-t)
      dNLds(5)=-0.125*(1-r)*(1+t)
      dNLds(6)=-0.125*(1+r)*(1+t)
      dNLds(7)=+0.125*(1+r)*(1+t)
      dNLds(8)=+0.125*(1-r)*(1+t)
   case default
      stop 'pb in dNNLds'
   end select
end if

end subroutine

!==========================================================

subroutine dNNLdt(r,s,t,dNLdt,mL,ndim,pair)
implicit none
integer, intent(in) :: mL,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNLdt(mL)
character(len=4), intent(in) :: pair

select case(pair)
case('q1p0','q1q1','q2q1')
   dNLdt(1)=-0.125*(1-r)*(1-s)
   dNLdt(2)=-0.125*(1+r)*(1-s)
   dNLdt(3)=-0.125*(1+r)*(1+s)
   dNLdt(4)=-0.125*(1-r)*(1+s)
   dNLdt(5)=+0.125*(1-r)*(1-s)
   dNLdt(6)=+0.125*(1+r)*(1-s)
   dNLdt(7)=+0.125*(1+r)*(1+s)
   dNLdt(8)=+0.125*(1-r)*(1+s)
case default
   stop 'pb in dNNLdt'
end select

end subroutine

!==========================================================
