!=================================================================

subroutine NNP(r,s,t,NP,mP,ndim,pair)
implicit none
integer, intent(in) :: mP,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: NP(mP)
character(len=4), intent(in) :: pair

if (ndim==2) then

   select case(pair)
   case('q1p0')
      NP(1)=1
   case('q1q1')
      NP(1)=0.25*(1-r)*(1-s)
      NP(2)=0.25*(1+r)*(1-s)
      NP(3)=0.25*(1+r)*(1+s)
      NP(4)=0.25*(1-r)*(1+s)
   case default
      stop 'pb in NNP'
   end select

end if

if (ndim==3) then

   select case(pair)
   case('q1p0')
      NP(1)=1
   case('q1q1')
      NP(1)=0.125*(1-r)*(1-s)*(1-t)
      NP(2)=0.125*(1+r)*(1-s)*(1-t)
      NP(3)=0.125*(1+r)*(1+s)*(1-t)
      NP(4)=0.125*(1-r)*(1+s)*(1-t)
      NP(5)=0.125*(1-r)*(1-s)*(1+t)
      NP(6)=0.125*(1+r)*(1-s)*(1+t)
      NP(7)=0.125*(1+r)*(1+s)*(1+t)
      NP(8)=0.125*(1-r)*(1+s)*(1+t)
   case default
      stop 'pb in NNP'
   end select

end if

end subroutine

