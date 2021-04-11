
function Bubble(r,s)
implicit none
real(8) r,s,Bubble
real(8), parameter :: beta=0.25d0
Bubble=(1-r**2)*(1-s**2)*(1+beta*(r+s))
end function

function dBubbledr(r,s)
implicit none
real(8) r,s,dBubbledr
real(8), parameter :: beta=0.25d0
dBubbledr=(s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
end function

function dBubbleds(r,s)
implicit none
real(8) r,s,dBubbleds
real(8), parameter :: beta=0.25d0
dBubbleds=(r**2-1)*(-beta+2*s*(beta*r+1)+3*beta*s**2)
end function

!=================================================================

subroutine NNV(r,s,t,NV,mV,ndim,pair)
!use global_parameters
implicit none
integer, intent(in) :: mV,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: NV(mV)
real(8), external :: Bubble
character(len=4), intent(in) :: pair

if (ndim==2) then

   select case(pair)
   case('q1p0')
      NV(1)=0.25*(1-r)*(1-s)
      NV(2)=0.25*(1+r)*(1-s)
      NV(3)=0.25*(1+r)*(1+s)
      NV(4)=0.25*(1-r)*(1+s)
   case('q1q1')
      NV(1)=0.25*(1-r)*(1-s)-0.25d0*Bubble(r,s)
      NV(2)=0.25*(1+r)*(1-s)-0.25d0*Bubble(r,s)
      NV(3)=0.25*(1+r)*(1+s)-0.25d0*Bubble(r,s)
      NV(4)=0.25*(1-r)*(1+s)-0.25d0*Bubble(r,s)
      NV(5)=Bubble(r,s)      
   case default
      stop 'pb in NNV'
   end select

end if

if (ndim==3) then

   select case(pair)
   case('q1p0')
      NV(1)=0.125*(1-r)*(1-s)*(1-t)
      NV(2)=0.125*(1+r)*(1-s)*(1-t)
      NV(3)=0.125*(1+r)*(1+s)*(1-t)
      NV(4)=0.125*(1-r)*(1+s)*(1-t)
      NV(5)=0.125*(1-r)*(1-s)*(1+t)
      NV(6)=0.125*(1+r)*(1-s)*(1+t)
      NV(7)=0.125*(1+r)*(1+s)*(1+t)
      NV(8)=0.125*(1-r)*(1+s)*(1+t)
   case default
      stop 'pb in NNV'
   end select



end if

end subroutine

!==========================================================

subroutine dNNVdr(r,s,t,dNVdr)
use global_parameters
implicit none
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNVdr(mV)
real(8), external :: dBubbledr

if (ndim==2) then

   select case(pair)
   case('q1p0')
      dNVdr(1)=-0.25*(1-s)
      dNVdr(2)=+0.25*(1-s)
      dNVdr(3)=+0.25*(1+s)
      dNVdr(4)=-0.25*(1+s)
   case('q1q1')
      dNVdr(1)=-0.25*(1-s)-0.25d0*dBubbledr(r,s)
      dNVdr(2)=+0.25*(1-s)-0.25d0*dBubbledr(r,s)
      dNVdr(3)=+0.25*(1+s)-0.25d0*dBubbledr(r,s)
      dNVdr(4)=-0.25*(1+s)-0.25d0*dBubbledr(r,s)
      dNVdr(5)=dBubbledr(r,s)      
   case default
      stop 'pb in NNV'
   end select

end if

end subroutine

!==========================================================

subroutine dNNVds(r,s,t,dNVds)
use global_parameters
implicit none
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNVds(mV)
real(8), external :: dBubbleds

if (ndim==2) then

   select case(pair)
   case('q1p0')
      dNVds(1)=-0.25*(1-r)
      dNVds(2)=-0.25*(1+r)
      dNVds(3)=+0.25*(1+r)
      dNVds(4)=+0.25*(1-r)
   case('q1q1')
      dNVds(1)=-0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNVds(2)=-0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNVds(3)=+0.25*(1+r)-0.25d0*dBubbleds(r,s)
      dNVds(4)=+0.25*(1-r)-0.25d0*dBubbleds(r,s)
      dNVds(5)=dBubbleds(r,s)      
   case default
      stop 'pb in NNV'
   end select

end if

end subroutine

!==========================================================






