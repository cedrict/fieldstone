!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{basis\_functions\_V.f90}
!@@ This file contains 3 functions: 
!==================================================================================================!

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
implicit none
integer, intent(in) :: mV,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: NV(mV)
real(8), external :: Bubble
character(len=4), intent(in) :: pair
real(8), parameter :: aa=8d0/27d0
real(8), parameter :: bb=10d0/21d0
real(8), parameter :: cc=4d0/21d0
real(8), parameter :: dd=64d0/63d0
real(8), parameter :: ee=8d0/63d0
real(8) b1,b2

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
   case('q1q1')
      b1=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1-r)*(1-s)*(1-t)  
      b2=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1+r)*(1+s)*(1+t)
      NV( 1)=0.125*(1-r)*(1-s)*(1-t) -aa*b1
      NV( 2)=0.125*(1+r)*(1-s)*(1-t) -aa*bb*b1-aa*cc*b2
      NV( 3)=0.125*(1+r)*(1+s)*(1-t) -aa*cc*b1-aa*bb*b2
      NV( 4)=0.125*(1-r)*(1+s)*(1-t) -aa*bb*b1-aa*cc*b2
      NV( 5)=0.125*(1-r)*(1-s)*(1+t) -aa*bb*b1-aa*cc*b2
      NV( 6)=0.125*(1+r)*(1-s)*(1+t) -aa*cc*b1-aa*bb*b2
      NV( 7)=0.125*(1+r)*(1+s)*(1+t) -aa*b2
      NV( 8)=0.125*(1-r)*(1+s)*(1+t) -aa*cc*b1-aa*bb*b2
      NV( 9)= dd*b1-ee*b2
      NV(10)=-ee*b1+dd*b2
   case default
      stop 'pb in NNV'
   end select
end if

end subroutine

!==========================================================

subroutine dNNVdr(r,s,t,dNVdr,mV,ndim,pair)
implicit none
integer, intent(in) :: mV,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNVdr(mV)
real(8), external :: dBubbledr
character(len=4), intent(in) :: pair

real(8), parameter :: aa=8d0/27d0
real(8), parameter :: bb=10d0/21d0
real(8), parameter :: cc=4d0/21d0
real(8), parameter :: dd=64d0/63d0
real(8), parameter :: ee=8d0/63d0
real(8) db1dr,db2dr

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
      stop 'pb in dNNVdr'
   end select
end if

if (ndim==3) then
   select case(pair)
   case('q1p0')
      dNVdr(1)=-0.125*(1-s)*(1-t)
      dNVdr(2)=+0.125*(1-s)*(1-t)
      dNVdr(3)=+0.125*(1+s)*(1-t)
      dNVdr(4)=-0.125*(1+s)*(1-t)
      dNVdr(5)=-0.125*(1-s)*(1+t)
      dNVdr(6)=+0.125*(1-s)*(1+t)
      dNVdr(7)=+0.125*(1+s)*(1+t)
      dNVdr(8)=-0.125*(1+s)*(1+t)
   case('q1q1')
      db1dr=(27d0/32d0)**3*(1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2) 
      db2dr=(27d0/32d0)**3*(1-s**2)*(1-t**2)*(1+s)*(1+t)*( 1-2*r-3*r**2)
      dNVdr(01)=-0.125*(1-s)*(1-t) -aa*db1dr  
      dNVdr(02)=+0.125*(1-s)*(1-t) -aa*bb*db1dr-aa*cc*db2dr
      dNVdr(03)=+0.125*(1+s)*(1-t) -aa*cc*db1dr-aa*bb*db2dr
      dNVdr(04)=-0.125*(1+s)*(1-t) -aa*bb*db1dr-aa*cc*db2dr
      dNVdr(05)=-0.125*(1-s)*(1+t) -aa*bb*db1dr-aa*cc*db2dr
      dNVdr(06)=+0.125*(1-s)*(1+t) -aa*cc*db1dr-aa*bb*db2dr
      dNVdr(07)=+0.125*(1+s)*(1+t) -aa*db2dr
      dNVdr(08)=-0.125*(1+s)*(1+t) -aa*cc*db1dr-aa*bb*db2dr
      dNVdr(09)= dd*db1dr-ee*db2dr
      dNVdr(10)=-ee*db1dr+dd*db2dr
   case default
      stop 'pb in dNNVdr'
   end select
end if

end subroutine

!==========================================================

subroutine dNNVds(r,s,t,dNVds,mV,ndim,pair)
implicit none
integer, intent(in) :: mV,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNVds(mV)
real(8), external :: dBubbleds
character(len=4), intent(in) :: pair

real(8), parameter :: aa=8d0/27d0
real(8), parameter :: bb=10d0/21d0
real(8), parameter :: cc=4d0/21d0
real(8), parameter :: dd=64d0/63d0
real(8), parameter :: ee=8d0/63d0
real(8) db1ds,db2ds

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
      stop 'pb in dNNVds'
   end select
end if

if (ndim==3) then
   select case(pair)
   case('q1p0')
      dNVds(1)=-0.125*(1-r)*(1-t)
      dNVds(2)=-0.125*(1+r)*(1-t)
      dNVds(3)=+0.125*(1+r)*(1-t)
      dNVds(4)=+0.125*(1-r)*(1-t)
      dNVds(5)=-0.125*(1-r)*(1+t)
      dNVds(6)=-0.125*(1+r)*(1+t)
      dNVds(7)=+0.125*(1+r)*(1+t)
      dNVds(8)=+0.125*(1-r)*(1+t)
   case('q1q1')
      db1ds=(27d0/32d0)**3*(1-r**2)*(1-t**2)*(1-r)*(1-t)*(-1-2*s+3*s**2)
      db2ds=(27d0/32d0)**3*(1-r**2)*(1-t**2)*(1+r)*(1+t)*( 1-2*s-3*s**2)
      dNVds(01)=-0.125*(1-r)*(1-t) -aa*db1ds
      dNVds(02)=-0.125*(1+r)*(1-t) -aa*bb*db1ds-aa*cc*db2ds
      dNVds(03)=+0.125*(1+r)*(1-t) -aa*cc*db1ds-aa*bb*db2ds
      dNVds(04)=+0.125*(1-r)*(1-t) -aa*bb*db1ds-aa*cc*db2ds
      dNVds(05)=-0.125*(1-r)*(1+t) -aa*bb*db1ds-aa*cc*db2ds
      dNVds(06)=-0.125*(1+r)*(1+t) -aa*cc*db1ds-aa*bb*db2ds
      dNVds(07)=+0.125*(1+r)*(1+t) -aa*db2ds
      dNVds(08)=+0.125*(1-r)*(1+t) -aa*cc*db1ds-aa*bb*db2ds
      dNVds(09)= dd*db1ds-ee*db2ds
      dNVds(10)=-ee*db1ds+dd*db2ds
   case default
      stop 'pb in dNNVds'
   end select
end if

end subroutine

!==========================================================

subroutine dNNVdt(r,s,t,dNVdt,mV,ndim,pair)
implicit none
integer, intent(in) :: mV,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNVdt(mV)
real(8), external :: dBubbledt
character(len=4), intent(in) :: pair

real(8), parameter :: aa=8d0/27d0
real(8), parameter :: bb=10d0/21d0
real(8), parameter :: cc=4d0/21d0
real(8), parameter :: dd=64d0/63d0
real(8), parameter :: ee=8d0/63d0
real(8) db1dt,db2dt

select case(pair)
case('q1p0')
   dNVdt(1)=-0.125*(1-r)*(1-s)
   dNVdt(2)=-0.125*(1+r)*(1-s)
   dNVdt(3)=-0.125*(1+r)*(1+s)
   dNVdt(4)=-0.125*(1-r)*(1+s)
   dNVdt(5)=+0.125*(1-r)*(1-s)
   dNVdt(6)=+0.125*(1+r)*(1-s)
   dNVdt(7)=+0.125*(1+r)*(1+s)
   dNVdt(8)=+0.125*(1-r)*(1+s)
case('q1q1')
   db1dt=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1-r)*(1-s)*(-1-2*t+3*t**2)
   db2dt=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1+r)*(1+s)*( 1-2*t-3*t**2)
   dNVdt(01)=-0.125*(1-r)*(1-s) -aa*db1dt
   dNVdt(02)=-0.125*(1+r)*(1-s) -aa*bb*db1dt-aa*cc*db2dt
   dNVdt(03)=-0.125*(1+r)*(1+s) -aa*cc*db1dt-aa*bb*db2dt
   dNVdt(04)=-0.125*(1-r)*(1+s) -aa*bb*db1dt-aa*cc*db2dt
   dNVdt(05)=+0.125*(1-r)*(1-s) -aa*bb*db1dt-aa*cc*db2dt
   dNVdt(06)=+0.125*(1+r)*(1-s) -aa*cc*db1dt-aa*bb*db2dt
   dNVdt(07)=+0.125*(1+r)*(1+s) -aa*db2dt
   dNVdt(08)=+0.125*(1-r)*(1+s) -aa*cc*db1dt-aa*bb*db2dt
   dNVdt(09)= dd*db1dt-ee*db2dt
   dNVdt(10)=-ee*db1dt+dd*db2dt

case default
   stop 'pb in dNNVdt'
end select

end subroutine

!==========================================================



