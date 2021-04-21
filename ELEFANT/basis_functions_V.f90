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
real(8) Nmr,Nlr,Nrr,Nls,Nms,Nrs,Nlt,Nmt,Nrt
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
   case('q2q1')
      NV(1)= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s-1.d0)
      NV(2)=      (1.d0-r**2) * 0.5d0*s*(s-1.d0)
      NV(3)= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s-1.d0)
      NV(4)= 0.5d0*r*(r-1.d0) *      (1.d0-s**2)
      NV(5)=      (1.d0-r**2) *      (1.d0-s**2)
      NV(6)= 0.5d0*r*(r+1.d0) *      (1.d0-s**2)
      NV(7)= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s+1.d0)
      NV(8)=      (1.d0-r**2) * 0.5d0*s*(s+1.d0)
      NV(9)= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s+1.d0)
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
   case('q2q1')
      Nlr=0.5d0*r*(r-1d0) ; Nls=0.5d0*s*(s-1d0) ; Nlt=0.5d0*t*(t-1d0) 
      Nmr=(1d0-r**2)      ; Nms=(1d0-s**2)      ; Nmt=(1d0-t**2)       
      Nrr=0.5d0*r*(r+1d0) ; Nrs=0.5d0*s*(s+1d0) ; Nrt=0.5d0*t*(t+1d0) 
      NV(01)= Nlr * Nls * Nlt 
      NV(02)= Nmr * Nls * Nlt 
      NV(03)= Nrr * Nls * Nlt 
      NV(04)= Nlr * Nms * Nlt 
      NV(05)= Nmr * Nms * Nlt 
      NV(06)= Nrr * Nms * Nlt 
      NV(07)= Nlr * Nrs * Nlt 
      NV(08)= Nmr * Nrs * Nlt 
      NV(09)= Nrr * Nrs * Nlt 
      NV(10)= Nlr * Nls * Nmt 
      NV(11)= Nmr * Nls * Nmt 
      NV(12)= Nrr * Nls * Nmt 
      NV(13)= Nlr * Nms * Nmt 
      NV(14)= Nmr * Nms * Nmt 
      NV(15)= Nrr * Nms * Nmt 
      NV(16)= Nlr * Nrs * Nmt 
      NV(17)= Nmr * Nrs * Nmt 
      NV(18)= Nrr * Nrs * Nmt 
      NV(19)= Nlr * Nls * Nrt 
      NV(20)= Nmr * Nls * Nrt 
      NV(21)= Nrr * Nls * Nrt 
      NV(22)= Nlr * Nms * Nrt 
      NV(23)= Nmr * Nms * Nrt 
      NV(24)= Nrr * Nms * Nrt 
      NV(25)= Nlr * Nrs * Nrt 
      NV(26)= Nmr * Nrs * Nrt 
      NV(27)= Nrr * Nrs * Nrt 
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
real(8) dNmr,dNlr,dNrr,Nls,Nms,Nrs,Nlt,Nmt,Nrt
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
   case('q2q1')
dNVdr(1)= 0.5d0*(2.d0*r-1.d0) * 0.5d0*t*(t-1)
dNVdr(2)=           (-2.d0*r) * 0.5d0*t*(t-1)
dNVdr(3)= 0.5d0*(2.d0*r+1.d0) * 0.5d0*t*(t-1)
dNVdr(4)= 0.5d0*(2.d0*r-1.d0) *   (1.d0-t**2)
dNVdr(5)=           (-2.d0*r) *   (1.d0-t**2)
dNVdr(6)= 0.5d0*(2.d0*r+1.d0) *   (1.d0-t**2)
dNVdr(7)= 0.5d0*(2.d0*r-1.d0) * 0.5d0*t*(t+1)
dNVdr(8)=           (-2.d0*r) * 0.5d0*t*(t+1)
dNVdr(9)= 0.5d0*(2.d0*r+1.d0) * 0.5d0*t*(t+1)

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
   case('q2q1')

      dNlr=r-0.5d0 ; Nls=0.5d0*s*(s-1d0) ; Nlt=0.5d0*t*(t-1d0) 
      dNmr=-2d0*r  ; Nms=(1d0-s**2)      ; Nmt=(1d0-t**2)       
      dNrr=r+0.5d0 ; Nrs=0.5d0*s*(s+1d0) ; Nrt=0.5d0*t*(t+1d0) 
      dNVdr(01)= dNlr * Nls * Nlt 
      dNVdr(02)= dNmr * Nls * Nlt 
      dNVdr(03)= dNrr * Nls * Nlt 
      dNVdr(04)= dNlr * Nms * Nlt 
      dNVdr(05)= dNmr * Nms * Nlt 
      dNVdr(06)= dNrr * Nms * Nlt 
      dNVdr(07)= dNlr * Nrs * Nlt 
      dNVdr(08)= dNmr * Nrs * Nlt 
      dNVdr(09)= dNrr * Nrs * Nlt 
      dNVdr(10)= dNlr * Nls * Nmt 
      dNVdr(11)= dNmr * Nls * Nmt 
      dNVdr(12)= dNrr * Nls * Nmt 
      dNVdr(13)= dNlr * Nms * Nmt 
      dNVdr(14)= dNmr * Nms * Nmt 
      dNVdr(15)= dNrr * Nms * Nmt 
      dNVdr(16)= dNlr * Nrs * Nmt 
      dNVdr(17)= dNmr * Nrs * Nmt 
      dNVdr(18)= dNrr * Nrs * Nmt 
      dNVdr(19)= dNlr * Nls * Nrt 
      dNVdr(20)= dNmr * Nls * Nrt 
      dNVdr(21)= dNrr * Nls * Nrt 
      dNVdr(22)= dNlr * Nms * Nrt 
      dNVdr(23)= dNmr * Nms * Nrt 
      dNVdr(24)= dNrr * Nms * Nrt 
      dNVdr(25)= dNlr * Nrs * Nrt 
      dNVdr(26)= dNmr * Nrs * Nrt 
      dNVdr(27)= dNrr * Nrs * Nrt 
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
real(8) Nmr,Nlr,Nrr,dNls,dNms,dNrs,Nlt,Nmt,Nrt
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
   case('q2q1')
      dNVds(1)= 0.5d0*r*(r-1.d0) * 0.5d0*(2.d0*s-1.d0)
      dNVds(2)=      (1.d0-r**2) * 0.5d0*(2.d0*s-1.d0)
      dNVds(3)= 0.5d0*r*(r+1.d0) * 0.5d0*(2.d0*s-1.d0)
      dNVds(4)= 0.5d0*r*(r-1.d0) *           (-2.d0*s)
      dNVds(5)=      (1.d0-r**2) *           (-2.d0*s)
      dNVds(6)= 0.5d0*r*(r+1.d0) *           (-2.d0*s)
      dNVds(7)= 0.5d0*r*(r-1.d0) * 0.5d0*(2.d0*s+1.d0)
      dNVds(8)=      (1.d0-r**2) * 0.5d0*(2.d0*s+1.d0)
      dNVds(9)= 0.5d0*r*(r+1.d0) * 0.5d0*(2.d0*s+1.d0)
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
   case('q2q1')
      Nlr=0.5d0*r*(r-1d0) ; dNls=s-0.5d0 ; Nlt=0.5d0*t*(t-1d0) 
      Nmr=(1d0-r**2)      ; dNms=-2d0*s  ; Nmt=(1d0-t**2)       
      Nrr=0.5d0*r*(r+1d0) ; dNrs=s+0.5d0 ; Nrt=0.5d0*t*(t+1d0) 
      dNVds(01)= Nlr * dNls * Nlt 
      dNVds(02)= Nmr * dNls * Nlt 
      dNVds(03)= Nrr * dNls * Nlt 
      dNVds(04)= Nlr * dNms * Nlt 
      dNVds(05)= Nmr * dNms * Nlt 
      dNVds(06)= Nrr * dNms * Nlt 
      dNVds(07)= Nlr * dNrs * Nlt 
      dNVds(08)= Nmr * dNrs * Nlt 
      dNVds(09)= Nrr * dNrs * Nlt 
      dNVds(10)= Nlr * dNls * Nmt 
      dNVds(11)= Nmr * dNls * Nmt 
      dNVds(12)= Nrr * dNls * Nmt 
      dNVds(13)= Nlr * dNms * Nmt 
      dNVds(14)= Nmr * dNms * Nmt 
      dNVds(15)= Nrr * dNms * Nmt 
      dNVds(16)= Nlr * dNrs * Nmt 
      dNVds(17)= Nmr * dNrs * Nmt 
      dNVds(18)= Nrr * dNrs * Nmt 
      dNVds(19)= Nlr * dNls * Nrt 
      dNVds(20)= Nmr * dNls * Nrt 
      dNVds(21)= Nrr * dNls * Nrt 
      dNVds(22)= Nlr * dNms * Nrt 
      dNVds(23)= Nmr * dNms * Nrt 
      dNVds(24)= Nrr * dNms * Nrt 
      dNVds(25)= Nlr * dNrs * Nrt 
      dNVds(26)= Nmr * dNrs * Nrt 
      dNVds(27)= Nrr * dNrs * Nrt 
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
real(8) Nmr,Nlr,Nrr,Nls,Nms,Nrs,dNlt,dNmt,dNrt
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
case('q2q1')
   Nlr=0.5d0*r*(r-1d0) ; Nls=0.5d0*s*(s-1d0) ; dNlt=t-0.5d0
   Nmr=(1d0-r**2)      ; Nms=(1d0-s**2)      ; dNmt=-2*t
   Nrr=0.5d0*r*(r+1d0) ; Nrs=0.5d0*s*(s+1d0) ; dNrt=t+0.5d0
   dNVdt(01)= Nlr * Nls * dNlt 
   dNVdt(02)= Nmr * Nls * dNlt 
   dNVdt(03)= Nrr * Nls * dNlt 
   dNVdt(04)= Nlr * Nms * dNlt 
   dNVdt(05)= Nmr * Nms * dNlt 
   dNVdt(06)= Nrr * Nms * dNlt 
   dNVdt(07)= Nlr * Nrs * dNlt 
   dNVdt(08)= Nmr * Nrs * dNlt 
   dNVdt(09)= Nrr * Nrs * dNlt 
   dNVdt(10)= Nlr * Nls * dNmt 
   dNVdt(11)= Nmr * Nls * dNmt 
   dNVdt(12)= Nrr * Nls * dNmt 
   dNVdt(13)= Nlr * Nms * dNmt 
   dNVdt(14)= Nmr * Nms * dNmt 
   dNVdt(15)= Nrr * Nms * dNmt 
   dNVdt(16)= Nlr * Nrs * dNmt 
   dNVdt(17)= Nmr * Nrs * dNmt 
   dNVdt(18)= Nrr * Nrs * dNmt 
   dNVdt(19)= Nlr * Nls * dNrt 
   dNVdt(20)= Nmr * Nls * dNrt 
   dNVdt(21)= Nrr * Nls * dNrt 
   dNVdt(22)= Nlr * Nms * dNrt 
   dNVdt(23)= Nmr * Nms * dNrt 
   dNVdt(24)= Nrr * Nms * dNrt 
   dNVdt(25)= Nlr * Nrs * dNrt 
   dNVdt(26)= Nmr * Nrs * dNrt 
   dNVdt(27)= Nrr * Nrs * dNrt 
case default
   stop 'pb in dNNVdt'
end select

end subroutine

!==========================================================
