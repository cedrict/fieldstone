!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine dNNNdr(r,s,t,dNdr,m,ndim,space,caller)

use module_constants, only: aa,bb,cc,dd,ee

implicit none
integer, intent(in) :: m,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNdr(m)
character(len=4), intent(in) :: space
integer, intent(in) :: caller
real(8), external :: dBubbledr
real(8) dNmr,dNlr,dNrr,Nls,Nms,Nrs,Nlt,Nmt,Nrt
real(8) db1dr,db2dr,N1s,N2s,N3s,N4s,dN1rdr,dN2rdr,dN3rdr,dN4rdr

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{dNNNdr}
!@@ Spaces supported so far:
!@@ \begin{itemize}
!@@ \item 2D: $Q_1$, $Q_2$, $Q_3$, $Q_1^+$, $P_1$, $P_1^+$, $P_2$, $P_2^+$
!@@ \item 3D: $Q_1$, $Q_2$, $Q_1^{++}$
!@@ \end{itemize}
!==================================================================================================!

!print *,'->',caller,space

if (ndim==2) then

   select case(space)
   case('__Q1')
      dNdr(1)=-0.25*(1-s)
      dNdr(2)=+0.25*(1-s)
      dNdr(3)=+0.25*(1+s)
      dNdr(4)=-0.25*(1+s)
   case('__Q2')
      dNdr(1)= 05d0*(2d0*r-1.d0) * 0.5d0*t*(t-1)
      dNdr(2)=          (-2d0*r) * 0.5d0*t*(t-1)
      dNdr(3)= 05d0*(2d0*r+1.d0) * 0.5d0*t*(t-1)
      dNdr(4)= 05d0*(2d0*r-1.d0) *    (1d0-t**2)
      dNdr(5)=          (-2d0*r) *    (1d0-t**2)
      dNdr(6)= 05d0*(2d0*r+1.d0) *    (1d0-t**2)
      dNdr(7)= 05d0*(2d0*r-1.d0) * 0.5d0*t*(t+1)
      dNdr(8)=          (-2d0*r) * 0.5d0*t*(t+1)
      dNdr(9)= 05d0*(2d0*r+1.d0) * 0.5d0*t*(t+1)
   case('__Q3')
      dN1rdr=( +1d0 +18d0*r -27d0*r**2)/16d0
      dN2rdr=(-27d0 -18d0*r +81d0*r**2)/16d0
      dN3rdr=(+27d0 -18d0*r -81d0*r**2)/16d0
      dN4rdr=( -1d0 +18d0*r +27d0*r**2)/16d0
      N1s=(-1d0      +s +9d0*s**2 - 9d0*s**3)/16d0
      N2s=(+9d0 -27d0*s -9d0*s**2 +27d0*s**3)/16d0
      N3s=(+9d0 +27d0*s -9d0*s**2 -27d0*s**3)/16d0
      N4s=(-1d0      -s +9d0*s**2 + 9d0*s**3)/16d0
      dNdr(01)=dN1rdr*N1s 
      dNdr(02)=dN2rdr*N1s 
      dNdr(03)=dN3rdr*N1s 
      dNdr(04)=dN4rdr*N1s
      dNdr(05)=dN1rdr*N2s 
      dNdr(06)=dN2rdr*N2s 
      dNdr(07)=dN3rdr*N2s 
      dNdr(08)=dN4rdr*N2s
      dNdr(09)=dN1rdr*N3s 
      dNdr(10)=dN2rdr*N3s 
      dNdr(11)=dN3rdr*N3s 
      dNdr(12)=dN4rdr*N3s
      dNdr(13)=dN1rdr*N4s 
      dNdr(14)=dN2rdr*N4s 
      dNdr(15)=dN3rdr*N4s 
      dNdr(16)=dN4rdr*N4s
   case('_Q1+')
      dNdr(1)=-0.25*(1-s)-0.25d0*dBubbledr(r,s)
      dNdr(2)=+0.25*(1-s)-0.25d0*dBubbledr(r,s)
      dNdr(3)=+0.25*(1+s)-0.25d0*dBubbledr(r,s)
      dNdr(4)=-0.25*(1+s)-0.25d0*dBubbledr(r,s)
      dNdr(5)=dBubbledr(r,s)      
   case('__P1')
      dNdr(1)=-1
      dNdr(2)=1
      dNdr(3)=0
   case('_P1+')
      dNdr(1)= -1-9*(1-2*r-s)*s
      dNdr(2)=  1-9*(1-2*r-s)*s
      dNdr(3)=   -9*(1-2*r-s)*s
      dNdr(4)=   27*(1-2*r-s)*s
   case('__P2')
      dNdr(1)= -3+4*r+4*s
      dNdr(2)= -1+4*r
      dNdr(3)= 0
      dNdr(4)= 4-8*r-4*s
      dNdr(5)= 4*s
      dNdr(6)= -4*s
   case('_P2+')
      dNdr(1)= r*(4-6*s)-3*s**2+7*s-3
      dNdr(2)= r*(4-6*s)-3*s**2+3*s-1
      dNdr(3)= -3*s*(2*r+s-1)
      dNdr(4)= 4*(3*s-1)*(2*r+s-1)
      dNdr(5)= 4*s*(6*r+3*s-2)
      dNdr(6)= 4*s*(6*r+3*s-4)
      dNdr(7)=-27*s*(2*r+s-1)
   case default
      stop 'unknown 2D space in dNNNdr'
   end select

else

   select case(space)
   case('__Q1')
      dNdr(1)=-0.125*(1-s)*(1-t)
      dNdr(2)=+0.125*(1-s)*(1-t)
      dNdr(3)=+0.125*(1+s)*(1-t)
      dNdr(4)=-0.125*(1+s)*(1-t)
      dNdr(5)=-0.125*(1-s)*(1+t)
      dNdr(6)=+0.125*(1-s)*(1+t)
      dNdr(7)=+0.125*(1+s)*(1+t)
      dNdr(8)=-0.125*(1+s)*(1+t)
   case('__Q2')
      dNlr=r-0.5d0 ; Nls=0.5d0*s*(s-1d0) ; Nlt=0.5d0*t*(t-1d0) 
      dNmr=-2d0*r  ; Nms=(1d0-s**2)      ; Nmt=(1d0-t**2)       
      dNrr=r+0.5d0 ; Nrs=0.5d0*s*(s+1d0) ; Nrt=0.5d0*t*(t+1d0) 
      dNdr(01)= dNlr * Nls * Nlt 
      dNdr(02)= dNmr * Nls * Nlt 
      dNdr(03)= dNrr * Nls * Nlt 
      dNdr(04)= dNlr * Nms * Nlt 
      dNdr(05)= dNmr * Nms * Nlt 
      dNdr(06)= dNrr * Nms * Nlt 
      dNdr(07)= dNlr * Nrs * Nlt 
      dNdr(08)= dNmr * Nrs * Nlt 
      dNdr(09)= dNrr * Nrs * Nlt 
      dNdr(10)= dNlr * Nls * Nmt 
      dNdr(11)= dNmr * Nls * Nmt 
      dNdr(12)= dNrr * Nls * Nmt 
      dNdr(13)= dNlr * Nms * Nmt 
      dNdr(14)= dNmr * Nms * Nmt 
      dNdr(15)= dNrr * Nms * Nmt 
      dNdr(16)= dNlr * Nrs * Nmt 
      dNdr(17)= dNmr * Nrs * Nmt 
      dNdr(18)= dNrr * Nrs * Nmt 
      dNdr(19)= dNlr * Nls * Nrt 
      dNdr(20)= dNmr * Nls * Nrt 
      dNdr(21)= dNrr * Nls * Nrt 
      dNdr(22)= dNlr * Nms * Nrt 
      dNdr(23)= dNmr * Nms * Nrt 
      dNdr(24)= dNrr * Nms * Nrt 
      dNdr(25)= dNlr * Nrs * Nrt 
      dNdr(26)= dNmr * Nrs * Nrt 
      dNdr(27)= dNrr * Nrs * Nrt 
   case('Q1++')
      db1dr=(27d0/32d0)**3*(1-s**2)*(1-t**2)*(1-s)*(1-t)*(-1-2*r+3*r**2) 
      db2dr=(27d0/32d0)**3*(1-s**2)*(1-t**2)*(1+s)*(1+t)*( 1-2*r-3*r**2)
      dNdr(01)=-0.125*(1-s)*(1-t) -aa*db1dr  
      dNdr(02)=+0.125*(1-s)*(1-t) -aa*bb*db1dr-aa*cc*db2dr
      dNdr(03)=+0.125*(1+s)*(1-t) -aa*cc*db1dr-aa*bb*db2dr
      dNdr(04)=-0.125*(1+s)*(1-t) -aa*bb*db1dr-aa*cc*db2dr
      dNdr(05)=-0.125*(1-s)*(1+t) -aa*bb*db1dr-aa*cc*db2dr
      dNdr(06)=+0.125*(1-s)*(1+t) -aa*cc*db1dr-aa*bb*db2dr
      dNdr(07)=+0.125*(1+s)*(1+t) -aa*db2dr
      dNdr(08)=-0.125*(1+s)*(1+t) -aa*cc*db1dr-aa*bb*db2dr
      dNdr(09)= dd*db1dr-ee*db2dr
      dNdr(10)=-ee*db1dr+dd*db2dr
   case default
      stop 'unknown 3D space in dNNNdr'
   end select

end if


end subroutine

!==================================================================================================!
!==================================================================================================!

function dBubbledr(r,s)
implicit none
real(8) r,s,dBubbledr
real(8), parameter :: beta=0.25d0
dBubbledr=(s**2-1)*(-beta+3*beta*r**2+2*r*(beta*s+1))
end function

