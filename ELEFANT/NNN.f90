!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine NNN(r,s,t,N,m,ndim,space,caller)

use module_constants, only: aa,bb,cc,dd,ee

implicit none
integer, intent(in) :: m,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: N(m)
character(len=4), intent(in) :: space
integer, intent(in) :: caller
real(8), external :: Bubble
real(8) Nmr,Nlr,Nrr,Nls,Nms,Nrs,Nlt,Nmt,Nrt
real(8) b1,b2
real(8) N1r,N1s,N2r,N2s,N3r,N3s,N4r,N4s,N5r,N5s

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{NNN}
!@@ Spaces supported so far:
!@@ \begin{itemize}
!@@ \item 2D: Q0, Q1, Q2, Q3, Q4, Q1+, P1, P1+, P2, P2+, P3
!@@ \item 3D: Q0, Q1, Q2, Q1++
!@@ \end{itemize}
!==================================================================================================!

!print *,'->',caller,space

if (ndim==2) then

   select case(space)
   case('__Q0')
      N(1)=1
   case('__Q1')
      N(1)=0.25*(1-r)*(1-s)
      N(2)=0.25*(1+r)*(1-s)
      N(3)=0.25*(1+r)*(1+s)
      N(4)=0.25*(1-r)*(1+s)
   case('__Q2')
      N(1)= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s-1.d0)
      N(2)=      (1.d0-r**2) * 0.5d0*s*(s-1.d0)
      N(3)= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s-1.d0)
      N(4)= 0.5d0*r*(r-1.d0) *      (1.d0-s**2)
      N(5)=      (1.d0-r**2) *      (1.d0-s**2)
      N(6)= 0.5d0*r*(r+1.d0) *      (1.d0-s**2)
      N(7)= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s+1.d0)
      N(8)=      (1.d0-r**2) * 0.5d0*s*(s+1.d0)
      N(9)= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s+1.d0)
   case('_Q1+')
      N(1)=0.25*(1-r)*(1-s)-0.25d0*Bubble(r,s)
      N(2)=0.25*(1+r)*(1-s)-0.25d0*Bubble(r,s)
      N(3)=0.25*(1+r)*(1+s)-0.25d0*Bubble(r,s)
      N(4)=0.25*(1-r)*(1+s)-0.25d0*Bubble(r,s)
      N(5)=Bubble(r,s)      
   case('__Q3')
      N1r=(-1    +r +9*r**2 - 9*r**3)/16
      N2r=(+9 -27*r -9*r**2 +27*r**3)/16
      N3r=(+9 +27*r -9*r**2 -27*r**3)/16
      N4r=(-1    -r +9*r**2 + 9*r**3)/16
      N1s=(-1    +s +9*s**2 - 9*s**3)/16
      N2s=(+9 -27*s -9*s**2 +27*s**3)/16
      N3s=(+9 +27*s -9*s**2 -27*s**3)/16
      N4s=(-1    -s +9*s**2 + 9*s**3)/16
      N(01)= N1r*N1s ; N(02)= N2r*N1s ; N(03)= N3r*N1s ; N(04)= N4r*N1s 
      N(05)= N1r*N2s ; N(06)= N2r*N2s ; N(07)= N3r*N2s ; N(08)= N4r*N2s 
      N(09)= N1r*N3s ; N(10)= N2r*N3s ; N(11)= N3r*N3s ; N(12)= N4r*N3s 
      N(13)= N1r*N4s ; N(14)= N2r*N4s ; N(15)= N3r*N4s ; N(16)= N4r*N4s
   case('__Q4')
      N1r=(    r -   r**2 -4*r**3 + 4*r**4)/6
      N2r=( -8*r +16*r**2 +8*r**3 -16*r**4)/6
      N3r=(1     - 5*r**2         + 4*r**4)
      N4r=(  8*r +16*r**2 -8*r**3 -16*r**4)/6
      N5r=(   -r -   r**2 +4*r**3 + 4*r**4)/6
      N1s=(    s -   s**2 -4*s**3 + 4*s**4)/6
      N2s=( -8*s +16*s**2 +8*s**3 -16*s**4)/6
      N3s=(1     - 5*s**2         + 4*s**4)
      N4s=(  8*s +16*s**2 -8*s**3 -16*s**4)/6
      N5s=(   -s -   s**2 +4*s**3 + 4*s**4)/6
      N(01)= N1r*N1s ; N(02)= N2r*N1s ; N(03)= N3r*N1s ; N(04)= N4r*N1s ; N(05)= N5r*N1s
      N(06)= N1r*N2s ; N(07)= N2r*N2s ; N(08)= N3r*N2s ; N(09)= N4r*N2s ; N(10)= N5r*N2s
      N(11)= N1r*N3s ; N(12)= N2r*N3s ; N(13)= N3r*N3s ; N(14)= N4r*N3s ; N(15)= N5r*N3s
      N(16)= N1r*N4s ; N(17)= N2r*N4s ; N(18)= N3r*N4s ; N(19)= N4r*N4s ; N(20)= N5r*N4s
      N(21)= N1r*N5s ; N(22)= N2r*N5s ; N(23)= N3r*N5s ; N(24)= N4r*N5s ; N(25)= N5r*N5s
   case('__P1')
      N(1)=1d0-r-s
      N(2)=r
      N(3)=s
   case('_P1+')
      N(1)=1-r-s-9*(1-r-s)*r*s
      N(2)=  r  -9*(1-r-s)*r*s
      N(3)=    s-9*(1-r-s)*r*s
      N(4)=     27*(1-r-s)*r*s
   case('__P2')
      N(1)= 1-3*r-3*s+2*r**2+4*r*s+2*s**2 
      N(2)= -r+2*r**2
      N(3)= -s+2*s**2
      N(4)= 4*r-4*r**2-4*r*s
      N(5)= 4*r*s 
      N(6)= 4*s-4*r*s-4*s**2
   case('_P2+')
      N(1)= (1-r-s)*(1-2*r-2*s+ 3*r*s)
      N(2)= r*(2*r -1 + 3*s-3*r*s-3*s**2 )
      N(3)= s*(2*s -1 + 3*r-3*r**2-3*r*s )
      N(4)= 4*(1-r-s)*r*(1-3*s)
      N(5)= 4*r*s*(-2+3*r+3*s)
      N(6)= 4*(1-r-s)*s*(1-3*r)
      N(7)= 27*(1-r-s)*r*s
   case('__P3')
      N( 1)=0.5*(2 -11*r - 11*s + 18*r**2 + 36*r*s + 18*s**2 -9*r**3 -27*r**2*s -27*r*s**2 -9*s**3)
      N( 2)=0.5*(18*r-45*r**2-45*r*s +27*r**3 +54*r**2*s+27*r*s**2  )
      N( 3)=0.5*(-9*r+36*r**2+9*r*s -27*r**3 -27*r**2*s  )
      N( 4)=0.5*(2*r-9*r**2+9*r**3  )
      N( 5)=0.5*(18*s -45*r*s-45*s**2+27*r**2*s+54*r*s**2+27*s**3  )
      N( 6)=0.5*(54*r*s-54*r**2*s-54*r*s**2   )
      N( 7)=0.5*(-9*r*s+27*r**2*s   )
      N( 8)=0.5*(-9*s+9*r*s+36*s**2-27*r*s**2-27*s**3  )
      N( 9)=0.5*(-9*r*s+27*r*s**2 )
      N(10)=0.5*(2*s-9*s**2+9*s**3  )
   case default
      stop 'unknown 2D space in NNN'
   end select

else ! ndim=3

   select case(space)
   case('__Q0')
      N(1)=1
   case('__Q1')
      N(1)=0.125*(1-r)*(1-s)*(1-t)
      N(2)=0.125*(1+r)*(1-s)*(1-t)
      N(3)=0.125*(1+r)*(1+s)*(1-t)
      N(4)=0.125*(1-r)*(1+s)*(1-t)
      N(5)=0.125*(1-r)*(1-s)*(1+t)
      N(6)=0.125*(1+r)*(1-s)*(1+t)
      N(7)=0.125*(1+r)*(1+s)*(1+t)
      N(8)=0.125*(1-r)*(1+s)*(1+t)
   case('__Q2')
      Nlr=0.5d0*r*(r-1d0) ; Nls=0.5d0*s*(s-1d0) ; Nlt=0.5d0*t*(t-1d0) 
      Nmr=(1d0-r**2)      ; Nms=(1d0-s**2)      ; Nmt=(1d0-t**2)       
      Nrr=0.5d0*r*(r+1d0) ; Nrs=0.5d0*s*(s+1d0) ; Nrt=0.5d0*t*(t+1d0) 
      N(01)= Nlr * Nls * Nlt 
      N(02)= Nmr * Nls * Nlt 
      N(03)= Nrr * Nls * Nlt 
      N(04)= Nlr * Nms * Nlt 
      N(05)= Nmr * Nms * Nlt 
      N(06)= Nrr * Nms * Nlt 
      N(07)= Nlr * Nrs * Nlt 
      N(08)= Nmr * Nrs * Nlt 
      N(09)= Nrr * Nrs * Nlt 
      N(10)= Nlr * Nls * Nmt 
      N(11)= Nmr * Nls * Nmt 
      N(12)= Nrr * Nls * Nmt 
      N(13)= Nlr * Nms * Nmt 
      N(14)= Nmr * Nms * Nmt 
      N(15)= Nrr * Nms * Nmt 
      N(16)= Nlr * Nrs * Nmt 
      N(17)= Nmr * Nrs * Nmt 
      N(18)= Nrr * Nrs * Nmt 
      N(19)= Nlr * Nls * Nrt 
      N(20)= Nmr * Nls * Nrt 
      N(21)= Nrr * Nls * Nrt 
      N(22)= Nlr * Nms * Nrt 
      N(23)= Nmr * Nms * Nrt 
      N(24)= Nrr * Nms * Nrt 
      N(25)= Nlr * Nrs * Nrt 
      N(26)= Nmr * Nrs * Nrt 
      N(27)= Nrr * Nrs * Nrt 
   case('Q1++')
      b1=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1-r)*(1-s)*(1-t)  
      b2=(27d0/32d0)**3*(1-r**2)*(1-s**2)*(1-t**2)*(1+r)*(1+s)*(1+t)
      N(01)=0.125*(1-r)*(1-s)*(1-t) -aa*b1
      N(02)=0.125*(1+r)*(1-s)*(1-t) -aa*bb*b1-aa*cc*b2
      N(03)=0.125*(1+r)*(1+s)*(1-t) -aa*cc*b1-aa*bb*b2
      N(04)=0.125*(1-r)*(1+s)*(1-t) -aa*bb*b1-aa*cc*b2
      N(05)=0.125*(1-r)*(1-s)*(1+t) -aa*bb*b1-aa*cc*b2
      N(06)=0.125*(1+r)*(1-s)*(1+t) -aa*cc*b1-aa*bb*b2
      N(07)=0.125*(1+r)*(1+s)*(1+t) -aa*b2
      N(08)=0.125*(1-r)*(1+s)*(1+t) -aa*cc*b1-aa*bb*b2
      N(09)= dd*b1-ee*b2
      N(10)=-ee*b1+dd*b2
   case default
      stop 'unknown 3D space in NNN'
   end select

end if

end subroutine

!==================================================================================================!
!==================================================================================================!

function Bubble(r,s)
implicit none
real(8) r,s,Bubble
real(8), parameter :: beta=0.25d0
Bubble=(1-r**2)*(1-s**2)*(1+beta*(r+s))
end function


