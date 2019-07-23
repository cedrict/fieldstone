program opla
implicit none

integer i
real(8) r,s,x,y,N1,N2,N3,N4,N5,N6,N7,N8,N9
integer, parameter :: np=12345
real(8), parameter :: x1=-1,y1=-2
real(8), parameter :: x3=3,y3=-1
real(8), parameter :: x9=2,y9=2
real(8), parameter :: x7=-3,y7=1
real(8), parameter :: x2=0.5d0*(x1+x3)
real(8), parameter :: y2=0.5d0*(y1+y3)
real(8), parameter :: x4=0.5d0*(x1+x7)
real(8), parameter :: y4=0.5d0*(y1+y7)
real(8), parameter :: x6=0.5d0*(x3+x9)
real(8), parameter :: y6=0.5d0*(y3+y9)
real(8), parameter :: x8=0.5d0*(x7+x9)
real(8), parameter :: y8=0.5d0*(y7+y9)
real(8), parameter :: x5=0.25d0*(x1+x3+x7+x9)
real(8), parameter :: y5=0.25d0*(y1+y3+y7+y9)

do i=1,np

   call random_number(r) ; r=(r-0.5)*2
   call random_number(s) ; s=(s-0.5)*2

   ! Q2 mapping

   N1= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s-1.d0)
   N2=      (1.d0-r**2) * 0.5d0*s*(s-1.d0)
   N3= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s-1.d0)
   N4= 0.5d0*r*(r-1.d0) *      (1.d0-s**2)
   N5=      (1.d0-r**2) *      (1.d0-s**2)
   N6= 0.5d0*r*(r+1.d0) *      (1.d0-s**2)
   N7= 0.5d0*r*(r-1.d0) * 0.5d0*s*(s+1.d0)
   N8=      (1.d0-r**2) * 0.5d0*s*(s+1.d0)
   N9= 0.5d0*r*(r+1.d0) * 0.5d0*s*(s+1.d0)

   x=N1*x1+N2*x2+N3*x3+N4*x4+N5*x5+N6*x6+N7*x7+N8*x8+N9*x9
   y=N1*y1+N2*y2+N3*y3+N4*y4+N5*y5+N6*y6+N7*y7+N8*y8+N9*y9

   write(123,*) r,s,x,y

   ! Q1 mapping

   N1=0.25*(1-r)*(1-s)
   N2=0.25*(1+r)*(1-s)
   N3=0.25*(1+r)*(1+s)
   N4=0.25*(1-r)*(1+s)

   x=N1*x1+N2*x3+N3*x9+N4*x7
   y=N1*y1+N2*y3+N3*y9+N4*y7

   write(124,*) r,s,x,y



end do

end program
