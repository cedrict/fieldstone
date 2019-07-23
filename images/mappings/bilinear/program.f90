program opla
implicit none

integer i
real(8) r,s,x,y,N1,N2,N3,N4
integer, parameter :: np=12345
real(8), parameter :: x1=-1,y1=-2
real(8), parameter :: x2=3,y2=-1
real(8), parameter :: x3=2,y3=2
real(8), parameter :: x4=-3,y4=1


do i=1,np

   call random_number(r) ; r=(r-0.5)*2
   call random_number(s) ; s=(s-0.5)*2

   N1=0.25*(1-r)*(1-s)
   N2=0.25*(1+r)*(1-s)
   N3=0.25*(1+r)*(1+s)
   N4=0.25*(1-r)*(1+s)

   x=N1*x1+N2*x2+N3*x3+N4*x4
   y=N1*y1+N2*y2+N3*y3+N4*y4

   write(123,*) r,s,x,y

end do

end program
