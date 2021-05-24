program opla
implicit none

integer iel,i
integer, parameter :: ncell=22
integer, parameter :: np=8*ncell
integer, parameter :: mpe=8

real(8) zeta
real(8), dimension(:),   allocatable :: x,y,z
integer, dimension(:,:), allocatable :: icon      ! connectivity array

!--------------------------

allocate(x(np)) ; x=0
allocate(y(np)) ; y=0
allocate(z(np)) ; z=0
allocate(icon(8,ncell)) ; icon=1

zeta=1.5

!--------------------------
iel=1

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=2

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=3

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=4

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=5

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=6

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=7

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=8

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=9

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= 2
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 2
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 2
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= 2
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 3
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 3
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 3
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 3

!--------------------------
iel=10

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 0
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= zeta 
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=11

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= zeta
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= zeta
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=12

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= zeta
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 0
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=13

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= zeta
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= zeta
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=14

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= zeta
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= zeta
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= zeta
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= zeta
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=15

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= zeta
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= zeta
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=16

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 0
x((iel-1)*mpe+3)= 2  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 2  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= zeta
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 2  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 2  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=17

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 2  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= zeta
x((iel-1)*mpe+2)= 2  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 1  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 1  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= zeta
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=18

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 1  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= zeta
x((iel-1)*mpe+2)= 1  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 0
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 1  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= 2
x((iel-1)*mpe+6)= 1  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 2
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 2
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= 2

!--------------------------
iel=19

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 0
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 1  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 1  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 0
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 0  ; z((iel-1)*mpe+5)= 1
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 1  ; z((iel-1)*mpe+6)= zeta
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 1  ; z((iel-1)*mpe+7)= zeta
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 0  ; z((iel-1)*mpe+8)= 1

!--------------------------
iel=20

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 1  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 2  ; z((iel-1)*mpe+2)= 1
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 2  ; z((iel-1)*mpe+3)= 1
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 1  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= zeta
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= zeta
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= zeta
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= zeta

!--------------------------
iel=21

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 2  ; z((iel-1)*mpe+1)= 1
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 0
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 0
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 2  ; z((iel-1)*mpe+4)= 1
x((iel-1)*mpe+5)= 2  ; y((iel-1)*mpe+5)= 2  ; z((iel-1)*mpe+5)= zeta
x((iel-1)*mpe+6)= 2  ; y((iel-1)*mpe+6)= 3  ; z((iel-1)*mpe+6)= 1
x((iel-1)*mpe+7)= 1  ; y((iel-1)*mpe+7)= 3  ; z((iel-1)*mpe+7)= 1
x((iel-1)*mpe+8)= 1  ; y((iel-1)*mpe+8)= 2  ; z((iel-1)*mpe+8)= zeta

!--------------------------
iel=22

do i=1,mpe
icon(i,iel)=(iel-1)*mpe+i
end do

x((iel-1)*mpe+1)= 3  ; y((iel-1)*mpe+1)= 0  ; z((iel-1)*mpe+1)= 0
x((iel-1)*mpe+2)= 3  ; y((iel-1)*mpe+2)= 3  ; z((iel-1)*mpe+2)= 0
x((iel-1)*mpe+3)= 0  ; y((iel-1)*mpe+3)= 3  ; z((iel-1)*mpe+3)= 0
x((iel-1)*mpe+4)= 0  ; y((iel-1)*mpe+4)= 0  ; z((iel-1)*mpe+4)= 0
x((iel-1)*mpe+5)= 3  ; y((iel-1)*mpe+5)= 1  ; z((iel-1)*mpe+5)= 1
x((iel-1)*mpe+6)= 3  ; y((iel-1)*mpe+6)= 2  ; z((iel-1)*mpe+6)= 1
x((iel-1)*mpe+7)= 0  ; y((iel-1)*mpe+7)= 2  ; z((iel-1)*mpe+7)= 1
x((iel-1)*mpe+8)= 0  ; y((iel-1)*mpe+8)= 1  ; z((iel-1)*mpe+8)= 1

!--------------------------------------------------

call output_for_paraview2 (np,ncell,x,y,z,icon)

end program




