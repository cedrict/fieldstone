!-----------------------------------
!Cases
!-1 - original incompressible bench
!1 - 2D Cartesian Linear
!2 - 2D Cartesian Sinusoidal
!3 - 1D Cartesian Linear
!4 - Arie van den Berg
!5 - 1D Cartesian Sinusoidal
!-----------------------------------

function uth (x,y,ibench)
implicit none
real(8) uth,x,y
integer ibench
select case(ibench)
case(-1)
uth = x**2 * (1.d0-x)**2 * (2.d0*y - 6.d0*y**2 + 4*y**3)
case(1)
uth = 1.d0/x 
case(2)
uth = 1.d0/cos(x)
case(3)
uth = 1.d0/x
case(4)
uth=1.d0-x
case(5)
uth = 1.d0/cos(x)
end select
end function

!-----------------------------------

function vth (x,y,ibench)
implicit none
real(8) vth,x,y
integer ibench
select case(ibench)
case(-1)
vth = -y**2 * (1.d0-y)**2 * (2.d0*x - 6.d0*x**2 + 4*x**3)
case(1)
vth = 1.d0/y
case(2)
vth = 1.d0/cos(y)
case(3)
vth=0.d0
case(4)
vth=0.d0
case(5)
vth=0.d0
end select
end function

!-----------------------------------

function pth (x,y,ibench)
implicit none
real(8) pth,x,y
integer ibench
select case(ibench)
case(-1)
pth = x*(1.d0-x)-1.d0/6.d0
case(1)
!pth = -(1.d0/x**2+1.d0/y**2)-x*y + 3.25
pth = -4.d0/(3.d0*x**2) - 4.d0/(3.d0*y**2) + x**2/2.d0 + y**2/2.d0  -1.d0
case(2)
pth = 4.d0*sin(x)/(3.d0*cos(x)**2)+4.d0*sin(y)/(3.d0*cos(y)**2) + sin(x) + sin(y) - 3.18834
case(3)
pth =  -4.d0/(3.d0*x**2) - x**2/2.d0 + 11.d0/6.d0
!pth =  -1.d0/(x**2) - x**2/2.d0 + 5.d0/3.d0
case(4)
pth = 10.d0*log(x-1) +290.9 
!pth = 1.d0/(1.d0-x)*x
case(5)
!pth = sin(x)/cos(x)**2-sin(x) - 0.391118
pth = 4.d0*sin(x)/(3.d0*cos(x)**2) - sin(x) -0.674723
end select
end function

!-----------------------------------

function rho(x,y,ibench)
implicit none
real(8) rho,x,y
integer ibench
select case(ibench)
case(-1)
rho=1
case(1)
rho=x*y
case(2)
rho = cos(x)*cos(y)
case(3)
rho=x
case(4)
rho = 1.d0/(1.d0-x)
case(5)
rho=cos(x)
end select
end function

!-----------------------------------

function drhodx(x,y,ibench)
implicit none
real(8) drhodx,x,y
integer ibench
select case(ibench)
case(-1)
drhodx=0
case(1)
drhodx=y
case(2)
drhodx=-sin(x)*cos(y)
case(3)
drhodx=1.d0
case(4)
drhodx=1.d0/(1.d0-x)**2
case(5)
drhodx=-sin(x)
end select
end function

!-----------------------------------

function drhody(x,y,ibench)
implicit none
real(8) drhody,x,y
integer ibench
select case(ibench)
case(-1)
drhody=0
case(1)
drhody=x
case(2)
drhody=-cos(x)*sin(y)
case(3)
drhody=0.d0
case(4)
drhody=0.d0
case(5)
drhody=0.d0
end select
end function

!-----------------------------------

function gx(x,y,ibench)
implicit none
real(8) gx,x,y
integer ibench
select case(ibench)
case(-1)
gx = ( (12.d0-24.d0*y)*x**4 + (-24.d0+48.d0*y)*x**3 + (-48.d0*y+72.d0*y**2-48.d0*y**3+12.d0)*x**2 &
   + (-2.d0+24.d0*y-72.d0*y**2+48.d0*y**3)*x + 1.d0-4.d0*y+12.d0*y**2-8.d0*y**3 )
case(1)
gx=1.d0/y
case(2)
gx = 1.d0/cos(y)
case(3)
gx=-1.d0
case(4)
gx=10.d0
case(5)
gx=-1.d0
end select
end function

!-----------------------------------

function gy(x,y,ibench)
implicit none
real(8) gy,x,y
integer ibench
select case(ibench)
case(-1)
gy= ( (8.d0-48.d0*y+48.d0*y**2)*x**3 + (-12.d0+72.d0*y-72*y**2)*x**2 + & 
    (4.d0-24.d0*y+48.d0*y**2-48.d0*y**3+24.d0*y**4)*x - 12.d0*y**2 + 24.d0*y**3 -12.d0*y**4)
case(1)
gy=1.d0/x
case(2)
gy=1.d0/cos(x)
case(3)
gy=0.d0
case(4)
gy=0.d0
case(5)
gy=0.d0
end select
end function

!-----------------------------------

function dudxth(x,y,ibench)
implicit none
real(8) dudxth,x,y
integer ibench
select case(ibench)
case(1)
dudxth=-1.d0/x**2
case(2)
dudxth= sin(x)/cos(x)**2
case(3)
dudxth = -1.d0/x**2
case(4)
dudxth=0.d0
case(5)
dudxth = sin(x)/(cos(x)**2)
end select
end function

!-----------------------------------

function dvdyth(x,y,ibench)
implicit none
real(8) dvdyth,x,y
integer ibench
select case(ibench)
case(1)
dvdyth=-1.d0/y**2
case(2)
dvdyth=sin(y)/cos(y)**2
case(3)
dvdyth=0.d0
case(4)
dvdyth=0.d0
case(5)
dvdyth=0.d0
end select
end function

!-----------------------------------

function phith(x,y,ibench)
implicit none
real(8) phith,x,y
integer ibench
select case(ibench)
case(1)
phith = -(1.d0/(y**2))*(-4.d0/(3.d0*y**2) + 2.d0/(3.d0*x**2))&
        -(1.d0/(x**2))*(-4.d0/(3.d0*x**2) + 2.d0/(3.d0*y**2))  
! phi = 4.d0/(3.d0*x**4)
case(2)
phith = sin(y)/cos(y)**2*(-2.d0*sin(x)/(3.d0*cos(x)**2)+4*sin(y)/(3*cos(y)**2)) &
      + sin(x)/cos(x)**2*(-2.d0*sin(y)/(3.d0*cos(y)**2)+4*sin(x)/(3*cos(x)**2))
case(3)
phith = 4.d0/(3.d0*x**4)
case(4)
phith=0
case(5)
phith = 4.d0*sin(x)**2/(3.d0*cos(x)**4)
end select
end function

!-----------------------------------

function dudyth(x,y,ibench)
implicit none
real(8) dudyth,x,y
integer ibench
select case(ibench)
case(1)
dudyth=0.d0
case(2)
dudyth=0.d0
case(3)
dudyth=0.d0
case(4)
case(5)
dudyth=0.d0
end select
end function

!-----------------------------------

function dvdxth(x,y,ibench)
implicit none
real(8) dvdxth,x,y
integer ibench
select case(ibench)
case(1)
dvdxth=0.d0
case(2)
dvdxth=0.d0
case(3)
dvdxth=0.d0
case(4)
case(5)
dvdxth=0.d0
end select
end function

!-----------------------------------
