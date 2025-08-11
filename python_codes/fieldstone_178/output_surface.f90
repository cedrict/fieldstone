!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!

subroutine output_surface(np,nel,x,y,viscosity,exx,eyy,exy,press,icon)

implicit none

integer, intent(in) :: np,nel
real(8), dimension(np), intent(in) :: x,y
real(8), dimension(nel), intent(in) :: viscosity,exx,eyy,exy,press
integer, dimension(4,nel) :: icon

real(8) xc,E2
integer iel

!==============================================!

open(unit=123,file='OUT/surface.dat')
write(123,*) '#1 x'
write(123,*) '#2 dotepsilon'
write(123,*) '#3 viscosity'
write(123,*) '#4 pressure'
write(123,*) '#5 dev stress 2nd inv.'

do iel=1,nel

   if (y(icon(4,iel))>0.99999*100.d3) then

   xc=0.25*sum(x(icon(1:4,iel)))

   E2= sqrt( 0.5d0*(exx(iel)**2+eyy(iel)**2)+exy(iel)**2 )

   write(123,*) xc,E2,viscosity(iel),press(iel),2.d0*E2*viscosity(iel)

   end if

end do

close(123)

end subroutine

!==============================================!
