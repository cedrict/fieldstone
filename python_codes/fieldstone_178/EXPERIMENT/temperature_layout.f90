!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!

subroutine temperature_layout(np,x,y,T)

implicit none

integer, intent(in) :: np
real(8), dimension(np), intent(in) :: x,y
real(8), dimension(np), intent(out) :: T 

integer :: ip
real(8) xi,yi

!==============================================!

do ip=1,np

   xi=x(ip)
   yi=y(ip)

   if (yi>70.d3) then

   T(ip)= 0 ! MODIFY 

   else

   T(ip)= 0 ! MODIFY

   end if

end do

T=T+273.15

end subroutine
