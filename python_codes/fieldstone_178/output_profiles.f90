!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!

subroutine output_profiles(np,nelx,nely,x,y,T,mat,density,viscosity,exx,eyy,exy,press,vpmode,icon)

implicit none

integer, intent(in) :: np,nelx,nely
real(8), dimension(np), intent(in) :: x,y,T
integer, dimension(nelx*nely), intent(in) :: mat
real(8), dimension(nelx*nely), intent(in) :: density,viscosity,exx,eyy,exy,press,vpmode 
integer, dimension(4,nelx*nely) :: icon

real(8) yc,E2,Tc
integer iel,i,j

!==============================================!

open(unit=123,file='OUT/profiles.dat')
open(unit=124,file='OUT/profile_left.dat')
open(unit=125,file='OUT/profile_right.dat')
open(unit=126,file='OUT/profile_middle.dat')
write(123,*) '#1 y'
write(123,*) '#2 mat'
write(123,*) '#3 density'
write(123,*) '#4 viscosity'
write(123,*) '#5 pressure'
write(123,*) '#6 dev stress 2nd inv.'
write(123,*) '#7 Temperature'
write(123,*) '#8 vpmode'

iel=0
do j=1,nely
do i=1,nelx
   iel=iel+1

   yc=0.25*sum(y(icon(1:4,iel)))
   Tc=0.25*sum(T(icon(1:4,iel)))

   E2= sqrt( 0.5d0*(exx(iel)**2+eyy(iel)**2)+exy(iel)**2 )

   write(123,*) yc,mat(iel),density(iel),viscosity(iel),press(iel),2.d0*E2*viscosity(iel),Tc,vpmode(iel)

   if (i==1) & 
   write(124,*) yc,mat(iel),density(iel),viscosity(iel),press(iel),2.d0*E2*viscosity(iel),Tc,vpmode(iel)

   if (i==nelx) & 
   write(125,*) yc,mat(iel),density(iel),viscosity(iel),press(iel),2.d0*E2*viscosity(iel),Tc,vpmode(iel)

   if (i==nelx/2) & 
   write(126,*) yc,mat(iel),density(iel),viscosity(iel),press(iel),2.d0*E2*viscosity(iel),Tc,vpmode(iel)

end do
end do

close(123)

end subroutine

!==============================================!
