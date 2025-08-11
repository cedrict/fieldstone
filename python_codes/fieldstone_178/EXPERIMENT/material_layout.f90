!==============================================!
!
!==============================================!

subroutine material_layout(np,nel,x,y,icon,mat)

implicit none

integer, intent(in) :: np,nel
real(8), dimension(np), intent(in) :: x,y
integer, dimension(nel), intent(out) :: mat
integer, dimension(4,nel) :: icon

integer iel
real(8) xc,yc

!==============================================!
! This subroutine fills the array mat which 
! contains the material number of the material 
! residing in each element
!==============================================!

do iel=1,nel ! loop over elements

   xc=0.25*sum(x(icon(1:4,iel))) ! x coordinate of element center
   yc=0.25*sum(y(icon(1:4,iel))) ! y coordinate of element center

   ! EXAMPLE - MODIFY IT!
   if (yc>50.d3) then
   mat(iel)=1
   else
   mat(iel)=2
   end if

   !if ( ..... ...... ) mat(iel)=4

end do

end subroutine

!==============================================!
