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

   if (yc>80.d3) then
   mat(iel)=1
   elseif (yc>70.d3) then
   mat(iel)=2
   else
   mat(iel)=3
   end if

   if (yc<68.e3 .and. yc>60.e3 .and. xc>=198.e3 .and. xc<=202.e3) mat(iel)=4

end do

end subroutine

!==============================================!
