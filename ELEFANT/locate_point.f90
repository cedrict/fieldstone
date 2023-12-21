!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine locate_point(x,y,z,ielt,r,s,t)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: r,s,t
integer, intent(inout) :: ielt

integer ielx,iely,ielz

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{locate\_point}
!@@ This file contains a few simple subroutines which deal with the localisation of a point 
!@@ in the mesh. The {\tt locate\_point} subroutine receives the coordinates of a point as argument 
!@@ and returns its reduced coordinates and the id of the element it sits in.
!@@ It relies on 3 other subroutines ({\tt find\_ielx\_r}, {\tt find\_iely\_s}, {\tt find\_ielz\_t})
!@@ which take as argument a coorinate (x,y,z) and return the corresponding reduced
!@@ coordinate (r,s,t) and the integer coordinate (ielx,iely,ielz).
!==================================================================================================!

if (geometry=='cartesian') then

   if (ndim==2) then
      call find_ielx_r(x,ielx,r)
      call find_iely_s(y,iely,s)
      t=0
      ielz=0
   else
      call find_ielx_r(x,ielx,r)
      call find_iely_s(y,iely,s)
      call find_ielz_t(z,ielz,t)
   end if
   call compute_iel(ielx,iely,ielz,ielt)

else

   stop 'locate_point: pb with geometry'

end if

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine compute_iel(ielx,iely,ielz,ielt)

use module_parameters

implicit none

integer, intent(in) :: ielx,iely,ielz
integer, intent(out) :: ielt

if (ndim==2) then
   ielt=nelx*(iely-1)+ielx
else
   ielt=nelx*nely*(ielz-1)+nelx*(iely-1)+ielx
end if

end subroutine

!==================================================================================================!

subroutine find_ielx_r (x,ielx,r)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: x
integer, intent(out) :: ielx
real(8), intent(out) :: r

real(8) xmin,xmax

ielx=x/Lx*nelx+1
xmin=mesh(ielx)%xV(1)
xmax=mesh(ielx)%xV(2)
r=((x-xmin)/(xmax-xmin)-0.5d0)*2.d0

end subroutine

!==================================================================================================!

subroutine find_iely_s (y,iely,s)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: y
integer, intent(out) :: iely
real(8), intent(out) :: s

real(8) ymin,ymax

iely=y/Ly*nely+1
ymin=mesh(iely)%yV(1)
ymax=mesh(iely)%yV(4)
s=((y-ymin)/(ymax-ymin)-0.5d0)*2.d0

end subroutine

!==================================================================================================!

subroutine find_ielz_t (z,ielz,t)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: z
integer, intent(out) :: ielz
real(8), intent(out) :: t

real(8) zmin,zmax

ielz=z/Lz*nelz+1
zmin=mesh(ielz)%zV(1)
zmax=mesh(ielz)%zV(5)
t=((z-zmin)/(zmax-zmin)-0.5d0)*2.d0

end subroutine

!==================================================================================================!
