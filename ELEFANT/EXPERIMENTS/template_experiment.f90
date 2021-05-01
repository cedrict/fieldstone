!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine declare_main_parameters

use global_parameters

implicit none

!----------------------------------------------------------



!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_material_properties

use global_parameters
use structures

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine material_model(x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz,imat,mode,&
                          eta,rho,hcond,hcapa,hprod)

use global_parameters
use structures
use constants

implicit none

real(8), intent(in) :: x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz
integer, intent(in) :: imat,mode
real(8), intent(out) :: eta,rho,hcond,hcapa,hprod

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

use global_parameters
use structures

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcV

use global_parameters
use structures

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcT

use global_parameters
use structures

implicit none

!----------------------------------------------------------

do iel=1,nel
   mesh(iel)%fix_T(:)=.false. 
   !left boundary
   do i=1,mT
      if (mesh(iel)%bnd1_node(i)) then
         mesh(iel)%fix_T(i)=.true. ; mesh(iel)%T(i)=0.d0
      end if
   end do
   !right boundary
   do i=1,mT
      if (mesh(iel)%bnd2_node(i)) then
         mesh(iel)%fix_T(i)=.true. ; mesh(iel)%T(i)=0.d0
      end if
   end do
   !bottom boundary
   do i=1,mT
      if (mesh(iel)%bnd3_node(i)) then
         mesh(iel)%fix_T(i)=.true. ; mesh(iel)%T(i)=0.d0
      end if
   end do
   !top boundary
   do i=1,mT
      if (mesh(iel)%bnd4_node(i)) then
         mesh(iel)%fix_T(i)=.true. ; mesh(iel)%T(i)=0.d0
      end if
   end do
end do


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine initial_temperature

use global_parameters
use structures

implicit none

!----------------------------------------------------------

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine analytical_solution(x,y,z,u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz

!----------------------------------------------------------

! your stuff here

u=0
v=0
w=0
p=0
T=0
exx=0
eyy=0
ezz=0
exy=0
exz=0
eyz=0

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine gravity_model(x,y,z,gx,gy,gz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: gx,gy,gz

!----------------------------------------------------------

gx=0
gy=0
gz=-1

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine test

use global_parameters 
use global_measurements 
use constants

implicit none

!----------------------------------------------------------

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine postprocessor_experiment

implicit none

!----------------------------------------------------------

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!
