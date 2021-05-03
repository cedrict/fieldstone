!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine declare_main_parameters

use module_parameters

implicit none

!----------------------------------------------------------

Lx=512d3
Ly=512d3

nelx=32
nely=32

!solve_stokes_system=.false.

use_penalty=.true.
penalty=1e6

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_material_properties

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine material_model(x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz,imat,mode,&
                          eta,rho,hcond,hcapa,hprod)

implicit none

real(8), intent(in) :: x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz
integer, intent(in) :: imat,mode
real(8), intent(out) :: eta,rho,hcond,hcapa,hprod

!----------------------------------------------------------

if (abs(x-256d3)<64d3 .and. abs(y-384d3)<64d3) then
   eta=1d22
   rho=3300
else
   eta=1d21
   rho=3200
end if

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcV

use module_mesh
use module_parameters

implicit none

integer k

!----------------------------------------------------------

do iel=1,nel
   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   !left boundary
   do k=1,mV
      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
      end if
   end do
   !right boundary
   do k=1,mV
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
      end if
   end do
   !bottom boundary
   do k=1,mV
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
      end if
   end do
   !top boundary
   do k=1,mV
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
      end if
   end do
end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcT

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine initial_temperature

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
gy=-10
gz=0

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine test

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
