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

nelx=96
nely=96

!solve_stokes_system=.false.

use_penalty=.true.
penalty=1e6
normalise_pressure=.true.

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

use module_parameters, only: dparam1,dparam2

implicit none

real(8), intent(in) :: x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz
integer, intent(in) :: imat,mode
real(8), intent(out) :: eta,rho,hcond,hcapa,hprod

!----------------------------------------------------------

if (abs(x-256d3)<64d3 .and. abs(y-384d3)<64d3) then
   eta=dparam2
   rho=dparam1-3200
else
   eta=1d21
   rho=3200-3200
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

use module_parameters
use module_mesh
use module_constants, only: eps,year

implicit none

integer k

!----------------------------------------------------------

do iel=1,nel
   do k=1,mV
      if (abs(mesh(iel)%xV(k)-256d3)<eps*Lx .and. abs(mesh(iel)%yV(k)-384d3)<eps*Ly) then
          write(*,*) 'middle_q',nelx,1e21/dparam2,&
                      mesh(iel)%q(k) / (dparam1-3200)/128d3/10,\
                      mesh(iel)%p(1) / (dparam1-3200)/128d3/10
          write(*,*) 'middle_v',nelx,1e21/dparam2,mesh(iel)%v(k) *1d21 / (dparam1-3200)
      end if
      !if (abs(mesh(iel)%xV(k)-256d3)<eps*Lx) then
      !   write(1234,*) mesh(iel)%yV(k),mesh(iel)%q(k)
      !   write(1235,*) mesh(iel)%yV(k),mesh(iel)%p(1)
      !end if
   end do
end do

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!
