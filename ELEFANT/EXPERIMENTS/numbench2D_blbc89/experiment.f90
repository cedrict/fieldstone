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

Lx=1
Ly=1

nelx=64
nely=64

use_penalty=.true.
penalty=1d7

use_T=.true.

nstep=200

CFL_nb=0.2

output_freq=25

!debug=.true.

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

rho=1-0.01d0*T

eta=1

hcond=1
hcapa=1
hprod=0

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

use module_parameters, only: iel,nel,mV
use module_mesh

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

use module_parameters, only: iel,nel,mT
use module_mesh

implicit none

integer k

do iel=1,nel
   mesh(iel)%fix_T(:)=.false. 
   !bottom boundary
   do k=1,mT
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_T(k)=.true. ; mesh(iel)%T(k)=1.d0
      end if
   end do
   !top boundary
   do k=1,mT
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_T(k)=.true. ; mesh(iel)%T(k)=0.d0
      end if
   end do
end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine initial_temperature

use module_parameters
use module_mesh
use module_constants, only: pi

implicit none

integer k

!----------------------------------------------------------

do iel=1,nel
   do k=1,mT
      mesh(iel)%T(k)=1d0-mesh(iel)%yT(k) -0.01d0*cos(pi*mesh(iel)%xT(k))*sin(pi*mesh(iel)%yT(k))
   end do
end do

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

real(8), parameter :: Ra=1d4
!----------------------------------------------------------

gx=0
gy=-1d2*Ra
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

use module_parameters, only: iel,nel
use module_mesh 
use module_statistics, only: vrms

implicit none

real(8) Nu

!----------------------------------------------------------

Nu=0d0
do iel=1,nel
   if (mesh(iel)%bnd4) then
      Nu=Nu- (mesh(iel)%qy(3)+mesh(iel)%qy(4))/2d0*mesh(iel)%hx
   end if
end do
write(999,*) vrms,Nu
call flush(999)

!----------------------------------------------------------

end subroutine

!==================================================================================================!
