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

ndim=2
Lx=1
Ly=1
Lz=4
nelx=20
nely=20
nelz=40
geometry='cartesian'
pair='q1p0'

penalty=1000d0
use_penalty=.true.

use_swarm=.true.
nmarker_per_dim=5
init_marker_random=.false.
nmat=3

debug=.false.

nxstripes=-3
nystripes=4

end subroutine

!==================================================================================================!

subroutine define_material_properties

use global_parameters
use structures

implicit none

!liquid
mat(1)%rho0=1
mat(1)%eta0=1

!sphere
mat(2)%rho0=2
mat(2)%eta0=1d3

!air
mat(3)%rho0=0.001
mat(3)%eta0=1d-3


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

eta=mat(imat)%eta0
rho=mat(imat)%rho0

end subroutine

!==================================================================================================!

subroutine material_layout 

use global_parameters
use structures

implicit none

integer im

if (use_swarm) then

   do im=1,nmarker

      swarm(im)%mat=1

      if (swarm(im)%y>0.75) swarm(im)%mat=3

      if ((swarm(im)%x-0.5d0)**2+(swarm(im)%y-0.6d0)**2<0.123456789**2) swarm(im)%mat=2

   end do

end if

end subroutine

!==================================================================================================!

subroutine define_bcV

use global_parameters
use structures

implicit none

integer i

do iel=1,nel

   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 

   !left boundary
   do i=1,4
      if (mesh(iel)%left_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
      end if
   end do
   !right boundary
   do i=1,4
      if (mesh(iel)%right_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
      end if
   end do
   !bottom boundary
   do i=1,4
      if (mesh(iel)%bottom_node(i)) then
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
   !top boundary
   do i=1,4
      if (mesh(iel)%top_node(i)) then
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
end do

end subroutine

!==================================================================================================!

subroutine define_bcT

use global_parameters
use structures

implicit none



end subroutine

!==================================================================================================!

subroutine temperature_layout

use global_parameters
use structures

implicit none



end subroutine

!==================================================================================================!

subroutine analytical_solution(x,y,z,u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz




end subroutine

!==================================================================================================!

subroutine gravity_model(x,y,z,gx,gy,gz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: gx,gy,gz

gx=0
gy=-1
gz=0

end subroutine

!==================================================================================================!

subroutine test

use global_parameters 
use global_measurements 
use constants

implicit none

vrms_test=0.

if (abs(vrms-vrms_test)/vrms_test<epsilon_test) then
   print *,'***** test passed *****'
else
   print *,'***** test FAILED *****'
end if

end subroutine

!==================================================================================================!

subroutine postprocessor_experiment

implicit none




end subroutine

!==================================================================================================!
