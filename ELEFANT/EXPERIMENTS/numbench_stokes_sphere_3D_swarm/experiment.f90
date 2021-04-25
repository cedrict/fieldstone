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

ndim=3
Lx=1
Ly=1
Lz=1

nelx=48
nely=48
nelz=48

use_penalty=.true.
penalty=20000

debug=.false.

use_swarm=.true.
nmat=2

solve_stokes_system=.false.

grav_pointmass=.true.

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_material_properties

use global_parameters
use structures

implicit none

!----------------------------------------------------------

!liquid
mat(1)%rho0=1-1
mat(1)%eta0=1

!sphere
mat(2)%rho0=2-1
mat(2)%eta0=2

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

eta=mat(imat)%eta0
rho=mat(imat)%rho0

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

use global_parameters
use structures

implicit none

integer im

!----------------------------------------------------------
! 1: fluid
! 2: sphere

do im=1,nmarker

   swarm(im)%mat=1

   if ((swarm(im)%x-0.5)**2+(swarm(im)%y-0.5)**2+(swarm(im)%z-0.5)**2<0.123456789d0**2) then
      swarm(im)%mat=2      
   end if

end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcV

use global_parameters
use structures

implicit none

!----------------------------------------------------------

integer k

do iel=1,nel

   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   mesh(iel)%fix_w(:)=.false. 

   do k=1,mV
      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd5_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd6_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
   end do
end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcT

use global_parameters
use structures

implicit none

!----------------------------------------------------------

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine temperature_layout

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
