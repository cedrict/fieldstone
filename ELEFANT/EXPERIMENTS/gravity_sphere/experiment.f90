!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine declare_main_parameters

use module_parameters
use module_gravity

implicit none

!----------------------------------------------------------

ndim=3
Lx=1d3
Ly=1d3
Lz=1d3

nelx=48
nely=48
nelz=48

solve_stokes_system=.false.

use_swarm=.true.
nmarker_per_dim=3 
nmat=2

grav_pointmass=.true.
!grav_prism=.true.
plane_height=Lz+0.1
plane_xmin=0
plane_ymin=0
plane_xmax=Lx
plane_ymax=Ly
plane_nnx=0
plane_nny=25

xbeg=Lx/2
xend=1.11d3
ybeg=Ly/2
yend=2.22d3
zbeg=Lz/2
zend=5.55d3
line_nnp=256


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_material_properties

use module_materials 

implicit none

!----------------------------------------------------------

!liquid
materials(1)%rho0=0
materials(1)%eta0=1

!sphere
materials(2)%rho0=100
materials(2)%eta0=2

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine material_model(x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz,imat,mode,&
                          eta,rho,hcond,hcapa,hprod)

use module_materials

implicit none

real(8), intent(in) :: x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz
integer, intent(in) :: imat,mode
real(8), intent(out) :: eta,rho,hcond,hcapa,hprod

!----------------------------------------------------------

eta=materials(imat)%eta0
rho=materials(imat)%rho0

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

use module_parameters
use module_swarm

implicit none

integer im

!----------------------------------------------------------
! 1: fluid
! 2: sphere

do im=1,nmarker

   swarm(im)%mat=1

   if ((swarm(im)%x-0.5*Lx)**2+(swarm(im)%y-0.5*Ly)**2+(swarm(im)%z-0.5*Lz)**2<500**2) then
      swarm(im)%mat=2      
   end if

end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcV

implicit none

!----------------------------------------------------------


!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcT

implicit none

!----------------------------------------------------------

! your stuff here

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
gy=0
gz=-1

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
