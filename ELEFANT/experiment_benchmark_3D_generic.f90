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

nelx=14
nely=14
nelz=14

use_penalty=.true.
penalty=1000

!debug=.true.

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_material_properties

use global_parameters
use structures

implicit none

!----------------------------------------------------------

! your stuff here

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

rho=1
eta=1

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

use global_parameters
use structures

implicit none

!----------------------------------------------------------

! your stuff here

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcV

use global_parameters
use structures

implicit none

!----------------------------------------------------------

integer k
real(8) dum,uth,vth,wth,pth

do iel=1,nel

   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   mesh(iel)%fix_w(:)=.false. 

   do k=1,mV

      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,dum,dum,dum,dum,dum,dum,dum,dum)

      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
      end if
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
      end if
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
      end if
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
      end if
      if (mesh(iel)%bnd5_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
      end if
      if (mesh(iel)%bnd6_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=uth
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=vth
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=wth
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

u=x*(1-x)*(1-2*y)*(1-2*z)
v=(1-2*x)*y*(1-y)*(1-2*z)
w=-2*(1-2*x)*(1-2*y)*z*(1-z)
p=(2*x-1)*(2*y-1)*(2*z-1)

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine gravity_model(x,y,z,gx,gy,gz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: gx,gy,gz

!----------------------------------------------------------

gx=4*(2*y-1)*(2*z-1)  !*(-1)
gy=4*(2*x-1)*(2*z-1)  !*(-1)
gz=-2*(2*x-1)*(2*y-1) !*(-1)

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
