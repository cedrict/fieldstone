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
Lx=1!4
Ly=1!3
Lz=1!2

nelx=16!4
nely=16!3
nelz=16!2

use_penalty=.true.
penalty=20000

debug=.false.

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

if ((x-0.5)**2+(y-0.5)**2+(z-0.5)**2<0.123456789d0**2) then
   eta=1d1
   rho=1.01d0
else
   eta=1d0
   rho=1.d0
end if

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
