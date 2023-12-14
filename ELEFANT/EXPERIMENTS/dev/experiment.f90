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

nelx=16
nely=16

spaceV='__Q1'
spaceP='__Q0'
mapping='__Q1'

debug=.true.

use_penalty=.true.
penalty=1d6


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

eta=1
rho=1

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

!----------------------------------------------------------

integer k

do iel=1,nel
   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   !left boundary
   do k=1,mV
      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
      end if
   end do
   !right boundary
   do k=1,mV
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
      end if
   end do
   !bottom boundary
   do k=1,mV
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
      end if
   end do
   !top boundary
   do k=1,mV
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
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
gy=-1
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
