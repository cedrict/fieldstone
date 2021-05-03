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

ndim=2
Lx=1
Ly=1

nelx=64
nely=64

use_penalty=.true.
penalty=1d7

debug=.false.

end subroutine

!==================================================================================================!

subroutine define_material_properties

implicit none

end subroutine

!==================================================================================================!

subroutine material_model(x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz,imat,mode,&
                          eta,rho,hcond,hcapa,hprod)

implicit none

real(8), intent(in) :: x,y,z,p,T,exx,eyy,ezz,exy,exz,eyz
integer, intent(in) :: imat,mode
real(8), intent(out) :: eta,rho,hcond,hcapa,hprod

eta=1
rho=1

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

implicit none


end subroutine

!==================================================================================================!

subroutine define_bcV

use module_parameters
use module_mesh

implicit none

integer i

do iel=1,nel
   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   !left boundary
   do i=1,mV
      if (mesh(iel)%bnd1_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
   !right boundary
   do i=1,mV
      if (mesh(iel)%bnd2_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
   !bottom boundary
   do i=1,mV
      if (mesh(iel)%bnd3_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
   !top boundary
   do i=1,mV
      if (mesh(iel)%bnd4_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
end do

end subroutine

!==================================================================================================!

subroutine define_bcT

implicit none



end subroutine

!==================================================================================================!

subroutine initial_temperature

implicit none



end subroutine

!==================================================================================================!

subroutine analytical_solution(x,y,z,u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz

real(8) dudxth,dvdxth,dudyth,dvdyth

u=x**2 * (1.d0-x)**2 * (2.d0*y - 6.d0*y**2 + 4*y**3)
v=-y**2 * (1.d0-y)**2 * (2.d0*x - 6.d0*x**2 + 4*x**3)
w=0d0
p=x*(1-x)-1d0/6d0

dudxth=4*x*y*(1-x)*(1-2*x)*(1-3*y+2*y**2)
dudyth=2*x**2*(1-x)**2*(1-6*y+6*y**2)
dvdxth=-2*y**2*(1-y)**2*(1-6*x+6*x**2)
dvdyth=-4*x*y*(1-y)*(1-2*y)*(1-3*x+2*x**2)

exx=dudxth
eyy=dvdyth
exy=0.5d0*(dudyth+dvdxth)

ezz=0d0
exz=0d0
eyz=0d0

T=0

end subroutine

!==================================================================================================!

subroutine gravity_model(x,y,z,gx,gy,gz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: gx,gy,gz

gx= ( (12.d0-24.d0*y)*x**4 + (-24.d0+48.d0*y)*x**3 + (-48.d0*y+72.d0*y**2-48.d0*y**3+12.d0)*x**2 &
    + (-2.d0+24.d0*y-72.d0*y**2+48.d0*y**3)*x + 1.d0-4.d0*y+12.d0*y**2-8.d0*y**3 )

gy=( (8.d0-48.d0*y+48.d0*y**2)*x**3 + (-12.d0+72.d0*y-72*y**2)*x**2 + &
     (4.d0-24.d0*y+48.d0*y**2-48.d0*y**3+24.d0*y**4)*x - 12.d0*y**2 + 24.d0*y**3 -12.d0*y**4)

gz=0

!gx=-gx
!gy=-gy

end subroutine

!==================================================================================================!

subroutine test

use module_statistics 
use module_constants

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
