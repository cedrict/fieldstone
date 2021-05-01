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

nelx=64
nely=64

use_penalty=.true.
penalty=1e8

solve_stokes_system=.false.

end subroutine

!==================================================================================================!

subroutine define_material_properties

use global_parameters
use structures

implicit none

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

eta=1
rho=1
  
if ((x-0.12345d0)**2+(y-0.23456d0)**2 < 0.05**2) then
   eta=1.d1
   rho=rho+2.d-3
end if   

if ((x-0.34567d0)**2+(y-0.6789d0)**2 < 0.08**2) then
   eta=1.d2
   rho=rho+1.75d-3
end if   

if ((x-0.76543d0)**2+(y-0.234d0)**2 < 0.12**2) then
   eta=1.d4
   rho=rho+1.5d-3
end if   

if ((x-0.54321d0)**2+(y-0.45678d0)**2 < 0.04**2) then
   eta=1.d6
   rho=rho+1.25d-3
end if   

if ((x-0.7123d0)**2+(y-0.8123d0)**2 < 0.07**2) then
   eta=1.d5
   rho=rho+0.25d-3
end if   

end subroutine

!==================================================================================================!

subroutine swarm_material_layout 

use global_parameters
use structures

implicit none


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
   do i=1,mV
      if (mesh(iel)%bnd1_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
      end if
   end do
   !right boundary
   do i=1,mV
      if (mesh(iel)%bnd2_node(i)) then
         mesh(iel)%fix_u(i)=.true. ; mesh(iel)%u(i)=0.d0
      end if
   end do
   !bottom boundary
   do i=1,mV
      if (mesh(iel)%bnd3_node(i)) then
         mesh(iel)%fix_v(i)=.true. ; mesh(iel)%v(i)=0.d0
      end if
   end do
   !top boundary
   do i=1,mV
      if (mesh(iel)%bnd4_node(i)) then
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

subroutine initial_temperature

use global_parameters
use structures

implicit none



end subroutine

!==================================================================================================!

subroutine analytical_solution(x,y,z,u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz)

implicit none

real(8), intent(in) :: x,y,z
real(8), intent(out) :: u,v,w,p,T,exx,eyy,ezz,exy,exz,eyz

u=0d0
v=0d0
w=0d0
p=0d0

exx=0d0
eyy=0d0
exy=0d0
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
