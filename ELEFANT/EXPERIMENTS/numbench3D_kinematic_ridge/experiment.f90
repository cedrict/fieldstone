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

ndim=3

solve_stokes_system=.false.

Lx=250.d3
Ly=200.d3
Lz=50.d3

nelx=64
nely=int(Ly/Lx*nelx)
nelz=int(Lz/Lx*nelx)

use_T=.true.

nstep=1


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

rho=3300d0

eta=1d22

hcond=2.5
hcapa=1200
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

use module_parameters
use module_mesh 
use module_constants

implicit none

integer k
real(8) xi,yi,xxx

!----------------------------------------------------------

do iel=1,nel

   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   mesh(iel)%fix_w(:)=.false. 

   do k=1,mV
      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
      if (mesh(iel)%bnd6_node(k)) then
         xi=mesh(iel)%xV(k)
         yi=mesh(iel)%yV(k)
         xxx=Lx/2.d0-atan((yi-Ly/2.)/Ly*1000)*25d3
         if (xi<=xxx) then
            mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=-1*cm/year
         else
            mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=+1*cm/year
         end if
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0d0
         mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0d0
      end if
   end do
end do

!----------------------------------------------------------

end subroutine

!==================================================================================================!

subroutine define_bcT

use module_parameters
use module_mesh 

implicit none

integer k

!----------------------------------------------------------

do iel=1,nel
   mesh(iel)%fix_T(:)=.false. 
   !bottom boundary
   do k=1,mT
      if (mesh(iel)%bnd5_node(k)) then
         mesh(iel)%fix_T(k)=.true. ; mesh(iel)%T(k)=1300.d0
      end if
   end do
   !top boundary
   do k=1,mT
      if (mesh(iel)%bnd6_node(k)) then
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

implicit none

integer k

!----------------------------------------------------------

do iel=1,nel
   do k=1,mT
      mesh(iel)%T(k)=-(mesh(iel)%zT(k)-Lz)/Lz*1300 
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
