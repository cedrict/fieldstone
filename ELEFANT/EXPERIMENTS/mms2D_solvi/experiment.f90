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

Lx=1d0
Ly=1d0

nelx=64
nely=64

use_penalty=.true.
penalty=1d4


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

rho  =1.d0

if (sqrt(x**2+y**2) < 0.2d0) then
   eta=1.d3
else
   eta=1.d0
end if


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
real(8), external :: u_inclusion,v_inclusion,p_inclusion

!----------------------------------------------------------

do iel=1,nel
   mesh(iel)%fix_u(:)=.false. 
   mesh(iel)%fix_v(:)=.false. 
   !left boundary
   do k=1,mV
      if (mesh(iel)%bnd1_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=u_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=v_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
      end if
   end do
   !right boundary
   do k=1,mV
      if (mesh(iel)%bnd2_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=u_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=v_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
      end if
   end do
   !bottom boundary
   do k=1,mV
      if (mesh(iel)%bnd3_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=u_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=v_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
      end if
   end do
   !top boundary
   do k=1,mV
      if (mesh(iel)%bnd4_node(k)) then
         mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=u_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
         mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=v_inclusion(mesh(iel)%xV(k),mesh(iel)%yV(k))
      end if
   end do
end do

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

real(8), external :: u_inclusion,v_inclusion,p_inclusion

!----------------------------------------------------------

! your stuff here

u=u_inclusion (x,y)
v=v_inclusion (x,y)
w=0
p=p_inclusion (x,y)
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

function u_inclusion (x,z)
implicit none
real(8) u_inclusion,x,z
real(8) gr,er,mm,mc,A,rc
complex iii,V_tot,ZZ,phi_z,d_phi_z,psi_z,conj_d_phi_z,conj_psi_z

gr  =  0.d0   ! Simple shear: gr=1, er=0
er  = -1.d0   ! Strain rate
mm  =  1.d0   ! Viscosity of matrix
mc  =  1.d3   ! Viscosity of clast
rc  =  0.2    ! Radius of clast
A   =   mm*(mc-mm)/(mc+mm) 
iii   =  cmplx(0,-1.d0)
ZZ  = x + iii*z;

if (sqrt(x**2 + z**2)<=rc) then  ! inside clast
    V_tot          =  (mm/(mc+mm))*(iii*gr+2*er)*conjg(ZZ)-(iii/2)*gr*ZZ;
else ! outside clast
    phi_z          = -(iii/2)*mm*gr*ZZ-(iii*gr+2*er)*A*rc**2*ZZ**(-1)
    d_phi_z        = -(iii/2)*mm*gr + (iii*gr+2*er)*A*rc**2/ZZ**2
    conj_d_phi_z   = conjg(d_phi_z)
    psi_z          = (iii*gr-2*er)*mm*ZZ-(iii*gr+2*er)*A*rc**4*ZZ**(-3.)
    conj_psi_z     = conjg(psi_z)
    V_tot          = (phi_z- ZZ*conj_d_phi_z - conj_psi_z) / (2*mm)
end if 

u_inclusion=real(V_tot)

end function


function v_inclusion (x,z)
implicit none
real(8) v_inclusion,x,z
real(8) gr,er,mm,mc,A,rc
complex iii,V_tot,ZZ,phi_z,d_phi_z,psi_z,conj_d_phi_z,conj_psi_z

gr  =  0.d0   ! Simple shear: gr=1, er=0
er  = -1.d0   ! Strain rate
mm  =  1.d0   ! Viscosity of matrix
mc  =  1.d3   ! Viscosity of clast
rc  =  0.2    ! Radius of clast
A   =   mm*(mc-mm)/(mc+mm)
iii   =  cmplx(0,-1.d0)
ZZ  = x + iii*z;

if (sqrt(x**2 + z**2)<=rc) then  ! inside clast
    V_tot          =  (mm/(mc+mm))*(iii*gr+2*er)*conjg(ZZ)-(iii/2)*gr*ZZ;
else ! outside clast
    phi_z          = -(iii/2)*mm*gr*ZZ-(iii*gr+2*er)*A*rc**2*ZZ**(-1)
    d_phi_z        = -(iii/2)*mm*gr + (iii*gr+2*er)*A*rc**2/ZZ**2
    conj_d_phi_z   = conjg(d_phi_z)
    psi_z          = (iii*gr-2*er)*mm*ZZ-(iii*gr+2*er)*A*rc**4*ZZ**(-3.)
    conj_psi_z     = conjg(psi_z)
    V_tot          = (phi_z- ZZ*conj_d_phi_z - conj_psi_z) / (2*mm)
end if

v_inclusion=-aimag(V_tot)

end function

function p_inclusion (x,z)
real(8) p_inclusion,x,z
real(8) gr,er,mm,mc,A
complex iii,V_tot,ZZ,phi_z,d_phi_z,psi_z,conj_d_phi_z,conj_psi_z

gr  =  0.d0   ! Simple shear: gr=1, er=0
er  = -1.d0   ! Strain rate
mm  =  1.d0   ! Viscosity of matrix
mc  =  1.d3   ! Viscosity of clast
rc  =  0.2    ! Radius of clast
A   =   mm*(mc-mm)/(mc+mm)
iii   =  cmplx(0,-1.d0)
ZZ  = x + iii*z;

if (sqrt(x**2 + z**2)<=rc) then  ! inside clast
   p_inclusion=0.d0
else ! outside clast
   p_inclusion = -2.d0*mm*(mc-mm)/(mc+mm)*real(rc**2./ZZ**2.*(iii*gr+2.d0*er))
end if

end function


