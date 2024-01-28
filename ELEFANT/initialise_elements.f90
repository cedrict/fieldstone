!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine initialise_elements

use module_parameters, only: mU,mV,mW,mT,mVel,mP,geometry,nel,iel,iproc,mmapping
use module_mesh 
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{initialise\_elements}
!@@ This subroutine allocates pretty much all element-based arrays (node coordinates,velocity,
!@@ strain rate, ...).
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a,i6)')  shift//'nel=',nel
write(*,'(a,3i3)') shift//'mU,mV,mW=',mU,mV,mW

allocate(mesh(nel))

do iel=1,nel  

   allocate(mesh(iel)%iconU(mU)) 
   allocate(mesh(iel)%xU(mU)) ; mesh(iel)%xU=0.d0
   allocate(mesh(iel)%yU(mU)) ; mesh(iel)%yU=0.d0
   allocate(mesh(iel)%zU(mU)) ; mesh(iel)%zU=0.d0
   allocate(mesh(iel)%u(mU)) ; mesh(iel)%u=0.d0

   allocate(mesh(iel)%iconV(mV)) 
   allocate(mesh(iel)%xV(mV)) ; mesh(iel)%xV=0.d0
   allocate(mesh(iel)%yV(mV)) ; mesh(iel)%yV=0.d0
   allocate(mesh(iel)%zV(mV)) ; mesh(iel)%zV=0.d0
   allocate(mesh(iel)%v(mV)) ; mesh(iel)%v=0.d0

   allocate(mesh(iel)%iconW(mW)) 
   allocate(mesh(iel)%xW(mW)) ; mesh(iel)%xW=0.d0
   allocate(mesh(iel)%yW(mW)) ; mesh(iel)%yW=0.d0
   allocate(mesh(iel)%zW(mW)) ; mesh(iel)%zW=0.d0
   allocate(mesh(iel)%w(mW)) ; mesh(iel)%w=0.d0

   allocate(mesh(iel)%iconVel(mvel)) 

   allocate(mesh(iel)%iconP(mP)) 
   allocate(mesh(iel)%p(mP))  ; mesh(iel)%p=0.d0
   allocate(mesh(iel)%xP(mP)) ; mesh(iel)%xP=0.d0
   allocate(mesh(iel)%yP(mP)) ; mesh(iel)%yP=0.d0
   allocate(mesh(iel)%zP(mP)) ; mesh(iel)%zP=0.d0

   select case (geometry)
   case('cartesian','john') 
      allocate(mesh(iel)%bnd1_Unode(mU))
      allocate(mesh(iel)%bnd2_Unode(mU))
      allocate(mesh(iel)%bnd3_Unode(mU))
      allocate(mesh(iel)%bnd4_Unode(mU))
      allocate(mesh(iel)%bnd5_Unode(mU))
      allocate(mesh(iel)%bnd6_Unode(mU))
      allocate(mesh(iel)%bnd1_Vnode(mV))
      allocate(mesh(iel)%bnd2_Vnode(mV))
      allocate(mesh(iel)%bnd3_Vnode(mV))
      allocate(mesh(iel)%bnd4_Vnode(mV))
      allocate(mesh(iel)%bnd5_Vnode(mV))
      allocate(mesh(iel)%bnd6_Vnode(mV))
      allocate(mesh(iel)%bnd1_Wnode(mW))
      allocate(mesh(iel)%bnd2_Wnode(mW))
      allocate(mesh(iel)%bnd3_Wnode(mW))
      allocate(mesh(iel)%bnd4_Wnode(mW))
      allocate(mesh(iel)%bnd5_Wnode(mW))
      allocate(mesh(iel)%bnd6_Wnode(mW))
   case('spherical')
      allocate(mesh(iel)%inner_Unode(mV))
      allocate(mesh(iel)%outer_Unode(mV))
      allocate(mesh(iel)%inner_Vnode(mV))
      allocate(mesh(iel)%outer_Vnode(mV))
      allocate(mesh(iel)%inner_Wnode(mV))
      allocate(mesh(iel)%outer_Wnode(mV))
      allocate(mesh(iel)%rV(mV))           ! PB ?
      allocate(mesh(iel)%thetaV(mV))           ! PB ?
      allocate(mesh(iel)%phiV(mV))           ! PB ?
      allocate(mesh(iel)%rP(mP))           ! PB ?
      allocate(mesh(iel)%thetaP(mP))           ! PB ?
      allocate(mesh(iel)%phiP(mP))           ! PB ?
   case default
      stop 'initialise_elements: unknown geometry'
   end select

   allocate(mesh(iel)%fix_u(mU))
   allocate(mesh(iel)%fix_v(mV))
   allocate(mesh(iel)%fix_w(mW))

   allocate(mesh(iel)%iconT(mT)) 
   allocate(mesh(iel)%xT(mT))
   allocate(mesh(iel)%yT(mT))
   allocate(mesh(iel)%zT(mT))
   allocate(mesh(iel)%T(mT)) ; mesh(iel)%T=0.d0
   allocate(mesh(iel)%qx(mT))
   allocate(mesh(iel)%qy(mT))
   allocate(mesh(iel)%qz(mT))
   allocate(mesh(iel)%fix_T(mT))

   allocate(mesh(iel)%xM(mmapping))
   allocate(mesh(iel)%yM(mmapping))
   allocate(mesh(iel)%zM(mmapping))
   allocate(mesh(iel)%iconM(mmapping))

end do

time_assemble_S=0d0
time_assemble_K=0d0
time_assemble_GT=0d0
time_assemble_RHS=0d0

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'initialise_elements:',elapsed,' s            |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
