!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_strain_rate

use module_parameters
use module_mesh 
use module_arrays, only: dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz
!use module_constants
!use module_arrays
use module_timing

implicit none

real(8) :: jcob
real(8) :: exx_min,exx_max,eyy_min,eyy_max,ezz_min,ezz_max
real(8) :: exy_min,exy_max,exz_min,exz_max,eyz_min,eyz_max
real(8), parameter :: rc=0,sc=0,tc=0

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_elemental\_strain\_rate}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

exx_min=+1d50 ; exx_max=-1d50 
eyy_min=+1d50 ; eyy_max=-1d50 
ezz_min=+1d50 ; ezz_max=-1d50 
exy_min=+1d50 ; exy_max=-1d50 
exz_min=+1d50 ; exz_max=-1d50 
eyz_min=+1d50 ; eyz_max=-1d50 

do iel=1,nel
   call compute_dNdx_dNdy_dNdz(rc,sc,tc,dNNNUdx,dNNNUdy,dNNNUdz,&
                                        dNNNVdx,dNNNVdy,dNNNVdz,&
                                        dNNNWdx,dNNNWdy,dNNNWdz,jcob)
   mesh(iel)%exx=sum(dNNNUdx*mesh(iel)%u)
   mesh(iel)%eyy=sum(dNNNVdy*mesh(iel)%v)
   mesh(iel)%ezz=sum(dNNNWdz*mesh(iel)%w)
   mesh(iel)%exy=0.5d0*sum(dNNNUdy*mesh(iel)%u + dNNNVdx*mesh(iel)%v)
   mesh(iel)%exz=0.5d0*sum(dNNNUdz*mesh(iel)%u + dNNNWdx*mesh(iel)%w)
   mesh(iel)%eyz=0.5d0*sum(dNNNVdz*mesh(iel)%v + dNNNWdy*mesh(iel)%w)

   exx_min=min(mesh(iel)%exx,exx_min) ; exx_max=max(mesh(iel)%exx,exx_max)
   eyy_min=min(mesh(iel)%eyy,eyy_min) ; eyy_max=max(mesh(iel)%eyy,eyy_max)
   ezz_min=min(mesh(iel)%ezz,ezz_min) ; ezz_max=max(mesh(iel)%ezz,ezz_max)
   exy_min=min(mesh(iel)%exy,exy_min) ; exy_max=max(mesh(iel)%exy,exy_max)
   exz_min=min(mesh(iel)%exz,exz_min) ; exz_max=max(mesh(iel)%exz,exz_max)
   eyz_min=min(mesh(iel)%eyz,eyz_min) ; eyz_max=max(mesh(iel)%eyz,eyz_max)

end do

write(*,'(a,2es10.3)') shift//'exx (m/M):',exx_min,eyy_max
!write(*,'(a,2es10.3)') shift//'eyy (m/M):',etaq_min,etaq_max

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_elemental_strain_rate:',elapsed,' s      |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
