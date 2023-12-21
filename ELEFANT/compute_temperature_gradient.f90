!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_temperature_gradient

use module_parameters
use module_mesh 
use module_arrays
use module_timing

implicit none

integer k,node
real(8) qx(NT),qy(NT),cc(NT),dNdx(mT),dNdy(mT),jcob

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_temperature\_gradient}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   qx=0d0
   qy=0d0
   cc=0d0

   do iel=1,nel
      do k=1,mT
         call compute_dNTdx_dNTdy(rT(k),sT(k),dNdx(1:mT),dNdy(1:mT),jcob)
         node=mesh(iel)%iconT(k)
         qx(node)=qx(node)+sum(dNdx*mesh(iel)%T(1:mT))         
         qy(node)=qy(node)+sum(dNdy*mesh(iel)%T(1:mT))         
         cc(node)=cc(node)+1d0
      end do
   end do

   qx=qx/cc
   qy=qy/cc

   do iel=1,nel
      do k=1,mT
         node=mesh(iel)%iconT(k)
         mesh(iel)%qx(k)=qx(node)
         mesh(iel)%qy(k)=qy(node)
         mesh(iel)%qz(k)=0d0
      end do
   end do

   write(*,'(a,2es13.5)') shift//'qx(m/M)  =',minval(qx),maxval(qx)
   write(*,'(a,2es13.5)') shift//'qy(m/M)  =',minval(qy),maxval(qy)

end if

!----------------------------------------------------------

if (ndim==3) then

   stop 'compute_temperature_gradient: 3D not done'

end if





!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'compute_temperature_gradient (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
