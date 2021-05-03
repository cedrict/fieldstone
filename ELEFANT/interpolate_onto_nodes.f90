!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine interpolate_onto_nodes

use module_parameters
use module_mesh 
use module_constants
use module_timing
use module_arrays, only: rV,sV,tV

implicit none

integer k,node
real(8) exx(NV),eyy(NV),ezz(NV),exy(NV),exz(NV),eyz(NV),ccc(NV)
real(8) dNdx(mV),dNdy(mV),dNdz(mV),jcob

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{interpolate\_onto\_nodes.f90}
!@@ This subroutine interpolates the components of the strain rate tensor on the velocity nodes.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   exx=0.d0
   eyy=0.d0
   exy=0.d0
   ccc=0.d0

   do iel=1,nel
      do k=1,mV
         call compute_dNdx_dNdy(rV(k),sV(k),dNdx,dNdy,jcob)
         node=mesh(iel)%iconV(k)
         exx(node)=exx(node)+sum(dNdx*mesh(iel)%u(1:mV))         
         eyy(node)=eyy(node)+sum(dNdy*mesh(iel)%v(1:mV))         
         exy(node)=exy(node)+sum(dNdx*mesh(iel)%v(1:mV))*0.5d0 &
                            +sum(dNdy*mesh(iel)%u(1:mV))*0.5d0 
         ccc(node)=ccc(node)+1d0
      end do
   end do

   exx=exx/ccc
   eyy=eyy/ccc
   exy=exy/ccc

   do iel=1,nel
      do k=1,mV
         node=mesh(iel)%iconV(k)
         mesh(iel)%exx(k)=exx(node)
         mesh(iel)%eyy(k)=eyy(node)
         mesh(iel)%exy(k)=exy(node)
      end do
   end do

end if

!--------------------------------------

if (ndim==3) then

   exx=0.d0
   eyy=0.d0
   ezz=0.d0
   exy=0.d0
   exz=0.d0
   eyz=0.d0
   ccc=0.d0

   do iel=1,nel
      do k=1,mV
         call compute_dNdx_dNdy_dNdz(rV(k),sV(k),tV(k),dNdx,dNdy,dNdz,jcob)
         node=mesh(iel)%iconV(k)
         exx(node)=exx(node)+sum(dNdx*mesh(iel)%u(1:mV))         
         eyy(node)=eyy(node)+sum(dNdy*mesh(iel)%v(1:mV))         
         ezz(node)=ezz(node)+sum(dNdz*mesh(iel)%w(1:mV))         
         exy(node)=exy(node)+sum(dNdx*mesh(iel)%v(1:mV))*0.5d0 &
                            +sum(dNdy*mesh(iel)%u(1:mV))*0.5d0 
         exz(node)=exz(node)+sum(dNdx*mesh(iel)%w(1:mV))*0.5d0 &
                            +sum(dNdz*mesh(iel)%u(1:mV))*0.5d0 
         eyz(node)=eyz(node)+sum(dNdy*mesh(iel)%w(1:mV))*0.5d0 &
                            +sum(dNdz*mesh(iel)%v(1:mV))*0.5d0 
         ccc(node)=ccc(node)+1d0
      end do
   end do

   exx=exx/ccc
   eyy=eyy/ccc
   ezz=ezz/ccc
   exy=exy/ccc
   exz=exz/ccc
   eyz=eyz/ccc

   do iel=1,nel
      do k=1,mV
         node=mesh(iel)%iconV(k)
         mesh(iel)%exx(k)=exx(node)
         mesh(iel)%eyy(k)=eyy(node)
         mesh(iel)%ezz(k)=ezz(node)
         mesh(iel)%exy(k)=exy(node)
         mesh(iel)%exz(k)=exz(node)
         mesh(iel)%eyz(k)=eyz(node)
      end do
   end do

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'interpolate_onto_nodes (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
