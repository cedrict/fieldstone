!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine recover_pressure_penalty

use module_parameters
use module_statistics 
use module_mesh 
use module_timing
use module_arrays, only : solP

implicit none

integer k,inode
real(8) rq,sq,tq,jcob,pp
real(8) dNdx(mV),dNdy(mV),dNdz(mV),div_v,q(NV),c(NV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{recover\_pressure\_penalty}
!@@ This is scheme 4 in Section~\ref{psmoothing} (see \stone~12) which was proven to be 
!@@ very cheap and very accurate. 
!==================================================================================================!

do iel=1,nel

   rq=0d0
   sq=0d0
   tq=0d0

   if (ndim==2) then
      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))
   else
      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))&
           +sum(dNdz(1:mV)*mesh(iel)%w(1:mV))
   end if 
   solP(iel)=-penalty*mesh(iel)%eta_avrg*div_v

end do

p_min=minval(solP)
p_max=maxval(solP)

if (normalise_pressure) then
   pp=sum(mesh(1:nel)%vol*solP(1:nel))/sum(mesh(1:nel)%vol)
   solP=solP-pp
end if

!----------------------------------------------------------
!transfer pressure onto elements

do iel=1,nel
   do k=1,mP
      inode=mesh(iel)%iconP(k)
      mesh(iel)%p(k)=solP(inode)
   end do
end do

!----------------------------------------------------------

q=0d0
c=0d0
do iel=1,nel
   do k=1,mV
      q(mesh(iel)%iconV(k))=q(mesh(iel)%iconV(k))+solP(iel)/mesh(iel)%vol
      c(mesh(iel)%iconV(k))=c(mesh(iel)%iconV(k))+      1d0/mesh(iel)%vol
   end do
end do
q=q/c

do iel=1,nel
   do k=1,mV
      mesh(iel)%q(k)=q(mesh(iel)%iconV(k))
   end do
end do

q_min=minval(q)
q_max=maxval(q)

!----------------------------------------------------------

if (debug) then
   do iel=1,nel
   do k=1,mV
      write(777,*) mesh(iel)%xV(k),mesh(iel)%yV(k),solP(iel),mesh(iel)%q(k)
   end do
   end do
end if

end subroutine

!==================================================================================================!
!==================================================================================================!
