!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine recover_pressure_penalty

use global_parameters
use global_measurements
use structures
use timing
use global_arrays, only : solP

implicit none

integer k
real(8) rq,sq,tq,etaq,jcob
real(8) dNdx(mV),dNdy(mV),dNdz(mV),div_v,q(NV),c(NV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{recover\_pressure\_penalty}
!@@ This is scheme 4 in Section~\ref{psmoothing} (see \stone~12) which was proven to be 
!@@ very cheap and very accurate. 
!@@ The viscosity at the reduced quadrature location 
!@@ is obtained by taking the maximum viscosity value carried by the quadrature points of 
!@@ the element. 
!==================================================================================================!

do iel=1,nel

   rq=0d0
   sq=0d0
   tq=0d0
   etaq=maxval(mesh(iel)%etaq(:))

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
   solP(iel)=-penalty*etaq*div_v

end do

p_min=minval(solP)
p_max=maxval(solP)


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
