!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine test_basis_functions

use global_parameters
use structures
use timing

implicit none

integer iq
real(8) NNNV(1:mV),rq,sq,tq,uq,vq,exxq,eyyq
real(8) dNdx(mV),dNdy(mV),dNdz(mV),jcob

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{test\_basis\_functions}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (debug) then

!constant field
write(*,*) 'constant u=v=1'
do iel=1,nel
   mesh(iel)%u=1
   mesh(iel)%v=1
   mesh(iel)%w=1
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      exxq=sum(dNdx(1:mV)*mesh(iel)%u(1:mV)) 
      eyyq=sum(dNdy(1:mV)*mesh(iel)%v(1:mV)) 
      print *,mesh(iel)%xq(iq),mesh(iel)%yq(iq),uq,vq,exxq,eyyq
   end do
end do

!linear field
write(*,*) 'linear'
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV
   mesh(iel)%v=mesh(iel)%yV
   mesh(iel)%w=mesh(iel)%zV
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      exxq=sum(dNdx(1:mV)*mesh(iel)%u(1:mV)) 
      eyyq=sum(dNdy(1:mV)*mesh(iel)%v(1:mV)) 
      print *,mesh(iel)%xq(iq),mesh(iel)%yq(iq),uq,vq,exxq,eyyq
   end do
end do



end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,*) '     -> test_basis_functions ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
