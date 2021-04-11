!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine postprocessors

use global_parameters
use structures
!use constants

implicit none

integer iq
real(8) avrg_u,avrg_v,avrg_w,vrms
real(8) uq,vq,wq,volume,NNNV(mV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{postprocessors.f90}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

avrg_u=0.d0
avrg_v=0.d0
avrg_w=0.d0
vrms=0.d0
volume=0.d0

do iel=1,nel
   do iq=1,nqel
      call NNV(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV))
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV))
      wq=sum(NNNV(1:mV)*mesh(iel)%w(1:mV))
      avrg_u=avrg_u+uq*mesh(iel)%JxWq(iq)
      avrg_v=avrg_v+vq*mesh(iel)%JxWq(iq)
      avrg_w=avrg_w+wq*mesh(iel)%JxWq(iq)
      vrms=vrms+(uq**2+vq**2+wq**2)*mesh(iel)%JxWq(iq)
      volume=volume+mesh(iel)%JxWq(iq)
   end do
end do

vrms=sqrt(vrms/volume)
avrg_u=avrg_u/volume
avrg_v=avrg_v/volume
avrg_w=avrg_w/volume

write(*,*) '     -> vrms=',vrms
write(*,*) '     -> avrg_u=',avrg_u
write(*,*) '     -> avrg_v=',avrg_v
write(*,*) '     -> avrg_w=',avrg_w






!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> postprocessors ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
