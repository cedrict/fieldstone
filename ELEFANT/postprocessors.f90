!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine postprocessors

use global_parameters
use global_measurements
use structures
use timing

implicit none

integer iq
real(8) uq,vq,wq,pq,qq,NNNV(mV),NNNP(mP),NNNT(mT)
real(8) uth,vth,wth,pth,Tth,dum

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{postprocessors.f90}
!@@ This subroutine computes the root mean square velocity
!@@ and each of the average velocity components. It also 
!@@ computes the volume using GLQ.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

avrg_u=0d0
avrg_v=0d0
avrg_w=0d0
vrms=0d0
volume=0d0
errv=0d0
errp=0d0
errq=0d0

do iel=1,nel
   do iq=1,nqel

      !compute uq,vq,wq
      call NNV(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV))
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV))
      wq=sum(NNNV(1:mV)*mesh(iel)%w(1:mV))

      !compute pq
      call NNP(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNP(1:mP),mP,ndim,pair)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP))

      !compute qq
      call NNT(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNT(1:mT),mT,ndim,pair)
      qq=sum(NNNT(1:mT)*mesh(iel)%q(1:mT))

      avrg_u=avrg_u+uq*mesh(iel)%JxWq(iq)
      avrg_v=avrg_v+vq*mesh(iel)%JxWq(iq)
      avrg_w=avrg_w+wq*mesh(iel)%JxWq(iq)
      vrms=vrms+(uq**2+vq**2+wq**2)*mesh(iel)%JxWq(iq)
      volume=volume+mesh(iel)%JxWq(iq)

      call analytical_solution(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                               uth,vth,wth,pth,Tth,dum,dum,dum,dum,dum,dum)
      errv=errv+((uq-uth)**2+(vq-vth)**2+(wq-wth)**2)*mesh(iel)%JxWq(iq)
      errp=errp+(pq-pth)**2*mesh(iel)%JxWq(iq)
      errq=errq+(qq-pth)**2*mesh(iel)%JxWq(iq)

   end do
end do

errv=sqrt(errv)
errp=sqrt(errp)
errq=sqrt(errq)
vrms=sqrt(vrms/volume)
avrg_u=avrg_u/volume
avrg_v=avrg_v/volume
avrg_w=avrg_w/volume

write(*,'(a,es12.5)') '        vrms   =',vrms
write(*,'(a,es12.5)') '        avrg_u =',avrg_u
write(*,'(a,es12.5)') '        avrg_v =',avrg_v
write(*,'(a,es12.5)') '        avrg_w =',avrg_w
write(*,'(a,es12.5)') '        volume =',volume
write(*,'(a,es12.5)') '        errv   =',errv
write(*,'(a,es12.5)') '        errp   =',errp
write(*,'(a,es12.5)') '        errq   =',errq

call postprocessor_experiment

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> postprocessors                   ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
