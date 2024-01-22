!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine postprocessors

use module_parameters
use module_statistics 
use module_mesh
use module_timing

implicit none

integer iq
real(8) uq,vq,wq,pq,Tq,NNNU(mU),NNNV(mV),NNNW(mW),NNNP(mP),NNNT(mT)
real(8) uth,vth,wth,pth,Tth,dum

integer, parameter :: caller_id01=1001
integer, parameter :: caller_id02=1002
integer, parameter :: caller_id03=1003
integer, parameter :: caller_id04=1004
integer, parameter :: caller_id05=1005

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{postprocessors.f90}
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
avrg_T=0d0
avrg_p=0d0
avrg_q=0d0
vrms=0d0
volume=0d0
errv=0d0
errp=0d0
errq=0d0
errT=0d0

do iel=1,nel
   do iq=1,nqel

      !compute uq,vq,wq
      call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNU(1:mU),mU,ndim,spaceU,caller_id01)
      uq=sum(NNNU(1:mU)*mesh(iel)%u(1:mU))

      call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNV(1:mV),mV,ndim,spaceV,caller_id02)
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV))

      if (ndim==3) then
         call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNW(1:mW),mW,ndim,spaceW,caller_id03)
         wq=sum(NNNW(1:mW)*mesh(iel)%w(1:mW))
      else
         wq=0d0
      end if

      !compute pq
      call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNP(1:mP),mP,ndim,spacePressure,caller_id04)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP))

      !compute Tq
      if (use_T) then
         call NNN(mesh(iel)%rq(iq),mesh(iel)%sq(iq),mesh(iel)%tq(iq),NNNT(1:mT),mT,ndim,spaceTemperature,caller_id05)
         Tq=sum(NNNT(1:mT)*mesh(iel)%T(1:mT))
      else
         Tq=0
      end if

      avrg_u=avrg_u+uq*mesh(iel)%JxWq(iq)
      avrg_v=avrg_v+vq*mesh(iel)%JxWq(iq)
      avrg_w=avrg_w+wq*mesh(iel)%JxWq(iq)
      avrg_T=avrg_T+Tq*mesh(iel)%JxWq(iq)
      avrg_p=avrg_p+pq*mesh(iel)%JxWq(iq)
      vrms=vrms+(uq**2+vq**2+wq**2)*mesh(iel)%JxWq(iq)
      volume=volume+mesh(iel)%JxWq(iq)

      call experiment_analytical_solution(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                               uth,vth,wth,pth,Tth,dum,dum,dum,dum,dum,dum)
      errv=errv+((uq-uth)**2+(vq-vth)**2+(wq-wth)**2)*mesh(iel)%JxWq(iq)
      errp=errp+(pq-pth)**2*mesh(iel)%JxWq(iq)
      errT=errT+(Tq-Tth)**2*mesh(iel)%JxWq(iq)

   end do
end do

errv=sqrt(errv)
errp=sqrt(errp)
errT=sqrt(errT)
vrms=sqrt(vrms/volume)
avrg_u=avrg_u/volume
avrg_v=avrg_v/volume
avrg_w=avrg_w/volume
avrg_T=avrg_T/volume
avrg_p=avrg_p/volume

if (solve_stokes_system) then 
             write(*,'(a,es12.5)') shift//'vrms =',vrms
             write(*,'(a,es12.5)') shift//'<u>  =',avrg_u
             write(*,'(a,es12.5)') shift//'<v>  =',avrg_v
if (ndim==3) write(*,'(a,es12.5)') shift//'<w>  =',avrg_w
             write(*,'(a,es12.5)') shift//'<p>  =',avrg_p
             write(*,'(a,es12.5)') shift//'<q>  =',avrg_q
             write(*,'(a,es12.5)') shift//'errv =',errv
             write(*,'(a,es12.5)') shift//'errp =',errp
             write(*,'(a,es12.5)') shift//'errq =',errq
end if

if (use_T)   write(*,'(a,es12.5)') shift//'errT =',errT
if (use_T)   write(*,'(a,es12.5)') shift//'<T>  =',avrg_T
             write(*,'(a,es12.5)') shift//'vol  =',volume

call experiment_postprocessor

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'postprocessors:',elapsed,' s                 |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
