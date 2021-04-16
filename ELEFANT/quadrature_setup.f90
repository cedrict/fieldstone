!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine quadrature_setup

use global_parameters
use structures
use constants
use timing

implicit none

integer iq,jq,kq,counter
real(8) rq,sq,tq,NNNV(mV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{quadrature\_setup.f90}
!@@ This subroutine allocates all GLQ-related arrays for each element.
!@@ It further computes the real $(x_q,y_q,z_q)$ and reduced $(r_q,s_q,t_q)$
!@@ coordinates of the GLQ points, and assigns them their weights.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

do iel=1,nel
   allocate(mesh(iel)%xq(nqel),mesh(iel)%yq(nqel),mesh(iel)%zq(nqel))
   allocate(mesh(iel)%weightq(nqel),mesh(iel)%JxWq(nqel))
   allocate(mesh(iel)%rq(nqel),mesh(iel)%sq(nqel),mesh(iel)%tq(nqel))
   allocate(mesh(iel)%gxq(nqel),mesh(iel)%gyq(nqel),mesh(iel)%gzq(nqel))
   allocate(mesh(iel)%rhoq(nqel),mesh(iel)%etaq(nqel))
   allocate(mesh(iel)%hcondq(nqel),mesh(iel)%hcapaq(nqel),mesh(iel)%hprodq(nqel))
   allocate(mesh(iel)%pq(nqel),mesh(iel)%thetaq(nqel))
end do

!--------------------------------------

if (ndim==2) then
   do iel=1,nel
      counter=0
      do iq=1,nq_per_dim
      do jq=1,nq_per_dim
         counter=counter+1
         rq=qcoords(iq)
         sq=qcoords(jq)
         call NNV(rq,sq,0,NNNV(1:mV),mV,ndim,pair)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xV(1:mV)*NNNV(1:mV))
         mesh(iel)%yq(counter)=sum(mesh(iel)%yV(1:mV)*NNNV(1:mV))
         mesh(iel)%zq(counter)=0.d0
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=0
      end do
      end do
   end do
end if

!--------------------------------------

if (ndim==3) then
   do iel=1,nel
      counter=0
      do iq=1,nq_per_dim
      do jq=1,nq_per_dim
      do kq=1,nq_per_dim
         counter=counter+1
         rq=qcoords(iq)
         sq=qcoords(jq)
         tq=qcoords(kq)
         call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xV(1:mV)*NNNV(1:mV))
         mesh(iel)%yq(counter)=sum(mesh(iel)%yV(1:mV)*NNNV(1:mV))
         mesh(iel)%zq(counter)=sum(mesh(iel)%zV(1:mV)*NNNV(1:mV))
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)*qweights(kq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=tq
      end do
      end do
      end do
   end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> quadrature_setup                 ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
