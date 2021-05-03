!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine quadrature_setup

use module_parameters
use module_mesh 
use module_constants
use module_timing

implicit none

integer iq,jq,kq,counter
real(8) rq,sq,tq,NNNL(mL),dNdx(mV),dNdy(mV),dNdz(mV),jcob
real(8), dimension(:), allocatable :: qcoords,qweights

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{quadrature\_setup.f90}
!@@ This subroutine allocates all GLQ-related arrays for each element.
!@@ It further computes the real $(x_q,y_q,z_q)$ and reduced $(r_q,s_q,t_q)$
!@@ coordinates of the GLQ points, and assigns them their weights.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

if (pair=='q1p0' .or. pair=='q1q1') then
   nq_per_dim=2
   allocate(qcoords(nq_per_dim))
   allocate(qweights(nq_per_dim))
   qcoords=(/-1d0/sqrt3,+1.d0/sqrt3/)
   qweights=(/1d0,1d0/)
end if

if (pair=='q2q1') then
   nq_per_dim=3
   allocate(qcoords(nq_per_dim))
   allocate(qweights(nq_per_dim))
   qcoords=(/-sqrt(3d0/5d0),0d0,sqrt(3d0/5d0)/)
   qweights=(/ 5d0/9d0,8d0/9d0,5d0/9d0 /)
end if

nqel=nq_per_dim**ndim

Nq=nqel*nel

!==============================================================================!

do iel=1,nel
   allocate(mesh(iel)%xq(nqel),mesh(iel)%yq(nqel),mesh(iel)%zq(nqel))
   allocate(mesh(iel)%weightq(nqel),mesh(iel)%JxWq(nqel))
   allocate(mesh(iel)%rq(nqel),mesh(iel)%sq(nqel),mesh(iel)%tq(nqel))
   allocate(mesh(iel)%gxq(nqel),mesh(iel)%gyq(nqel),mesh(iel)%gzq(nqel))
   allocate(mesh(iel)%rhoq(nqel),mesh(iel)%etaq(nqel))
   allocate(mesh(iel)%hcondq(nqel),mesh(iel)%hcapaq(nqel),mesh(iel)%hprodq(nqel))
   allocate(mesh(iel)%pq(nqel),mesh(iel)%thetaq(nqel))
   mesh(iel)%JxWq(:)=0d0
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
         call NNL(rq,sq,0,NNNL(1:mV),mV,ndim,pair)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xV(1:mV)*NNNL(1:mV))
         mesh(iel)%yq(counter)=sum(mesh(iel)%yV(1:mV)*NNNL(1:mV))
         mesh(iel)%zq(counter)=0.d0
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=0
         call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
         mesh(iel)%JxWq(counter)=jcob*mesh(iel)%weightq(counter)
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
         call NNL(rq,sq,tq,NNNL(1:mV),mV,ndim,pair)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xV(1:mV)*NNNL(1:mV))
         mesh(iel)%yq(counter)=sum(mesh(iel)%yV(1:mV)*NNNL(1:mV))
         mesh(iel)%zq(counter)=sum(mesh(iel)%zV(1:mV)*NNNL(1:mV))
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)*qweights(kq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=tq
         call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
         mesh(iel)%JxWq(counter)=jcob*mesh(iel)%weightq(counter)
      end do
      end do
      end do
   end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'quadrature_setup (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
