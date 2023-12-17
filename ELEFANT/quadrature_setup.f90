!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine quadrature_setup

use module_parameters, only: ndim,mmapping,nq_per_dim,Nq,nel,nqel,iproc,iel,debug,spaceV,mapping
use module_mesh 
use module_constants, only: sqrt3,qc4a,qc4b,qw4a,qw4b
use module_timing

implicit none

integer iq,jq,kq,counter
real(8) rq,sq,tq,jcob
real(8) NNNM(mmapping),dNNNMdx(mmapping),dNNNMdy(mmapping),dNNNMdz(mmapping)
real(8), dimension(:), allocatable :: qcoords,qweights

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{quadrature\_setup.f90}
!@@ This subroutine allocates all GLQ-related arrays for each element.
!@@ It further computes the real $(x_q,y_q,z_q)$ and reduced $(r_q,s_q,t_q)$
!@@ coordinates of the GLQ points, and assigns them their weights and
!@@ jacobian values.
!@@ The required constants for the higher order quadrature schemes are in 
!@@ {\filenamefont module\_constants.f90}.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

if (spaceV=='__Q1') then
   nq_per_dim=2
   allocate(qcoords(nq_per_dim))
   allocate(qweights(nq_per_dim))
   qcoords=(/-1d0/sqrt3,+1.d0/sqrt3/)
   qweights=(/1d0,1d0/)
end if

if (spaceV=='__Q2') then
   nq_per_dim=3
   allocate(qcoords(nq_per_dim))
   allocate(qweights(nq_per_dim))
   qcoords=(/-sqrt(3d0/5d0),0d0,sqrt(3d0/5d0)/)
   qweights=(/5d0/9d0,8d0/9d0,5d0/9d0/)
end if

if (spaceV=='__Q3') then
   nq_per_dim=4
   allocate(qcoords(nq_per_dim))
   allocate(qweights(nq_per_dim))
   qcoords=(/-qc4a,-qc4b,qc4b,qc4a/)
   qweights=(/qw4a,qw4b,qw4b,qw4a/)
end if

nqel=nq_per_dim**ndim

Nq=nqel*nel

!===============================================================================

do iel=1,nel
   allocate(mesh(iel)%xq(nqel))      ; mesh(iel)%xq(:)=0 
   allocate(mesh(iel)%yq(nqel))      ; mesh(iel)%yq(:)=0 
   allocate(mesh(iel)%zq(nqel))      ; mesh(iel)%zq(:)=0 
   allocate(mesh(iel)%JxWq(nqel))    ; mesh(iel)%JxWq(:)=0 
   allocate(mesh(iel)%weightq(nqel)) ; mesh(iel)%weightq(:)=0 
   allocate(mesh(iel)%rq(nqel))      ; mesh(iel)%rq(:)=0 
   allocate(mesh(iel)%sq(nqel))      ; mesh(iel)%sq(:)=0 
   allocate(mesh(iel)%tq(nqel))      ; mesh(iel)%tq(:)=0 
   allocate(mesh(iel)%gxq(nqel))     ; mesh(iel)%gxq(:)=0 
   allocate(mesh(iel)%gyq(nqel))     ; mesh(iel)%gyq(:)=0 
   allocate(mesh(iel)%gzq(nqel))     ; mesh(iel)%gzq(:)=0 
   allocate(mesh(iel)%pq(nqel))      ; mesh(iel)%pq(:)=0 
   allocate(mesh(iel)%tempq(nqel))   ; mesh(iel)%tempq(:)=0 
   allocate(mesh(iel)%etaq(nqel))    ; mesh(iel)%etaq(:)=0 
   allocate(mesh(iel)%rhoq(nqel))    ; mesh(iel)%rhoq(:)=0 
   allocate(mesh(iel)%hcondq(nqel))  ; mesh(iel)%hcondq(:)=0 
   allocate(mesh(iel)%hcapaq(nqel))  ; mesh(iel)%hcapaq(:)=0 
   allocate(mesh(iel)%hprodq(nqel))  ; mesh(iel)%hprodq(:)=0 
end do

!===============================================================================

if (ndim==2) then
   do iel=1,nel
      counter=0
      do iq=1,nq_per_dim
      do jq=1,nq_per_dim
         counter=counter+1
         rq=qcoords(iq)
         sq=qcoords(jq)
         call NNN(rq,sq,0.d0,NNNM,mmapping,ndim,mapping)
         !print *,'------'
         !print *,iel,counter,rq,sq
         !print *,iel,counter,mesh(iel)%xM
         !print *,iel,counter,NNNM
         mesh(iel)%xq(counter)=sum(mesh(iel)%xM*NNNM)
         mesh(iel)%yq(counter)=sum(mesh(iel)%yM*NNNM)
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         call compute_dNdx_dNdy(rq,sq,dNNNMdx,dNNNMdy,jcob)
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
         call NNN(rq,sq,tq,NNNM,mmapping,ndim,mapping)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xM*NNNM)
         mesh(iel)%yq(counter)=sum(mesh(iel)%yM*NNNM)
         mesh(iel)%zq(counter)=sum(mesh(iel)%zM*NNNM)
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)*qweights(kq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=tq
         call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNNNMdx,dNNNMdy,dNNNMdz,jcob)
         mesh(iel)%JxWq(counter)=jcob*mesh(iel)%weightq(counter)
      end do
      end do
      end do
   end do
end if

if (debug) then
write(2345,*) limit//'quadrature_setup'//limit
write(2345,*) 'nq_per_dim=',nq_per_dim
write(2345,*) 'nqel=',nqel
write(2345,*) 'Nq=',Nq
write(2345,*) minval(mesh(1)%xq),maxval(mesh(1)%xq)
write(2345,*) minval(mesh(1)%yq),maxval(mesh(1)%yq)
write(2345,*) minval(mesh(1)%zq),maxval(mesh(1)%zq)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'quadrature_setup (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
