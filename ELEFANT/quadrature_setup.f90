!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine quadrature_setup

use module_parameters, only: ndim,mmapping,nqpts,NQ,nel,nqel,iproc,iel,debug,spaceVelocity,mapping
use module_mesh 
use module_constants, only: sqrt3
use module_timing
use module_quadrature

implicit none

integer iq,jq,kq,counter
real(8) rq,sq,tq
real(8) NNNM(mmapping)
real(8), dimension(:), allocatable :: qcoords,qweights
integer, parameter :: caller_id01=601

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{quadrature\_setup.f90}
!@@ This subroutine allocates all GLQ-related arrays for each element.
!@@ It further computes the real coordinates $(x_q,y_q,z_q)$ and reduced 
!@@ coordinates $(r_q,s_q,t_q)$ of the GLQ points, and assigns them their weights and
!@@ jacobian values.
!@@ The required constants for the quadrature schemes are in 
!@@ {\filenamefont module\_quadrature.f90}.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!----------------------------------------------------------
! compute nqel and NQ
!----------------------------------------------------------

select case(spaceVelocity)
case('__Q1','__Q2','__Q3','_Q1+','Q1++','_Q1F')
   nqel=nqpts**ndim
case('__P1','__P2','__P3','_P1+','_P2+')
   nqel=nqpts
case default
   stop 'quadrature_setup: spaceVelocity not supported yet'
end select

NQ=nqel*nel ! total number of quadrature points

write(*,'(a,i5)') shift//'nqel=',nqel
write(*,'(a,i5)') shift//'NQ=',NQ

!----------------------------------------------------------
! allocate quadrature-related arrays for each element
!----------------------------------------------------------

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
   
allocate(qcoords(nqpts))
allocate(qweights(nqpts))

!----------------------------------------------------------
select case(spaceVelocity)
case('__Q1','__Q2','__Q3','_Q1+','Q1++','_Q1F')

   !compute qcoords & qweights
   select case(nqpts)
   case(1)
      qcoords=(/0.d0/)
      qweights=(/2.d0/)
   case(2)
      qcoords=(/-qc2a,qc2a/)
      qweights=(/qw2a,qw2a/)
   case(3)
      qcoords=(/-qc3a,qc3b,qc3a/)
      qweights=(/qw3a,qw3b,qw3a/)
   case(4)
      qcoords=(/-qc4a,-qc4b,qc4b,qc4a/)
      qweights=(/qw4a,qw4b,qw4b,qw4a/)
   case(5)
      qcoords=(/-qc5a,-qc5b,qc5c,qc5b,qc5a/)
      qweights=(/qw5a,qw5b,qw5c,qw5b,qw5a/)
   case(6)
      qcoords=(/-qc6a,-qc6b,-qc6c,qc6c,qc6b,qc6a/)
      qweights=(/qw6a,qw6b,qw6c,qw6c,qw6b,qw6a/)
   case(7)
      qcoords=(/-qc7a,-qc7b,-qc7c,qc7d,qc7c,qc7b,qc7a/)
      qweights=(/qw7a,qw7b,qw7c,qw7d,qw7c,qw7b,qw7a/)
   case(8)
      qcoords=(/-qc8a,-qc8b,-qc8c,-qc8d,qc8d,qc8c,qc8b,qc8a/)
      qweights=(/qw8a,qw8b,qw8c,qw8d,qw8d,qw8c,qw8b,qw8a/)
   case(9)
      qcoords=(/-qc9a,-qc9b,-qc9c,-qc9d,qc9e,qc9d,qc9c,qc9b,qc9a/)
      qweights=(/qw9a,qw9b,qw9c,qw9d,qw9e,qw9d,qw9c,qw9b,qw9a/)
   case(10)
      qcoords=(/-qc10a,-qc10b,-qc10c,-qc10d,-qc10e,qc10e,qc10d,qc10c,qc10b,qc10a/)
      qweights=(/qw10a,qw10b,qw10c,qw10d,qw10e,qw10e,qw10d,qw10c,qw10b,qw10a/)
   case default
      stop 'quadrature_setup: nqpts not supported for Q elts' 
   end select

   !-------------------------------------------------------

   if (ndim==2) then
   do iel=1,nel
      counter=0
      do iq=1,nqpts
      do jq=1,nqpts
         counter=counter+1
         rq=qcoords(iq)
         sq=qcoords(jq)
         call NNN(rq,sq,0.d0,NNNM,mmapping,ndim,mapping,caller_id01)
         !print *,'------'
         !print *,iel,counter,rq,sq
         !print *,iel,counter,mesh(iel)%xM
         !print *,iel,counter,NNNM
         mesh(iel)%xq(counter)=sum(mesh(iel)%xM*NNNM)
         mesh(iel)%yq(counter)=sum(mesh(iel)%yM*NNNM)
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
      end do
      end do
   end do
   end if

   !-------------------------------------------------------

   if (ndim==3) then
   do iel=1,nel
      counter=0
      do iq=1,nqpts
      do jq=1,nqpts
      do kq=1,nqpts
         counter=counter+1
         rq=qcoords(iq)
         sq=qcoords(jq)
         tq=qcoords(kq)
         call NNN(rq,sq,tq,NNNM,mmapping,ndim,mapping,caller_id01)
         mesh(iel)%xq(counter)=sum(mesh(iel)%xM*NNNM)
         mesh(iel)%yq(counter)=sum(mesh(iel)%yM*NNNM)
         mesh(iel)%zq(counter)=sum(mesh(iel)%zM*NNNM)
         mesh(iel)%weightq(counter)=qweights(iq)*qweights(jq)*qweights(kq)
         mesh(iel)%rq(counter)=rq
         mesh(iel)%sq(counter)=sq
         mesh(iel)%tq(counter)=tq
      end do
      end do
      end do
   end do
   end if

case('__P1','__P2','__P3','_P1+','_P2+')

   !compute qcoords & qweights
   if (ndim==2) then
   select case(nqpts)
   case(3)
      do iel=1,nel
         mesh(iel)%rq(1)=1d0/6d0 ; mesh(iel)%sq(1)=1d0/6d0 ; mesh(iel)%weightq(1)=1d0/6d0
         mesh(iel)%rq(2)=2d0/3d0 ; mesh(iel)%sq(2)=1d0/6d0 ; mesh(iel)%weightq(2)=1d0/6d0
         mesh(iel)%rq(3)=1d0/6d0 ; mesh(iel)%sq(3)=2d0/3d0 ; mesh(iel)%weightq(3)=1d0/6d0
      end do
   case(6)
      do iel=1,nel
         mesh(iel)%rq(1)=0.091576213509771 ; mesh(iel)%sq(1)=0.091576213509771 ; mesh(iel)%weightq(1)=0.109951743655322/2.0
         mesh(iel)%rq(2)=0.816847572980459 ; mesh(iel)%sq(2)=0.091576213509771 ; mesh(iel)%weightq(2)=0.109951743655322/2.0
         mesh(iel)%rq(3)=0.091576213509771 ; mesh(iel)%sq(3)=0.816847572980459 ; mesh(iel)%weightq(3)=0.109951743655322/2.0
         mesh(iel)%rq(4)=0.445948490915965 ; mesh(iel)%sq(4)=0.445948490915965 ; mesh(iel)%weightq(4)=0.223381589678011/2.0
         mesh(iel)%rq(5)=0.108103018168070 ; mesh(iel)%sq(5)=0.445948490915965 ; mesh(iel)%weightq(5)=0.223381589678011/2.0
         mesh(iel)%rq(6)=0.445948490915965 ; mesh(iel)%sq(6)=0.108103018168070 ; mesh(iel)%weightq(6)=0.223381589678011/2.0
      end do
    case(7) !5th order
      do iel=1,nel
         mesh(iel)%rq(1)=0.1012865073235 ; mesh(iel)%sq(1)=0.1012865073235 ; mesh(iel)%weightq(1)=0.0629695902724 
         mesh(iel)%rq(2)=0.7974269853531 ; mesh(iel)%sq(2)=0.1012865073235 ; mesh(iel)%weightq(2)=0.0629695902724 
         mesh(iel)%rq(3)=0.1012865073235 ; mesh(iel)%sq(3)=0.7974269853531 ; mesh(iel)%weightq(3)=0.0629695902724 
         mesh(iel)%rq(4)=0.4701420641051 ; mesh(iel)%sq(4)=0.0597158717898 ; mesh(iel)%weightq(4)=0.0661970763942 
         mesh(iel)%rq(5)=0.4701420641051 ; mesh(iel)%sq(5)=0.4701420641051 ; mesh(iel)%weightq(5)=0.0661970763942
         mesh(iel)%rq(6)=0.0597158717898 ; mesh(iel)%sq(6)=0.4701420641051 ; mesh(iel)%weightq(6)=0.0661970763942
         mesh(iel)%rq(7)=0.3333333333333 ; mesh(iel)%sq(7)=0.3333333333333 ; mesh(iel)%weightq(7)=0.1125000000000
      end do
    case(12) !6th order
      do iel=1,nel
         mesh(iel)%rq( 1)=0.24928674517091 ; mesh(iel)%sq( 1)=0.24928674517091 ; mesh(iel)%weightq( 1)=0.11678627572638/2
         mesh(iel)%rq( 2)=0.24928674517091 ; mesh(iel)%sq( 2)=0.50142650965818 ; mesh(iel)%weightq( 2)=0.11678627572638/2
         mesh(iel)%rq( 3)=0.50142650965818 ; mesh(iel)%sq( 3)=0.24928674517091 ; mesh(iel)%weightq( 3)=0.11678627572638/2
         mesh(iel)%rq( 4)=0.06308901449150 ; mesh(iel)%sq( 4)=0.06308901449150 ; mesh(iel)%weightq( 4)=0.05084490637021/2
         mesh(iel)%rq( 5)=0.06308901449150 ; mesh(iel)%sq( 5)=0.87382197101700 ; mesh(iel)%weightq( 5)=0.05084490637021/2
         mesh(iel)%rq( 6)=0.87382197101700 ; mesh(iel)%sq( 6)=0.06308901449150 ; mesh(iel)%weightq( 6)=0.05084490637021/2
         mesh(iel)%rq( 7)=0.31035245103378 ; mesh(iel)%sq( 7)=0.63650249912140 ; mesh(iel)%weightq( 7)=0.08285107561837/2
         mesh(iel)%rq( 8)=0.63650249912140 ; mesh(iel)%sq( 8)=0.05314504984482 ; mesh(iel)%weightq( 8)=0.08285107561837/2
         mesh(iel)%rq( 9)=0.05314504984482 ; mesh(iel)%sq( 9)=0.31035245103378 ; mesh(iel)%weightq( 9)=0.08285107561837/2
         mesh(iel)%rq(10)=0.63650249912140 ; mesh(iel)%sq(10)=0.31035245103378 ; mesh(iel)%weightq(10)=0.08285107561837/2
         mesh(iel)%rq(11)=0.31035245103378 ; mesh(iel)%sq(11)=0.05314504984482 ; mesh(iel)%weightq(11)=0.08285107561837/2
         mesh(iel)%rq(12)=0.05314504984482 ; mesh(iel)%sq(12)=0.63650249912140 ; mesh(iel)%weightq(12)=0.08285107561837/2
      end do
    case(13) !7th order
      do iel=1,nel
         mesh(iel)%rq( 1)=0.33333333333333 ; mesh(iel)%sq( 1)=0.33333333333333 ; mesh(iel)%weightq( 1)=-0.14957004446768/2  !suspicious!!
         mesh(iel)%rq( 2)=0.26034596607904 ; mesh(iel)%sq( 2)=0.26034596607904 ; mesh(iel)%weightq( 2)=0.17561525743321/2
         mesh(iel)%rq( 3)=0.26034596607904 ; mesh(iel)%sq( 3)=0.47930806784192 ; mesh(iel)%weightq( 3)=0.17561525743321/2
         mesh(iel)%rq( 4)=0.47930806784192 ; mesh(iel)%sq( 4)=0.26034596607904 ; mesh(iel)%weightq( 4)=0.17561525743321/2
         mesh(iel)%rq( 5)=0.06513010290222 ; mesh(iel)%sq( 5)=0.06513010290222 ; mesh(iel)%weightq( 5)=0.05334723560884/2
         mesh(iel)%rq( 6)=0.06513010290222 ; mesh(iel)%sq( 6)=0.86973979419557 ; mesh(iel)%weightq( 6)=0.05334723560884/2
         mesh(iel)%rq( 7)=0.86973979419557 ; mesh(iel)%sq( 7)=0.06513010290222 ; mesh(iel)%weightq( 7)=0.05334723560884/2
         mesh(iel)%rq( 8)=0.31286549600487 ; mesh(iel)%sq( 8)=0.63844418856981 ; mesh(iel)%weightq( 8)=0.07711376089026/2
         mesh(iel)%rq( 9)=0.63844418856981 ; mesh(iel)%sq( 9)=0.04869031542532 ; mesh(iel)%weightq( 9)=0.07711376089026/2
         mesh(iel)%rq(10)=0.04869031542532 ; mesh(iel)%sq(10)=0.31286549600487 ; mesh(iel)%weightq(10)=0.07711376089026/2
         mesh(iel)%rq(11)=0.63844418856981 ; mesh(iel)%sq(11)=0.31286549600487 ; mesh(iel)%weightq(11)=0.07711376089026/2
         mesh(iel)%rq(12)=0.31286549600487 ; mesh(iel)%sq(12)=0.04869031542532 ; mesh(iel)%weightq(12)=0.07711376089026/2
         mesh(iel)%rq(13)=0.04869031542532 ; mesh(iel)%sq(13)=0.63844418856981 ; mesh(iel)%weightq(13)=0.07711376089026/2
      end do
   case(16) 
      do iel=1,nel
         mesh(iel)%rq( 1)=0.33333333333333 ; mesh(iel)%sq( 1)=0.33333333333333 ; mesh(iel)%weightq( 1)=0.14431560767779/2
         mesh(iel)%rq( 2)=0.45929258829272 ; mesh(iel)%sq( 2)=0.45929258829272 ; mesh(iel)%weightq( 2)=0.09509163426728/2
         mesh(iel)%rq( 3)=0.45929258829272 ; mesh(iel)%sq( 3)=0.08141482341455 ; mesh(iel)%weightq( 3)=0.09509163426728/2
         mesh(iel)%rq( 4)=0.08141482341455 ; mesh(iel)%sq( 4)=0.45929258829272 ; mesh(iel)%weightq( 4)=0.09509163426728/2
         mesh(iel)%rq( 5)=0.17056930775176 ; mesh(iel)%sq( 5)=0.17056930775176 ; mesh(iel)%weightq( 5)=0.10321737053472/2
         mesh(iel)%rq( 6)=0.17056930775176 ; mesh(iel)%sq( 6)=0.65886138449648 ; mesh(iel)%weightq( 6)=0.10321737053472/2
         mesh(iel)%rq( 7)=0.65886138449648 ; mesh(iel)%sq( 7)=0.17056930775176 ; mesh(iel)%weightq( 7)=0.10321737053472/2
         mesh(iel)%rq( 8)=0.05054722831703 ; mesh(iel)%sq( 8)=0.05054722831703 ; mesh(iel)%weightq( 8)=0.03245849762320/2
         mesh(iel)%rq( 9)=0.05054722831703 ; mesh(iel)%sq( 9)=0.89890554336594 ; mesh(iel)%weightq( 9)=0.03245849762320/2
         mesh(iel)%rq(10)=0.89890554336594 ; mesh(iel)%sq(10)=0.05054722831703 ; mesh(iel)%weightq(10)=0.03245849762320/2
         mesh(iel)%rq(11)=0.26311282963464 ; mesh(iel)%sq(11)=0.72849239295540 ; mesh(iel)%weightq(11)=0.02723031417443/2
         mesh(iel)%rq(12)=0.72849239295540 ; mesh(iel)%sq(12)=0.00839477740996 ; mesh(iel)%weightq(12)=0.02723031417443/2
         mesh(iel)%rq(13)=0.00839477740996 ; mesh(iel)%sq(13)=0.26311282963464 ; mesh(iel)%weightq(13)=0.02723031417443/2
         mesh(iel)%rq(14)=0.72849239295540 ; mesh(iel)%sq(14)=0.26311282963464 ; mesh(iel)%weightq(14)=0.02723031417443/2
         mesh(iel)%rq(15)=0.26311282963464 ; mesh(iel)%sq(15)=0.00839477740996 ; mesh(iel)%weightq(15)=0.02723031417443/2
         mesh(iel)%rq(16)=0.00839477740996 ; mesh(iel)%sq(16)=0.72849239295540 ; mesh(iel)%weightq(16)=0.02723031417443/2
      end do
   case default
      stop 'quadrature_setup: nqpts not supported for P elts' 
   end select

   do iel=1,nel
      do kq=1,nqel
         call NNN(mesh(iel)%rq(kq),mesh(iel)%sq(kq),0.d0,NNNM,mmapping,ndim,mapping,caller_id01)
         mesh(iel)%xq(kq)=sum(mesh(iel)%xM*NNNM)
         mesh(iel)%yq(kq)=sum(mesh(iel)%yM*NNNM)
      end do
   end do

   else ! ndim

      stop 'quadrature_setup: 3D tetrahedra not implemented'

   end if


case default
   stop 'quadrature_setup: spaceVelocity not supported yet'
end select

deallocate(qcoords)
deallocate(qweights)

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'quadrature_setup'//limit
write(2345,*) 'nqpts=',nqpts
write(2345,*) 'nqel=',nqel
write(2345,*) 'NQ=',NQ
write(2345,*) minval(mesh(1)%xq),maxval(mesh(1)%xq)
write(2345,*) minval(mesh(1)%yq),maxval(mesh(1)%yq)
write(2345,*) minval(mesh(1)%zq),maxval(mesh(1)%zq)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'quadrature_setup:',elapsed,' s               |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
