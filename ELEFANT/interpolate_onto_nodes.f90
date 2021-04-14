!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine interpolate_onto_nodes

use global_parameters
use structures
use constants
use timing

implicit none

integer k,node
real(8), dimension(:), allocatable :: exx,eyy,ezz,exy,exz,eyz,counter,q
real(8) dNdx(mV),dNdy(mV),dNdz(mV),jcob

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{interpolate\_onto\_nodes.f90}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   allocate(q(NT))   ; q=0.d0
   allocate(exx(NT)) ; exx=0.d0
   allocate(eyy(NT)) ; eyy=0.d0
   allocate(exy(NT)) ; exy=0.d0
   allocate(counter(NT)) ; counter=0.d0


   do iel=1,nel
      do k=1,ncorners
         call compute_dNdx_dNdy(rcorners(k),scorners(k),dNdx,dNdy,jcob)
         node=mesh(iel)%iconV(k)
         exx(node)=exx(node)+sum(dNdx*mesh(iel)%u(1:mV))         
         eyy(node)=eyy(node)+sum(dNdy*mesh(iel)%v(1:mV))         
         exy(node)=exy(node)+sum(dNdx*mesh(iel)%v(1:mV))*0.5d0 &
                            +sum(dNdy*mesh(iel)%u(1:mV))*0.5d0 
         if (pair=='q1p0') q(node)=q(node)+mesh(iel)%p(1)
         counter(node)=counter(node)+1
      end do
   end do

   exx=exx/counter
   eyy=eyy/counter
   exy=exy/counter
   q=q/counter

   do iel=1,nel
      do k=1,ncorners
         node=mesh(iel)%iconV(k)
         mesh(iel)%exx(k)=exx(node)
         mesh(iel)%eyy(k)=eyy(node)
         mesh(iel)%exy(k)=exy(node)
         if (pair=='q1p0') mesh(iel)%q(k)=q(node)
      end do
   end do

   deallocate(q,exx,eyy,exy,counter)

end if

!--------------------------------------

if (ndim==3) then

   allocate(q(NT))   ; q=0.d0
   allocate(exx(NT)) ; exx=0.d0
   allocate(eyy(NT)) ; eyy=0.d0
   allocate(ezz(NT)) ; ezz=0.d0
   allocate(exy(NT)) ; exy=0.d0
   allocate(exz(NT)) ; exz=0.d0
   allocate(eyz(NT)) ; eyz=0.d0
   allocate(counter(NT)) ; counter=0.d0

   do iel=1,nel
      do k=1,ncorners
         call compute_dNdx_dNdy_dNdz(rcorners(k),scorners(k),tcorners(k),dNdx,dNdy,dNdz,jcob)
         node=mesh(iel)%iconV(k)
         exx(node)=exx(node)+sum(dNdx*mesh(iel)%u(1:mV))         
         eyy(node)=eyy(node)+sum(dNdy*mesh(iel)%v(1:mV))         
         ezz(node)=ezz(node)+sum(dNdz*mesh(iel)%w(1:mV))         
         exy(node)=exy(node)+sum(dNdx*mesh(iel)%v(1:mV))*0.5d0 &
                            +sum(dNdy*mesh(iel)%u(1:mV))*0.5d0 
         exz(node)=exz(node)+sum(dNdx*mesh(iel)%w(1:mV))*0.5d0 &
                            +sum(dNdz*mesh(iel)%u(1:mV))*0.5d0 
         eyz(node)=eyz(node)+sum(dNdy*mesh(iel)%w(1:mV))*0.5d0 &
                            +sum(dNdz*mesh(iel)%v(1:mV))*0.5d0 
         if (pair=='q1p0') q(node)=q(node)+mesh(iel)%p(1)
         counter(node)=counter(node)+1
      end do
   end do

   exx=exx/counter
   eyy=eyy/counter
   ezz=ezz/counter
   exy=exy/counter
   exz=exz/counter
   eyz=eyz/counter
   q=q/counter

   do iel=1,nel
      do k=1,ncorners
         node=mesh(iel)%iconV(k)
         mesh(iel)%exx(k)=exx(node)
         mesh(iel)%eyy(k)=eyy(node)
         mesh(iel)%ezz(k)=ezz(node)
         mesh(iel)%exy(k)=exy(node)
         mesh(iel)%exz(k)=exz(node)
         mesh(iel)%eyz(k)=eyz(node)
         if (pair=='q1p0') mesh(iel)%q(k)=q(node)
      end do
   end do

   deallocate(q,exx,eyy,ezz,exy,exz,eyz,counter)

end if


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> interpolate_onto_nodes ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
