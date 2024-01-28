!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_blocks_CSR

use module_parameters, only: iproc,NU,NV,NW,mU,mV,mW,ndim,spaceVelocity
use module_mesh 
use module_arrays, only: Unode_belongs_to,Vnode_belongs_to
use module_sparse, only: csrKxx,csrKxy,csrKxz,csrKyx,csrKyy,csrKyz,csrKzx,csrKzy,csrKzz
use module_timing

implicit none

integer ip,jp,iel,NZ,i,k,nsees
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_blocks\_CSR}
!@@ probably some improvement to be found by only recomputing the ia,ja of all blocks only 
!@@ for a few specific velocity spaces. otherwise do it once and copy paste!
!@@ also the values in ja are not ordered from small to large.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!
!attribute sizes to all blocks

csrKxx%nr=NU ; csrKxy%nr=NU ; csrKxy%nr=NU
csrKxx%nc=NU ; csrKxy%nc=NV ; csrKxz%nc=NW

csrKyx%nr=NV ; csrKyy%nr=NV ; csrKyy%nr=NV
csrKyx%nc=NU ; csrKyy%nc=NV ; csrKyz%nc=NW

csrKzx%nr=NW ; csrKzy%nr=NW ; csrKzz%nr=NW
csrKzx%nc=NU ; csrKzy%nc=NV ; csrKzz%nc=NW

write(*,'(a,i6,i6)') shift//'Kxx size:',csrKxx%NR,csrKxx%NC
write(*,'(a,i6,i6)') shift//'Kxy size:',csrKxy%NR,csrKxy%NC
if (ndim==3) &
write(*,'(a,i6,i6)') shift//'Kxz size:',csrKxz%NR,csrKxz%NC

write(*,'(a,i6,i6)') shift//'Kyx size:',csrKyx%NR,csrKyx%NC
write(*,'(a,i6,i6)') shift//'Kyy size:',csrKyy%NR,csrKyy%NC
if (ndim==3) &
write(*,'(a,i6,i6)') shift//'Kyz size:',csrKyz%NR,csrKyz%NC

if (ndim==3) &
write(*,'(a,i6,i6)') shift//'Kzx size:',csrKzx%NR,csrKzx%NC
if (ndim==3) &
write(*,'(a,i6,i6)') shift//'Kzy size:',csrKzy%NR,csrKzy%NC
if (ndim==3) &
write(*,'(a,i6,i6)') shift//'Kzz size:',csrKzz%NR,csrKzz%NC

!----------------------------------------------------------
! Kxx
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NU))
NZ=0

do ip=1,NU
   alreadyseen=.false.
   do k=1,Unode_belongs_to(1,ip)
      iel=Unode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do
call cpu_time(t4)

write(*,'(a,i6,a,f7.3,a)') shift//'Kxx: NZ=',NZ,' | ',t4-t3,'s'

csrKxx%NZ=NZ

deallocate(alreadyseen)

!----------------------------------------------------------
! Kxy
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NV))
NZ=0

do ip=1,NU
   alreadyseen=.false.
   do k=1,Unode_belongs_to(1,ip)
      iel=Unode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do
call cpu_time(t4)

write(*,'(a,i6,a,f7.3,a)') shift//'Kxy: NZ=',NZ,' | ',t4-t3,'s'

csrKxy%NZ=NZ

deallocate(alreadyseen)

!----------------------------------------------------------
! Kyx
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NU))
NZ=0
do ip=1,NV
   alreadyseen=.false.
   do k=1,Vnode_belongs_to(1,ip)
      iel=Vnode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do
call cpu_time(t4) 

write(*,'(a,i6,a,f7.3,a)') shift//'Kyx: NZ=',NZ,' | ',t4-t3,'s'

csrKyx%NZ=NZ

deallocate(alreadyseen)

!---------------------------------------------------------- 
! Kyy
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NV))
NZ=0
do ip=1,NV
   alreadyseen=.false.
   do k=1,Vnode_belongs_to(1,ip)
      iel=Vnode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do
call cpu_time(t4) 

write(*,'(a,i6,a,f7.3,a)') shift//'Kyy: NZ=',NZ,' | ',t4-t3,'s'

csrKyy%NZ=NZ

deallocate(alreadyseen)

!----------------------------------------------------------
! allocate ia,ja,mat arrays
!----------------------------------------------------------

allocate(csrKxx%ia(csrKxx%NR+1)) ; csrKxx%ia=0 
allocate(csrKxx%ja(csrKxx%NZ))   ; csrKxx%ja=0 
allocate(csrKxx%mat(csrKxx%NZ))  ; csrKxx%mat=0d0 

allocate(csrKyy%ia(csrKyy%NR+1)) ; csrKyy%ia=0 
allocate(csrKyy%ja(csrKyy%NZ))   ; csrKyy%ja=0 
allocate(csrKyy%mat(csrKyy%NZ))  ; csrKyy%mat=0d0 

allocate(csrKxy%ia(csrKxy%NR+1)) ; csrKxy%ia=0 
allocate(csrKxy%ja(csrKxy%NZ))   ; csrKxy%ja=0 
allocate(csrKxy%mat(csrKxy%NZ))  ; csrKxy%mat=0d0 

allocate(csrKyx%ia(csrKyx%NR+1)) ; csrKyx%ia=0 
allocate(csrKyx%ja(csrKyx%NZ))   ; csrKyx%ja=0 
allocate(csrKyx%mat(csrKyx%NZ))  ; csrKyx%mat=0d0 

if (ndim==3) then

allocate(csrKxz%ia(csrKxz%NR+1)) ; csrKxz%ia=0 
allocate(csrKxz%ja(csrKxz%NZ))   ; csrKxz%ja=0 
allocate(csrKxz%mat(csrKxz%NZ))  ; csrKxz%mat=0d0 

allocate(csrKyz%ia(csrKyz%NR+1)) ; csrKyz%ia=0 
allocate(csrKyz%ja(csrKyz%NZ))   ; csrKyz%ja=0 
allocate(csrKyz%mat(csrKyz%NZ))  ; csrKyz%mat=0d0 

allocate(csrKzz%ia(csrKzz%NR+1)) ; csrKzz%ia=0 
allocate(csrKzz%ja(csrKzz%NZ))   ; csrKzz%ja=0 
allocate(csrKzz%mat(csrKzz%NZ))  ; csrKzz%mat=0d0 

allocate(csrKzx%ia(csrKzx%NR+1)) ; csrKzx%ia=0 
allocate(csrKzx%ja(csrKzx%NZ))   ; csrKzx%ja=0 
allocate(csrKzx%mat(csrKzx%NZ))  ; csrKzx%mat=0d0 

allocate(csrKzy%ia(csrKzy%NR+1)) ; csrKzy%ia=0 
allocate(csrKzy%ja(csrKzy%NZ))   ; csrKzy%ja=0 
allocate(csrKzy%mat(csrKzy%NZ))  ; csrKzy%mat=0d0 

end if

!----------------------------------------------------------
! fill ia,ja arrays
!----------------------------------------------------------
         
call cpu_time(t3)
allocate(alreadyseen(NU))
NZ=0
csrKxx%ia(1)=1
do ip=1,NU
   nsees=0
   alreadyseen=.false.
   do k=1,Unode_belongs_to(1,ip)
      iel=Unode_belongs_to(1+k,ip)
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            nz=nz+1
            csrKxx%ja(nz)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do    
   csrKxx%ia(ip+1)=csrKxx%ia(ip)+nsees    
end do    
deallocate(alreadyseen)    
call cpu_time(t4) ; write(*,'(f10.3,a)') t4-t3,'s'

!print *,csrKxx%ia
!print *,csrKxx%ja

write(*,'(a,2i9)') shift//'csrKxx%ia',minval(csrKxx%ia), maxval(csrKxx%ia)
write(*,'(a,2i9)') shift//'csrKxx%ja',minval(csrKxx%ja), maxval(csrKxx%ja)

select case(spaceVelocity)

!-----------
case('_Q1F')

   stop 'setup_K_blocks_CSR: pb'

!-------------------------------------------------------------
case('__Q1','__Q2','__Q3','__P1','_P1+','__P2','__P2+','__P3')

   csrKxy%ia=csrKxx%ia
   csrKxy%ja=csrKxx%ja

   csrKyx%ia=csrKxx%ia
   csrKyx%ja=csrKxx%ja

   csrKyy%ia=csrKxx%ia
   csrKyy%ja=csrKxx%ja

   csrKzz%ia=csrKxx%ia
   csrKzz%ja=csrKxx%ja

   csrKxz%ia=csrKxx%ia
   csrKxz%ja=csrKxx%ja

   csrKzx%ia=csrKxx%ia
   csrKzx%ja=csrKxx%ja

   csrKyz%ia=csrKxx%ia
   csrKyz%ja=csrKxx%ja

   csrKzy%ia=csrKxx%ia
   csrKzy%ja=csrKxx%ja

case default

   stop 'setup_K_blocks_CSR: unknown spaceVelocity'

end select


stop 'the end'

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_blocks_CSR:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
