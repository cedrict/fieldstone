!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_matrix_COO

use module_parameters, only: NfemVel,NU,NV,NW,mU,mV,mW,iproc,ndim
use module_mesh 
use module_sparse, only : cooK
use module_arrays, only: Unode_belongs_to,Vnode_belongs_to,Wnode_belongs_to
use module_timing

implicit none

integer :: nsees,nz,ip,i,jp,k,iel
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_matrix\_COO}
!@@ This subroutine allocates arrays ia, ja, col, row, and mat of cooK, 
!@@ and builds arrays ia, ja, col, row.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

cooK%NR=NfemVel
cooK%NC=NfemVel

write(*,'(a,i11,a)') shift//'cooK%NR=',cooK%NR
write(*,'(a,i11,a)') shift//'cooK%NC=',cooK%NC

!----------------------------------------------------------
! compute NZ
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NU+NV+NW))
NZ=0

! xx,xy,xz blocks

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
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do


! yx,yy,yz blocks

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
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do


! zx,zy,zz blocks

do ip=1,NW
   alreadyseen=.false.
   do k=1,Wnode_belongs_to(1,ip)
      iel=Wnode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do
end do
         
deallocate(alreadyseen)

call cpu_time(t4) 

cooK%NZ=NZ

!----------------------------------------------------------
! fill arrays ia,ja
!----------------------------------------------------------

!csrK%NZ=(csrK%NZ-csrK%N)/2+csrK%N
!write(*,'(a)')       shift//'CSR matrix format SYMMETRIC  '
write(*,'(a,i11,a,f7.3,a)') shift//'cooK%NZ      =',cooK%NZ,' | ',t4-t3,'s'

allocate(cooK%ia(cooK%NR+1)) ; cooK%ia=0 
allocate(cooK%ja(cooK%NZ))   ; cooK%ja=0 
allocate(cooK%mat(cooK%NZ))  ; cooK%mat=0 
allocate(cooK%snr(15*cooK%NZ))  ; cooK%snr=0 
allocate(cooK%rnr(15*cooK%NZ))  ; cooK%rnr=0 

call cpu_time(t3)
allocate(alreadyseen(NU+NV+NW))
NZ=0
cooK%ia(1)=1
cooK%snr(1)=1
cooK%rnr(1)=1


do ip=1,NU
   nsees=0
   alreadyseen=.false.
   do k=1,Unode_belongs_to(1,ip)
      iel=Unode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do    
   cooK%ia(ip+1)=cooK%ia(ip)+nsees    
end do    

do ip=1,NV
   nsees=0
   alreadyseen=.false.
   do k=1,Vnode_belongs_to(1,ip)
      iel=Vnode_belongs_to(1+k,ip)
      !-------- 
      do i=1,mU
         jp=mesh(iel)%iconU(i)
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip+NU
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip+NU
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            cooK%ja(NZ)=jp
            cooK%snr(NZ)=ip+NU
            cooK%rnr(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do    
   cooK%ia(NU+ip+1)=cooK%ia(NU+ip)+nsees    
end do    

if (ndim==3) stop 'setup_K_matrix_COO: not finished'

deallocate(alreadyseen)    

call cpu_time(t4) ; write(*,'(a,f10.3,a)') shift//'ia,ja time:',t4-t3,'s'

write(*,'(a,i9)' ) shift//'NZ=',NZ
write(*,'(a,2i9)') shift//'cooK%ia',minval(cooK%ia), maxval(cooK%ia)
write(*,'(a,2i9)') shift//'cooK%ja',minval(cooK%ja), maxval(cooK%ja)

!print *,cooK%ia
!print *,'---------'
!print *,cooK%ja
!print *,'---------'
!print *,cooK%snr(1:cooK%NZ)
!print *,'---------'
!print *,cooK%rnr(1:cooK%NZ)
!do i=1,cooK%NZ
!   write(333,*) cooK%snr(i),cooK%rnr(i)
!end do

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_matrix_COO:',elapsed,' s             |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
