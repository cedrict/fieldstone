!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_matrix_CSR

use module_parameters, only: NfemVel,NU,NV,NW,mU,mV,mW,iproc
use module_mesh 
use module_sparse, only : csrK
use module_arrays, only: Unode_belongs_to,Vnode_belongs_to,Wnode_belongs_to
use module_timing

implicit none

integer :: nsees,nz,ip,i,jp,k,iel
logical, dimension(:), allocatable :: alreadyseen

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_matrix\_CSR}
!@@ This subroutine allocates arrays ia, ja, and mat of csrK, 
!@@ and builds arrays ia and ja.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

csrK%N=NfemVel

write(*,'(a,i11,a)') shift//'csrK%N       =',csrK%N,' '

!----------------------------------------------------------
! compute NZ
!----------------------------------------------------------

call cpu_time(t3)
allocate(alreadyseen(NU+NV+NW))
NZ=0

!xx,xy,xz
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


!yx,yy,yz
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


!zx,zy,zz
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

csrK%NZ=NZ

!----------------------------------------------------------
! fill arrays ia,ja
!----------------------------------------------------------

!csrK%NZ=(csrK%NZ-csrK%N)/2+csrK%N
!write(*,'(a)')       shift//'CSR matrix format SYMMETRIC  '
write(*,'(a,i11,a,f7.3,a)') shift//'csrK%NZ      =',csrK%NZ,' | ',t4-t3,'s'

allocate(csrK%ia(csrK%N+1)) ; csrK%ia=0 
allocate(csrK%ja(csrK%NZ))  ; csrK%ja=0 
allocate(csrK%mat(csrK%NZ)) ; csrK%mat=0 

call cpu_time(t3)
allocate(alreadyseen(NU+NV+NW))
NZ=0
csrK%ia(1)=1

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
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do    
   csrK%ia(ip+1)=csrK%ia(ip)+nsees    
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
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mV
         jp=mesh(iel)%iconV(i)+NU
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
      !-------- 
      do i=1,mW
         jp=mesh(iel)%iconW(i)+NU+NV
         if (.not.alreadyseen(jp)) then
            NZ=NZ+1
            csrK%ja(NZ)=jp
            nsees=nsees+1
            alreadyseen(jp)=.true.
         end if
      end do
   end do    
   csrK%ia(NU+ip+1)=csrK%ia(NU+ip)+nsees    
end do    

deallocate(alreadyseen)    

call cpu_time(t4) ; write(*,'(a,f10.3,a)') shift//'ia,ja time:',t4-t3,'s'

write(*,'(a,i9)' ) shift//'NZ=',NZ
write(*,'(a,2i9)') shift//'csrK%ia',minval(csrK%ia), maxval(csrK%ia)
write(*,'(a,2i9)') shift//'csrK%ja',minval(csrK%ja), maxval(csrK%ja)

print *,csrK%ia

print *,csrK%ja

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_matrix_CSR:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
