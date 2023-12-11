!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_GT

use module_parameters
use module_mesh 
use module_timing
use module_sparse, only : csrGT

implicit none

integer inode,k,nz,i,ii,nsees

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_GT.f90}
!@@ This subroutine is executed if {\sl use\_penalty} is False.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (.not.use_penalty) then

csrGT%nr=NfemP ! number of rows
csrGT%nc=NfemV ! number of columns

if (spaceP=='__Q0' .or. spaceP=='__P0') then ! is pressure discontinuous
   csrGT%NZ=mV*ndofV*nel
else
   stop 'matrix_setup_GT: not done for q1q1'
end if

write(*,'(a,i8)') shift//'matrix GT%nr=',csrGT%nr
write(*,'(a,i8)') shift//'matrix GT%nc=',csrGT%nc
write(*,'(a,i8)') shift//'matrix GT%NZ=',csrGT%nz

allocate(csrGT%ia(csrGT%nr+1))
allocate(csrGT%ja(csrGT%NZ))  
allocate(csrGT%mat(csrGT%NZ)) 

if (spaceP=='__Q0' .or. spaceP=='__P0') then ! is pressure discontinuous

   nz=0
   csrGT%ia(1)=1
   do iel=1,nel      ! iel indicates the row in the matrix
      nsees=0
      do i=1,mV
         inode=mesh(iel)%iconV(i)
         do k=1,ndofV
            ii=ndofV*(inode-1) + k ! column address in the matrix
            nz=nz+1
            csrGT%ja(nz)=ii
            nsees=nsees+1
         end do
      end do
      csrGT%ia(iel+1)=csrGT%ia(iel)+nsees
   end do

end if

if (debug) then
write(2345,*) limit//'matrix_setup_GT'//limit
write(2345,*) 'csrGT%nz=',csrGT%nz
write(2345,*) 'csrGT%ia (m/M)',minval(csrGT%ia), maxval(csrGT%ia)
write(2345,*) 'csrGT%ja (m/M)',minval(csrGt%ja), maxval(csrGT%ja)
write(2345,*) 'csrGT%ia ',csrGT%ia
end if

end if !use_penalty

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'matrix_setup_GT (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
