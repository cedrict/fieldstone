!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_GT

use global_parameters
use structures
use timing

implicit none

integer inode,k,nz,i,ii,nsees

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_GT.f90}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

csrGT%nr=NfemP ! number of rows
csrGT%nc=NfemV ! number of columns

if (pair=='q1p0') then ! is pressure discontinuous
   csrGT%NZ=mV*ndofV*nel
else

   stop 'matrix_setup_GT: not done for q1q1'

end if

write(*,'(a,i8)') '        matrix GT%NZ=',csrGT%nz

allocate(csrGT%ia(csrGT%nr+1))
allocate(csrGT%ja(csrGT%NZ))  
allocate(csrGT%mat(csrGT%NZ)) 

if (pair=='q1p0') then ! is pressure discontinuous

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

   if (debug) then
   write(*,*) '          nz=',nz
   write(*,*) '          csrGT%ia (m/M)',minval(csrGT%ia), maxval(csrGT%ia)
   write(*,*) '          csrGT%ja (m/M)',minval(csrGt%ja), maxval(csrGT%ja)
   end if

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> matrix_setup_GT                  ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
