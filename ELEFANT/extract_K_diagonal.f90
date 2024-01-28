!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine extract_K_diagonal

use module_parameters, only: iproc,NfemVel,K_storage,NU,NV,NW
use module_sparse, only: csrK,csrKxx,csrKyy,csrKzz
use module_arrays, only: Kdiag,K_matrix,Kxx_matrix,Kyy_matrix,Kzz_matrix,&
                         Kxxdiag,Kyydiag,Kzzdiag
use module_timing

implicit none

integer :: i,k

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{extract\_K\_diagonal}
!@@ This subroutine extracts the diagonal of the matrix $\K$ or 
!@@ of the $\K_{xx}$, $\K_{yy}$, $\K_{zz}$ blocks.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a,a)') shift//'K_storage: ',K_storage

Kdiag=0d0

select case(K_storage)

!__________________
case('matrix_FULL')

   do i=1,NfemVel
      Kdiag(i)=K_matrix(i,i)
   end do

   write(*,'(a,2es12.4)') shift//'Kdiag (m,M):',minval(Kdiag),maxval(Kdiag)

!___________________
case('matrix_MUMPS')

   Kdiag=1d0

!_________________
case('matrix_CSR')

   do i=1,NfemVel
      do k=1,csrK%ia(i),csrK%ia(i+1)-1
         if (csrK%ja(k)==i) Kdiag(i)=csrK%mat(k)
      end do
   end do

   write(*,'(a,2es12.4)') shift//'Kdiag (m,M):',minval(Kdiag),maxval(Kdiag)

!_________________
case('blocks_FULL')

   do i=1,NU
      Kxxdiag(i)=Kxx_matrix(i,i)
   end do
   do i=1,NV
      Kyydiag(i)=Kyy_matrix(i,i)
   end do
   do i=1,NW
      Kzzdiag(i)=Kzz_matrix(i,i)
   end do

   write(*,'(a,2es12.4)') shift//'Kxxdiag (m,M):',minval(Kxxdiag),maxval(Kxxdiag)
   write(*,'(a,2es12.4)') shift//'Kyydiag (m,M):',minval(Kyydiag),maxval(Kyydiag)
   write(*,'(a,2es12.4)') shift//'Kzzdiag (m,M):',minval(Kzzdiag),maxval(Kzzdiag)

!___________________
case('block_MUMPS')

   Kxxdiag=1d0
   Kyydiag=1d0
   Kzzdiag=1d0

!_________________
case('blocks_CSR')

   do i=1,NU
      do k=1,csrKxx%ia(i),csrKxx%ia(i+1)-1
         if (csrKxx%ja(k)==i) Kxxdiag(i)=csrKxx%mat(k)
      end do
   end do
   do i=1,NV
      do k=1,csrKyy%ia(i),csrKyy%ia(i+1)-1
         if (csrKyy%ja(k)==i) Kyydiag(i)=csrKyy%mat(k)
      end do
   end do
   do i=1,NW
      do k=1,csrKzz%ia(i),csrKzz%ia(i+1)-1
         if (csrKzz%ja(k)==i) Kzzdiag(i)=csrKzz%mat(k)
      end do
   end do

   write(*,'(a,2es12.4)') shift//'Kxxdiag (m,M):',minval(Kxxdiag),maxval(Kxxdiag)
   write(*,'(a,2es12.4)') shift//'Kyydiag (m,M):',minval(Kyydiag),maxval(Kyydiag)
   write(*,'(a,2es12.4)') shift//'Kzzdiag (m,M):',minval(Kzzdiag),maxval(Kzzdiag)

!_________________
case default

   stop 'extract_K_diagonal: unknown K_storage value'

end select

!print *,Kdiag


!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'extract_K_diagonal:',elapsed,' s             |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
