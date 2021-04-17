!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{spmv\_kernels}
!@@ This file contains the Sparse Matrix-Vector multiplication kernels (see Section~\ref{ss:spmv}).
!==================================================================================================!

subroutine spmv (nr,nc,nz,x,y,mat,ja,ia)

implicit none

integer, intent(in)  :: nr,nc,nz
real(8), intent(in)  :: x(nc), mat(nz)
real(8), intent(out) :: y(nr)
integer, intent(in)  :: ja(nz),ia(nr+1)
real(8) t
integer i, k


do i = 1,nr
   t = 0.0d0
   do k=ia(i), ia(i+1)-1
      t = t + mat(k)*x(ja(k))
   end do
   y(i) = t
end do

end subroutine


!==================================================================================================!
!==================================================================================================!
! this version of the subroutine does not zero y 
! so that it adds the result to the y vector 
! passed as argument

subroutine spmv_add (nr,nc,nz,x,y,mat,ja,ia)

implicit none

integer, intent(in)  :: nr,nc,nz
real(8), intent(in)  :: x(nc), mat(nz)
real(8), intent(inout) :: y(nr)
integer, intent(in)  :: ja(nz),ia(nr+1)
integer i, k

do i = 1,nr
   do k=ia(i), ia(i+1)-1
      y(i) = y(i) + mat(k)*x(ja(k))
   end do
end do

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine spmvT (nr,nc,nz,x,y,mat,ja,ia)

implicit none

integer nr,nc,nz, ja(nz), ia(nr+1)
real(8)  x(nr), y(nc), mat(nz)
integer i, k

y=0.d0

do i = 1,nr
   do k=ia(i), ia(i+1)-1
      y(ja(k))=y(ja(k))+mat(k)*x(i)
   end do
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
! this version of the subroutine does not zero y 
! so that it adds the result to the y vector 
! passed as argument

subroutine spmvT_add (nr,nc,nz,x,y,mat,ja,ia)

implicit none

integer nr,nc,nz, ja(nz), ia(nr+1)
real(8)  x(nr), y(nc), mat(nz)
integer i, k

do i = 1,nr
   do k=ia(i), ia(i+1)-1
      y(ja(k))=y(ja(k))+mat(k)*x(i)
   end do
end do

end subroutine


!==================================================================================================!
!==================================================================================================!

subroutine spmv_symm (nr,nz, x, y, mat,ja,ia)
implicit none
integer nz,nr, ja(nz), ia(nr+1)
real(8)  x(nr), y(nr), mat(nz)
integer i, k,jak

y(1:nr)=0

do i=1,nr
   do k=ia(i), ia(i+1)-1
      jak=ja(k)
      y(i)=y(i)+mat(k)*x(jak)
      if (i/=jak) then
         y(jak)=y(jak)+mat(k)*x(i)
      end if
   end do
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
! this version of the subroutine does not zero y 
! so that it adds the result to the y vector 
! passed as argument

subroutine spmv_symm_add (nr,nz, x, y, mat,ja,ia)
implicit none
integer nz,nr, ja(nz), ia(nr+1)
real(8)  x(nr), y(nr), mat(nz)
integer i, k,jak

do i=1,nr
   do k=ia(i), ia(i+1)-1
      jak=ja(k)
      y(i)=y(i)+mat(k)*x(jak)
      if (i/=jak) then
         y(jak)=y(jak)+mat(k)*x(i)
      end if
   end do
end do

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine spmv_symm_omp (nr,nz, x, y, mat,ja,ia)
implicit none
integer nz,nr, ja(nz), ia(nr+1)
real(8)  x(nr), y(nr), mat(nz),matk
integer i, k,jak
integer, parameter :: chunksize=4

y(1:nr)=0
!$omp PARALLEL 
!$omp DO PRIVATE(i, k, jak,matk)
do i=1,nr
   do k=ia(i), ia(i+1)-1
      jak=ja(k)
      matk=mat(k)
      y(i)=y(i)+matk*x(jak)
      if (i/=jak) then
         y(jak)=y(jak)+matk*x(i)
      end if
   end do
end do
!$omp end DO
!$omp END PARALLEL 

end subroutine

!==================================================================================================!
!==================================================================================================!
! this subroutine multiplies a matrix by a vector bu compensates for the 
! facte that only half of the matrix is stored (it is symmetric)
! not tested !!!!


subroutine spmv2 (nr,nc,nz,x,y,mat,ja,ia)

implicit none

integer, intent(in)  :: nr,nc,nz
real(8), intent(in)  :: x(nc), mat(nz)
real(8), intent(out) :: y(nr)
integer, intent(in)  :: ja(nz),ia(nr+1)
integer i, k

y(1:nr)=0

do i = 1,nr

   ! normal, but not with diagona terms
   do k=ia(i), ia(i+1)-1
      !if (i/=ja(k)) then
      y(i) = y(i) + mat(k)*x(ja(k))
      !end if
   end do

   ! transpose
   do k=ia(i), ia(i+1)-1
      if (i/=ja(k)) then
      y(ja(k))=y(ja(k))+mat(k)*x(i)
      end if
   end do


end do

end subroutine

