subroutine JacobiSolver (A,b,N,sol,tol,maxiter)

implicit none

integer, intent(in) :: N            ! size of the system
real(8), intent(inout) :: A(N,N)    ! matrix 
real(8), intent(inout) :: b(N)      ! right hand size
real(8), intent(inout) :: sol(N)    ! initial guess
real(8), intent(in) :: tol          ! tolerance
integer, intent(in) :: maxiter      ! max nb of iterations 

!-------------------------------------------------------
! This subroutine solves the system A.x=b (where A is a
! square full matrix) by means of the Jacobi method.
! The matrix is first decomposed as A= L + D + U and 
! the iterations are as follows:
! x^{k+1} = D^{-1} ( b - (L+U) x^k )
! The solution of the system gets returned in the rhs b
!-------------------------------------------------------

integer i
real(8), dimension(:), allocatable :: diag
real(8), dimension(:), allocatable :: solmem
real(8) resid

!----------------------

allocate(diag(N))
allocate(solmem(N))

!----------------------
!build diagonal vector
! remove diagonal from A
!----------------------

do i=1,N
   diag(i)=A(i,i)
   A(i,i)=0.d0
end do

print *,minval(abs(diag))
!------------------------
! iterate
!------------------------

solmem=sol

do i=1,maxiter

   sol=(b-matmul(A,solmem))/diag 

   resid=maxval(abs(solmem-sol))!/maxval(abs(sol))

   write(777,*) i,resid,minval(sol),maxval(sol)

   !if (resid<tol) then
   !   print *,'JacobiSolver conv.: iters=',i,' diff= ',resid
   !   exit
   !end if

   solmem=sol

end do

b=sol

!------------------------

deallocate(solmem)
deallocate(diag)

end subroutine



