subroutine solve_linpack(A,b,np)
!This function uses LINPACK to solve the equation A*x=b
integer, intent(in) :: np
real(8), dimension(np,np):: A !A matrix
real(8), dimension(np) :: b   !b vector
real(8), dimension(:),allocatable :: work  ! work array for LAPACK
integer, dimension(:),allocatable :: ipvt   ! pivot indices
integer info,job
real(8) rcond
external dgesl
external dgefa

info=0
rcond=0
job=0
allocate(work(Np))
allocate(ipvt(Np))
call DGECO(A,np,np,ipvt,rcond,work)
call DGESL (A, np, np, ipvt, B, job)

deallocate(ipvt)
deallocate(work)

end subroutine
