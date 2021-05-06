!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_energy

use module_parameters 
use module_arrays
use module_sparse, only: csrA 
use module_mesh 
use module_timing

implicit none

integer, parameter :: maxits=1000
integer, parameter :: size_krylov_subspace=100
real(8), parameter :: rtol=1.d-14
real(8), parameter :: atol=1.d50 

integer :: k
real(8) :: sol(NfemT)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{solve\_energy}
!@@ This routine solves the energy equation using a GMRES solver obtained from 
!@@ \url{https://people.sc.fsu.edu/~jburkardt/f_src/mgmres/mgmres.html} 
!==================================================================================================!
! We rely here on the pmgmres\_ilu\_cr subroutine with the following arguments:
! Input, integer ( kind = 4 ) N, the order of the linear system.
! Input, integer ( kind = 4 ) NZ_NUM, the number of nonzero matrix values.
! Input, integer ( kind = 4 ) IA(N+1), JA(NZ_NUM), the row and column indices
! of the matrix values.  The row vector has been compressed.
! Input, real ( kind = 8 ) A(NZ_NUM), the matrix values.
! Input/output, real ( kind = 8 ) X(N); on input, an approximation to
! the solution.  On output, an improved approximation.
! Input, real ( kind = 8 ) RHS(N), the right hand side of the linear system.
! Input, integer ( kind = 4 ) ITR_MAX, the maximum number of (outer) iterations to take.
! Input, integer ( kind = 4 ) MR, the maximum number of (inner) iterations 
! to take.  MR must be less than N.
! Input, real ( kind = 8 ) TOL_ABS, an absolute tolerance applied to the current residual.
! Input, real ( kind = 8 ) TOL_REL, a relative tolerance comparing the
! current residual to the initial residual.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

sol=0d0

write(*,'(a)') '...........................................'
call pmgmres_ilu_cr (csrA%N,csrA%NZ,csrA%ia,csrA%ja,csrA%mat,sol,rhs_b,& 
                     maxits,               &
                     size_krylov_subspace, &
                     atol,                 &
                     rtol)
write(*,'(a)') '...........................................'

write(*,'(a,2es12.4)') '        -> T (m/M)',minval(sol),maxval(sol)
write(1237,'(2es12.4)') minval(sol),maxval(sol)
call flush(1237)

!transfer solution to elements

do iel=1,nel
   do k=1,mT
      mesh(iel)%T(k)=sol(mesh(iel)%iconT(k))
   end do
end do

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'solve_energy (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
