!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine inner_solver(rhs,guess,solution)

use module_parameters, only: inner_solver_type,NfemVel,nproc,iproc
use module_timing
use module_arrays, only : K_matrix,Kdiag
use module_sparse, only : csrK,cooK
use module_MUMPS

implicit none

real(8), intent(inout) :: rhs(NfemVel)
real(8), intent(in)    :: guess(NfemVel)
real(8), intent(out)   :: solution(NfemVel)

integer, dimension(:,:), allocatable :: ha
integer, dimension(:), allocatable :: rnr,snr
integer :: iflag(10), ifail,nn1,nn,job
integer :: icount1,icount2,imode(3),verbose
integer, dimension(:), allocatable :: ipvt
real(8), dimension(:), allocatable :: pivot
real(8), dimension(:), allocatable :: mat 
real(8) :: pta,ta,tf,ptf,ts,pts,aflag(8)
real(8), dimension(:),   allocatable :: work
real(8) rcond

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{inner\_solver}
!@@ This subroutine solves the system $\K\cdot \vec{\cal V} = \vec{f}$. The matrix is 
!@@ implicit passed as argument via the module but the rhs and the guess vector are 
!@@ passed as arguments.
!@@ Which solver is used is determined by the value of the {\tt inner\_solver\_type} parameter.
!@@ There are four options:
!@@ \begin{itemize}
!@@ \item 'MUMPS' (specific storage)
!@@ \item 'LINPACK' (full matrix storage)
!@@ \item 'Y12M' (CSR storage): Y12M is a package of Fortran subroutines. It was developed at
!@@ the Regional Computing Centre at the University of Copenhagen (RECKU) by \textcite{zlws81}. One
!@@ can obtain the code from Netlib and the documentation is at {\tt https://www.netlib.org/y12m/doc}.
!@@ About {\tt ifail}: Error diagnostic parameter. The content of parameter IFAIL is modified  
!@@ by subroutine Y12MA.  On exit IFAIL = 0 if the subroutine has not detected any error.  
!@@ Positive  values  of IFAIL on  exit  show  that some error has been
!@@ detected by the subroutine. 
!@@ \item 'PCG' (symmetric CSR storage ?)
!@@ \end{itemize}
!@@ Note that there is directsolver.f90 too that is for now not used.
!@@ I should also explore MA28 by Duff ! 
!==================================================================================================!

write(*,'(a,a)') shift//'inner_solver_type=',inner_solver_type

select case(inner_solver_type)

!-----------------
case('LINPACK') 

   job=0
   allocate(work(NfemVel))
   allocate(ipvt(NfemVel))
   call DGECO (K_matrix, NfemVel, NfemVel, ipvt, rcond, work)
   call DGESL (K_matrix, NfemVel, NfemVel, ipvt, rhs, job)
   deallocate(ipvt)
   deallocate(work)

   solution=rhs

!-----------------
case('MUMPS') 

   verbose=0

   !-----------------------
   ! Specify element entry

   idV%ICNTL(5) = 1 ! elemental format
 
   !------------------------------
   ! level of global info printing
   ! ICNTL(4):
   !   1 : Only error messages printed. 
   !   2 : Errors, warnings, and main statistics printed. 
   !   3 : Errors and warnings and terse diagnostics (only first ten entries of arrays) printed.
   !   4: Errors and warnings and information on input and output parameters printed.
 
   idV%ICNTL(3) = -1020
   idV%ICNTL(4) = 2 

   !-------------------------
   ! matrix scaling strategy

   idV%ICNTL(6)=1

   !------------------
   ! ordering choice 
   ! 0 AMD used 
   ! 2 AMF used
   ! 3 SCOTCH
   ! 4 PORD
   ! 5 METIS
   ! 6 QAMD
   ! 7 automatic choice

   idV%ICNTL(7) = 5 ! use METIS

   ! additional space

   !idV%ICNTL(14)=1

   !----------------
   ! analysis phase 
   !----------------

   if (imode(1)==1) then

      CALL SYSTEM_CLOCK(iCOUNT1,COUNT_RATE) 

      !-------------------------------
      idV%JOB = 1 ; CALL DMUMPS(idV)
      !-------------------------------

      CALL SYSTEM_CLOCK(iCOUNT2,COUNT_RATE) 

      if (idV%INFOG(1)/=0) print *,idV%INFOG(1)
      if (idV%INFOG(1)/=0) call errordisplay1

      ta=(icount2-icount1)/dble(count_rate)

      if (verbose==1) then
      if (nproc==1) then
         write(*,'(a,f8.4,a)') shift//'analysis time ',ta,'s             ||'
      else
         !call mpi_reduce(ta,pta,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
         pta=pta/nproc
         if (iproc==0) write(*,'(a,f8.4,a)') shift//'avrg. analysis time ',pta,'s       ||'
      end if
      end if

   end if

   !---------------------
   ! factorisation phase 
   !---------------------

   if (imode(2)==1) then

      CALL SYSTEM_CLOCK(iCOUNT1,COUNT_RATE) 

      !-------------------------------
      idV%JOB = 2 ; CALL DMUMPS(idV)
      !-------------------------------

      CALL SYSTEM_CLOCK(iCOUNT2,COUNT_RATE) 

      if (idV%INFOG(1)/=0) print *,idV%INFOG(1)
      if (idV%INFOG(1)/=0) call errordisplay2 

      tf=(icount2-icount1)/dble(count_rate)

      if (verbose==1) then
      if (nproc==1) then
         write(*,'(a,f8.4,a)') shift//'factor.  time ',tf,'s             ||'
      else
         !call mpi_reduce(tf,ptf,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
         ptf=ptf/nproc
         if (iproc==0) write(*,'(a,f8.4,a)') shift//'avrg. factor.  time ',ptf,'s       ||'
      end if
      end if

   end if

   !-------------
   ! solve phase
   !-------------

   if (imode(3)==1) then

      CALL SYSTEM_CLOCK(iCOUNT1,COUNT_RATE) 

      !-------------------------------
      idV%JOB = 3 ; CALL DMUMPS(idV)
      !-------------------------------

      CALL SYSTEM_CLOCK(iCOUNT2,COUNT_RATE) 
      if (idV%INFOG(1)/=0) call errordisplay3

      ts=(icount2-icount1)/dble(count_rate)

      if (verbose==1) then
      if (nproc==1) then 
         write(*,'(a,f8.4,a)') shift//'solution time ',ts,'s             ||'
         write(*,'(a,i6,a)') shift//'MUMPS: est. RAM for fact.:',idV%info(15),'Mb  ||'
         write(*,'(a,i6,a)') shift//'MUMPS: eff. RAM for fact.:',idV%info(22),'Mb  ||'
         write(*,'(a,i6,a)') shift//'MUMPS: eff. RAM for sol. :',idV%info(26),'Mb  ||'
      else 
         !call mpi_reduce(ts,pts,1,mpi_double_precision,mpi_sum,0,mpi_comm_world,ierr)
         pts=pts/nproc
         if (iproc==0) then 
            write(*,'(a,f8.4,a)') shift//'avrg. solution time ',pts,'s       ||'
            write(*,'(a,i7,a)') shift//'estimated RAM for fact:',idV%infog(17),'Mb (sum over all MPI processes)'
            write(*,'(a,i7,a)') shift//'RAM allocated for fact:',idV%infog(19),'Mb (sum over all MPI processes)'
            write(*,'(a,i7,a)') shift//'effective RAM for fact:',idV%infog(22),'Mb (sum over all MPI processes)'
            write(*,'(a,i7,a)') shift//'effective RAM for sol :',idV%infog(31),'Mb (sum over all MPI processes)'
         end if
      end if
      end if

      solution=idV%rhs

   end if

!-----------------
case('Y12M')

   write(*,'(a)') shift//'Y12M solver'
   write(*,'(a,i5)') shift//'NfemVel=',NfemVel
   write(*,'(a,i5)') shift//'cooK%NZ=',cooK%NZ

   aflag=0
   iflag=0
   allocate(ha(NfemVel,11))
   allocate(pivot(NfemVel))
   nn=15*cooK%NZ
   nn1=15*cooK%NZ

   call y12maf(NfemVel,cooK%NZ,cooK%mat,cooK%snr,nn,cooK%rnr,nn1,pivot,ha,NfemVel,aflag,iflag,rhs,ifail)

   if (ifail/=0) print *,'ifail=',ifail
   if (ifail/=0) stop 'inner_solver: problem with Y12M solver'

   solution=rhs

   deallocate(ha)
   deallocate(pivot)
   deallocate(mat)
   deallocate(rnr)
   deallocate(snr)

!-----------------
case('PCG')

   call pcg_solver_csr(csrK,guess,rhs,Kdiag)

!-------------
case default

   stop 'inner_solver: unknown inner_solver_type'

end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine errordisplay1
print *,'******************'
print *,'*  MUMPS ERROR   *'
print *,'*  ANALYSIS      *'
print *,'******************'
stop
end

subroutine errordisplay2
print *,'******************'
print *,'*  MUMPS ERROR   *'
print *,'*  FACTORISATION *'
print *,'******************'
stop
end

subroutine errordisplay3
print *,'******************'
print *,'*  MUMPS ERROR   *'
print *,'*  SOLVE         *'
print *,'******************'
stop
end
