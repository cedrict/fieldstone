!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine inner_solver(rhs,guess,solution)

use module_parameters, only: inner_solver_type,NfemV,nproc,iproc
use module_timing
use module_sparse, only : csrK
use module_MUMPS

implicit none

real(8), intent(inout) :: rhs(NfemV)
real(8), intent(in)    :: guess(NfemV)
real(8), intent(out)   :: solution(NfemV)

integer, dimension(:,:), allocatable :: ha
integer, dimension(:), allocatable :: rnr,snr
integer :: iflag(10), ifail,nn1,nn
integer :: icount1,icount2,countrate,imode(3),verbose,ierr
real(8), dimension(:), allocatable :: pivot
real(8), dimension(:), allocatable :: mat 
real(8) :: pta,ta,tf,ptf,ts,pts,aflag(8)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{inner\_solver}
!@@ This subroutine solves the system $\K\cdot \vec{\cal V} = \vec{f}$. The matrix is 
!@@ implicit passed as argument via the module but the rhs and the guess vector are 
!@@ passed as arguments.
!@@ If MUMPS is used the system is solved via MUMPS (the guess is vector
!@@ is then neglected). Same if y12m solver is used. 
!@@ Otherwise a call is made to the {\tt pcg\_solver\_csr} subroutine.
!==================================================================================================!

select case(inner_solver_type)

!-------------
case('_MUMPS') 

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

!-------------
case('__y12m')

   aflag=0
   iflag=0
   allocate(ha(NfemV,11))
   allocate(pivot(NfemV))
   allocate(mat(15*csrK%NZ)) ; mat=0d0
   allocate(snr(15*csrK%NZ)) 
   allocate(rnr(15*csrK%NZ)) 
   nn=size(snr)
   nn1=size(rnr)

   mat(1:csrK%NZ)=csrK%mat(1:csrK%NZ)
   rnr(1:csrK%NZ)=csrK%rnr(1:csrK%NZ)
   snr(1:csrK%NZ)=csrK%snr(1:csrK%NZ)

   call y12maf(NfemV,csrK%NZ,mat,snr,nn,rnr,nn1,pivot,ha,NfemV,aflag,iflag,rhs,ifail)

   if (ifail/=0) print *,'ifail=',ifail
   if (ifail/=0) stop 'inner_solver: problem with y12m solver'

   solution=rhs

   deallocate(ha)
   deallocate(pivot)
   deallocate(mat)
   deallocate(rnr)
   deallocate(snr)

!-------------
case('__pcg')

   !call pcg_solver_csr(csrK,guess,rhs,Kdiag)

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
