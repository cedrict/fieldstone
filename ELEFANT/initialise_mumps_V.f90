!=================================================================================================!
!=================================================================================================!
!                                                                                                 !
! ELEFANT                                                                        C. Thieulot      !
!                                                                                                 !
!=================================================================================================!
!=================================================================================================!

subroutine initialise_mumps_V

use module_parameters

implicit none

!#ifdef UseMUMPS
!include 'mpif.h'
!#endif

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{initialise\_mumps\_V}
!@@
!==================================================================================================!

!#ifdef UseMUMPS

!idV%COMM = MPI_COMM_WORLD     ! Define a communicator for the package 

!idV%SYM = 1                   ! Ask for symmetric matrix storage 

!if (nproc>1) then
!   idV%PAR=0                  ! Host not working 
!else
!   idV%par=1                  ! Host working 
!end if

!idV%JOB = -1                  ! Initialize an instance of the package 

!call DMUMPS(idV)              ! MUMPS initialisation

!IF (idV%INFOG(1).LT.0) THEN
!WRITE(6,'(A,A,I6,A,I9)') " ERROR RETURN: ",&
!            "  idV%INFOG(1)= ", idV%INFOG(1),&
!            "  idV%INFOG(2)= ", idV%INFOG(2)
!END IF

!#endif

!==============================================================================!

!if (iproc==0) write(*,'(a,f6.2,a)') 'initialise_mumps_V (',elapsed,' s)'

end subroutine

!==================================================================================================!
!==================================================================================================!


