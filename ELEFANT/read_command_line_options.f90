!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine read_command_line_options

use module_parameters
use module_timing

implicit none

integer :: option_ID
integer :: argc,numarg
logical :: err_detected
character(len=255) :: arg

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{read_command_line_options}
!@@
!==================================================================================================!

if (iproc==0) then

!==============================================================================!

err_detected=.false.
option_ID=1
numarg = command_argument_count()
if (numarg>0) then
   write(*,'(a,i2)') shift//'number of passed arguments:',numarg
   argc=command_argument_count()
   do while (option_ID <= argc)
      call getarg(option_ID,arg)

      !------------------------------------------
      if (arg=='-nelx') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nelx 
         write(*,'(a,i4)') shift//'read nelx=',nelx
      !------------------------------------------
      elseif (arg=='-nely') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nely 
         write(*,'(a,i4)') shift//'read nely=',nely
      !------------------------------------------
      elseif (arg=='-nelz') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nelz 
         write(*,'(a,i4)') shift//'read nelz=',nelz

      !------------------------------------------
      elseif (arg=='-dparam1') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) dparam1 
         write(*,*) ' read dparam1 as argument: ',dparam1
      !------------------------------------------
      else if (arg=='-dparam2') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) dparam2 
         write(*,*) ' read dparam2 as argument: ',dparam2
      !------------------------------------------
      else if (arg=='-dparam3') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) dparam3 
         write(*,*) ' read dparam3 as argument: ',dparam3




      else
         err_detected=.true.
         exit
      end if

      option_ID=option_ID+1

   end do

   if (err_detected) then
   write(*,'(a)') 'unknown command line option',arg
   stop
   end if

end if





write(*,'(a)') 'read_command_line_options'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
