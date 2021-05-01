!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine read_command_line_options

use global_parameters
use structures, only: shift
!use constants

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

      if (arg=='-nelx') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nelx 
         write(*,'(a,i4)') shift//'read nelx=',nelx
      elseif (arg=='-nely') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nely 
         write(*,'(a,i4)') shift//'read nely=',nely
      elseif (arg=='-nelz') then
         option_ID=option_ID+1
         call getarg(option_ID,arg)
         read(arg,*) nelz 
         write(*,'(a,i4)') shift//'read nelz=',nelz
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
