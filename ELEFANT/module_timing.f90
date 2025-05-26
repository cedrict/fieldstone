module module_timing
implicit none
integer counti,countf,count_rate
real(8) elapsed
character(len=42), parameter :: shift='                                        |'
character(len=42), parameter :: limit='*****************************************'

real(8) :: t3,t4
real(8) :: time_assemble_K
real(8) :: time_assemble_GT
real(8) :: time_assemble_S
real(8) :: time_assemble_RHS

contains

real(8) :: t1,t2

subroutine tic()
  implicit none
  call cpu_time(t1)
end subroutine tic

subroutine toc()
  implicit none
  call cpu_time(t2)
  ! if (rank==0) print*,"Time Taken -->", real(t2-t1)
  print*,"Time Taken -->", real(t2-t1)
end subroutine toc

end module
