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
end module
