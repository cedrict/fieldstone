module enumerates

implicit none

! test meshes
integer, parameter :: testSquare = 1
integer, parameter :: testRectangle = 2
integer, parameter :: testSquareFocused = 3
integer, parameter :: testWobble = 4
integer, parameter :: testStitch = 5
integer, parameter :: testMultiGrid = 6

! application
integer, parameter :: fivePatches = 11
integer, parameter :: fivePatchesAutostitch = 12
integer, parameter :: thirteenPatches = 13

! edge definitions to make calls to stitching more intuitive
integer, parameter :: edge12 = 1
integer, parameter :: edge23 = 2
integer, parameter :: edge43 = 3
integer, parameter :: edge14 = 4

! use to indicate how to write elements to file
integer, parameter :: clockWise = 1
integer, parameter :: counterClockWise = 2

! element types
integer            :: elementType
integer, parameter :: Q0 = 1
integer, parameter :: Q1 = 2
integer, parameter :: Q2 = 3
integer, parameter :: Q3 = 4
integer, parameter :: Q4 = 5

character(len=2)   :: meshNames(5)

contains

subroutine initializeMeshNames

implicit none

meshNames(Q0) = "Q0"
meshNames(Q1) = "Q1"
meshNames(Q2) = "Q2"
meshNames(Q3) = "Q3"
meshNames(Q4) = "Q4"

end subroutine


end module
