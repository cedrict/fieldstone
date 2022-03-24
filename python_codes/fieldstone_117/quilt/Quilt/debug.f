module debug

contains

subroutine printSection(section)

use circleMod
implicit none

type(circleSection) :: section

write(*,*) "------ section ----------------"
write(*,*) "from: ", section%from%x, ",", section%from%y
write(*,*) "to:   ", section%to%x, ",", section%to%y
write(*,*) "around center", section%center%x, ",", section%center%y
write(*,*) "radius", section%radius
write(*,*) "curvature: ", section%curvature
write(*,*) "-------------------------------"

end subroutine

!-----------------------------------------------------------------------

subroutine showEdges(patchID)
! very useful for debugging, 
! as here the actual stitching is shown, 
! including its many mistakes

use stitching

implicit none

integer :: patchID
integer :: i, ni

ni = size(edges(patchID)%pointIDs,1)

do i = 1, ni
    write(*,*) edges(patchID)%pointIDs(i, :)
enddo

end subroutine

!-----------------------------------------------------------------------


subroutine printConnectivity()

use meshData, only: Q1connectivity, countedElements

implicit none

integer :: iElement

do iElement = 1, countedElements
    write(*,*) "conn", iElement, Q1connectivity(iElement,:)
enddo

end subroutine

end module
