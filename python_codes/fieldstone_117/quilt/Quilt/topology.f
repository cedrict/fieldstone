module topology

integer :: elementType  ! q1, q2, etc...

contains

subroutine setMeshOrientation(orientation)

use meshData

implicit none

integer :: orientation
integer :: iElement

do iElement = 1, nTotalElements
   call setElementOrientation(Q1connectivity(iElement,:), orientation)
enddo

end subroutine


!-----------------------------------------------------------------------

subroutine invertIntegerArray(n, A)
! reverse the order of array elements

implicit none

integer :: n
integer :: A(n), temp(n)

integer :: i

temp = A

do i=1,n
    A(i) = temp(n-i+1)
enddo

end subroutine


!-----------------------------------------------------------------------

subroutine setElementOrientation(conn, direction)

! ensure that an element is either in clockwise or in 
! counter clockwise direction as desired.

use enumerates
use meshData,   only: Q1Points
use pointMod,   only: point

implicit none

integer :: conn(4), direction
integer :: tmp

type(point) :: p1, p2, p3

p1 = Q1Points(conn(1))
p2 = Q1Points(conn(2))
p3 = Q1Points(conn(3))

if (getElementOrientation(p1, p2, p3) .ne. direction) then
    ! element no in desired direction. 
    ! Switch points 1 and 3 to adjust.
   tmp = conn(3)
   conn(3) = conn(1)
   conn(1) = tmp
endif


end subroutine

!-----------------------------------------------------------------------

integer function getElementOrientation(p1, p2, p3)

use enumerates
use pointMod, only: point

implicit none

type(point) :: p1, p2, p3
type(point) :: v1, v2

v1%x = p3%x - p1%x
v1%y = p3%y - p1%y

v2%x = p3%x - p2%x
v2%y = p3%y - p2%y

! test direction using cross product
if (v1%x * v2%y - v1%y * v2%x .lt. 0) then
    getElementOrientation = clockWise
else
    getElementOrientation = counterClockWise
endif

end function

end module
