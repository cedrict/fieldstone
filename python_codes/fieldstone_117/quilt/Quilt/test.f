subroutine runTests()

implicit none

call testCrossCircleSections()
call testCircleCenterFromRadius()
call testMinorDeterminant()
call testElementOrientation()

end subroutine

!----------------------------------------------

subroutine testCrossCircleSections()

! check whether we can correctly compute the
! points where two circles cross.

use pointMod,  only: point
use circleMod, only: circleSection

implicit none

type(circleSection) :: sectionA, sectionB
type(point)         :: crossPoint1, crossPoint2
type(point)         :: referencePointA, referencePointB

logical             :: OK
logical, external   :: pointsMatch

sectionA%center%x = 2d0
sectionA%center%y = 0d0
sectionB%center%x = 14d0
sectionB%center%y = 0d0

sectionA%radius = 5d0
sectionB%radius = 13d0

sectionA%curvature = 1d0
sectionB%curvature = 1d0


call whereDoTwoSectionsCross(sectionA, sectionB, &
                            crossPoint1, crossPoint2)

referencePointA%x =  2.0
referencePointA%y =  5.0
referencePointB%x =  2.0
referencePointB%y = -5.0

if ((pointsMatch(crossPoint1, referencePointA) .and. &
     pointsMatch(crossPoint2, referencePointB)) .or. &
    (pointsMatch(crossPoint1, referencePointB) .and. &
     pointsMatch(crossPoint2, referencePointA))) then
   OK = .true.
else
    OK = .false.
endif

if (OK) then
   write(*,*) "Testing intersection of circles:  OK"
else
    write(*,*) "Testing intersection of circles:  FAIL"
endif

end subroutine

!---------------------------------------------------------------

subroutine testCircleCenterFromRadius()

use circleMod, only: circleSection
use pointMod,  only: point

implicit none

type(circleSection) :: section
type(point)         :: referencePoint

logical             :: OK
logical, external   :: doublesMatch
logical, external   :: pointsMatch


! test 1

section%from%x =  4d0
section%from%y =  3d0
section%to%x   =  4d0
section%to%y   = -3d0
section%curvature  = 1d0

referencePoint%x = 0d0
referencePoint%y = 0d0

call midPointOfCircle(section)

if (pointsMatch(section%center, referencePoint) .and. &
   doublesMatch(section%radius, 5d0)) then
    write(*,*) "Testing circles radius 1:         OK"
else
    write(*,*) "Testing circles radius 1:         FAIL"
endif

! test 2

section%from%x =  4d0
section%from%y =  3d0
section%to%x   =  4d0
section%to%y   = -3d0
section%curvature  = -1d0

referencePoint%x = 8d0
referencePoint%y = 0d0

call midPointOfCircle(section)

if (pointsMatch(section%center, referencePoint) .and. &
   doublesMatch(section%radius,   5d0)) then
    write(*,*) "Testing circles radius 2:         OK"
else
    write(*,*) "Testing circles radius 2:         FAIL"
endif

! test 3

section%from%x =  4d0
section%from%y = -3d0
section%to%x   =  4d0
section%to%y   =  3d0
section%curvature  = -1d0

referencePoint%x = 0d0
referencePoint%y = 0d0

call midPointOfCircle(section)

if (pointsMatch(section%center, referencePoint) .and. &
   doublesMatch(section%radius,   5d0)) then
    write(*,*) "Testing circles radius 3:         OK"
else
    write(*,*) "Testing circles radius 3:         FAIL"
endif

! test 4

section%from%x =  4d0
section%from%y =  3d0
section%to%x   = -4d0
section%to%y   =  3d0
section%curvature  = -2d0

referencePoint%x = 0d0
referencePoint%y = 0d0

call midPointOfCircle(section)

if (pointsMatch(section%center, referencePoint) .and. &
   doublesMatch(section%radius, 5d0)) then
    write(*,*) "Testing circles radius 4:         OK"
else
    write(*,*) "Testing circles radius 4:         FAIL"
endif

! test 5

section%from%x =  0d0
section%from%y =  5d0
section%to%x   =  5d0
section%to%y   =  0d0
section%curvature  = 5d0 - 2.5 * sqrt(2d0)

referencePoint%x = 0d0
referencePoint%y = 0d0

call midPointOfCircle(section)

if (pointsMatch(section%center, referencePoint) .and. &
   doublesMatch(section%radius, 5d0)) then
    write(*,*) "Testing circles radius 5:         OK"
else
    write(*,*) "Testing circles radius 5:         FAIL"
endif


end subroutine

!---------------------------------------------------------------

subroutine testMinorDeterminant()

implicit none

double precision :: mat4x4(4,4)
double precision :: minorDet

double precision, external :: minorDeterminant4x4
logical, external          :: doublesMatch

mat4x4(1,1) = 6d0
mat4x4(1,2) = 7d0
mat4x4(1,3) = 9d0
mat4x4(1,4) = 8d0

mat4x4(2,1) = 2d0
mat4x4(2,2) = 1d0
mat4x4(2,3) = 1d0
mat4x4(2,4) = 1d0

mat4x4(3,1) = 20d0
mat4x4(3,2) = 2d0
mat4x4(3,3) = 4d0
mat4x4(3,4) = 1d0

mat4x4(4,1) = 34d0
mat4x4(4,2) = 5d0
mat4x4(4,3) = 3d0
mat4x4(4,4) = 1d0


if (doublesMatch(minorDeterminant4x4(mat4x4,1,2), -60d0) .and. &
    doublesMatch(minorDeterminant4x4(mat4x4,1,3), 40d0) .and. &
    doublesMatch(minorDeterminant4x4(mat4x4,1,1), -10d0)) then
    write(*,*) "Testing minor determinant:        OK"
else
    write(*,*) "Testing minor determinant:        FAIL"
endif

end subroutine

!---------------------------------------------------------------

subroutine testElementOrientation

use enumerates
use pointMod, only: point
use topology, only: getElementOrientation

implicit none

type(point) :: p1, p2, p3

! build unit triangle, clockwise
p1%x = 0d0
p1%y = 0d0

p2%x = 0d0
p2%y = 1d0

p3%x = 1d0
p3%y = 0d0

if (getElementOrientation(p1, p2, p3) .eq. clockWise) then
    write(*,*) "Testing element direction 1:      OK"
else
    write(*,*) "Testing element direction 1:      FAIL"
endif

! build a counter clockwise element

p2%x = 8d0
p2%y = 1d0

p3%x = 1d0
p3%y = 8d0

if (getElementOrientation(p1, p2, p3) .eq. counterClockWise) then
    write(*,*) "Testing element direction 1:      OK"
else
    write(*,*) "Testing element direction 1:      FAIL"
endif




end subroutine

!---------------------------------------------------------------

logical function pointsMatch(p1, p2)

use pointMod, only: point

implicit none

type(point) :: p1, p2
double precision, parameter :: eps = 1e-9

if (abs(p1%x - p2%x) .lt. eps .and. &
    abs(p1%y - p2%y) .lt. eps) then
   pointsMatch = .true.
else
    pointsMatch   = .false.
endif

end function

!---------------------------------------------------------------

logical function doublesMatch(a, b)

implicit none

double precision            :: a, b
double precision, parameter :: eps = 1e-9

if (abs(a - b) .lt. eps) then
    doublesMatch = .true.
else
    doublesMatch = .false.
endif

end function

!---------------------------------------------------------------



subroutine testStuff

use pointMod
use circleMod
implicit none

! test circle intersector

type(circleSection) :: section
type(circleSection) :: sectionA, sectionB

type(point)      :: centerA, centerB, crossPoint1, crossPoint2
double precision :: radiusA, radiusB

type(point)      :: pointA, pointB, pointC, pointD
type(point), allocatable :: points(:)
integer          :: iPoint

double precision :: mat4x4(4,4)
double precision :: minorDet
double precision, external :: minorDeterminant4x4

!integer, parameter :: nElems12 = 200
!integer, parameter :: nElems23 = 150

!integer, parameter :: nPatchPoints = (nElems12+1) * (nElems23+1)
!integer, parameter :: nPatchElements = nElems12 * nElems23

!integer            :: iRow
!integer            :: connectivity(nPatchElements,4)
!integer            :: connectivity(nElems12, nElems23, 4)
!type(point)        :: patchPoints(nElems12 + 1 , nElems23 + 1)

double precision   :: interpolation(11)

integer, parameter :: nlist = 7
double precision   :: list(nlist)

allocate(points(11))



write(*,*) "-- D 1 -------------------------------------------------"

section%from%x = 3d0
section%from%y = 4d0
section%to%x = 4d0
section%to%y = 3d0
section%curvature = 0d0

call pointsOnCircleSection(section, &
                           .true., .true., 11, &
                           points)

write(*,*) "Points should go from [3,4] to [4,3] in steps of 0.1 on straight line"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "-- D 2 -------------------------------------------------"

section%from%x = 3d0
section%from%y = 4d0
section%to%x = 4d0
section%to%y = 3d0
section%curvature = 0d0

call pointsOnCircleSection(section, &

                           .false., .false., 11, &
                           points)

write(*,*) "Same but with edges -1"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "-- D 3 -------------------------------------------------"

section%from%x = 1d0
section%from%y = 0d0
section%to%x = 0d0
section%to%y = 1d0
section%curvature = - 1d0 + 0.5 * sqrt(2d0)

call pointsOnCircleSection(section, &
                           .true., .true., 11, &
                           points)

write(*,*) "quarter circle around origin rom [1,0] to [0,1]"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "-- D 3 -------------------------------------------------"

section%from%x = 10d0
section%from%y = 0d0
section%to%x = 0d0
section%to%y = 0d0
section%curvature = -2d0

call pointsOnCircleSection(section, &
                           .true., .true., 11, &
                           points)

write(*,*) "Section from [10,0] to [0,0], with middle point in [5,2]"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "-- D 4 -------------------------------------------------"

section%from%x = 0d0
section%from%y = 10d0
section%to%x = 0d0
section%to%y = 0d0
section%curvature = -2d0

call pointsOnCircleSection(section, &
                           .true., .true., 11, &
                           points)

write(*,*) "Section from [0,10] to [0,0], with middle point in [-2,5]"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "-- D 5 -------------------------------------------------"

section%from%x = 10d0
section%from%y = 10d0
section%to%x = -10d0
section%to%y = -10d0
section%curvature = 2d0 * sqrt(2d0)

call pointsOnCircleSection(section, &
                           .true., .true., 11, &
                           points)

write(*,*) "Section from [10,10] to [-10,-10], with middle point in [2,-2]"
do iPoint=1,11
    write(*,*) "point ", iPoint, "has coords: ", points(iPoint)%x, points(iPoint)%y
enddo

write(*,*) "--  E 1 -------------------------------------------------"


call linSpace(11, 0d0, 10d0, interpolation)
write(*,*) "should have numbers from 0 to 10 in d.p."
write(*,*) interpolation

write(*,*) "--  F 1 -------------------------------------------------"

! horizontal line through a circle

! the circle
radiusA   = 5d0
centerA%x = 1d0
centerA%y = 3d0

! the line
pointA%x = 2d0
pointA%y = 7d0
pointB%x = 3d0
pointB%y = 7d0

call WhereDoLineAndCircleCross(radiusA, centerA, pointA, pointB, &
                                     crossPoint1, crossPoint2)

write(*,*) "points should be -2,7 and 4,7, and are:", &
            crossPoint1%x, crossPoint1%y, "and", &
            crossPoint2%x, crossPoint2%y


write(*,*) "--  F 2 -------------------------------------------------"

! vertical line through a circle

! the circle
radiusA   = 5d0
centerA%x = 1d0
centerA%y = 3d0

! the line
pointA%x = 1d0
pointA%y = 4d0
pointB%x = 1d0
pointB%y = 8d0

call WhereDoLineAndCircleCross(radiusA, centerA, pointA, pointB, &
                                     crossPoint1, crossPoint2)

write(*,*) "points should be 1,8 and 1,-2, and are:", &
            crossPoint1%x, crossPoint1%y, "and", &
            crossPoint2%x, crossPoint2%y

write(*,*) "--  G 1 -------------------------------------------------"


! the first line
pointA%x = 1d0
pointA%y = 4d0
pointB%x = 1d0
pointB%y = 8d0

! the seconde line
pointC%x = 3d0
pointC%y = 3d0
pointD%x = 5d0
pointD%y = 5d0

call WhereDoTwoLinesCross(pointA, pointB, pointC, pointD, crossPoint1, crossPoint2)


write(*,*) "points should both be 1,1 and are:", &
            crossPoint1%x, crossPoint1%y, "and", &
            crossPoint2%x, crossPoint2%y


write(*,*) "--  G 2 -------------------------------------------------"

! points reversed
call WhereDoTwoLinesCross(pointC, pointD, pointA, pointB, crossPoint1, crossPoint2)


write(*,*) "points should both be 1,1 and are:", &
            crossPoint1%x, crossPoint1%y, "and", &
            crossPoint2%x, crossPoint2%y




write(*,*) "--  H 1 -------------------------------------------------"


do iPoint = 1, nlist
    list(iPoint) = 0.7 * iPoint - 1.5
enddo



write(*,*) "fresh list", list

call compressDoubles(nlist, list, 1.4d0)

write(*,*) "final list", list

write(*,*) "--  I 1 -------------------------------------------------"




end subroutine

