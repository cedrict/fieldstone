subroutine onePatch(patchID, &
                    nElems12, nElems23, &
                    p1, p2, p3, p4, &
                    ca, cb, cc, cd, &
                    fa, fb, fc, fd)

use stitching
use pointMod
use circleMod
use meshData, only: nTotalElements, &
                    countedPoints
use debug

implicit none

! makes one single patch, with corner points and radii:

!  p1 -------- p2
!   |    ca    |
!   | cd    cb |
!   |    cc    |
!  p4 -------- p3
!
! There are nElems12 elements on the line from p1 to p2
! There are nElems23 elements on the line from p2 to p3
!
! Positive curvatures are all point outward (so radii are all pointing inward)
!
! The points are returned in the matrix : points
! And en connectivity matrix            : connectivity

integer          :: patchID
integer          :: nElems12, nElems23
type(point)      :: p1, p2, p3, p4
double precision :: ca, cb, cc, cd   ! curvatures
double precision :: fa, fb, fc, fd   ! focusing

type(point)      :: points(nElems12+1, nElems23+1)
integer          :: connectivity(nElems12, nElems23 , 4)

type(circleSection) :: verticalSections(nElems12 + 1)
type(circleSection) :: horizontalSections(nElems23 + 1)

double precision :: curvaturesa2c(nElems23 + 1)
double precision :: curvaturesd2b(nElems12 + 1)

integer          :: i, j
integer          :: pickMe
integer          :: iPoint, iSection
integer          :: iCurve, jCurve
integer          :: iElement, jElement
type(point)      :: iCenter, jCenter

type(point)      :: matchPoint
type(point)      :: crossPoint1, crossPoint2



! initiliaze
points(:,:)%x = 0d0
points(:,:)%y = 0d0
connectivity = 0

! update the global meshData
!nTotalElements = nTotalElements + nElems12 * nElems23


! assume for now that the curvatures change linearly between opposite sides.
! This may have to be adapted later for strongly deformed patches
! It does work well with patches that have straight line edges, though,
! which could not be done well will radii

call linSpace(nElems23 + 1, ca, cc, horizontalSections(:)%curvature)
call linSpace(nElems12 + 1, cd, cb, verticalSections(:)%curvature)

! start with the circumference.
! The support points created here along those circumference arcs are 
! required to compute the interior.

! first the sections have to be set up:
horizontalSections(1)%from = p1
horizontalSections(1)%to = p2
horizontalSections(nElems23 + 1)%from = p4
horizontalSections(nElems23 + 1)%to = p3
verticalSections(1)%from = p1
verticalSections(1)%to = p4
verticalSections(nElems12 + 1)%from = p2
verticalSections(nElems12 + 1)%to = p3

! section from P1 to p2
call pointsOnCircleSection(horizontalSections(1),    .true., .true., nElems12 + 1, fa, points(:,1))
! section from P4 to p3
call pointsOnCircleSection(horizontalSections(nElems23 + 1), .true., .true., nElems12 + 1, fc, points(:,nElems23 + 1))

! section from P1 to p4
call pointsOnCircleSection(verticalSections(1),      .true., .true., nElems23 + 1, fd, points(1,:))
! section from P2 to p3
call pointsOnCircleSection(verticalSections(nElems12 + 1),   .true., .true., nElems23 + 1, fb, points(nElems12 + 1,:))

! now fill up the middle,
! by determine where all the circles cross.

! first fill the section arrays with starting and finishing points
! (from and to), that are stored in the circumference

! fill the horizontal sections
do iCurve = 2, nElems23
    horizontalSections(iCurve)%from = points(1, iCurve)
    horizontalSections(iCurve)%to   = points(nElems12 + 1, iCurve)

    ! fill in radius from points and curvature
    call midPointOfCircle(horizontalSections(iCurve))
enddo


! fill the vertical sections
do iCurve = 2, nElems12
    verticalSections(iCurve)%from = points(iCurve, 1)
    verticalSections(iCurve)%to   = points(iCurve, nElems23 + 1)

    ! fill in radius from points and curvature
    call midPointOfCircle(verticalSections(iCurve))
enddo

! now select the crossing points of those circle section
! for each combination of horizontal and vertical section:

do iCurve = 2, nElems12  ! loop over the vertical sections in outer loop
    do jCurve = 2, nElems23 ! and over the horizontal sections in the inner loop

        call whereDoTwoSectionsCross(verticalSections(iCurve), &
                                     horizontalSections(jCurve), &
                                    crossPoint1, crossPoint2)

        ! for each combination of two sections, there will be two crossings.
        ! (if both the sections are straight line, there will be 1 points,
        !  but both points will have the same value then)
        ! Select the proper one, which is closest to where we are looking.
        ! this is not a 100% fool proof, but it needs a pretty big fool
        ! to trip it.

        matchPoint%x = 0.25 * (verticalSections(iCurve)%from%x + &
                               verticalSections(iCurve)%to%x + &
                               horizontalSections(jCurve)%from%x + &
                               horizontalSections(jCurve)%to%x)

        matchPoint%y = 0.25 * (verticalSections(iCurve)%from%y + &
                               verticalSections(iCurve)%to%y + &
                               horizontalSections(jCurve)%from%y + &
                               horizontalSections(jCurve)%to%y)

        call closestPoint(matchPoint, crossPoint1, crossPoint2, points(iCurve, jCurve))

    enddo
enddo


! make the elements

do iElement = 1, nElems23
    do jElement = 1, nElems12
        connectivity(jElement, iElement, 1) = (iElement-1) * (nElems12+1) + jElement
        connectivity(jElement, iElement, 2) = (iElement-1) * (nElems12+1) + jElement+1
        connectivity(jElement, iElement, 3) = (iElement)   * (nElems12+1) + jElement+1
        connectivity(jElement, iElement, 4) = (iElement)   * (nElems12+1) + jElement
    enddo
enddo

call fillEdges(patchID, nElems12, nElems23, countedPoints)

call addPatchToGlobal(nElems12, nElems23, points, connectivity)

end subroutine

!-----------------------------------------------------------------------

subroutine fillEdges(thisOne, nElems12, nElems23, startAt)

use stitching

implicit none

integer :: thisOne, startAt

integer :: nElems12, nElems23
integer :: n12, n23
integer :: i12, i23

integer :: size1, size2

!     edge(patchID)%pointIDs(i,j)
!
! 2   _____________   3
!    |     j=1     |
!    |             |
!    |i=1          |i=max
!    |             |
!    |_____________|
! 1       j=max       4
n12 = nElems12 + 1
n23 = nElems23 + 1

allocate(edges(thisOne)%pointIDs(n23, n12))

edges(thisOne)%pointIDs = 0

! start counting at 1

size1 = size(edges(thisOne)%pointIDs,1)
size2 = size(edges(thisOne)%pointIDs,2)

do i12 = 1, n12
    edges(thisOne)%pointIDs(1,i12)   = i12 + startAt
    edges(thisOne)%pointIDs(n23,i12) = i12 + (n23-1)*n12 + startAt
enddo

do i23 = 1, n23
    edges(thisOne)%pointIDs(i23,n12)   = i23 * n12 + startAt
    edges(thisOne)%pointIDs(i23,1) = (i23-1) * n12  + 1 + startAt
enddo

end subroutine



!-----------------------------------------------------------------------

subroutine closestPoint(compareTo, p1, p2, closest)

use pointMod

implicit none

type(point) :: compareTo, p1, p2, closest
double precision, external :: distanceBetweenTwoPoints

if (distanceBetweenTwoPoints(p1, compareTo) .gt. &
    distanceBetweenTwoPoints(p2, compareTo)) then
    closest = p2
else
    closest = p1
endif

end subroutine

!-----------------------------------------------------------------------

function distanceBetweenTwoPoints(p1, p2)

use pointMod

implicit none

double precision :: distanceBetweenTwoPoints
type(point) :: p1, p2

distanceBetweenTwoPoints = sqrt((p1%x - p2%x)**2 + (p1%y - p2%y)**2)

end function

!-----------------------------------------------------------------------

subroutine linSpace(n, min, max, array)
! makes an array with linear steps, like the linSpace matlab command.
implicit none

integer          :: n
double precision :: min, max
double precision :: array(n)

integer          :: i
double precision :: stepSize

stepSize = (max - min) / dble(n-1)
do i = 1,n
    array(i) = min + dble(i-1) * stepSize
enddo

end subroutine


!-----------------------------------------------------------------------

subroutine pointsOnCircleSection(section, &
                                 includeStartPoint, includeEndPoint, nPoints, &
                                 focus, &
                                 points)

! points in clockwise direction on the circle section

! if radius = 0 -> straight line.

use circleMod
use pointMod

implicit none

double precision, parameter :: pi = 4d0 * atan(1d0)

type(circleSection)         :: section

type(point)                 :: startPoint
type(point)                 ::   endPoint
double precision            :: radius
integer                     :: nPoints
logical                     :: isLine
logical                     :: includeStartPoint
logical                     :: includeEndPoint
double precision            :: focus


type(point)                 :: centerPoint
type(point)                 :: points(nPoints)
type(point)                 :: dxdy

integer                     :: iStart, iEnd, iPoint
double precision            :: dx, dy

double precision            :: angles(nPoints)
double precision            :: angleStep

if (focus.eq.0d0) then
    stop "focus on a patch is set to 0. Must be positive."
endif


iStart = 0
iEnd   = nPoints

! is not always necessary, but it makes the process more robust
call midPointOfCircle(section)


! set the default coords to -1
! This will later be used to filter out
! only the used points


do iPoint = 1, nPoints
    points(iPoint)%x = -1d0
    points(iPoint)%y = -1d0
enddo

if (includeStartPoint) then
    points(1) = section%from
endif

if (includeEndPoint) then
    points(nPoints) = section%to
endif

! fill the middle points here:
if (section%radius.eq.0d0) then
    ! the points are not actually on a circle section, but on a straight line
    ! ignore the centerXY point.
    dx = (section%to%x - section%from%x) / dble(nPoints - 1)
    dy = (section%to%y - section%from%y) / dble(nPoints - 1)

    do iPoint = 2, nPoints-1
        points(iPoint)%x = section%from%x + (iPoint-1) * dx
        points(iPoint)%y = section%from%y + (iPoint-1) * dy
    enddo

    call compressDoubles(nPoints, points(:)%x, focus)
    call compressDoubles(nPoints, points(:)%y, focus)

else
    ! compute the circle section

    ! angle starts at -pi / 2 at 6 o'clock and goes counter clockwise
    ! through pi / 2 at 12 o'clock and to 3 pi / 2 again at 6 o'clock.

    ! starting angle
    dx = section%from%x - section%center%x
    dy = section%from%y - section%center%y

    if (dx .gt. 0d0) then
        angles(1) = atan(dy / dx)
    else if (dx .lt. 0d0) then
        angles(1) = pi + atan(dy / dx)
    else if (dx .eq. 0d0 .and. dy .gt. 0) then
        angles(1) = pi / 2d0
    else
        angles(1) = -pi / 2d0
    endif

    ! final angle
    dx = section%to%x - section%center%x
    dy = section%to%y - section%center%y

    if (dx .gt. 0d0) then
        angles(nPoints) = atan(dy / dx)
    else if (dx .lt. 0d0) then
        angles(nPoints) = pi + atan(dy / dx)
    else if (dx .eq. 0d0 .and. dy .gt. 0) then
        angles(nPoints) = pi / 2d0
    else
        angles(nPoints) = -pi / 2d0
    endif

    ! we can run into the situation that one angle is one side
    ! of the 6 o'clock mark and the other angle on the other side.
    ! the angle will no step all the way around the near full circle
    ! if we would not prevent this.
    if (angles(nPoints) - angles(1) .gt. pi) then
        angles(nPoints) = angles(nPoints) - 2d0 * pi
    endif
    if (angles(1) - angles(nPoints) .gt. pi) then
        angles(1) = angles(1) - 2d0 * pi
    endif

    ! step size  (what if this becomes negative?...)
    angleStep = (angles(nPoints) - angles(1)) / dble(nPoints-1)

    ! set all angles
    do iPoint = 2, nPoints-1
        angles(iPoint) = angles(1) + (iPoint-1) * angleStep
    enddo

    call compressDoubles(nPoints, angles, focus)

    ! create points
    do iPoint = 2, nPoints-1
        points(iPoint)%x = section%center%x + section%radius * cos(angles(iPoint))
        points(iPoint)%y = section%center%y + section%radius * sin(angles(iPoint))
    enddo

endif


end subroutine

!-----------------------------------------------------------------------

subroutine compressDoubles(n, list, s)
! subroutine takes an equidistant array of double precision numbers,
! and compresses them to either one side or the other,
! by scaling them to the 0-1 interval,
! projecting them on y=x^s
! and converting it back to the original range.
! for s higher than 1, the points will move toward lower coordinates
! for s lower than 1, the points will move toward higher coordinates

integer          :: n
double precision :: list(n)   ! original list
double precision :: list01(n) ! list on 0-1 interval
double precision :: s

double precision :: a,b
integer          :: i


! if the values are all the same, because we have
! the x coordinates of a vertical line, or
! the y coordinates of a horizontal line
! do not do this.
if (list(1).eq.list(n)) then
    return
endif


! define a,b so that a * list(i) + b = list01(i)

a = 1/(list(n) - list(1))
b = -list(1) * a

do i=1,n
    list01(i) = a * list(i) + b
enddo

! do the transformation
do i=1,n
     list01(i) = list01(i)**s
enddo

! redefine a,b so that a * list01(i) + b = list(i)
a = list(n) - list(1)
b = list(1)

do i=2,n-1
    list(i) = a * list01(i) + b
enddo

end subroutine

!-----------------------------------------------------------------------

subroutine midPointOfCircle(section)


use pointMod
use circleMod

implicit none

type(circleSection)  :: section


double precision :: dx, dy, dr
type(point)      :: midPoint, thirdPoint

double precision :: halfBaseLength
double precision :: dxToThird, dyToThird

double precision :: M(4,4)
double precision, external :: minorDeterminant4x4


if (section%curvature .eq. 0d0) then

    section%center%x = 0
    section%center%y = 0
    section%radius=0d0

else

    dx = 0.5 * (section%to%x - section%from%x)
    dy = 0.5 * (section%to%y - section%from%y)

    midPoint%x = section%from%x + dx
    midPoint%y = section%from%y + dy

    halfBaseLength = sqrt(dx**2 + dy**2)

    dxToThird = -dy * section%curvature / halfBaseLength
    dyToThird =  dx * section%curvature / halfBaseLength

    thirdPoint%x = midPoint%x + dxToThird
    thirdPoint%y = midPoint%y + dyToThird

    ! shamelessly stolen from:
    ! https://math.stackexchange.com/questions/213658/get-the-equation-of-a-circle-when-given-3-points

    ! first row does not matter, because is will be excluded form all the minors
    M(1,1) = 1
    M(1,2) = 2
    M(1,3) = 3
    M(1,4) = 4

    M(2,1) = section%from%x**2 + section%from%y**2
    M(2,2) = section%from%x
    M(2,3) = section%from%y
    M(2,4) = 1d0

    M(3,1) = section%to%x**2 + section%to%y**2
    M(3,2) = section%to%x
    M(3,3) = section%to%y
    M(3,4) = 1d0

    M(4,1) = thirdPoint%x**2 + thirdPoint%y**2
    M(4,2) = thirdPoint%x
    M(4,3) = thirdPoint%y
    M(4,4) = 1d0

    section%center%x =  0.5 * minorDeterminant4x4(M,1,2) / minorDeterminant4x4(M,1,1)
   section%center%y = -0.5 * minorDeterminant4x4(M,1,3) / minorDeterminant4x4(M,1,1)
    section%radius = sqrt(section%center%x**2 + section%center%y**2 + &
                          minorDeterminant4x4(M,1,4) / minorDeterminant4x4(M,1,1))

endif

end subroutine

!-----------------------------------------------------------------------

function minorDeterminant4x4(M4x4, mi, mj)

implicit none

double precision :: minorDeterminant4x4

double precision :: M4x4(4,4)
integer          :: mi, mj ! line and column to be excluded

double precision :: M3x3(3,3)
double precision , external :: determinant3x3

integer          :: i, j, filli, fillj

! construct submatrix

filli = 1
fillj = 1

do i = 1,4
    if (i.ne.mi) then
        do j= 1,4
            if (j.ne.mj) then
                M3x3(filli,fillj) = M4x4(i,j)
                fillj = fillj + 1
            endif
        enddo
        filli = filli + 1
        fillj = 1
    endif
enddo

minorDeterminant4x4 = determinant3x3(M3x3)

end function

!-----------------------------------------------------------------------

function determinant3x3(A)

implicit none

double precision :: A(3,3), determinant3x3

determinant3x3 = A(1,1) * (A(2,2)*A(3,3)-A(2,3)*A(3,2)) &
               - A(2,1) * (A(1,2)*A(3,3)-A(1,3)*A(3,2)) &
               + A(3,1) * (A(1,2)*A(2,3)-A(1,3)*A(2,2))

end function

!-----------------------------------------------------------------------

subroutine whereDoTwoSectionsCross(sectionA, sectionB, &
                                   crossPoint1, crossPoint2)

use circleMod
use pointMod

implicit none

type(circleSection) :: sectionA, sectionB
type(point)         :: crossPoint1, crossPoint2

if (sectionA%curvature.ne.0d0 .and. sectionB%curvature.ne.0d0) then

    call WhereDoTwoCirclesCross(sectionA%radius, sectionA%center, &
                                sectionB%radius, sectionB%center, &
                                crossPoint1, crossPoint2)

else if (sectionA%curvature.eq.0d0 .and. sectionB%curvature.ne.0d0) then

    call WhereDoLineAndCircleCross(sectionB%radius, sectionB%center, &
                                   sectionA%from, sectionA%to, &
                                   crossPoint1, crossPoint2)

else if (sectionA%curvature.ne.0d0 .and. sectionB%curvature.eq.0d0) then

    call WhereDoLineAndCircleCross(sectionA%radius, sectionA%center, &
                                   sectionB%from, sectionB%to, &
                                   crossPoint1, crossPoint2)

else if (sectionA%curvature.eq.0d0 .and. sectionB%curvature.eq.0d0) then

    call WhereDoTwoLinesCross(sectionA%from, sectionA%to, &
                              sectionB%from, sectionB%to, &
                              crossPoint1, crossPoint2)

else
    STOP "This should not happen... Cross point exception"
endif

end subroutine

!-----------------------------------------------------------------------

subroutine WhereDoTwoLinesCross(p1, p2, q1, q2, crossPoint1, crossPoint2)

use pointMod

implicit none

type(point)     :: p1, p2, q1, q2, crossPoint1, crossPoint2

double precision :: a1, a2, b1, b2


! todo, check if no numerical issues for veeeeery 
! steep lines, which can realistically occur.

if (p1%x .eq. p2%x .and. q1%x .eq. q2%x) then
    stop "both crossing lines vertical. Does not work"

else if (p1%x .eq. p2%x) then
    ! the p line is vertical
    a2 = (q2%y - q1%y)/(q2%x - q1%x)
    b2 = q1%y - a2 * q1%x

    write(*,*) "a2, b2", a2, b2

    crossPoint1%x = p1%x
    crossPoint1%y = a2 * crossPoint1%x + b2

else if (q1%x .eq. q2%x) then
    ! the q line is vertical 
    a1 = (p2%y - p1%y)/(p2%x - p1%x)
    b1 = p1%y - a1 * p1%x

    crossPoint1%x = q1%x
    crossPoint1%y = a1 * crossPoint1%x + b1

else
    ! two non-vertical lines

    ! reduce both lines to the form of y = ax + b
    a1 = (p2%y - p1%y)/(p2%x - p1%x)
    a2 = (q2%y - q1%y)/(q2%x - q1%x)

    b1 = p1%y - a1 * p1%x
    b2 = q1%y - a2 * q1%x

    crossPoint1%x = (b2 - b1) / (a1 - a2)
    crossPoint1%y = a1 * crossPoint1%x + b1

endif

crossPoint2 = crossPoint1

end subroutine

!-----------------------------------------------------------------------

subroutine WhereDoTwoCirclesCross(radiusA, centerA, radiusB, centerB, &
                                  crossPoint1, crossPoint2)

! shamelessly stolen from:
! https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect

use pointMod

implicit none

type(point)      :: centerA, centerB
double precision :: radiusA, radiusB
type(point)      :: crossPoint1, crossPoint2

! abbreviations to follow math exchange:
double precision :: Rsq, x1, y1, x2, y2, r1, r2
double precision :: f1, f2, f2a, f2b


x1 = centerA%x
y1 = centerA%y
x2 = centerB%x
y2 = centerB%y

Rsq = (x1 - x2)**2 + (y1 - y2)**2

r1 = radiusA
r2 = radiusB

f1 = (r1**2 - r2**2) / (2d0 * Rsq)
f2a = 2d0 * (r1**2 + r2**2)    / Rsq
f2b =    (r1**2 - r2**2)**2 / Rsq**2

f2 = 0.5 * (f2a - f2b - 1d0)**0.5

crossPoint1%x = 0.5 * (x1 + x2) + f1 * (x2 - x1) + f2 * (y2 - y1)
crossPoint1%y = 0.5 * (y1 + y2) + f1 * (y2 - y1) + f2 * (x1 - x2)

crossPoint2%x = 0.5 * (x1 + x2) + f1 * (x2 - x1) - f2 * (y2 - y1)
crossPoint2%y = 0.5 * (y1 + y2) + f1 * (y2 - y1) - f2 * (x1 - x2)

end subroutine

!-----------------------------------------------------------------------

subroutine WhereDoLineAndCircleCross(radius, center, p1, p2, &
                                     crossPoint1, crossPoint2)

! find points crossPoint1, crossPoint2 where a line through points p1 and p2 crosses
! a circle with center and radius

! solution shamelessly stolen from:
! https://www.embibe.com/exams/intersection-between-circle-and-line/

use pointMod

implicit none

double precision :: radius
type(point)      :: p1, p2, l1, l2, center
type(point)      :: crossPoint1, crossPoint2

logical          :: flipped

double precision :: la,lb  ! to write the line to y = ax + b
double precision :: ca, cb, cc ! to write circle into x^2 + y^2 + ca x + cb y - cc = 0
double precision :: a, b, c, D

if (p1%x .eq. p2%x) then
    ! line is vertical
    ! change coordinates to horizontal by flipping and x and y, and flipping
    ! back at the end
    call switchPoint(p1)
    call switchPoint(p2)
    call switchPoint(center)
    flipped = .true.
else
    flipped = .false.
endif

! from:
! (x - cx)^2 + (y - cy)^2 - r^2 = 0
! we get
! x^2 - 2 x cx + cx^2 + y^2 - 2 y cy + cy^2 - r^2 = 0
! so that
! x^2 + y^2 - (2 cx)x - (2 cy)y - (r^2 - cx^2 - cy^2) = 0

la = (p2%y - p1%y)/(p2%x - p1%x)
lb =  p1%y - la * p1%x

ca = -2*center%x
cb = -2*center%y
cc = radius**2 - center%x**2 - center%y**2

! substitute y by la x + lb from the line.

! now solve x^2 + (la x + lb)^2 + ca x + cb (la x + lb) - cc = 0
! It forms an equation of ax^2 + bx + c = 0, with

a = 1d0 + la**2
b = 2d0 * la * lb + ca + cb * la
c = lb**2 + cb * lb - cc

D = b**2 - 4d0 * a * c

if (D.le.0) then
    stop "Determinant non-positive when computing crossing of line and circle"
else
    crossPoint1%x = (-b + sqrt(D))/(2*a)
    crossPoint2%x = (-b - sqrt(D))/(2*a)

    crossPoint1%y = la * crossPoint1%x + lb
    crossPoint2%y = la * crossPoint2%x + lb
endif

if (flipped) then
    ! check y now, because we switched, and flip back
    call switchPoint(p1)
    call switchPoint(p2)
    call switchPoint(center)
    call switchPoint(crossPoint1)
    call switchPoint(crossPoint2)
endif

end subroutine

!-----------------------------------------------------------------------

subroutine switchPoint(p)
! switch x and y

use pointMod

implicit none

type(point)      :: p
double precision :: temp

temp = p%x
p%x  = p%y
p%y  = temp

end subroutine

!-----------------------------------------------------------------------

subroutine addPatchToGlobal(ni, nj, patchPoints, patchConnectivity)

use meshData
use pointMod

implicit none

integer     :: ni, nj
type(point) :: patchPoints(ni+1, nj+1)
integer     :: patchConnectivity(ni, nj, 4)

integer     :: iPoint, jPoint
integer     :: iElement, jElement

integer     :: startConnAt

startConnAt = countedPoints

do jPoint = 1, nj+1
    do iPoint = 1, ni+1
        countedPoints = countedPoints + 1
        Q1Points(countedPoints) = patchPoints(iPoint,jPoint)
    enddo
enddo

do jElement = 1, nj
    do iElement = 1, ni
        countedElements = countedElements + 1
        Q1connectivity(countedElements,:) = &
          patchConnectivity(iElement,jElement,:) + startConnAt
    enddo
enddo

end subroutine
