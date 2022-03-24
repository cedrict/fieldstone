module highq

! This module turns a basic Q1 mesh into a higher order mesh
! by adding points and adapting the connectivity.
! It follows the connectivity convention from Fieldstone,
! see https://github.com/cedrict/fieldstone

implicit none

contains

subroutine q1toqn()

! this turns the connectivity of q1 elements into qn elements by adding
! points and creating a new connectivity

use pointMod, only: point
use meshData, only: nTotalPointsAfterStitching, &
                    Q1connectivity, &
                    Q1Points, &
                    finalConnectivity, &
                    finalPoints, &
                    finalnPoints, &
                    countedElements
use enumerates
use debug

implicit none

! lookup table for the neighbor points, needed to find the adges of the graph

integer            :: nPointsPerElement
integer            :: n

integer, parameter :: maxNeighnbors = 8
integer :: nNeighbors(nTotalPointsAfterStitching)
integer :: neighborIDs(nTotalPointsAfterStitching, maxNeighnbors)
integer :: sideLinkIDs(nTotalPointsAfterStitching, maxNeighnbors)
integer :: existingSideID
integer :: nSides

type element
integer :: elementID
integer :: sideID
logical :: reverse
type(point)              :: from, to
integer                  :: fromID, toID
end type

type side
type(point), allocatable :: pointsOnSide(:)
integer,     allocatable :: pointIDs(:)
integer                  :: sideID
type(element)            :: elements(2)
logical                  :: reverse
type(point)              :: from, to
integer                  :: fromID, toID
end type

type(side),  allocatable :: sides(:)

type sidePerElement
type(side)               :: sides(4)
end type

type(sidePerElement), allocatable :: sidesPerElement(:)

integer :: iElement, iSide, iPoint
integer :: pos

double precision         :: xstart, xend, dx, xm
double precision         :: ystart, yend, dy, ym
integer                  :: p1, p2
integer                  :: pointCount
integer                  :: addThis, baseNum
integer                  :: sideID, elementID, sideNr
type(point)              :: from, to
integer                  :: fromID, toID

integer                  :: nNewPointsPerSide
integer                  :: nPointsPerElem
integer                  :: nNewPoints

logical                  :: reverse

type(point)              :: p(25) ! to shorten syntax of higher order elements.
type(point)              :: pm

nNeighbors = 0
neighborIDs = 0
sideLinkIDs = 0

if (elementType.eq.Q2) then
    n = 1
else if (elementType.eq.Q3) then
    n = 2
else if (elementType.eq.Q4) then
    n = 3
else
    stop "elements other than Q2, Q3 and Q4 not yet implemented"
endif

! loop through connectivity and store the neighbors that are found,
! based on the lowest index alone. So the outerd edges are stored
! once, but the inner edges separating two elements are also
! stored only once.

nSides = 0

do iElement = 1, countedElements
    do iSide = 1,4

        if (iSide.lt.4) then
            p1 = q1Connectivity(iElement, iSide)
            p2 = q1Connectivity(iElement, iSide+1)
        else
            ! closing segment of the elements
            p1 = q1Connectivity(iElement, 4)
            p2 = q1Connectivity(iElement, 1)
        endif

        baseNum = min(p1, p2)
        addThis = max(p1, p2)        

        call positionInArray(addThis, maxNeighnbors, neighborIDs(baseNum,:), pos)

        if (pos.eq.-1) then
            ! we have found a new side!
            nSides = nSides + 1
            nNeighbors(baseNum) = nNeighbors(baseNum) + 1
            neighborIDs(baseNum, nNeighbors(baseNum)) = addThis
            sideLinkIDs(baseNum, nNeighbors(baseNum)) = nSides
        endif

    enddo

enddo

! determine size for the new mesh
nPointsPerElement = (n+2)**2
nNewPoints = n * nSides + n**2 * countedElements
FinalnPoints = nTotalPointsAfterStitching + nNewPoints

allocate(finalConnectivity(countedElements, nPointsPerElement))
allocate(FinalPoints(FinalnPoints))

! index the sides, and administrate to which elements
! they belong, in a loop that is very similar to the one above
allocate(sides(nSides))

sideNr = 0
nNeighbors = 0
neighborIDs = 0

! initialize all element IDs to 0,
! because we select on them later and not all will be filled:
do iSide = 1, nSides
    do iElement = 1,2
        sides(iSide)%elements(iElement)%elementID = 0
    enddo
enddo

do iElement = 1, countedElements
    do iSide = 1,4

        if (iSide.lt.4) then
            p1 = q1Connectivity(iElement, iSide)
            p2 = q1Connectivity(iElement, iSide+1)
        else
            ! closing segment of the elements
            p1 = q1Connectivity(iElement, 4)
            p2 = q1Connectivity(iElement, 1)
        endif

        baseNum = min(p1, p2)
        addThis = max(p1, p2)

        call positionInArray(addThis, &
                             maxNeighnbors, &
                             neighborIDs(baseNum,:), &
                             pos)

        if (pos.eq.-1) then
            ! we have found a new side!
            sideNr = sideNr + 1

            nNeighbors(baseNum) = nNeighbors(baseNum) + 1
            neighborIDs(baseNum, nNeighbors(baseNum)) = addThis

            sides(sideNr)%elements(1)%elementID = iElement
            sides(sideNr)%elements(1)%sideID = iSide
            sides(sideNr)%elements(1)%from = Q1Points(baseNum)
            sides(sideNr)%elements(1)%to = Q1Points(addThis)
            sides(sideNr)%elements(1)%fromID = baseNum
            sides(sideNr)%elements(1)%toID = addThis
            sides(sideNr)%elements(1)%reverse = &
               isPointDirectionReversed(q1Connectivity(iElement,:), baseNum, addThis, p1, p2)

        else

            ! we have found the second element belonging to an existing side.
            ! We identify which side, by using the sideLinkIDs table,
            ! which is only there for this purpose.
            
            existingSideID = sideLinkIDs(basenum, pos)

            sides(existingSideID)%elements(2)%elementID = iElement
            sides(existingSideID)%elements(2)%sideID = iSide
            sides(existingSideID)%elements(2)%from = Q1Points(baseNum)
            sides(existingSideID)%elements(2)%to = Q1Points(addThis)
            sides(existingSideID)%elements(2)%fromID = baseNum 
            sides(existingSideID)%elements(2)%toID = addThis 
            sides(existingSideID)%elements(2)%reverse = &
              isPointDirectionReversed(q1Connectivity(iElement,:), baseNum, addThis, p1, p2)

        endif
    
    enddo
enddo


! now we have the lookup table of what elements
! belong to which sides. We use this to build a reverse lookup
! table to find out which sides belong to which elements.

allocate(sidesPerElement(countedElements))
do iElement = 1, countedElements
    do iSide = 1, 4
        sidesPerElement(iElement)%sides(iSide)%sideID = 0
        sidesPerElement(iElement)%sides(iSide)%reverse = .false.
        sidesPerElement(iElement)%sides(iSide)%from%x = 0
        sidesPerElement(iElement)%sides(iSide)%from%y = 0
        sidesPerElement(iElement)%sides(iSide)%to%x = 0
        sidesPerElement(iElement)%sides(iSide)%to%y = 0
        sidesPerElement(iElement)%sides(iSide)%fromID = 0
        sidesPerElement(iElement)%sides(iSide)%toID = 0
    enddo
enddo

do iSide = 1, nSides

    do iElement = 1, 2

        elementID = sides(iSide)%elements(iElement)%elementID

        ! It is possible that we are checking the second
        ! element of a side that has no second element,
        ! because the side is on the outside border of the domain
        ! and only has one element.
        ! In that case, do nothing.
        if (elementID .ne. 0) then

            sideID = sides(iSide)%elements(iElement)%sideID
            reverse = sides(iSide)%elements(iElement)%reverse
            from = sides(iSide)%elements(iElement)%from
            to = sides(iSide)%elements(iElement)%to
            fromID = sides(iSide)%elements(iElement)%fromID
            toID = sides(iSide)%elements(iElement)%toID

            sidesPerElement(elementID)%sides(sideID)%sideID = iSide
            sidesPerElement(elementID)%sides(sideID)%reverse = reverse
            sidesPerElement(elementID)%sides(sideID)%from = from
            sidesPerElement(elementID)%sides(sideID)%to = to
            sidesPerElement(elementID)%sides(sideID)%fromID = fromID
            sidesPerElement(elementID)%sides(sideID)%toID = toID

        endif
    enddo
enddo


! We transfer al of the old coordinate list to the new coordinate list,
! and then we append the new points.

FinalPoints(1:nTotalPointsAfterStitching) = q1Points
pointCount = nTotalPointsAfterStitching
nNewPointsPerSide = n

! With the lookup table complete, we can add the extra points
! needed. We start by adding points to the sides.

do iSide = 1, nSides

!    write(*,*) "side ", iSide, &
!                "goes from point", sides(iSide)%elements(1)%fromID, &
!                "to point", sides(iSide)%elements(1)%toID

    xstart = sides(iSide)%elements(1)%from%x
    ystart = sides(iSide)%elements(1)%from%y

    xend   = sides(iSide)%elements(1)%to%x
    yend   = sides(iSide)%elements(1)%to%y

    dx = (xend - xstart) / (n+1)
    dy = (yend - ystart) / (n+1)

    allocate(sides(iSide)%pointsOnSide(nNewPointsPerSide))
    allocate(sides(iSide)%pointIDs(nNewPointsPerSide))
    sides(iSide)%pointsOnSide%x = 0d0
    sides(iSide)%pointsOnSide%y = 0d0
    sides(iSide)%pointIDs = 0


    do iPoint = 1, nNewPointsPerSide

        pointCount = pointCount + 1

        ! put the point in the global list
        FinalPoints(pointCount)%x = xstart + dx * dble(iPoint)
        FinalPoints(pointCount)%y = ystart + dy * dble(iPoint)

        ! but also in the the list for this particular side.
        sides(iSide)%pointsOnSide(iPoint) = FinalPoints(pointCount)
        sides(iSide)%pointIDs(iPoint) = pointCount
    enddo

enddo

! now we have the points on the sides fixed, 
! we can fill the elements and wrap up.

do iElement = 1, countedElements

!    write(*,*) "-------------------------------------------------"
!    write(*,*) "building conn of elem: ", iElement


    if (elementType .eq. Q2) then

        ! base q1 connectivity:

        !  4---3
        !  |   |        
        !  1---2

        ! follow connectivity convention:

        !  4---7---3
        !  |   |   |
        !  8---9---6
        !  |   |   |
        !  1---5---2



        ! Q2 elements. We have all the edge midpoints,
        ! and need only the single midpoint to fill it up.
        ! take the average between points 1 and 3

        xm = (q1Points(q1Connectivity(iElement,1))%x + &
              q1Points(q1Connectivity(iElement,3))%x) * 0.5
        ym = (q1Points(q1Connectivity(iElement,1))%y + &
              q1Points(q1Connectivity(iElement,3))%y) * 0.5

        pointCount = pointCount + 1

        ! put the new point in the global list
        FinalPoints(pointCount)%x = xm
        FinalPoints(pointCount)%y = ym

        ! set the original corners
        finalConnectivity(iElement,1:4) = q1Connectivity(iElement,:)

        ! set the midPoints
        do iSide = 1,4
            sideID = sidesPerElement(iElement)%sides(iSide)%sideID

            ! because we have only single point here,
            ! we do not  yet have to check whether they go in
            ! reverse order or not.
            ! Do this for higher order elements.
            finalConnectivity(iElement,4+iSide) = &
              sides(sideID)%pointIDs(1)
        enddo

        ! and set the center point
        finalConnectivity(iElement,9) = pointCount

    else if (elementType .eq. Q3) then

        ! base q1 connectivity:

        !  2---1
        !  |   |
        !  3---4

        ! create Q3 following connectivity convention from Fieldstone:

        ! 13--14--15--16
        !  |   |   |   |
        !  9--10--11--12
        !  |   |   |   |
        !  5---6---7---8
        !  |   |   |   |
        !  1---2---3---4

        ! create the four central points and 
        ! add them to the final point array
        p(1) = q1Points(q1Connectivity(iElement,3))
        p(4) = q1Points(q1Connectivity(iElement,4))
        p(16) = q1Points(q1Connectivity(iElement,1))
        p(13) = q1Points(q1Connectivity(iElement,2))

        xm = (p(1)%x + p(16)%x) / 2d0
        ym = (p(1)%y + p(16)%y) / 2d0

        p(6)%x  = (2d0 * xm + p(1)%x)  / 3d0
        p(6)%y  = (2d0 * ym + p(1)%y)  / 3d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(6)

        p(7)%x  = (2d0 * xm + p(4)%x) / 3d0
        p(7)%y  = (2d0 * ym + p(4)%y) / 3d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(7)

        p(10)%x = (2d0 * xm + p(13)%x) / 3d0
        p(10)%y = (2d0 * ym + p(13)%y) / 3d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(10)

        p(11)%x = (2d0 * xm + p(16)%x) / 3d0
        p(11)%y = (2d0 * ym + p(16)%y) / 3d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(11)


        ! fill the connectivity of this element
        ! start with the corners
        finalConnectivity(iElement,  1) = q1Connectivity(iElement,1)
        finalConnectivity(iElement,  4) = q1Connectivity(iElement,2)
        finalConnectivity(iElement, 16) = q1Connectivity(iElement,3)
        finalConnectivity(iElement, 13) = q1Connectivity(iElement,4)

        ! If a side has the reverse flag, the points go from
        ! the highest points number to the lowest point number.
        ! So we compare the local direction 

        ! do the side from point 1 to 2  (1 to 4 for Q2)
        sideID = sidesPerElement(iElement)%sides(1)%sideID
!        write(*,*) "side 1 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(1)%reverse

        if (sidesPerElement(iElement)%sides(1)%reverse) then
            finalConnectivity(iElement,  2) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement,  3) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement,  2) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement,  3) = sides(sideID)%pointIDs(2)
        endif

        ! do the side from point 2 to 3  (4 to 16 for Q2)
        sideID = sidesPerElement(iElement)%sides(2)%sideID
!        write(*,*) "side 2 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(2)%reverse

        if (sidesPerElement(iElement)%sides(2)%reverse) then
            finalConnectivity(iElement,  8) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 12) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement,  8) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 12) = sides(sideID)%pointIDs(2)
        endif

        ! do the side from point 4 to 3  (13 to 16 for Q2)
        sideID = sidesPerElement(iElement)%sides(3)%sideID
!        write(*,*) "side 3 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(3)%reverse

        if (sidesPerElement(iElement)%sides(3)%reverse) then
            finalConnectivity(iElement, 14) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 15) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement, 14) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 15) = sides(sideID)%pointIDs(2)
        endif

        ! do the side from point 1 to 4  (1 to 13 for Q2)
        sideID = sidesPerElement(iElement)%sides(4)%sideID
!        write(*,*) "side 4 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(4)%reverse
        if (sidesPerElement(iElement)%sides(4)%reverse) then
            finalConnectivity(iElement, 5) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 9) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement, 5) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 9) = sides(sideID)%pointIDs(2)
        endif

        ! and finally the center
        finalConnectivity(iElement,  6) = pointCount
        finalConnectivity(iElement,  7) = pointCount - 1
        finalConnectivity(iElement, 10) = pointCount - 2
        finalConnectivity(iElement, 11) = pointCount - 3


    else if (elementType .eq. Q4) then

        ! base q1 connectivity:

        !  4---3
        !  |   |        
        !  1---2

        ! create Q4 following connectivity convention from Fieldstone:

        ! 21--22--23--24--25
        !  |   |   |   |   |
        ! 16--17--18--19--20
        !  |   |   |   |   |
        ! 11--12--13--14--15
        !  |   |   |   |   |
        !  6---7---8---9--10
        !  |   |   |   |   |
        !  1---2---3---4---5

! create the four central points and 
        ! add them to the final point array
        p(1) = q1Points(q1Connectivity(iElement,3))
        p(5) = q1Points(q1Connectivity(iElement,4))
        p(25) = q1Points(q1Connectivity(iElement,1))
        p(21) = q1Points(q1Connectivity(iElement,2))

        ! set midpoint
        p(13)%x  = (p(25)%x + p(1)%x)  / 2d0
        p(13)%y  = (p(25)%y + p(1)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(13)

        ! set midpoints of radiants diagonals
        p(7)%x   = (p(1)%x  + p(13)%x)  / 2d0
        p(7)%y   = (p(1)%y  + p(13)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(7)

        p(9)%x   = (p(5)%x  + p(13)%x)  / 2d0
        p(9)%y   = (p(5)%y  + p(13)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(9)

        p(19)%x  = (p(25)%x + p(13)%x)  / 2d0
        p(19)%y  = (p(25)%y + p(13)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(19)

        p(17)%x  = (p(21)%x + p(13)%x)  / 2d0
        p(17)%y  = (p(21)%y + p(13)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(17)

        ! set neighbors in cardinal directions of midpoint
        p(8)%x   = (p(7)%x  + p(9)%x)   / 2d0
        p(8)%y   = (p(7)%y  + p(9)%y)   / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(8)

        p(14)%x  = (p(9)%x  + p(19)%x)  / 2d0
        p(14)%y  = (p(9)%y  + p(19)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(14)

        p(18)%x  = (p(19)%x + p(17)%x)  / 2d0
        p(18)%y  = (p(19)%y + p(17)%y)  / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(18)

        p(12)%x  = (p(17)%x + p(7)%x)   / 2d0
        p(12)%y  = (p(17)%y + p(7)%y)   / 2d0
        pointCount = pointCount + 1
        FinalPoints(pointCount) = p(12)



        ! fill the connectivity of this element
        ! start with the corners
        finalConnectivity(iElement,  1) = q1Connectivity(iElement,1)
        finalConnectivity(iElement,  5) = q1Connectivity(iElement,2)
        finalConnectivity(iElement, 25) = q1Connectivity(iElement,3)
        finalConnectivity(iElement, 21) = q1Connectivity(iElement,4)

        ! If a side has the reverse flag, the points go from
        ! the highest points number to the lowest point number.
        ! So we compare the local direction 

        ! do the side from point 1 to 2  (1 to 4 for Q2)
        sideID = sidesPerElement(iElement)%sides(1)%sideID
!        write(*,*) "side 1 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(1)%reverse

        if (sidesPerElement(iElement)%sides(1)%reverse) then
            finalConnectivity(iElement,  2) = sides(sideID)%pointIDs(3)
            finalConnectivity(iElement,  3) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement,  4) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement,  2) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement,  3) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement,  4) = sides(sideID)%pointIDs(3)
        endif

    ! do the side from point 2 to 3  (4 to 16 for Q2)
        sideID = sidesPerElement(iElement)%sides(2)%sideID
!        write(*,*) "side 2 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(2)%reverse

        if (sidesPerElement(iElement)%sides(2)%reverse) then
            finalConnectivity(iElement, 10) = sides(sideID)%pointIDs(3)
            finalConnectivity(iElement, 15) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 20) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement, 10) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 15) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 20) = sides(sideID)%pointIDs(3)
        endif

    ! do the side from point 4 to 3  (13 to 16 for Q2)
        sideID = sidesPerElement(iElement)%sides(3)%sideID
!        write(*,*) "side 3 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(3)%reverse

        if (sidesPerElement(iElement)%sides(3)%reverse) then
            finalConnectivity(iElement, 22) = sides(sideID)%pointIDs(3)
            finalConnectivity(iElement, 23) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 24) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement, 22) = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 23) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 24) = sides(sideID)%pointIDs(3)
        endif

    ! do the side from point 1 to 4  (1 to 13 for Q2)
        sideID = sidesPerElement(iElement)%sides(4)%sideID
!        write(*,*) "side 4 ",sideID,"has points", &
!                    sides(sideID)%pointIDs(1), sides(sideID)%pointIDs(2), &
!                    "reverse: ", sidesPerElement(iElement)%sides(4)%reverse
        if (sidesPerElement(iElement)%sides(4)%reverse) then
            finalConnectivity(iElement, 6)  = sides(sideID)%pointIDs(3)
            finalConnectivity(iElement, 11) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 16) = sides(sideID)%pointIDs(1)
        else
            finalConnectivity(iElement, 6)  = sides(sideID)%pointIDs(1)
            finalConnectivity(iElement, 11) = sides(sideID)%pointIDs(2)
            finalConnectivity(iElement, 16) = sides(sideID)%pointIDs(3)
        endif

        ! and finally the center

        ! the inner 3 x 3 point square has been generated in the 
        ! following sequence:

        !  5---8---4
        !  |   |   |
        !  9---1---7
        !  |   |   |
        !  2---6---3



        finalConnectivity(iElement, 19) = pointCount - 7
        finalConnectivity(iElement, 18) = pointCount - 3
        finalConnectivity(iElement, 17) = pointCount - 6
        finalConnectivity(iElement, 14) = pointCount - 0
        finalConnectivity(iElement, 13) = pointCount - 8
        finalConnectivity(iElement, 12) = pointCount - 2
        finalConnectivity(iElement,  9) = pointCount - 4
        finalConnectivity(iElement,  8) = pointCount - 1
        finalConnectivity(iElement,  7) = pointCount - 5

    else
        stop "Other element types than Q1...4 still to be programmed."
    endif

enddo

finalnPoints = pointCount


end subroutine

!-------------------------------------------------------------------

logical function isPointDirectionReversed(conn, p1, p2, o1, o2)

use enumerates

implicit none

integer :: conn(4)
integer :: p1, p2
integer :: o1, o2 ! o from original


! set default:
isPointDirectionReversed = .false.

if ((p1.eq.conn(2) .and. p2.eq.conn(1)) .or. & ! reverse on edge12
    (p1.eq.conn(3) .and. p2.eq.conn(2)) .or. & ! reverse on edge23
    (p1.eq.conn(3) .and. p2.eq.conn(4)) .or. & ! reverse on edge43
    (p1.eq.conn(4) .and. p2.eq.conn(1))) then  ! reverse on edge14

    isPointDirectionReversed = .true.
endif

if (isPointDirectionReversed) then
    if (p1 .gt. p2) then
        isPointDirectionReversed = .false.
    endif
else
    if (p1 .gt. p2) then
        isPointDirectionReversed = .true.
    endif
endif


end function


!-------------------------------------------------------------------


subroutine positionInArray(num, arraysize, array, pos)

implicit none

integer :: num, arraysize
integer :: array(arraysize)
integer :: pos

integer :: iCheck

pos = -1

do iCheck = 1, arraysize
    if (array(iCheck) .eq. num) then
        pos = iCheck
        return
    endif
enddo

end subroutine


end module
