subroutine removeConnectivityGaps()

use meshData
use pointMod
use stitching, only: removedPointIDs
use debug

! Remove the gaps caused by the stitching.
! Several points have been replaced by their duplicates in other patches.
! They are no longer needed and must be removed.

implicit none

integer     :: nRemovedPoints

integer     :: iPoint, jPoint, iElement, iPatch
integer     :: size12, size23
integer, allocatable :: replaceLookup(:)

if (nStitches.eq.0) then
    ! there is only a single patch and there was no stitching
    ! causing connectivity gaps that have to be fixed.
    nTotalPointsAfterStitching = nTotalPointsBeforeStitching

else
    
    nRemovedPoints = 0

    ! create array of replacement values of all points

    allocate(replaceLookup(countedPoints))
    replaceLookup = 0

    do iPoint = 1, nTotalPointsBeforeStitching
        if (removedPointIDs(iPoint).ne.0) then
            nRemovedPoints = nRemovedPoints + 1
            replaceLookup(iPoint) = 0
        else
            replaceLookup(iPoint) = iPoint - nRemovedPoints
        endif
    enddo

    do iElement = 1, nTotalElements
        do iPoint = 1,4
            Q1connectivity(iElement, iPoint) = &
              replaceLookup(Q1connectivity(iElement, iPoint))
        enddo
    enddo


!    write(*,*) "replaceLookup 1"
!    do iPoint = 1,countedPoints
!        write(*,*) iPoint, replaceLookup(iPoint)
!    enddo

    ! update the unstitched edges
    do iPatch = 1, nPatches

!        write(*,*) "patch", iPatch, "before update"
!        call showEdges(iPatch)

        size12 = size(edges(iPatch)%pointIDs,2)
        size23 = size(edges(iPatch)%pointIDs,1)

        do iPoint=1,size23
            do jPoint=1,size12
                if (edges(iPatch)%pointIDs(iPoint, jPoint).gt.0) then
                    ! 0 is center, not relevant
                    ! negative = islready stitched
                    ! positive = still to be replaced

!                    write(*,*) 'replace ', iPoint, jPoint, &
!                      edges(iPatch)%pointIDs(iPoint, jPoint), 'by', &
!                      replaceLookup(edges(iPatch)%pointIDs(iPoint, jPoint))

                    edges(iPatch)%pointIDs(iPoint, jPoint) = &
                        replaceLookup(edges(iPatch)%pointIDs(iPoint, jPoint))

                else if (edges(iPatch)%pointIDs(iPoint, jPoint).lt.0) then
                    edges(iPatch)%pointIDs(iPoint, jPoint) = &
                        -replaceLookup(-edges(iPatch)%pointIDs(iPoint, jPoint))


                endif
            enddo
        enddo

!        write(*,*) "patch", iPatch, "after update"
!        call showEdges(iPatch)

    enddo

    ! make a different lookup table for the points,
    ! because the lookup here goes the other way.

    nTotalPointsAfterStitching = 0
    replaceLookup = 0
    nRemovedPoints = 0

    do iPoint = 1, nTotalPointsBeforeStitching
        if (removedPointIDs(iPoint).eq.0) then
            nTotalPointsAfterStitching = nTotalPointsAfterStitching + 1
            replaceLookup(nTotalPointsAfterStitching) = iPoint
        endif
    enddo

!    write(*,*) "replaceLookup 2"
!    do iPoint = 1,countedPoints
!        write(*,*) iPoint, replaceLookup(iPoint)
!    enddo

    do iPoint = 1, nTotalPointsAfterStitching
        Q1Points(iPoint) = Q1Points(replaceLookup(iPoint))
    enddo

    deallocate(replaceLookup)

endif

end subroutine

!-----------------------------------------------------------------------

subroutine replaceInConn(n, replaceThis, byThis)

use meshData, only: Q1connectivity

! yes, this overlaps too much with stitch, but the goal
! is fundamentally different, so the separate subs are justified.

implicit none

integer :: n, replaceThis, byThis

integer :: iElement, iPoint

do iElement = 1, n
    do iPoint = 1, 4
        if (Q1connectivity(iElement, iPoint) .eq. replaceThis) then
            Q1connectivity(iElement, iPoint) = byThis
        endif
    enddo
enddo

end subroutine

!-----------------------------------------------------------------------
subroutine stitch(edgeLength, oldPatchID, oldEdgeID, newPatchID, newEdgeID, reverse)

use meshData, only: countedElements, &
                    Q1connectivity, &
                    nStitches
use enumerates
use stitching

implicit none

integer :: edgeLength
logical :: reverse
integer :: oldPatchID, oldEdgeID, newPatchID, newEdgeID

integer :: old(edgeLength), new(edgeLength)
integer :: iElement, iPoint, iEdge

integer :: size1old, size2old
integer :: size1new, size2new
integer :: mySizeOld, mySizeNew

nStitches = nStitches + 1

size1old = size(edges(oldPatchID)%pointIDs,1)
size2old = size(edges(oldPatchID)%pointIDs,2)
size1new = size(edges(newPatchID)%pointIDs,1)
size2new = size(edges(newPatchID)%pointIDs,2)


if      (oldEdgeID .eq. edge12) then
    old = edges(oldPatchID)%pointIDs(1,:)
    mySizeOld = size2old

else if (oldEdgeID .eq. edge23) then
    old = edges(oldPatchID)%pointIDs(:,size2old)
    mySizeOld = size1old

else if (oldEdgeID .eq. edge43) then
    old = edges(oldPatchID)%pointIDs(size1Old,:)
    mySizeOld = size2old

else if (oldEdgeID .eq. edge14) then
    old = edges(oldPatchID)%pointIDs(:,1)
    mySizeOld = size1old

else
    write(*,*) "Stitch does not recognize edge ID for patch", oldPatchID
endif

if      (newEdgeID .eq. edge12) then
    new = edges(newPatchID)%pointIDs(1,:)
    mySizeNew = size2new

else if (newEdgeID .eq. edge23) then
    new = edges(newPatchID)%pointIDs(:,size2new)
    mySizeNew = size1new

else if (newEdgeID .eq. edge43) then
    new = edges(newPatchID)%pointIDs(size1New,:)
    mySizeNew = size2new

else if (newEdgeID .eq. edge14) then
    new = edges(newPatchID)%pointIDs(:,1)
    mySizeNew = size1new

else
    write(*,*) "Stitch does not recognize edge ID for patch", oldPatchID
endif

if (mySizeOld .ne. mySizeNew) then

    write(*,*) "Trying to stitch patch ", oldPatchID, "on side", oldEdgeID, "of length", mySizeOld, &
                          "witch patch ", newPatchID, "on side", newEdgeID, "of length", mySizeNew
    STOP "Sides to be stitched must have the same length."
endif


! search through the the connectivity and replace old entries by new ones.
! old ones are points from the original patch.
! new ones are points from the patch to which this on is attached




! mark removed, which is used later to fix the connectivity
do iEdge = 1, edgeLength
    if (old(iEdge).gt.0) then
        removedPointIDs(old(iEdge)) = 1
    endif
enddo


! Update connectivity
do iEdge = 1, edgeLength
    if (old(iEdge).gt.0) then
        do iElement = 1, countedElements
            do iPoint = 1, 4
                if (reverse) then
                    if (Q1connectivity(iElement, iPoint) .eq. abs(old(iEdge))) then
                        Q1connectivity(iElement, iPoint) = abs(new(edgeLength-iEdge+1))

                    endif
                else
                    if (Q1connectivity(iElement, iPoint) .eq. abs(old(iEdge))) then
                        Q1connectivity(iElement, iPoint) = abs(new(iEdge))
                    endif
                endif
            enddo
        enddo
    endif
enddo


! Replace the old edge by the new one
do iEdge = 1, edgeLength
    if (reverse) then
        if (old(iEdge).gt.0) then
            old(iEdge) = -abs(new(edgeLength - iEdge + 1))
        endif
    else
        if (old(iEdge).gt.0) then
            old(iEdge) = -abs(new(iEdge))
        endif
    endif
enddo

! Put the old array back in the original matrix.

if      (oldEdgeID .eq. edge12) then
    edges(oldPatchID)%pointIDs(1,:) = old
else if (oldEdgeID .eq. edge23) then
    edges(oldPatchID)%pointIDs(:,size2old) = old
else if (oldEdgeID .eq. edge43) then
    edges(oldPatchID)%pointIDs(size1Old,:) = old
else if (oldEdgeID .eq. edge14) then
    edges(oldPatchID)%pointIDs(:,1) = old
endif


end subroutine

!-----------------------------------------------------------------------

logical function twoPointsMatch(pointA, pointB)

use pointMod

implicit none

double precision, parameter :: eps = 1e-6

type(point) :: pointA, pointB

if (abs(pointA%x - pointB%x) .lt. eps .and. &
    abs(pointA%y - pointB%y) .lt. eps) then
    twoPointsMatch = .true.
else
    twoPointsMatch = .false.
endif

end function

!-----------------------------------------------------------------------

subroutine autoStitch()

use meshData
use stitching
use enumerates

! automatically stitches all adjacent sides.
! If no open areas between patches are required,
! this is the answer

implicit none

integer :: fromPatch, toPatch
!logical, external :: twoPointsMatch


integer :: f1, f2, f3, f4
integer :: t1, t2, t3, t4

integer :: n12f, n23f, n12t, n23t
integer :: edgeLength, fromEdge, toEdge
logical :: reverse

do fromPatch = 1, nPatches-1
    do toPatch = fromPatch + 1 , nPatches

        edgeLength = 0
        fromEdge = 0
        toEdge = 0

        n12f = size(edges(fromPatch)%pointIDs,2)
        n23f = size(edges(fromPatch)%pointIDs,1)

        n12t = size(edges(toPatch)%pointIDs,2)
        n23t = size(edges(toPatch)%pointIDs,1)

        f1 = abs(edges(fromPatch)%pointIDs(1,1))
        f2 = abs(edges(fromPatch)%pointIDs(1, n12f))
        f3 = abs(edges(fromPatch)%pointIDs(n23f,n12f))
        f4 = abs(edges(fromPatch)%pointIDs(n23f,1))

        t1 = abs(edges(toPatch)%pointIDs(1,1))
        t2 = abs(edges(toPatch)%pointIDs(1,n12t))
        t3 = abs(edges(toPatch)%pointIDs(n23t,n12t))
        t4 = abs(edges(toPatch)%pointIDs(n23t,1))

        call whichEdgesMatch(f1, f2, f3, f4, &
                             t1, t2, t3, t4, &
                             toEdge, fromEdge, reverse)

        if      (toEdge.eq.edge12 .or. toEdge.eq.edge43) then
            edgeLength = n12t
        else if (toEdge.eq.edge23 .or. toEdge.eq.edge14) then
            edgeLength = n23t
        endif

        if (fromEdge .ne. 0 .and. &
              toEdge .ne. 0) then
            call stitch(edgeLength, toPatch, toEdge, fromPatch, fromEdge, reverse)
        endif
    enddo
enddo

end subroutine

!-----------------------------------------------------------------------

subroutine whichEdgesMatch(f1, f2, f3, f4, t1, t2, t3, t4, toEdge, fromEdge, reverse)

use enumerates

implicit none

integer :: f1, f2, f3, f4
integer :: t1, t2, t3, t4

integer :: fromEdge, toEdge
logical :: reverse

fromEdge = 0
toEdge = 0


call checkSide(f1, f2, t1, t2, edge12, edge12, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f2, t2, t3, edge12, edge23, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f2, t4, t3, edge12, edge43, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f2, t1, t4, edge12, edge14, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return

call checkSide(f2, f3, t1, t2, edge23, edge12, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f2, f3, t2, t3, edge23, edge23, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f2, f3, t4, t3, edge23, edge43, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f2, f3, t1, t4, edge23, edge14, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return

call checkSide(f4, f3, t1, t2, edge43, edge12, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f4, f3, t2, t3, edge43, edge23, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f4, f3, t4, t3, edge43, edge43, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f4, f3, t1, t4, edge43, edge14, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return

call checkSide(f1, f4, t1, t2, edge14, edge12, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f4, t2, t3, edge14, edge23, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f4, t4, t3, edge14, edge43, toEdge, fromEdge, reverse)
if (fromEdge .ne. 0) return
call checkSide(f1, f4, t1, t4, edge14, edge14, toEdge, fromEdge, reverse)
! return anyway

end subroutine

!-----------------------------------------------------------------------

subroutine checkSide(fa, fb, ta, tb, ifMatchEdgeFrom, ifMatchEdgeTo, &
                     toEdge, fromEdge, reverse)

use meshdata, only: Q1Points

implicit none

integer :: fa, fb, ta, tb, ifMatchEdgeFrom, ifMatchEdgeTo
integer :: toEdge, fromEdge
logical :: reverse

logical, external :: twoPointsMatch

if      (twoPointsMatch(Q1Points(fa),Q1Points(ta)) .and. &
         twoPointsMatch(Q1Points(fb),Q1Points(tb))) then
    fromEdge = ifMatchEdgeFrom 
    toEdge = ifMatchEdgeTo
    reverse = .false.
else if (twoPointsMatch(Q1Points(fa),Q1Points(tb)) .and. &
         twoPointsMatch(Q1Points(fb),Q1Points(ta))) then
    fromEdge = ifMatchEdgeFrom
    toEdge = ifMatchEdgeTo 
    reverse = .true.
endif

end subroutine

!-----------------------------------------------------------------------
