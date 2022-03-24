module meshData

use pointMod
use stitching, only: edges, removedPointIDs

implicit none

integer                  :: nPoints, nElements, nPatches, nStitches

integer,     allocatable :: Q1connectivity(:,:)
type(point), allocatable :: Q1Points(:)

integer,     allocatable :: finalConnectivity(:,:)
type(point), allocatable :: finalPoints(:)
integer                  :: finalnPoints


integer                  :: countedPoints
integer                  :: countedElements

integer                  :: nUniquePoints

integer                  :: nTotalElements
integer                  :: nTotalPointsBeforeStitching
integer                  :: nTotalPointsAfterStitching

double precision, allocatable :: elementArea(:)
double precision, allocatable :: elementAspectRatio(:)

contains

subroutine initializeMesh()

implicit none

integer    :: iPoint

nPoints = 0
nElements = 0
countedPoints = 0
countedElements = 0
nUniquePoints = 0

nStitches = 0

nTotalPointsAfterStitching = 0

if (nTotalPointsBeforeStitching.eq.0) then
    stop "called initializeMesh before setting nTotalPointsBeforeStitching"
endif

if (nTotalElements.eq.0) then
    stop "called initializeMesh before setting nTotalElements"
endif

if (nPatches.eq.0) then
    stop "called initializeMesh before setting nPatches"
endif

allocate(Q1Points(nTotalPointsBeforeStitching))
allocate(Q1connectivity(nTotalElements,4))
allocate(edges(nPatches))
allocate(removedPointIDs(nTotalPointsBeforeStitching))

do iPoint = 1, nTotalPointsBeforeStitching
    Q1Points(iPoint)%x = 0d0
    Q1Points(iPoint)%y = 0d0
enddo
Q1connectivity = 0
! edges can only be set to 0 once edges(i)%pointIDs(:,:) has been allocated
removedPointIDs = 0

end subroutine

subroutine elementInfo()

implicit none

integer :: iElement

double precision :: p1x, p1y
double precision :: p2x, p2y  
double precision :: p3x, p3y  
double precision :: p4x, p4y  

double precision :: v1x, v1y  
double precision :: v2x, v2y  

double precision :: l1, l2  

allocate(elementArea(nTotalElements))
allocate(elementAspectRatio(nTotalElements))

do iElement = 1, nTotalElements

    p1x = Q1Points(Q1connectivity(iElement,1))%x
    p1y = Q1Points(Q1connectivity(iElement,1))%y
    p2x = Q1Points(Q1connectivity(iElement,2))%x
    p2y = Q1Points(Q1connectivity(iElement,2))%y
    p3x = Q1Points(Q1connectivity(iElement,3))%x
    p3y = Q1Points(Q1connectivity(iElement,3))%y
    p4x = Q1Points(Q1connectivity(iElement,4))%x
    p4y = Q1Points(Q1connectivity(iElement,4))%y

    ! first the area
    v1x = p3x - p1x
    v1y = p3y - p1y
    v2x = p3x - p2x
    v2y = p3y - p2y

    elementArea(iElement) = abs(v1x * v2y - v1y * v2x)

    ! and the aspect ratio is computed as the
    ! length of the longest diagonal divided my the
    ! length of the shortest diagonal

    l1 = sqrt((p1x - p3x)**2 + (p1y - p3y)**2)
    l2 = sqrt((p2x - p4x)**2 + (p2y - p4y)**2)

    elementAspectRatio(iElement) = l1 / l2
    if (elementAspectRatio(iElement) .lt. 1d0) then
        elementAspectRatio(iElement) = &
           1d0 / elementAspectRatio(iElement)
    endif

enddo

end subroutine

subroutine cleanMeshdata()

implicit none

if (allocated(Q1connectivity)) then
	deallocate(Q1connectivity)
endif
if (allocated(Q1Points)) then
deallocate(Q1Points)
endif
if (allocated(finalConnectivity)) then
deallocate(finalConnectivity)
endif
if (allocated(finalPoints)) then
deallocate(finalPoints)
endif
if (allocated(elementArea)) then
deallocate(elementArea)
endif
if (allocated(elementAspectRatio)) then
deallocate(elementAspectRatio)
endif

end subroutine


end module

