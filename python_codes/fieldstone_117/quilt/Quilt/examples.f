! This file contains multiple examples of meshes made by Quilt.
! Feel free to add your own.

!-----------------------------------------------------------------------

subroutine meshTestSquare()

use topology
use enumerates
use meshData
use pointMod
use stitching, only: edges
use debug
use io, only: writeMeshtoFile

implicit none

type(point)        :: patchCorners(4)

integer, parameter :: nElems12 = 2
integer, parameter :: nElems23 = 2

! variables from meshData
nPatches = 1
nTotalPointsBeforeStitching = (nElems12+1) * (nElems23+1)
nTotalElements = nElems12 * nElems23

call initializeMesh()

patchCorners(1)%x = 0d0
patchCorners(1)%y = 0d0

patchCorners(2)%x = 0d0
patchCorners(2)%y = 6d0

patchCorners(3)%x = 6d0
patchCorners(3)%y = 6d0

patchCorners(4)%x = 6d0
patchCorners(4)%y = 0d0

call onePatch(1, nElems12, nElems23, &
              patchCorners(1), patchCorners(2), patchCorners(3), patchCorners(4), &
              0.0d0, 0.0d0, 0d0, 0d0,  &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

call setMeshOrientation(counterClockWise)

call writeMeshtoFile()

end subroutine


!-----------------------------------------------------------------------

subroutine meshTestRectangle()

use meshData
use pointMod
use stitching, only: edges
use io,    only: writeMeshtoFile

implicit none

type(point)        :: patchCorners(4)

integer, parameter :: nElems12 = 4
integer, parameter :: nElems23 = 4

! variables from meshData
nPatches = 1
nTotalPointsBeforeStitching = (nElems12+1) * (nElems23+1)
nTotalElements = nElems12 * nElems23

call initializeMesh()

patchCorners(1)%x = 0d0
patchCorners(1)%y = 0d0

patchCorners(2)%x = 0d0
patchCorners(2)%y = 1d0

patchCorners(3)%x = 1d0
patchCorners(3)%y = 1d0

patchCorners(4)%x = 1d0
patchCorners(4)%y = 0d0

call onePatch(1, nElems12, nElems23, &
              patchCorners(1), patchCorners(2), patchCorners(3), patchCorners(4), &
              0.0d0, 0.0d0, 0d0, 0d0,  &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

call writeMeshtoFile()

end subroutine


!-----------------------------------------------------------------------

subroutine meshTestSquareFocused()

use meshData
use pointMod
use stitching, only: edges
use io,    only: writeMeshtoFile

implicit none

type(point)        :: patchCorners(4)

integer, parameter :: nElems12 = 15
integer, parameter :: nElems23 = 15

! variables from meshData
nPatches = 1
nTotalPointsBeforeStitching = (nElems12+1) * (nElems23+1)
nTotalElements = nElems12 * nElems23

call initializeMesh()

patchCorners(1)%x = 0d0
patchCorners(1)%y = 0d0

patchCorners(2)%x = 0d0
patchCorners(2)%y = 1d0

patchCorners(3)%x = 1d0
patchCorners(3)%y = 1d0

patchCorners(4)%x = 1d0
patchCorners(4)%y = 0d0

call onePatch(1, nElems12, nElems23, &
              patchCorners(1), patchCorners(2), patchCorners(3), patchCorners(4), &
              0.0d0, 0.0d0, 0d0, 0d0,  &
              1.4d0, 1.4d0, 1.4d0, 1.4d0)

call writeMeshtoFile()

end subroutine

!-----------------------------------------------------------------------

subroutine meshTestWobble

use meshData
use pointMod
use stitching
use io,    only: writeMeshtoFile

type(point)        :: pointA, pointB, pointC, pointD

integer, parameter :: nElems12 = 10
integer, parameter :: nElems23 = 10

! variables from meshData
nPatches = 1
nTotalPointsBeforeStitching = (nElems12+1) * (nElems23+1)
nTotalElements = nElems12 * nElems23

call initializeMesh()

pointA%x = 0d0
pointA%y = 10d0

pointB%x = 15d0
pointB%y = 10d0

pointC%x = 15d0
pointC%y = 0d0

pointD%x = 0d0
pointD%y = 0d0

call onePatch(1, nElems12, nElems23, &
              pointA,pointB,pointC,pointD, &
              0.0d0, 0.5d0, 3d0, -2.11d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

call writeMeshtoFile()


end subroutine


!-----------------------------------------------------------------------

subroutine meshTestStitch

! stich three patches into a hexagon
! The edges should be stitched properly and can be verified.

use enumerates
use meshData
use pointMod
use stitching
use topology,  only: setMeshOrientation
use debug
use io,    only: writeMeshtoFile

implicit none

type(point)      :: cornerPoints(7)

integer :: iPatch
integer, parameter :: nElems12 = 4
integer, parameter :: nElems23 = 4

! variables from meshData
nPatches = 3
nTotalPointsBeforeStitching = 3 * (nElems12+1) * (nElems23+1)
nTotalElements = 3 * nElems12 * nElems23

call initializeMesh()

cornerPoints(1)%x = 0d0
cornerPoints(1)%y = 0d0

cornerPoints(2)%x = 0d0
cornerPoints(2)%y = 4d0

cornerPoints(3)%x = 4d0
cornerPoints(3)%y = 4d0

cornerPoints(4)%x = 4d0
cornerPoints(4)%y = 0d0

cornerPoints(5)%x = 4d0
cornerPoints(5)%y = 8d0

cornerPoints(6)%x = 8d0
cornerPoints(6)%y = 8d0

cornerPoints(7)%x = 8d0
cornerPoints(7)%y = 4d0

!----- patch 1
call onePatch(1, nElems12, nElems23, &
              cornerPoints(1),cornerPoints(2),cornerPoints(3),cornerPoints(4), &
              0.0d0, 0.0d0,0.0d0,0.0d0,  &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

!----- patch 2
call onePatch(2, nElems12, nElems23, &
              cornerPoints(2),cornerPoints(5),cornerPoints(6),cornerPoints(3), &
              0.0d0, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

!----- patch 3
call onePatch(3, nElems12, nElems23, &
              cornerPoints(4),cornerPoints(3),cornerPoints(6),cornerPoints(7), &
              0.0d0, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! edge between patch 1 (from 2 to 3) and patch 2 (from 1 to 4)
call stitch(nElems23+1, 2, edge14, 1, edge23, .false.)

! edge between patch 1 (from 4 to 3) and patch 3 (from 1 to 2)
call stitch(nElems23+1, 3, edge12, 1, edge43, .false.)

! edge between patch 2 (from 4 to 3) and patch 3 (from 2 to 3)
call stitch(nElems23+1, 3, edge23, 2, edge43, .false.)


call setMeshOrientation(counterClockWise)

call writeMeshToFile()

end subroutine

!-----------------------------------------------------------------------

subroutine meshFivePatches()

use enumerates
use meshData
use pointMod
use stitching
use debug
use io,    only: writeMeshtoFile

implicit none

type(point)      :: cornerPoints(10)

double precision, parameter :: circleRadius = 1d0
double precision, parameter :: squareSize = 2d0

double precision, parameter :: pi = 4d0 * atan(1d0)

integer, parameter :: nElems12 = 20
integer, parameter :: nElems23 = 20

double precision, parameter :: circleCurvature = circleRadius - circleRadius * (cos(pi/8d0))

! variables from meshData
nPatches = 5
nTotalPointsBeforeStitching = nPatches * (nElems12+1) * (nElems23+1)
nTotalElements = nPatches * nElems12 * nElems23

call initializeMesh()

cornerPoints(1)%x = 0d0
cornerPoints(1)%y = 0d0

cornerPoints(2)%x = 0d0
cornerPoints(2)%y = 0.5d0 * circleRadius

cornerPoints(3)%x = 0.4d0 * circleRadius
cornerPoints(3)%y = 0.4d0 * circleRadius

cornerPoints(4)%x = 0.5d0 * circleRadius
cornerPoints(4)%y = 0d0

cornerPoints(5)%x = 0d0
cornerPoints(5)%y = circleRadius

cornerPoints(6)%x = circleRadius / sqrt(2d0)
cornerPoints(6)%y = circleRadius / sqrt(2d0)

cornerPoints(7)%x = circleRadius
cornerPoints(7)%y = 0d0

cornerPoints(8)%x = 0d0
cornerPoints(8)%y = squareSize

cornerPoints(9)%x = squareSize
cornerPoints(9)%y = squareSize

cornerPoints(10)%x = squareSize
cornerPoints(10)%y = 0d0


! patch 1, central patch
call onePatch(1, nElems12, nElems23, &
              cornerPoints(1),cornerPoints(2),cornerPoints(3),cornerPoints(4), &
              0.0d0, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 2, top inside circle
call onePatch(2, nElems12, nElems23, &
              cornerPoints(2),cornerPoints(5),cornerPoints(6),cornerPoints(3), &
              0.0d0, circleCurvature,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 3, right inside circle
call onePatch(3, nElems12, nElems23, &
              cornerPoints(4),cornerPoints(3),cornerPoints(6),cornerPoints(7), &
              0.0d0, 0.0d0,-circleCurvature,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 4, left outside circle
call onePatch(4, nElems12, nElems23, &
              cornerPoints(5),cornerPoints(8),cornerPoints(9),cornerPoints(6), &
              0.0d0, 0.0d0,0.0d0,circleCurvature, &
              1.3d0, 1.0d0, 1.3d0, 1.0d0)

! patch 5, right outside circle
call onePatch(5, nElems12, nElems23, &
              cornerPoints(7),cornerPoints(6),cornerPoints(9),cornerPoints(10), &
              -circleCurvature, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.3d0, 1.0d0, 1.3d0)

! stitch the edges together

! 1, between patches 1 (p23) and 2 (p14))
call stitch(nElems23+1, 2, edge14, 1, edge23, .false.)

! 2, between patches 1 (p43) and 3 (p12)
call stitch(nElems23+1, 3, edge12, 1, edge43, .false.)

! 3, between patches 2 (p43) and 3 (p23)
call stitch(nElems23+1, 3, edge23, 2, edge43, .false.)

! 4, between patches 2 (p23) and 4 (p14)
call stitch(nElems23+1, 4, edge14, 2, edge23, .false.)

! 5, between patches 3 (p43) and 5 (p12)
call stitch(nElems23+1, 5, edge12, 3, edge43, .false.)

! 6, between patches 4 (p43) and 5 (p23)
call stitch(nElems23+1, 5, edge23, 4, edge43, .false.)



call writeMeshToFile()


end subroutine

!-----------------------------------------------------------------------

subroutine meshFivePatchesAutostitch()

use enumerates
use meshData
use pointMod
use stitching
use debug
use io,    only: writeMeshtoFile

implicit none

type(point) :: cornerPoints(10)

double precision, parameter :: circleRadius = 1d0
double precision, parameter :: squareSize = 2d0

double precision, parameter :: pi = 4d0 * atan(1d0)

integer, parameter :: nElems12 = 20
integer, parameter :: nElems23 = 20

double precision, parameter :: circleCurvature = circleRadius - circleRadius * (cos(pi/8d0))

! variables from meshData
nPatches = 5
nTotalPointsBeforeStitching = nPatches * (nElems12+1) * (nElems23+1)
nTotalElements = nPatches * nElems12 * nElems23

call initializeMesh()

cornerPoints(1)%x = 0d0
cornerPoints(1)%y = 0d0

cornerPoints(2)%x = 0d0
cornerPoints(2)%y = 0.5d0 * circleRadius

cornerPoints(3)%x = 0.4d0 * circleRadius
cornerPoints(3)%y = 0.4d0 * circleRadius

cornerPoints(4)%x = 0.5d0 * circleRadius
cornerPoints(4)%y = 0d0

cornerPoints(5)%x = 0d0
cornerPoints(5)%y = circleRadius

cornerPoints(6)%x = circleRadius / sqrt(2d0)
cornerPoints(6)%y = circleRadius / sqrt(2d0)

cornerPoints(7)%x = circleRadius
cornerPoints(7)%y = 0d0

cornerPoints(8)%x = 0d0
cornerPoints(8)%y = squareSize

cornerPoints(9)%x = squareSize
cornerPoints(9)%y = squareSize

cornerPoints(10)%x = squareSize
cornerPoints(10)%y = 0d0

! patch 1, central patch
call onePatch(1, nElems12, nElems23, &
              cornerPoints(1),cornerPoints(2),cornerPoints(3),cornerPoints(4), &
              0.0d0, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 2, top inside circle
call onePatch(2, nElems12, nElems23, &
              cornerPoints(2),cornerPoints(5),cornerPoints(6),cornerPoints(3), &
              0.0d0, circleCurvature,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 3, right inside circle
call onePatch(3, nElems12, nElems23, &
              cornerPoints(4),cornerPoints(3),cornerPoints(6),cornerPoints(7), &
              0.0d0, 0.0d0,-circleCurvature,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 4, left outside circle
call onePatch(4, nElems12, nElems23, &
              cornerPoints(5),cornerPoints(8),cornerPoints(9),cornerPoints(6), &
              0.0d0, 0.0d0,0.0d0,circleCurvature, &
              1.3d0, 1.0d0, 1.3d0, 1.0d0)

! patch 5, right outside circle
call onePatch(5, nElems12, nElems23, &
              cornerPoints(7),cornerPoints(6),cornerPoints(9),cornerPoints(10), &
              -circleCurvature, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.3d0, 1.0d0, 1.3d0)

! stitch the edges together with auto stitch.
call autoStitch()

call writeMeshToFile()


end subroutine



!-----------------------------------------------------------------------

subroutine meshThirteenPatches()

use topology
use enumerates
use meshData
use pointMod
use stitching
use debug
use io,    only: writeMeshtoFile

implicit none

type(point)      :: cornerPoints(22)

double precision, parameter :: circleRadius = 0.2d0
double precision, parameter :: squareSize = 1d0

double precision, parameter :: pi = 4d0 * atan(1d0)


! base element count for the central square!
integer, parameter :: nElemsBase = 35
integer, parameter :: nElemsDiagSpoke = int(nElemsBase * 0.8)
integer, parameter :: nElemsSmall = int(nElemsBase * 0.5)
integer, parameter :: nElemsFill = int( (squareSize - (2.5 * circleRadius)) * &
                                        nElemsBase / (1.57 * circleRadius) )


integer, parameter :: nPointsBase = nElemsBase + 1
integer, parameter :: nPointsDiagSpoke = nElemsDiagSpoke + 1
integer, parameter :: nPointsSmall = nElemsSmall + 1
integer, parameter :: nPointsFill = nElemsFill + 1

double precision, parameter :: circleCurvature = circleRadius - circleRadius * (cos(pi/8d0))
double precision, parameter :: wideCurvature = circleCurvature
double precision, parameter :: smallCurvature = circleCurvature * 0.3

double precision, parameter :: shortFocus = 1.3
double precision, parameter :: longFocus = 1.2
double precision, parameter :: edgeFocus = 1.1

integer :: tempInt, iElement

! variables from meshData
nPatches = 13


nTotalPointsBeforeStitching = 3 * nPointsBase * nPointsBase + & ! three inner circle patches
                 2 * nPointsBase    * nPointsDiagSpoke + & ! first ring
                 nPointsSmall * nPointsSmall + & ! keystone
                 2 * nPointsBase    * nPointsSmall  + & ! rest of second ring
                 2 * nPointsBase * nPointsFill + & ! long ones of outer ring
                 2 * nPointsSmall * nPointsFill +  & ! middle ones of outer ring
                 nPointsFill * nPointsFill ! central one of outer ring

nTotalElements = 3 * nElemsBase * nElemsBase + & ! three inner circle patches
                 2 * nElemsBase * nElemsDiagSpoke + & ! first ring
                 nElemsSmall * nElemsSmall + & ! keystone
                 2 * nElemsBase * nElemsSmall + & ! rest of second ring
                 2 * nElemsBase * nElemsFill + & ! long ones of outer ring
                 2 * nElemsSmall * nElemsFill + & ! middle ones of outer ring
                 nElemsFill * nElemsFill ! central one of outer ring

call initializeMesh()

! main points that define the corners of the patches

cornerPoints(1)%x = 0d0
cornerPoints(1)%y = 0d0

cornerPoints(2)%x = 0d0
cornerPoints(2)%y = 0.5d0 * circleRadius

cornerPoints(3)%x = 0.4d0 * circleRadius
cornerPoints(3)%y = 0.4d0 * circleRadius

cornerPoints(4)%x = 0.5d0 * circleRadius
cornerPoints(4)%y = 0d0

cornerPoints(5)%x = 0d0
cornerPoints(5)%y = circleRadius

cornerPoints(6)%x = circleRadius / sqrt(2d0)
cornerPoints(6)%y = circleRadius / sqrt(2d0)

cornerPoints(7)%x = circleRadius
cornerPoints(7)%y = 0d0

cornerPoints(8)%x = 0d0
cornerPoints(8)%y = 1.84d0 * circleRadius

cornerPoints(9)%x = 2.1d0 * circleRadius / sqrt(2d0)
cornerPoints(9)%y = 2.1d0 * circleRadius / sqrt(2d0)

cornerPoints(10)%x = 1.84d0 * circleRadius
cornerPoints(10)%y = 0d0

cornerPoints(11)%x = 0d0
cornerPoints(11)%y = 2.5d0 * circleRadius

cornerPoints(12)%x = 1.57d0 * circleRadius
cornerPoints(12)%y = 2.5d0 * circleRadius

cornerPoints(13)%x = 2.5d0 * circleRadius
cornerPoints(13)%y = 2.5d0 * circleRadius

cornerPoints(14)%x = 2.5d0 * circleRadius
cornerPoints(14)%y = 1.57d0 * circleRadius

cornerPoints(15)%x = 2.5d0 * circleRadius
cornerPoints(15)%y = 0d0

cornerPoints(16)%x = 0d0
cornerPoints(16)%y = squareSize

cornerPoints(17)%x = 1.57d0 * circleRadius
cornerPoints(17)%y = squareSize 

cornerPoints(18)%x = 2.5d0 * circleRadius
cornerPoints(18)%y = squareSize

cornerPoints(19)%x = squareSize
cornerPoints(19)%y = squareSize

cornerPoints(20)%x = squareSize
cornerPoints(20)%y = 2.5d0 * circleRadius

cornerPoints(21)%x = squareSize
cornerPoints(21)%y = 1.57d0 * circleRadius

cornerPoints(22)%x = squareSize
cornerPoints(22)%y = 0d0

! the patches themselves

!----------- inside circle

! patch 1, central patch
call onePatch(1, nElemsBase, nElemsBase, &
              cornerPoints(1),cornerPoints(2),cornerPoints(3),cornerPoints(4), &
              0.0d0, 0.0d0,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 2, top inside circle
call onePatch(2, nElemsBase, nElemsBase, &
              cornerPoints(2),cornerPoints(5),cornerPoints(6),cornerPoints(3), &
              0.0d0, circleCurvature,0.0d0,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 3, right inside circle
call onePatch(3, nElemsBase, nElemsBase, &
              cornerPoints(4),cornerPoints(3),cornerPoints(6),cornerPoints(7), &
              0.0d0, 0.0d0,-circleCurvature,0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

!----------- first ring 


! patch 4, left outside circle
call onePatch(4, nElemsDiagSpoke, nElemsBase, &
              cornerPoints(5),cornerPoints(8),cornerPoints(9),cornerPoints(6), &
              0.0d0, wideCurvature,0.0d0,circleCurvature, &
              1.2d0, 1.0d0, 1.2d0, 1.0d0)

! patch 5, right outside circle
call onePatch(5, nElemsBase, nElemsDiagSpoke, &
              cornerPoints(7),cornerPoints(6),cornerPoints(9),cornerPoints(10), &
              -circleCurvature, 0.0d0,-wideCurvature,0.0d0, &
              1.0d0, 1.2d0, 1.0d0, 1.2d0)

! ----------- second ring 

! patch 6, left outside circle
call onePatch(6, nElemsSmall, nElemsBase, &
              cornerPoints(8),cornerPoints(11),cornerPoints(12),cornerPoints(9), &
              0.0d0, 0d0, -smallCurvature, wideCurvature, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 7, left outside circle
call onePatch(7, nElemsSmall, nElemsSmall, &
              cornerPoints(9),cornerPoints(12),cornerPoints(13),cornerPoints(14), &
              -smallCurvature, 0.0d0, 0.0d0,smallCurvature, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 8, left outside circle
call onePatch(8, nElemsBase, nElemsSmall, &
              cornerPoints(10),cornerPoints(9),cornerPoints(14),cornerPoints(15), &
              -wideCurvature, smallCurvature, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! ---- outer field

! patch 9, left outside circle
call onePatch(9, nElemsFill, nElemsBase, &
              cornerPoints(11),cornerPoints(16),cornerPoints(17),cornerPoints(12), &
              0.0d0, 0.0d0, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 10, left outside circle
call onePatch(10, nElemsFill, nElemsSmall, &
              cornerPoints(12),cornerPoints(17),cornerPoints(18),cornerPoints(13), &
              0.0d0, 0.0d0, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 11, left outside circle
call onePatch(11, nElemsFill, nElemsFill, &
              cornerPoints(13),cornerPoints(18),cornerPoints(19),cornerPoints(20), &
              0.0d0, 0.0d0, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 12, left outside circle
call onePatch(12, nElemsSmall, nElemsFill, &
              cornerPoints(14),cornerPoints(13),cornerPoints(20),cornerPoints(21), &
              0.0d0, 0.0d0, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)

! patch 13, left outside circle
call onePatch(13, nElemsBase, nElemsFill, &
              cornerPoints(15),cornerPoints(14),cornerPoints(21),cornerPoints(22), &
              0.0d0, 0.0d0, 0.0d0, 0.0d0, &
              1.0d0, 1.0d0, 1.0d0, 1.0d0)


! stitch the edges together with auto stitch.
call autoStitch()

! turn connectivity counter clockwise by exchanging points 1 and 3
do iElement = 1, nTotalElements
    tempInt = Q1connectivity(iElement, 1)
    Q1connectivity(iElement, 1) = Q1connectivity(iElement, 3)
    Q1connectivity(iElement, 3) = tempInt
enddo

call setMeshOrientation(counterClockWise)

call writeMeshToFile()

end subroutine

