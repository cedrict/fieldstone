module io

logical :: writeSimple
logical :: writeParaview
logical :: writeGMT

contains

subroutine writeMeshToFile()

use stitching,  only: cleanStitchData
use meshData,   only: cleanMeshData, &
                      finalConnectivity, &
                      finalPoints, &
                      finalnPoints, &
                      nTotalElements, &
                      Q1Points, &
                      Q1connectivity, &
                      nTotalPointsAfterStitching                      
use enumerates, only: Q0, Q1, &
                      elementType, &
                      meshNames, &
                      initializeMeshNames
use highQ,      only: q1toqn

implicit none

integer :: iElement

! The elements are clockwise by default.
! If counter clockwise direction is desired, this has to be rotated.

call initializeMeshNames()

! remove artifacts from the stitching
call removeConnectivityGaps()

if (nTotalElements.eq.0) then
    stop "Attempting to write a mesh without elements."
endif

if (nTotalPointsAfterStitching.eq.0) then
    stop "Attempting to write a mesh without points."
endif

!-------------------------------------------------------------------
! If the example requires it, upgrade the elements
! to a different order.

if (elementType.eq.Q0) then
    allocate(finalConnectivity(nTotalElements,1))
    allocate(finalPoints(nTotalElements))
    do iElement = 1, nTotalElements
        ! take the mid point of each element as its only point
        finalPoints(iElement)%x = sum(Q1Points(Q1connectivity(iElement,:))%x) / 4d0
        finalPoints(iElement)%y = sum(Q1Points(Q1connectivity(iElement,:))%y) / 4d0
        finalConnectivity(iElement,1) = iElement
    enddo
    finalnPoints = nTotalElements

else if (elementType.eq.Q1) then
    ! no upgrading required. This is the default
    allocate(finalConnectivity(nTotalElements,4))
    allocate(finalPoints(nTotalPointsAfterStitching))
    finalConnectivity = Q1connectivity
    finalPoints = Q1Points
    finalnPoints = nTotalPointsAfterStitching
else
    ! q1toqn takes care of the allocation of
    ! finalConnectivity and finalPoints
    call q1toqn()
endif

!-------------------------------------------------------------------

write(*,*) "----------------------------------------------------------"
write(*,*) "Created mesh with: "
write(*,*) finalnPoints, "nodal points and"
write(*,*) nTotalElements, "elements of type " // meshNames(elementType)
write(*,*) "----------------------------------------------------------"



if (writeSimple) then
    call writeMesh2Simple
endif
if (writeParaview) then
    call writeMesh2Paraview
endif
if (writeGMT) then
    call writeMesh2GMT
endif


! clean up allocations, in case another mesh is requested in the same run

call cleanStitchData
call cleanMeshData

end subroutine

!-----------------------------------------------------------------

subroutine writeMesh2GMT()

use meshData,   only: nTotalElements, &
                      Q1connectivity, &
                      nTotalPointsAfterStitching, &
                      Q1Points, &
                      finalConnectivity, &
                      finalPoints, &
                      finalnPoints, &
                      nPatches
use stitching,  only: edges
use enumerates, only: Q0, &
                      elementType, &
                      meshNames
use debug

! for GMT we need a file of points coordinates and one with connectivity

implicit none

integer            :: iElement
integer, parameter :: fpGrid = 62
integer, parameter :: fpAllPoints = 63
integer, parameter :: fpEdges = 64
integer, parameter :: fpMain = 65

integer            :: size12, size23
integer            :: iPatch, iPoint

character(len=14)  :: gridFileName
character(len=19)  :: allPointsFileName
character(len=15)  :: edgesFileName
character(len=14)  :: GMTFileName

! plot data
integer, parameter :: nLargeTicks=5

double precision   :: xmin, xmax
double precision   :: ymin, ymax

integer, parameter :: stringLength = 20
character(len=7)   :: stringFormat = "(e14.6)"


character(len=stringLength) :: xMinString, xMaxString
character(len=stringLength) :: yMinString, yMaxString   
character(len=stringLength) :: bigXTickString, smallXTickString
character(len=stringLength) :: bigYTickString, smallYTickString
double precision   :: bigXTick, smallXTick
double precision   :: bigYTick,    smallYTick
double precision, parameter :: xPadding = 0.02d0
double precision, parameter :: yPadding = 0.02d0


gridFileName      =  "GMTgrid_"      // meshNames(elementType) // ".dat"
allPointsFileName =  "GMTallPoints_" // meshNames(elementType) // ".dat"
edgesFileName     =  "GMTedges_"     // meshNames(elementType) // ".dat"

GMTFileName       =  "GMTmain_"      // meshNames(elementType) // ".gmt"


! write a grid file for a simple quadrilateral plot
open(unit = fpgrid, file = gridFileName)
do iElement = 1, nTotalElements

    write(fpgrid,"(1a)") ">"

    if (elementType.eq.Q0) then
        write(fpgrid,*) FinalPoints(iElement)%x, FinalPoints(iElement)%y
    else 
        do iPoint = 1,4
            write(fpgrid,*) Q1Points(Q1connectivity(iElement, iPoint))%x, &
                            Q1Points(Q1connectivity(iElement, iPoint))%y 
        enddo
    endif
enddo

close(fpgrid)

! write a file with all the points

open(unit = fpAllPoints, file = allPointsFileName)
do iPoint = 1, FinalnPoints
    write(fpAllPoints,*) FinalPoints(iPoint)%x, FinalPoints(iPoint)%y
enddo
close(fpAllPoints)

! write a file with the edges, 
! so it easy to plot where meshes have been stitched


! not yet working. Must be fixed if edges between patches 
! must be plotted. Not really important right now, but nice to have

open(unit = fpEdges, file = edgesFileName)


if (elementType .ne. Q0) then

  do iPatch = 1, nPatches

  

!    write(*,*) "final patch ", iPatch
!    call showEdges(iPatch)

    size12 = size(edges(iPatch)%pointIDs,2)
    size23 = size(edges(iPatch)%pointIDs,1)

    write(fpEdges,"(1a)") ">"

    do iPoint=1,size23
        write(fpEdges,*) FinalPoints(abs(edges(iPatch)%pointIDs(iPoint,1)))%x, &
                         FinalPoints(abs(edges(iPatch)%pointIDs(iPoint,1)))%y
    enddo

    do iPoint=2,size12-1
        write(fpEdges,*) FinalPoints(abs(edges(iPatch)%pointIDs(size23,iPoint)))%x, &
                         FinalPoints(abs(edges(iPatch)%pointIDs(size23,iPoint)))%y
    enddo

    do iPoint=1,size23
        write(fpEdges,*) FinalPoints(abs(edges(iPatch)%pointIDs(size23-iPoint+1,size12)))%x, &
                         FinalPoints(abs(edges(iPatch)%pointIDs(size23-iPoint+1,size12)))%y
    enddo

    do iPoint=2,size12-1
        write(fpEdges,*) FinalPoints(abs(edges(iPatch)%pointIDs(1,size12-iPoint+1)))%x, &
                         FinalPoints(abs(edges(iPatch)%pointIDs(1,size12-iPoint+1)))%y
    enddo

    ! close the loop
    write(fpEdges,*) FinalPoints(abs(edges(iPatch)%pointIDs(1,1)))%x, &
                     FinalPoints(abs(edges(iPatch)%pointIDs(1,1)))%y

   enddo

endif

close(fpEdges)

!-------------------------------------


! write the actual GMT script to plot the postscript files of the mesh

! first we need to estimate a decent tick size and edges of the domain.

xmin = minval(Q1Points(:)%x)
xmax = maxval(Q1Points(:)%x)
ymin = minval(Q1Points(:)%y)
ymax = maxval(Q1Points(:)%y)

call setLinearTicks(xmin, xmax, nLargeTicks, 0.1d0, bigXTick)
call setLinearTicks(ymin, ymax, nLargeTicks, 0.1d0, bigYTick)

smallXtick = 0.5 * bigXTick
smallYtick = 0.5 * bigYTick

write(xMinString,stringFormat) xMin - (xMax - xMin) * xPadding
write(xMaxString,stringFormat) xMax + (xMax - xMin) * xPadding
write(yMinString,stringFormat) yMin - (yMax - yMin) * yPadding
write(yMaxString,stringFormat) yMax + (yMax - yMin) * yPadding


write(bigXTickString,stringFormat) bigXtick
write(smallXTickString,stringFormat) smallXtick
write(bigYTickString,stringFormat) bigYtick
write(smallYTickString,stringFormat) smallYtick




open(unit = fpMain, file = GMTFileName)

write(fpMain,*) "#!/bin/sh -x"

write(fpMain,"(a)") "xmin=" // trim(adjustl(xMinString))
write(fpMain,"(a)") "xmax=" // trim(adjustl(xMaxString))

write(fpMain,"(a)") "ymin=" // trim(adjustl(yMinString))
write(fpMain,"(a)") "ymax=" // trim(adjustl(yMaxString))


write(fpMain,"(a)") "PROJ=-JX15c/15c"
write(fpMain,"(a)") "LABEL=-Ba" // trim(adjustl(bigXTickString))   // "f" &
                                // trim(adjustl(smallXTickString)) // &
                      ":'x':/a" // trim(adjustl(bigYTickString))   // "f" &
                                // trim(adjustl(smallYTickString)) //":'y':"

write(fpMain,"(a)") "VIEW="

write(fpMain,"(a)") "VERBOSE="
write(fpMain,"(a)") "FRAME=-R${xmin}/${xmax}/${ymin}/${ymax}"
write(fpMain,"(a)") "gmt set PS_MEDIA A4"

write(fpMain,"(a)") "gmt psbasemap     ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -K > grid.ps || exit 1"
write(fpMain,"(a)") "gmt psxy " // gridFileName // &
                    " ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -W0.1,black -K -O >> grid.ps || exit 1"

write(fpMain,"(a)") "gmt psbasemap     ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -K > allPoints.ps || exit 1"
write(fpMain,"(a)") "gmt psxy " // gridFileName // &
                    " ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -W0.1,black -K  -O >> allPoints.ps || exit 1"
write(fpMain,"(a)") "gmt psxy " // allPointsFileName // &
                    " ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -Sp   -K -O >> allPoints.ps || exit 1"


if (elementType .ne. Q0) then
! edges still does not really work if the edges are not existing points
write(fpMain,"(a)") "gmt psbasemap     ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -K > patches.ps || exit 1"
write(fpMain,"(a)") "gmt psxy " // gridFileName // &
                    " ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -W0.1,black -K  -O >> patches.ps || exit 1"
write(fpMain,"(a)") "gmt psxy " // edgesFileName // &
                    " ${PROJ} ${FRAME} ${LABEL} ${VIEW} ${VERBOSE} -W1.0,red  -K -O >> patches.ps || exit 1"
endif

close(fpMain)

!-------------------------------------

write(*,*) "Written grid for GMT to            " // gridFileName
write(*,*) "Written edges for GMT to           " // edgesFileName
write(*,*) "Written all points for GMT to      " // allPointsFileName
write(*,*) "Written GMT file to generate plots " // GMTFileName
write(*,*) "----------------------------------------------------------"


end subroutine

!-----------------------------------------------------------------


subroutine writeMesh2Paraview()

use meshData,   only: nTotalElements, &
                      Q1connectivity, &
                      nTotalPointsAfterStitching, &
                      Q1Points, &
                      finalConnectivity, &
                      finalPoints, &
                      finalnPoints, &
                      elementArea, &
                      elementAspectRatio, &
                      elementInfo
use enumerates
use pointMod
use debug

implicit none

character(len=15)  :: vtuFileName 
character(len=19)  :: connectivityFileName
character(len=18)  :: coordinatesFileName


character(len=12)  :: nPoints_char, nElements_char
integer, parameter :: fpvtu = 42
integer, parameter :: fpcoords = 43
integer, parameter :: fpconn = 44

integer            :: iPoint, iElement

integer            :: nPointsPerElem


! The elements are clockwise by default.
! If counter clockwise direction is desired, this has to be rotated.


vtuFileName =          "paraview_"     // meshNames(elementType) // ".vtu"

! compute area and aspect ratio of elements
call elementInfo()

write (nPoints_char, *) finalnPoints
write (nElements_char, *) nTotalElements

open(unit=fpvtu, file=vtuFileName)

! XML header
write(fpvtu,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(fpvtu,*) '   <UnstructuredGrid>'
write(fpvtu,*) '      <Piece NumberOfPoints="'//nPoints_char//'" NumberOfCells="'//nElements_char//' ">'

! element info
write(fpvtu,*) '         <CellData Scalars="scalars">'
write(fpvtu,*) '             <DataArray type="Float32" NumberOfComponents="1" Name="Area" Format="ascii">'
do iElement = 1, nTotalElements
    write(fpvtu,*) elementArea(iElement)
enddo
write(fpvtu,*) '             </DataArray>'
write(fpvtu,*) '             <DataArray type="Float32" NumberOfComponents="1" Name="Aspect ratio" Format="ascii">'
do iElement = 1, nTotalElements
    write(fpvtu,*) elementAspectRatio(iElement)
enddo
write(fpvtu,*) '             </DataArray>'
write(fpvtu,*) '         </CellData>'


! coordinates of the mesh points
write(fpvtu,*) '         <Points>'
write(fpvtu,*) '             <DataArray type="Float32" NumberOfComponents="3" Name="Vertex coordinates" Format="ascii">'

do iPoint = 1, FinalnPoints
    write(fpvtu,*) FinalPoints(iPoint)%x, FinalPoints(iPoint)%y, 0
enddo
write(fpvtu,*) '            </DataArray>'
write(fpvtu,*) '         </Points>'

! connectivity
write(fpvtu,*) '         <Cells>'
write(fpvtu,*) '             <DataArray type="UInt64" Name="connectivity" Format="ascii">'

do iElement = 1, nTotalElements
    write(fpvtu,*) Finalconnectivity(iElement, :)-1
enddo
write(fpvtu,*) '            </DataArray>'

! necessary evil
write(fpvtu,*) '            <DataArray type="UInt32" Name="offsets" Format="ascii">'
do iElement = 1, nTotalElements
   if (elementType.eq.Q0) then
       write(fpvtu,*)  1 * iElement
   else if (elementType.eq.Q1) then
       write(fpvtu,*)  4 * iElement
   else if (elementType.eq.Q2) then
       write(fpvtu,*)  9 * iElement
   else if (elementType.eq.Q3) then
       write(fpvtu,*) 16 * iElement
   else if (elementType.eq.Q4) then
       write(fpvtu,*) 25 * iElement
   else 
      stop "whaaaa, unknown element"
   endif
enddo
write(fpvtu,*) '            </DataArray>'

write(fpvtu,*) '            <DataArray type="Int32" Name="types" Format="ascii">'
do iElement = 1, nTotalElements-1
   ! 1 is paraviewinese for a single point.
   ! 9 is paraviewinese for quad element.
   ! 4 is paraviewinese for a polyline
   ! see page 9 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
   if (elementType.eq.Q0) then
       write(fpvtu,'(a2)',advance="no") '1 '
   else if (elementType.eq.Q1) then
       write(fpvtu,'(a2)',advance="no") '9 ' ! write them all on one line to keep it compact
   else if (elementType.eq.Q2 .or. &
            elementType.eq.Q3 .or. &
            elementType.eq.Q4) then
       write(fpvtu,'(a2)',advance="no") '4 ' ! write them all on one line to keep it compact
   else
      stop "whaaaa, unknown element"
   endif
enddo
if (nTotalElements.gt.0) then
   if (elementType.eq.Q0) then
      write(fpvtu,'(a1)') '1'
   else if (elementType.eq.Q1) then
      write(fpvtu,'(a1)') '9'
   else if (elementType.eq.Q2 .or. &
            elementType.eq.Q3 .or. &
            elementType.eq.Q4) then
      write(fpvtu,'(a1)') '4'
   else
      stop "whaaaa, unknown element"
   endif
endif
write(fpvtu,*) '            </DataArray>'

write(fpvtu,*) '         </Cells>'
write(fpvtu,*) '      </Piece>'
write(fpvtu,*) '   </UnstructuredGrid>'
write(fpvtu,*) '</VTKFile>'

close(fpvtu)

write(*,*) "Written mesh to VTU file           " // vtuFileName
write(*,*) "----------------------------------------------------------"

end subroutine

!-----------------------------------------------------------------------

subroutine writeMesh2Simple()

use meshData,   only: FinalnPoints, &
                      nTotalElements, &
                      FinalPoints, &
                      Finalconnectivity
use enumerates, only: elementType, &
                      meshNames

implicit none

integer :: iPoint, iElement
integer, parameter :: fpcoords = 60
integer, parameter :: fpconn   = 61

character(len=19)  :: connectivityFileName
character(len=18)  :: coordinatesFileName

connectivityFileName = "connectivity_" // meshNames(elementType) // ".dat"
coordinatesFileName =  "coordinates_"  // meshNames(elementType) // ".dat"

open(unit=fpcoords, file=coordinatesFileName)
write(fpcoords,*) FinalnPoints
do iPoint = 1, FinalnPoints
    write(fpcoords,*) FinalPoints(iPoint)%x, FinalPoints(iPoint)%y
enddo
close(fpcoords)


open(unit=fpconn, file=connectivityFileName)
write(fpconn,*) nTotalElements
do iElement = 1, nTotalElements
    write(fpconn,*) Finalconnectivity(iElement, :)
enddo
close(fpconn)

write(*,*) "Written mesh points coordinates to " // coordinatesFileName
write(*,*) "Written mesh connectivity to       " // connectivityFileName
write(*,*) "----------------------------------------------------------"


end subroutine

!-----------------------------------------------------------------------
! Below here are help routines to generates domain and tick size 
! for the GMT plot.



subroutine setLinearTicks(min, max, nLargeTicks, margin, tickSize)

implicit none

double precision   :: min, max
integer            :: nLargeTicks
double precision   :: tickSize
double precision   :: margin ! padding on eah side, as a fraction of the total interval length

double precision   :: intervalLength
double precision   :: totalRange, powerRange, factor, power, rangeModified

!double precision   :: bigYtick, smallYtick

intervalLength = max - min

totalRange = (max - min) * (1d0 + 2d0 * margin) / dble(nLargeTicks)
powerRange = log10(totalRange)
factor = dble(nint(-powerRange+1.0))
power = dble(10**factor)
rangeModified = dble(nint(totalRange * power))

tickSize = rangeModified / power

end subroutine





end module
