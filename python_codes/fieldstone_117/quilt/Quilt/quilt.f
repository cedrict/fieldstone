program quilt

!--------------------------------------------------------------------------------
! Welcome to Quilt, the patchwork mesher.
!
! This program makes quadrilateral meshes.
! It does this my making (possible deformed, possibly multiple) 
! patches of quadrilaterals and stitching those together to form
! one big mesh.
!
! These patches are constructed of circle sections and straight 
! lines to enable a wide spectrum of possible meshes.
!
! A few examples (that are also used for testing this code)
! are in file examples.f.
! Such an example can be run by selecting the proper meshType in
! this main program below. Extended documentation on the two main 
! subroutines 'patch' and 'stitch' follows below.
!
!
! Written by Lukas van de Wiel, 02 Feb 2022.
!            (l.y.vandewiel@uu.nl)
!
!--------------------------------------------------------------------------------
!
! The workflow of this program is as follows:
! 1: define the corner points of each patch
! 2: create all of the patches
! 3: stitch the patches
! 4: write the mesh to file.
!
!--------------------------------------------------------------------------------
!
! More details on these four steps are give below:
! 1: This code contains a derived type 'point',
!    that has a x and a y member. This class is used
!    extensive throughout the code. The corner points
!    must be stored in those form. See the examples
!    for how this works.
!
! 2: A patch is created with a call to onePatch. It contains a lot 
!    of variables:
!
!      Input:
!
!      patchID:        the sequence number of the patch
!      nElems12:       the number of elements from point 1 to 2
!      nElems23:       the number of elements from point 2 to 3
!      p1, p2, p3, p4: the corner points.
!      ca, cb, cc, cd: the curvature of the side. This quantifies 
!                      how much the circle section sticks out at
!                      middle between the two points it connects.
!                      A positive curvature implies a curve that 
!                      extends to the left, and a negative curvature
!                      implies a curve to the right.
!                      These 'from' and 'to' points are:
!                      1 to 2, 2 to 3, 4 to 3, 1 to 4.
!                      So two opposite sides go in the same direction,
!                      in stead of the from-to cycling round.
!      fa, fb, fc, fd: The focus of the points on each side.
!                      The equidistand positions are scaled to the
!                      [0,1] interval, and projected on the function
!                      y = x^f
!                      When fx is 1, the points on side x are
!                      equidistant. For smaller f, the points get closer
!                      to the 'to' point, and for bigger f, they shift
!                      closer to the 'from' point. This is very convenient
!                      for strongly deformed patches and helps to
!                      prevent highly elongated elements.
!
!      Output:
!
!      points:         A two dimensional array containing the points
!                      of the new patch.
!      connectivity:   The connectivity of the elements, also in a 2D array.
!
!    p2 ----------------------------- p3
!     |   |   |   |   |   |   |   |   |
!     |-         nElems23 = 8        -|
!     |                               |
!     |-                             -|
!     |  nElems12 = 5                 |
!     |-                             -|
!     |                               |
!     |-                             -|
!     |   |   |   |   |   |   |   |   |
!    p1 ----------------------------- p4
!
! 3: stitching the patches
!
!    There are two possible ways to stitch:
!    1) if all sides of patches that are adjacent are connected, they can
!       be stitched using a call to: autoStitch() that takes no arguments.
!    2) if not all adjacent are connected, such as might be the case with
!       with slippery interfaces, the sides that actually connected must
!       be stitched manually using a command to stitch() for every stitch.
!
!    Two patches are stitched together with a call to the commant: stitch.
!    During this process, the points numbers of adjacent patches are matched,
!    by replacing the point IDs from one patch by those from the other.
!    The connectivity is adjusted to accomodate this change.
!
!    It contains a lot of variables:
!
!      Input: 
!
!      edgeLength:     The number of element on the side being stitched
!      oldPatchID:     The number of one patch. Numbers start counting at 1
!      oldEdgeID:      The edge of this patch. This can be one of:
!                      edge12, edge23, edge14, edge43, that correspond to
!                      integers in enumerates.f
!      newPatchID:     The number of the other patch.
!      newEdgeID:      The edge of this patch.
!      reverse:        The logical to indicate whether the from and to points
!                      of both sides run in the same direction (.false.) or 
!                      in opposite direction (.true.)
!
!      Output:
!                      Output is stored in the global data in the meshData 
!                      module. The subroutine itself provides no output.
!
!    The patches must be stitched in such a way that higher number
!    points are overwritten by lower number points.
!    As points numbers increase by the patch sequence, this means that
!    Higher number patches must always be overwritten by lower number 
!    patches.
!
!    To accomplish this, the 'old' patch in the call to stitch must always
!    be the higher number patch, and the 'new' one always be the lower one.
!    On corners where many patches can mean, a single points can propagate
!    through multiple patches. To ensuire this works properly, first all
!    the stitches involving patch 1 must be executed, than all remaining
!    stitched with patch 2, then all remaining stitched with patch 3,
!    and so down the sequence.
!
! 4: A call to subroutine writeMeshToFile writes the mesh to file:
!
!    It takes no arguments, but uses the global data in the meshdata module
!
!    The mesh writes data to three files:
!
!    1) awesomeMesh.vtu       That can be used to display the mesh in Paraview.
!                             A z-coordinate of 0 added to display the 2D meshes
!                             in this 3D-tool
!    2) pointCoordinates.dat
!    3) connectivity.dat      Starts counting at 1, as opposed to Paraview which
!                             starts at 0.
!--------------------------------------------------------------------------------

use enumerates
use io

implicit none

! run several test routines... or not.
!logical,     parameter   :: testMe = .true.
logical,     parameter   :: testMe = .false.

! test meshes are not modified by the parameters here
! (See enumerates.f for these values, and examples.f for the actual routines)
!integer,     parameter   :: meshType = testSquare
!integer,     parameter   :: meshType = testRectangle
!integer,     parameter   :: meshType = testSquareFocused
!integer,     parameter   :: meshType = testWobble
!integer,     parameter   :: meshType = testStitch
!integer,     parameter   :: meshType = testMultiGrid

!integer,     parameter   :: meshType = fivePatches
!integer,     parameter   :: meshType = fivePatchesAutostitch
integer,     parameter   :: meshType = thirteenPatches

! select one or more output format
writeSimple   = .true.
writeParaview = .true.
writeGMT      = .true.

!-------------------------------
elementType = Q0
!-------------------------------  

if (testMe) then
    call runTests()
else

    if      (meshType.eq.testSquare) then
        call meshTestSquare()
    else if (meshType.eq.testRectangle) then
        call meshTestRectangle()
    else if (meshType.eq.testSquareFocused) then
        call meshTestSquareFocused()
    else if (meshType.eq.testWobble) then
        call meshTestWobble()
    else if (meshType.eq.testStitch) then
        call meshTestStitch()
    else if (meshType.eq.testMultiGrid) then
        elementType = Q1
        call meshTestStitch()
        elementType = Q2
        call meshTestStitch()
    else if (meshType.eq.fivePatches) then
        call meshFivePatches()
    else if (meshType.eq.fivePatchesAutostitch) then
        call meshFivePatchesAutostitch() 
    else if (meshType.eq.thirteenPatches) then
        call meshThirteenPatches()
    endif

endif


end program
