module pointMod

implicit none

type point
double precision :: x, y
end type

end module

!-------------------

module circleMod

! a circle section has a start point and an en point.
! called 'from' and  'to', because 'end' is a reserved word.
! The curvature is the distance from the point between from and to,
! to the circle. When this number is positive, the circle goes to the left
! negative indicates circle passing to the right.
! Left and right are relative when walking from 'from' to 'to'.
! Curvature of 0d0 implies a straight line.

use pointMod

implicit none

type circleSection
type(point)      :: from, to
double precision :: curvature
double precision :: radius
type(point)      :: center
end type

end module

!-------------------

module stitching
! patchEdges are stored because they are very convenient for 
! stitching the patches together

! An allocatable of allocatables is not performance-optimal,
! but it is so very convenient, and it usually fast enough anyway.

implicit none

type patchEdges
integer, allocatable :: pointIDs(:,:)
end type

type(patchEdges), allocatable :: edges(:)

integer, allocatable :: removedPointIDs(:)

contains

!-------------------

subroutine cleanStitchData()

implicit none

deallocate(edges)
deallocate(removedPointIDs)

end subroutine

end module

!-------------------
