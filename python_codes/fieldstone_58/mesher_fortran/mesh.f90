program circleMesh

implicit none

integer,          parameter :: nLayers     = 4
double precision, parameter :: outerRadius = 10d0

integer :: nPoints
integer :: nElems
integer :: nPointsOnCircle

integer :: iLayer
integer :: iPoint, storedPoints
integer :: iElem, storedElems
integer :: iBlock
integer :: iSection
integer :: iInner, iOuter

double precision :: radius

double precision,allocatable :: coordinates(:,:)
integer,         allocatable :: connectivity(:,:)

double precision             :: pi

real*8 scalex,scaley,scalez
integer, parameter :: mpe=3
integer, parameter :: iunit=123
character(len=12) velocity_discretisation

velocity_discretisation='P1_2D'
scalex=1
scaley=1
scalez=1


pi = 4d0 * atan(1d0)

nPoints = 1 + 3 * nLayers * (nLayers+1) ! Euler's trick :-)
nElems = 6 * nLayers * nLayers


allocate(coordinates(3,nPoints))
allocate(connectivity(3,nElems))

! initialize to zero
coordinates = 0d0
connectivity = 0


! set coordinates
storedPoints = 1

do iLayer = 1,nLayers
    radius = outerRadius * dble(iLayer)/dble(nLayers)
    nPointsOnCircle = 6*iLayer
    do iPoint = 1,nPointsOnCircle
        storedPoints = storedPoints + 1
        ! Coordinates are created, starting at twelve o'clock, 
        ! going in clockwise direction
        coordinates(1,storedPoints) = radius * sin(2d0 * pi * dble(iPoint-1) / dble(nPointsOnCircle))
        coordinates(2,storedPoints) = radius * cos(2d0 * pi * dble(iPoint-1) / dble(nPointsOnCircle))
    enddo
enddo

! set connectivity

storedElems = 0

! first layer by hand
connectivity(:,1) = (/1,2,3/)
connectivity(:,2) = (/1,3,4/)  
connectivity(:,3) = (/1,4,5/)
connectivity(:,4) = (/1,5,6/)
connectivity(:,5) = (/1,6,7/)
connectivity(:,6) = (/1,7,2/)

storedElems = 6


iInner = 2
iOuter = 8


do iLayer = 2,nLayers
    nPointsOnCircle = 6*iLayer
    do iSection = 1,5
        do iBlock = 1,iLayer-1
            storedElems = storedElems + 1    
            connectivity(:,storedElems) = (/iInner, iOuter, iOuter + 1/)

            storedElems = storedElems + 1
            connectivity(:,storedElems) = (/iInner, iInner+1, iOuter + 1/)

            iInner = iInner + 1
            iOuter = iOuter + 1
        enddo
        
        storedElems = storedElems + 1
        connectivity(:,storedElems) = (/iInner, iOuter, iOuter + 1/)

        iOuter = iOuter + 1

    enddo

    ! do the closing section. This has some extra difficulty where it is attached to the starting point

    ! first do the regular blocks within the section
    do iBlock = 1,iLayer - 2
            storedElems = storedElems + 1
            connectivity(:,storedElems) = (/iInner, iOuter, iOuter + 1/)

            storedElems = storedElems + 1
            connectivity(:,storedElems) = (/iInner, iInner+1, iOuter + 1/)

            iInner = iInner + 1
            iOuter = iOuter + 1
    enddo
    
    ! do the last block, which shares an inner point with the first section
    storedElems = storedElems + 1
    connectivity(:,storedElems) = (/iInner, iOuter, iOuter + 1/)

    storedElems = storedElems + 1
    connectivity(:,storedElems) = (/iInner, iOuter+1, iInner + 1 - 6*(iLayer-1)/)
    
    ! last element, closing the layer.
    storedElems = storedElems + 1
    connectivity(:,storedElems) = (/iInner+1, iInner + 1 - 6*(iLayer-1), iOuter + 1/)

    iInner = iInner + 1
    iOuter = iOuter + 2

enddo

print *,storedelems,nElems

!do iElem =1,storedElems
!    write(*,*) connectivity(:,iElem)
!enddo


open(unit=iunit,file='gridV.vtu',status='replace',form='formatted')
write(iunit,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(iunit,*) '<UnstructuredGrid>'
write(iunit,*) '<Piece NumberOfPoints="',nPoints,'" NumberOfCells="',nElems,'">'


write(iunit,*) '<Points>'
call write_positions(nPoints,coordinates(1,:),coordinates(2,:),coordinates(3,:),iunit,scalex,scaley,scalez)
write(iunit,*) '</Points>'


write(iunit,*) '<Cells>'
call write_icon(mpe,nElems,connectivity,iunit,velocity_discretisation)
call write_offsets(mpe,nElems,iunit,velocity_discretisation)
call write_types(mpe,nElems,iunit,velocity_discretisation)
write(iunit,*) '</Cells>'

!.............................

write(iunit,*) '</Piece>'
write(iunit,*) '</UnstructuredGrid>'
write(iunit,*) '</VTKFile>'
close(iunit)





end program




