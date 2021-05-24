subroutine output_for_paraview2 (np,nel,x,y,z,icon)
implicit none
integer np,nel
real(8), dimension(np)    :: x,y,z
integer, dimension(8,nel) :: icon

integer i,iel

!=======================================


open(unit=123,file='OUT/refined.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',np,'" NumberOfCells="',nel,'">'
!.............................
!write(123,*) '<PointData Scalars="scalars">'

!write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
!do i=1,np
!write(123,*) Vx(i),Vy(i),0
!end do
!write(123,*) '</DataArray>'

!write(123,*) '</PointData>'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,np
write(123,*) x(i),y(i),z(i)
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
!write(123,*) '<CellData Scalars="scalars">'
!write(123,*) '<DataArray type="Float32" Name="crtype" Format="ascii">'
!do iel=1,nel
!write(123,*) crtype(iel)
!end do
!write(123,*) '</DataArray>'
!write(123,*) '</CellData>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
write(123,*) icon(1,iel)-1,icon(2,iel)-1,icon(3,iel)-1,icon(4,iel)-1,&
             icon(5,iel)-1,icon(6,iel)-1,icon(7,iel)-1,icon(8,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*8,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (12,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)


end subroutine
