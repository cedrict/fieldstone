!=======================================

subroutine output_for_paraview_visu (grids) 
use structures
implicit none
type (grid) grids
integer i,iel

!=======================================

open(unit=123,file='OUT/visu.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',grids%np,'" NumberOfCells="',grids%nel,'">'
!.............................
write(123,*) '<PointData Scalars="scalars">'

write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
do i=1,grids%np
write(123,*) grids%u(i),grids%v(i),0
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="u" Format="ascii">'
do i=1,grids%np
write(123,*) grids%u(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="v" Format="ascii">'
do i=1,grids%np
write(123,*) grids%v(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="divv" Format="ascii">'
do i=1,grids%np
write(123,*) grids%divv(i)
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" Name="pressure" Format="ascii">'
do i=1,grids%np
write(123,*) grids%p(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="rho" Format="ascii">'
do i=1,grids%np
write(123,*) grids%rho(i)
end do
write(123,*) '</DataArray>'

write(123,*) '</PointData>'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,grids%np
write(123,*) grids%x(i),grids%y(i),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,grids%nel
write(123,*) grids%icon(1,iel)-1,grids%icon(2,iel)-1,grids%icon(3,iel)-1,grids%icon(4,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,grids%nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (9,iel=1,grids%nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)


end subroutine
