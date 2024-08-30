subroutine output_for_paraview (gridd,which)
use structures
implicit none
type (grid) gridd
integer which
integer i,iel

!=======================================

select case(which)
case(0)
open(unit=123,file='OUT/visu_b.vtu',status='replace',form='formatted')
case(1)
open(unit=123,file='OUT/visu_u.vtu',status='replace',form='formatted')
case(2)
open(unit=123,file='OUT/visu_v.vtu',status='replace',form='formatted')
case(3)
open(unit=123,file='OUT/visu_p.vtu',status='replace',form='formatted')
end select

!=======================================

write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',gridd%np,'" NumberOfCells="',gridd%nel,'">'
!.............................
write(123,*) '<PointData Scalars="scalars">'

if (allocated(gridd%field)) then
write(123,*) '<DataArray type="Float32" Name="field" Format="ascii">'
do i=1,gridd%np   
write(123,*) gridd%field(i) 
end do
write(123,*) '</DataArray>'
end if

if (allocated(gridd%rho)) then
write(123,*) '<DataArray type="Float32" Name="rho" Format="ascii">'
do i=1,gridd%np   
write(123,*) gridd%rho(i) 
end do
write(123,*) '</DataArray>'
end if

if (allocated(gridd%bc)) then
write(123,*) '<DataArray type="Float32" Name="bc" Format="ascii">'
do i=1,gridd%np
if (gridd%bc(i)) then
write(123,*) 1
else
write(123,*) 0
end if 
end do
write(123,*) '</DataArray>'
end if

write(123,*) '</PointData>'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,gridd%np
write(123,*) gridd%x(i),gridd%y(i),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,gridd%nel
write(123,*) gridd%icon(1,iel)-1,gridd%icon(2,iel)-1,gridd%icon(3,iel)-1,gridd%icon(4,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,gridd%nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (9,iel=1,gridd%nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

end subroutine
