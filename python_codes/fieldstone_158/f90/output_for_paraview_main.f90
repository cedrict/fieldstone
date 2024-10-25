subroutine output_for_paraview_main (np,nel,x,y,icon,rho,mueff)
implicit none
integer np,nel,which
real(8), dimension(np)    :: x,y,rho,mueff
integer, dimension(4,nel) :: icon

integer i,iel

!=======================================



!print *,np,nel

open(unit=123,file='OUT/visu_m.vtu',status='replace',form='formatted')

write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',np,'" NumberOfCells="',nel,'">'
!.............................
write(123,*) '<PointData Scalars="scalars">'


write(123,*) '<DataArray type="Float32" Name="rho" Format="ascii">'
do i=1,np
write(123,*) rho(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="mueff" Format="ascii">'
do i=1,np
write(123,*) mueff(i)
end do
write(123,*) '</DataArray>'











write(123,*) '</PointData>'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,np
write(123,*) x(i),y(i),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
write(123,*) icon(1,iel)-1,icon(2,iel)-1,icon(3,iel)-1,icon(4,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (9,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)


end subroutine
