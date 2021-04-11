subroutine export_mesh

use global_parameters
use structures

open(unit=123,file='meshV.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',4*nel,'" NumberOfCells="',nel,'">'
!.............................
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%xV(1),mesh(iel)%yV(1),0.d0
   write(123,*) mesh(iel)%xV(2),mesh(iel)%yV(2),0.d0
   write(123,*) mesh(iel)%xV(3),mesh(iel)%yV(3),0.d0
   write(123,*) mesh(iel)%xV(4),mesh(iel)%yV(4),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
   write(123,*) (iel-1)*4,(iel-1)*4+1,(iel-1)*4+2,(iel-1)*4+3 
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (5,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)


end subroutine
