subroutine output_qpoints
use global_parameters
use structures

implicit none

integer iq

open(unit=123,file='qpoints.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',Nq,'" NumberOfCells="',Nq,'">'


write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq)
   end do
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'

!-------------------
write(123,*) '<PointData Scalars="scalars">'
!-----
write(123,*) '<DataArray type="Float32" Name="r" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%rq(iq)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="s" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%sq(iq)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="t" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%tq(iq)
   end do
end do
write(123,*) '</DataArray>'







write(123,*) '</PointData>'
!-------------------



write(123,*) '<Cells>'

write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iq=1,Nq
write(123,'(i8)') iq-1
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
do iq=1,Nq
write(123,'(i8)') iq
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
do iq=1,Nq
write(123,'(i1)') 1
end do
write(123,*) '</DataArray>'

write(123,*) '</Cells>'

write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)















end subroutine
