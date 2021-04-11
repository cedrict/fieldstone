!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine export_swarm

use global_parameters
use structures
!use constants

implicit none

integer im

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{\tt template}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


open(unit=123,file='markers.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',nmarker,'" NumberOfCells="',nmarker,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<PointData Scalars="scalars">'
!-----
write(123,*) '<DataArray type="Float32" Name="r" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%r
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="s" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%s
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="t" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%t
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="mat" Format="ascii">'
do im=1,nmarker
write(123,'(i4)') swarm(im)%mat
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="paint" Format="ascii">'
do im=1,nmarker
write(123,*) swarm(im)%paint
end do
write(123,*) '</DataArray>'


!-----
write(123,*) '</PointData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Points>'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do im=1,nmarker
write(123,'(3es12.4)') swarm(im)%x,swarm(im)%y,swarm(im)%z
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '</Points>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do im=1,nmarker
write(123,'(i8)') im-1
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
do im=1,nmarker
write(123,'(i8)') im
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
do im=1,nmarker
write(123,'(i1)') 1
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '</Cells>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)










!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> export_swarm ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
