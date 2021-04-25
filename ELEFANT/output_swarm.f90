!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_swarm

use global_parameters
use structures
use timing

implicit none

integer im

logical, parameter :: output_rst=.false.

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_swarm.f90}
!@@ This subroutine produces the {\filenamefont swarm.vtu} file in the 
!@@ {\foldernamefont OUTPUT} folder which contains the 
!@@ swarm of particles with all their properties.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (use_swarm) then

open(unit=123,file='OUTPUT/swarm.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',nmarker,'" NumberOfCells="',nmarker,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<PointData Scalars="scalars">'
!-----
if (output_rst) then
write(123,*) '<DataArray type="Float32" Name="r" Format="ascii">'
do im=1,nmarker
write(123,'(f5.2)') swarm(im)%r
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_rst) then
write(123,*) '<DataArray type="Float32" Name="s" Format="ascii">'
do im=1,nmarker
write(123,'(f5.2)') swarm(im)%s
end do
write(123,*) '</DataArray>'
end if
!-----
if (output_rst .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="t" Format="ascii">'
do im=1,nmarker
write(123,'(f5.2)') swarm(im)%t
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="mat" Format="ascii">'
do im=1,nmarker
write(123,'(f5.2)') swarm(im)%mat*(1+0.1*swarm(im)%paint)
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="paint" Format="ascii">'
do im=1,nmarker
write(123,'(f5.2)') swarm(im)%paint
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="rho0" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') mat(swarm(im)%mat)%rho0
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="eta0" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') mat(swarm(im)%mat)%eta0
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="iel" Format="ascii">'
do im=1,nmarker
write(123,'(i7)') swarm(im)%iel
end do
write(123,*) '</DataArray>'
!-----
write(123,'(a)') '<DataArray type="Float32" Name="eta" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%eta
end do
write(123,'(a)') '</DataArray>'
!-----
write(123,'(a)') '<DataArray type="Float32" Name="rho" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%rho
end do
write(123,'(a)') '</DataArray>'
!-----
if (use_T) then
write(123,'(a)') '<DataArray type="Float32" Name="hcapa" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%hcapa
end do
write(123,'(a)') '</DataArray>'
end if
!-----
if (use_T) then
write(123,'(a)') '<DataArray type="Float32" Name="hcond" Format="ascii">'
do im=1,nmarker
write(123,'(es12.4)') swarm(im)%hcond
end do
write(123,'(a)') '</DataArray>'
end if
!-----
if (use_T) then
write(123,*) '<DataArray type="Float32" Name="hprod" Format="ascii">'
do im=1,nmarker
write(123,'(f12.4)') swarm(im)%hprod
end do
write(123,*) '</DataArray>'
end if
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

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') '     >> output_swarm                     ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
