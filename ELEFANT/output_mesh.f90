!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_mesh

use module_parameters
use module_mesh
use module_timing

implicit none

integer i

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_mesh.f90}
!@@ This subroutine produces the {\filenamefont meshV.vtu} file which only 
!@@ contains the corner nodes.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (debug) then

open(unit=123,file='OUTPUT/meshV.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',ncorners*nel,'" NumberOfCells="',nel,'">'
!.............................
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do i=1,ncorners
      write(123,'(es12.4)') mesh(iel)%xV(i),mesh(iel)%yV(i),mesh(iel)%zV(i)
   end do
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
   write(123,*) ( (iel-1)*ncorners+i-1,i=1,ncorners) 
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*ncorners,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
if (ndim==2) write(123,*) (9,iel=1,nel)
if (ndim==3) write(123,*) (12,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_mesh (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
