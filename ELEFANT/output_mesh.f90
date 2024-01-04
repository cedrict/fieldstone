!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_mesh

use module_parameters, only: mU,mV,mW,mP,iel,nel,iproc,ndim,mmapping
use module_mesh
use module_timing
use module_export_vtu

implicit none

integer i,k

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{output\_mesh.f90}
!@@ This subroutine produces the {\filenamefont meshV.vtu} and {\filenamefont meshP.vtu} files. 
!@@ See subroutine output\_solution for more info.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

open(unit=123,file='OUTPUT/nodesU.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mU*nel,'" NumberOfCells="',nel*mU,'">'
!------------------
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mU
      write(123,'(3es12.4)') mesh(iel)%xU(k),mesh(iel)%yU(k),mesh(iel)%zU(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!------------------
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
write(123,*) (i-1,i=1,nel*mU)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (i,i=1,nel*mU)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (1,i=1,nel*mU)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesU.vtu'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/nodesV.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mV*nel,'" NumberOfCells="',nel*mV,'">'
!------------------
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(3es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!------------------
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
write(123,*) (i-1,i=1,nel*mV)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (i,i=1,nel*mV)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (1,i=1,nel*mV)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesV.vtu'

!----------------------------------------------------------

if (ndim>2) then
open(unit=123,file='OUTPUT/nodesW.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mW*nel,'" NumberOfCells="',nel*mW,'">'
!------------------
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mW
      write(123,'(3es12.4)') mesh(iel)%xW(k),mesh(iel)%yW(k),mesh(iel)%zW(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!------------------
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
write(123,*) (i-1,i=1,nel*mW)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (i,i=1,nel*mW)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (1,i=1,nel*mW)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesW.vtu'
end if
!----------------------------------------------------------

open(unit=123,file='OUTPUT/nodesP.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mP*nel,'" NumberOfCells="',nel*mP,'">'
!------------------
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mP
      write(123,'(3es12.4)') mesh(iel)%xP(k),mesh(iel)%yP(k),mesh(iel)%zP(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!------------------
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
write(123,*) (i-1,i=1,nel*mP)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (i,i=1,nel*mP)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (1,i=1,nel*mP)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesP.vtu'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/nodesM.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mmapping*nel,'" NumberOfCells="',nel*mmapping,'">'
!------------------
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      write(123,'(3es12.4)') mesh(iel)%xM(k),mesh(iel)%yM(k),mesh(iel)%zM(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!------------------
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
write(123,*) (i-1,i=1,nel*mmapping)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (i,i=1,nel*mmapping)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (1,i=1,nel*mmapping)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesM.vtu'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/nodesU.ascii',status='replace',form='formatted')
do iel=1,nel
   do k=1,mU
      write(123,'(3es12.4)') mesh(iel)%xU(k),mesh(iel)%yU(k),mesh(iel)%zU(k)
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesU.ascii'

open(unit=123,file='OUTPUT/nodesV.ascii',status='replace',form='formatted')
do iel=1,nel
   do k=1,mU
      write(123,'(3es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesV.ascii'

if (ndim>2) then
open(unit=123,file='OUTPUT/nodesW.ascii',status='replace',form='formatted')
do iel=1,nel
   do k=1,mU
      write(123,'(3es12.4)') mesh(iel)%xW(k),mesh(iel)%yW(k),mesh(iel)%zW(k)
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesW.ascii'
end if

open(unit=123,file='OUTPUT/nodesP.ascii',status='replace',form='formatted')
do iel=1,nel
   do k=1,mP
      write(123,'(3es12.4)') mesh(iel)%xP(k),mesh(iel)%yP(k),mesh(iel)%zP(k)
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesP.ascii'

open(unit=123,file='OUTPUT/nodesM.ascii',status='replace',form='formatted')
do iel=1,nel
   do k=1,mmapping
      write(123,'(3es12.4)') mesh(iel)%xM(k),mesh(iel)%yM(k),mesh(iel)%zM(k)
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/nodesM.ascii'

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_mesh:',elapsed,' s                    |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
