!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_qpoints

use global_parameters
use structures
use timing

implicit none

integer iq

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_qpoints}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

open(unit=123,file='OUTPUT/qpoints.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',Nq,'" NumberOfCells="',Nq,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq)
   end do
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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
!-----
write(123,*) '<DataArray type="Float32" Name="rho" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%rhoq(iq)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="eta" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%etaq(iq)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="JxW" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,*) mesh(iel)%JxWq(iq)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="gravity" Format="ascii">'
do iel=1,nel
   do iq=1,nqel
      write(123,'(3f12.4)') mesh(iel)%gxq(iq),mesh(iel)%gyq(iq),mesh(iel)%gzq(iq)
   end do
end do
write(123,*) '</DataArray>'


!-----
write(123,*) '</PointData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iq=1,Nq
write(123,'(i8)') iq-1
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
do iq=1,Nq
write(123,'(i8)') iq
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
do iq=1,Nq
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

write(*,'(a,f6.2,a)') 'output_qpoints (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
