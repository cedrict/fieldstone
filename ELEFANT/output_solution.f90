subroutine output_solution

use global_parameters
use structures

open(unit=123,file='solution.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',4*nel,'" NumberOfCells="',nel,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<CellData Scalars="scalars">'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: left" Format="ascii">'
do iel=1,nel
if (mesh(iel)%left) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: right" Format="ascii">'
do iel=1,nel
if (mesh(iel)%right) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: bottom" Format="ascii">'
do iel=1,nel
if (mesh(iel)%bottom) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: top" Format="ascii">'
do iel=1,nel
if (mesh(iel)%top) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="hx" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%hx
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="hy" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%hy
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="nmarker" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%nmarker
end do
write(123,*) '</DataArray>'

!-----
write(123,*) '<DataArray type="Float32" Name="least squares: a_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%a_rho
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: b_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%b_rho
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: c_rho" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%c_rho
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: a_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%a_eta
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: b_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%b_eta
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: c_eta" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%c_eta
end do
write(123,*) '</DataArray>'





!-----
write(123,*) '</CellData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<PointData Scalars="scalars">'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%u(1),mesh(iel)%v(1),mesh(iel)%w(1)
   write(123,*) mesh(iel)%p(2),mesh(iel)%v(2),mesh(iel)%w(2)
   write(123,*) mesh(iel)%p(3),mesh(iel)%v(3),mesh(iel)%w(3)
   write(123,*) mesh(iel)%p(4),mesh(iel)%v(4),mesh(iel)%w(4)
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="pressure" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%p(1)
   write(123,*) mesh(iel)%p(2)
   write(123,*) mesh(iel)%p(3)
   write(123,*) mesh(iel)%p(4)
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="temperature" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%T(1)
   write(123,*) mesh(iel)%T(2)
   write(123,*) mesh(iel)%T(3)
   write(123,*) mesh(iel)%T(4)
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: left" Format="ascii">'
do iel=1,nel
   if (mesh(iel)%left_node(1)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%left_node(2)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%left_node(3)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%left_node(4)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: right" Format="ascii">'
do iel=1,nel
   if (mesh(iel)%right_node(1)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%right_node(2)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%right_node(3)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%right_node(4)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: bottom" Format="ascii">'
do iel=1,nel
   if (mesh(iel)%bottom_node(1)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%bottom_node(2)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%bottom_node(3)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%bottom_node(4)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: top" Format="ascii">'
do iel=1,nel
   if (mesh(iel)%top_node(1)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%top_node(2)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%top_node(3)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%top_node(4)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="fix_u" Format="ascii">'
do iel=1,nel
   if (mesh(iel)%fix_u(1)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%fix_u(2)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%fix_u(3)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
   if (mesh(iel)%fix_u(4)) then
   write(123,*) 1
   else
   write(123,*) 0
   end if
end do
write(123,*) '</DataArray>'

!-----
write(123,*) '<DataArray type="Float32" Name="density" Format="ascii">'
if (use_markers) then
   do iel=1,nel
      do i=1,4
         write(123,*) mesh(iel)%a_rho+&
                      mesh(iel)%b_rho*(mesh(iel)%xV(i)-mesh(iel)%xc)+&
                      mesh(iel)%c_rho*(mesh(iel)%yV(i)-mesh(iel)%yc)
         end do
   end do
else
   do iel=1,nel
      write(123,*) mesh(iel)%rhoq(1)   
      write(123,*) mesh(iel)%rhoq(2)   ! putting rhoq on nodes for visu
      write(123,*) mesh(iel)%rhoq(3)   ! ordering wrong!
      write(123,*) mesh(iel)%rhoq(4)   
   end do
end if
write(123,*) '</DataArray>'

!-----
write(123,*) '<DataArray type="Float32" Name="viscosity" Format="ascii">'
if (use_markers) then
   do iel=1,nel
      do i=1,4
         write(123,*) mesh(iel)%a_eta+&
                      mesh(iel)%b_eta*(mesh(iel)%xV(i)-mesh(iel)%xc)+&
                      mesh(iel)%c_eta*(mesh(iel)%yV(i)-mesh(iel)%yc)
         end do
   end do
else
   do iel=1,nel
      write(123,*) mesh(iel)%etaq(1)   
      write(123,*) mesh(iel)%etaq(2)   ! putting rhoq on nodes for visu
      write(123,*) mesh(iel)%etaq(3)   ! ordering wrong!
      write(123,*) mesh(iel)%etaq(4)   
   end do
end if
write(123,*) '</DataArray>'













!-----
write(123,*) '</PointData>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
   write(123,*) (iel-1)*4,(iel-1)*4+1,(iel-1)*4+2,(iel-1)*4+3 
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,nel)
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (5,iel=1,nel)
write(123,*) '</DataArray>'
!-----
write(123,*) '</Cells>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

end subroutine
