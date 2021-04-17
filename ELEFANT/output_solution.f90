!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_solution

use global_parameters
use structures
use timing

implicit none

integer k
real(8) uth,vth,wth,dum

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_solution}
!@@ This subroutine generates the {\filenamefont solution.vtu} in the {\foldernamefont OUTPUT}
!@@ folder. It also generates the basic ascii file {\filenamefont solution.ascii}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

open(unit=123,file='OUTPUT/solution.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',ncorners*nel,'" NumberOfCells="',nel,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,*) mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
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
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: back" Format="ascii">'
do iel=1,nel
if (mesh(iel)%back) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
end if
!-----
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: front" Format="ascii">'
do iel=1,nel
if (mesh(iel)%front) then
   write(123,*) 1
else
   write(123,*) 0
end if
end do
write(123,*) '</DataArray>'
end if
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
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="hz" Format="ascii">'
do iel=1,nel
   write(123,*) mesh(iel)%hz
end do
write(123,*) '</DataArray>'
end if
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
   do k=1,ncorners
      write(123,*) mesh(iel)%u(k),mesh(iel)%v(k),mesh(iel)%w(k)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="pressure (p)" Format="ascii">'
do iel=1,nel
   if (pair=='q1p0') then
      do k=1,ncorners
         write(123,*) mesh(iel)%p(1)
      end do
   else
      do k=1,ncorners
         write(123,*) mesh(iel)%p(k)
      end do
   end if
end do
write(123,*) '</DataArray>'
!-----
if (use_T) then 
write(123,*) '<DataArray type="Float32" Name="temperature (T)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,*) mesh(iel)%T(k)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="pressure (q)" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      write(123,*) mesh(iel)%q(k)
   end do
end do
write(123,*) '</DataArray>'

!-----
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: back" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%back_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: front" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%front_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: left" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%left_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: right" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%right_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: bottom" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%bottom_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="boundary: top" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%top_node(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="fix_u" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_u(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="fix_v" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_v(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
!-----
if (ndim==3) then
write(123,*) '<DataArray type="Float32" Name="fix_w" Format="ascii">'
do iel=1,nel
   do k=1,ncorners
      if (mesh(iel)%fix_w(k)) then
         write(123,*) 1
      else
         write(123,*) 0
      end if
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="density" Format="ascii">'
if (use_swarm) then
   do iel=1,nel
      do k=1,ncorners
         write(123,*) mesh(iel)%a_rho+&
                      mesh(iel)%b_rho*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                      mesh(iel)%c_rho*(mesh(iel)%yV(k)-mesh(iel)%yc)
         end do
   end do
else
   do iel=1,nel
      do k=1,ncorners
         write(123,*) mesh(iel)%rhoq(k) ! putting rhoq on nodes for visu ordering wrong!
      end do
   end do
end if
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="viscosity" Format="ascii">'
if (use_swarm) then
   do iel=1,nel
      do k=1,4
         write(123,*) mesh(iel)%a_eta+&
                      mesh(iel)%b_eta*(mesh(iel)%xV(k)-mesh(iel)%xc)+&
                      mesh(iel)%c_eta*(mesh(iel)%yV(k)-mesh(iel)%yc)
         end do
   end do
else
   do iel=1,nel
      do k=1,ncorners
         write(123,*) mesh(iel)%etaq(k) ! putting etaq on nodes for visu ordering wrong!
      end do
   end do
end if
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity (analytical)" Format="ascii">'
do iel=1,nel
   do k=1,4
      call analytical_solution(mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                               uth,vth,wth,dum,dum,dum,dum,dum,dum,dum,dum)
      write(123,*) uth,vth,wth
   end do
end do
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



open(unit=123,file="OUTPUT/solution.ascii",action="write")
write(123,*) '#1,2,3  x,y,z '
write(123,*) '#4,5,6  u,v,w '
write(123,*) '#7      q '
write(123,*) '#8      T '

do iel=1,nel 
   do k=1,ncorners
      write(123,*) mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),&
                   mesh(iel)%u(k),mesh(iel)%v(k),mesh(iel)%w(k),&
                   mesh(iel)%q(k),mesh(iel)%T(k)
   end do
end do
close(123)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> output_solution                  ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
