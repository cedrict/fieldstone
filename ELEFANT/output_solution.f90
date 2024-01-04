!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_solution

use module_constants, only: zero
use module_parameters, only: mapping,mmapping,nel,ndim,spaceU,spaceV,spaceW,mU,mV,mW,mP,mT,&
                             spacePressure,spaceTemperature,output_freq,debug,iproc,cistep,&
                             iel,use_swarm,solve_stokes_system,use_T
use module_arrays, only: rmapping,smapping,tmapping 
use module_statistics 
use module_mesh
use module_timing

implicit none

integer, external :: conv_l1_to_int

integer i,k
real(8) uth,vth,wth,pth,dum
real(8) NNNU(mU),NNNV(mV),NNNW(mW),NNNP(mP),NNNT(mT)
integer, parameter :: caller_id01=901
integer, parameter :: caller_id02=902
integer, parameter :: caller_id03=903
integer, parameter :: caller_id04=904
integer, parameter :: caller_id05=905
logical, parameter :: output_boundary_indicators=.false. 

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{output\_solution}
!@@ This subroutine generates the {\filenamefont solution.vtu} in the {\foldernamefont OUTPUT}
!@@ folder. It also generates the basic ascii file {\filenamefont solution.ascii}
!@@ It actually export each element separately inside the vtu file (more flexible approach
!@@ but more costly in terms of memory).
!@@ Since some element pairs (like the Rannacher-Turek) do not even have velocity nodes at the 
!@@ corners I have decided to use the mapping nodes as support for the output.
!@@ In the following array the cell types are shown corresponding to each mapping space:
!@@ \begin{center}
!@@ \begin{tabular}{lcc}
!@@ \hline
!@@ & 2D & 3D \\
!@@ \hline
!@@ $P_1$ & 5 & 10 \\
!@@ $P_2$ & 22 & 24 \\
!@@ $Q_1$ & 9 & 12 \\
!@@ $Q_2$ & 23 & 25 \\
!@@ \hline
!@@ \end{tabular}
!@@ \end{center}
!==================================================================================================!

call system_clock(counti,count_rate)

!==============================================================================!

if (iproc==0) then


open(unit=123,file='OUTPUT/u_'//cistep//'.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mU*nel,'" NumberOfCells="',nel*mU,'">'
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mU
      write(123,'(3es12.4)') mesh(iel)%xU(k),mesh(iel)%yU(k),mesh(iel)%zU(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
write(123,*) '<PointData Scalars="scalars">'
write(123,*) '<DataArray type="Float32" Name="u" Format="ascii">'
do iel=1,nel
   do k=1,mU
      write(123,'(es12.4)') mesh(iel)%u(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,*) '</PointData>'
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
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!----------------------------------------------------------

open(unit=123,file='OUTPUT/v_'//cistep//'.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mV*nel,'" NumberOfCells="',nel*mV,'">'
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(3es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
write(123,*) '<PointData Scalars="scalars">'
write(123,*) '<DataArray type="Float32" Name="u" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(es12.4)') mesh(iel)%v(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,*) '</PointData>'
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
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!----------------------------------------------------------

if (ndim>2) then
open(unit=123,file='OUTPUT/w_'//cistep//'.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mW*nel,'" NumberOfCells="',nel*mW,'">'
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mW
      write(123,'(3es12.4)') mesh(iel)%xW(k),mesh(iel)%yW(k),mesh(iel)%zW(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
write(123,*) '<PointData Scalars="scalars">'
write(123,*) '<DataArray type="Float32" Name="u" Format="ascii">'
do iel=1,nel
   do k=1,mW
      write(123,'(es12.4)') mesh(iel)%w(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,*) '</PointData>'
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
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)
end if

!----------------------------------------------------------

open(unit=123,file='OUTPUT/solution_'//cistep//'.vtu',status='replace',form='formatted')
write(123,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,'(a)') '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mmapping*nel,'" NumberOfCells="',nel,'">'
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
write(123,'(a)') '<CellData Scalars="scalars">'
!-----
if (debug .or. output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 1" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd1_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
if (debug .or. output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 2" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd2_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
if (debug .or. output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 3" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd3_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
if (debug .or. output_boundary_indicators) then
write(123,*) '<DataArray type="Float32" Name="boundary: 4" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd4_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
if (debug .or. output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 5" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd5_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
if (debug .or. output_boundary_indicators .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="boundary: 6" Format="ascii">'
do iel=1,nel ; write(123,'(i1)') conv_l1_to_int(mesh(iel)%bnd6_elt) ; end do
write(123,*) '</DataArray>'
end if
!-----
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Name="hx,hy,hz" Format="ascii">'
do iel=1,nel
   write(123,'(3es12.4)') mesh(iel)%hx,mesh(iel)%hy,mesh(iel)%hz
end do
write(123,'(a)') '</DataArray>'
!-----
write(123,'(a)') '<DataArray type="Float32" Name="vol" Format="ascii">'
do iel=1,nel ; write(123,'(es12.4)') mesh(iel)%vol ; end do
write(123,'(a)') '</DataArray>'
!-----
if (use_swarm) then
write(123,'(a)') '<DataArray type="Float32" Name="nmarker" Format="ascii">'
do iel=1,nel ; write(123,'(i4)') mesh(iel)%nmarker ; end do
write(123,'(a)') '</DataArray>'
end if
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: a_rho" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%a_rho ; end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: b_rho" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%b_rho ; end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: c_rho" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%c_rho ; end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: a_eta" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%a_eta ; end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: b_eta" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%b_eta ; end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="least squares: c_eta" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%c_eta ; end do
write(123,*) '</DataArray>'
!-----
write(123,'(a)') '<DataArray type="Float32" Name="rho (avrg)" Format="ascii">'
do iel=1,nel ; write(123,'(f12.5)') mesh(iel)%rho_avrg ; end do
write(123,'(a)') '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="eta (avrg)" Format="ascii">'
do iel=1,nel ; write(123,'(es12.4)') mesh(iel)%eta_avrg ; end do
write(123,*) '</DataArray>'
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="exx" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%exx ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="eyy" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%eyy ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="exy" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%exy ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="ezz" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%ezz ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="exz" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%exz ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system .and. ndim==3) then
write(123,*) '<DataArray type="Float32" Name="eyz" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%eyz ; end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="div(v)" Format="ascii">'
do iel=1,nel ; write(123,*) mesh(iel)%exx+mesh(iel)%eyy+mesh(iel)%ezz ; end do
write(123,*) '</DataArray>'
end if
!-----
write(123,*) '</CellData>'
!------------------
write(123,*) '<PointData Scalars="scalars">'
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      call NNN(rmapping(k),smapping(k),tmapping(k),NNNU,mU,ndim,spaceU,caller_id01)
      call NNN(rmapping(k),smapping(k),tmapping(k),NNNV,mV,ndim,spaceV,caller_id02)
      call NNN(rmapping(k),smapping(k),tmapping(k),NNNW,mW,ndim,spaceW,caller_id03)
      write(123,'(3es12.4)') sum(NNNU*mesh(iel)%u),sum(NNNV*mesh(iel)%v),sum(NNNW*mesh(iel)%w)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
if (solve_stokes_system) then
write(123,*) '<DataArray type="Float32" Name="pressure" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      call NNN(rmapping(k),smapping(k),tmapping(k),NNNP,mP,ndim,spacePressure,caller_id04)
      write(123,'(es12.4)') sum(NNNP*mesh(iel)%p)
   end do
end do
write(123,*) '</DataArray>'
end if 
!-----
write(123,*) '<DataArray type="Float32" Name="rho" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      write(123,'(es12.4)') mesh(iel)%a_rho+&
                            mesh(iel)%b_rho*(mesh(iel)%xM(k)-mesh(iel)%xc)+&
                            mesh(iel)%c_rho*(mesh(iel)%yM(k)-mesh(iel)%yc)+&
                            mesh(iel)%d_rho*(mesh(iel)%zM(k)-mesh(iel)%zc)
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="eta" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      write(123,'(es12.4)') mesh(iel)%a_eta+&
                            mesh(iel)%b_eta*(mesh(iel)%xM(k)-mesh(iel)%xc)+&
                            mesh(iel)%c_eta*(mesh(iel)%yM(k)-mesh(iel)%yc)+&
                            mesh(iel)%d_eta*(mesh(iel)%zM(k)-mesh(iel)%zc)
   end do
end do
write(123,*) '</DataArray>'
!-----
if (use_T) then 
write(123,*) '<DataArray type="Float32" Name="temperature (T)" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      call NNN(rmapping(k),smapping(k),tmapping(k),NNNT,mT,ndim,spaceTemperature,caller_id05)
      write(123,'(es12.4)') sum(NNNT*mesh(iel)%T)
   end do
end do
write(123,*) '</DataArray>'
end if
!-----
!if (use_T) then 
!write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="heat flux" Format="ascii">'
!do iel=1,nel
!   do k=1,mV
!      write(123,'(3es12.4)') mesh(iel)%qx(k),mesh(iel)%qy(k),mesh(iel)%qz(k)
!   end do
!end do
!write(123,*) '</DataArray>'
!end if
!-----
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity (analytical)" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      call experiment_analytical_solution(mesh(iel)%xM(k),mesh(iel)%yM(k),mesh(iel)%zM(k),&
                                          uth,vth,wth,dum,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(3es12.5)') uth,vth,wth
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Float32" Name="pressure (analytical)" Format="ascii">'
do iel=1,nel
   do k=1,mmapping
      call experiment_analytical_solution(mesh(iel)%xM(k),mesh(iel)%yM(k),mesh(iel)%zM(k),&
                               uth,vth,wth,pth,dum,dum,dum,dum,dum,dum,dum)
      write(123,'(es12.4)') pth
   end do
end do
write(123,*) '</DataArray>'
!-----
write(123,*) '</PointData>'
!------------------
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
select case(mapping)
case('__Q1','__P1','__P2')
   if (ndim==2) then 
      do iel=1,nel
         write(123,*) ( (iel-1)*mmapping+k-1,k=1,mmapping) 
      end do
   else
      stop 'output_mesh: pb1'
   end if
case('__Q2')
   if (ndim==2) then 
      do iel=1,nel
         write(123,*) (iel-1)*mmapping+1-1,(iel-1)*mmapping+3-1,(iel-1)*mmapping+9-1,&
                      (iel-1)*mmapping+7-1,(iel-1)*mmapping+2-1,(iel-1)*mmapping+6-1,&
                      (iel-1)*mmapping+8-1,(iel-1)*mmapping+4-1,(iel-1)*mmapping+5-1
      end do
   else
      stop 'output_mesh: pb2'
   end if
case default
   stop 'pb in output_solution: connectivity not done'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
select case(mapping)
case('__Q1')
   if (ndim==2) write(123,*) (iel*4,iel=1,nel)
   if (ndim==3) write(123,*) (iel*8,iel=1,nel)
case('__Q2')
   if (ndim==2) write(123,*) (iel*9,iel=1,nel)
   if (ndim==3) write(123,*) (iel*27,iel=1,nel)
case('__P1')
   if (ndim==2) write(123,*) (iel*3,iel=1,nel)
   if (ndim==3) write(123,*) (iel*4,iel=1,nel)
case('__P2')
   if (ndim==2) write(123,*) (iel*6,iel=1,nel)
   if (ndim==3) write(123,*) (iel*10,iel=1,nel)
case default
   stop 'pb in output_solution: cell offset unknown'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
select case(mapping)
case('__Q1')
   if (ndim==2) write(123,*) (9,iel=1,nel)
   if (ndim==3) write(123,*) (12,iel=1,nel)
case('__Q2')
   if (ndim==2) write(123,*) (23,iel=1,nel)
   if (ndim==3) write(123,*) (25,iel=1,nel)
case('__P1')
   if (ndim==2) write(123,*) (5,iel=1,nel)
   if (ndim==3) write(123,*) (10,iel=1,nel)
case('__P2')
   if (ndim==2) write(123,*) (22,iel=1,nel)
   if (ndim==3) write(123,*) (24,iel=1,nel)
case default
   stop 'pb in output_solution: cell type unknown'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '</Cells>'
!------------------
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)

!----------------------------------------------------------

open(unit=123,file="OUTPUT/u_"//cistep//".ascii",action="write")
write(123,*) '#x,y,z,u '
do iel=1,nel 
   do k=1,mU
      write(123,'(4es12.4)') mesh(iel)%xU(k),mesh(iel)%yU(k),mesh(iel)%zU(k),mesh(iel)%u(k)
   end do
end do
close(123)

open(unit=123,file="OUTPUT/v_"//cistep//".ascii",action="write")
write(123,*) '#x,y,z,v'
do iel=1,nel 
   do k=1,mV
      write(123,'(4es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k),mesh(iel)%v(k)
   end do
end do
close(123)

if (ndim==3) then
open(unit=123,file="OUTPUT/w_"//cistep//".ascii",action="write")
write(123,*) '#x,y,z,w'
do iel=1,nel 
   do k=1,mW
      write(123,'(4es12.4)') mesh(iel)%xW(k),mesh(iel)%yW(k),mesh(iel)%zW(k),mesh(iel)%w(k)
   end do
end do
close(123)
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_solution (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!

function conv_l1_to_int (b)
implicit none
logical(1), intent(in) :: b
integer conv_l1_to_int
if (b) then
conv_l1_to_int=1
else
conv_l1_to_int=0
end if
end function
