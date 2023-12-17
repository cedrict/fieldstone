!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_mesh

use module_parameters, only: mV,spaceV,debug,iel,nel,iproc,ndim
use module_mesh
use module_timing

implicit none

integer k

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{output\_mesh.f90}
!@@ This subroutine produces the {\filenamefont meshV.vtu} file which only 
!@@ contains the corner nodes (4 in 2D, 8 in 3D).
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (debug) then

open(unit=123,file='OUTPUT/meshV.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,'(a,i8,a,i7,a)') '<Piece NumberOfPoints="',mV*nel,'" NumberOfCells="',nel,'">'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,'(a)') '<Points>'
write(123,'(a)') '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do iel=1,nel
   do k=1,mV
      write(123,'(3es12.4)') mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
   end do
end do
write(123,'(a)') '</DataArray>'
write(123,'(a)') '</Points>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '<Cells>'
!-----
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
select case(spaceV)
case('__Q1')
   if (ndim==2) then 
      do iel=1,nel
         write(123,*) ( (iel-1)*mV+k-1,k=1,mV) 
      end do
   else
      stop 'output_mesh: pb1'
   end if
case('__Q2')
   if (ndim==2) then 
      do iel=1,nel
         write(123,*) (iel-1)*mV+1-1,&
                      (iel-1)*mV+3-1,&
                      (iel-1)*mV+9-1,&
                      (iel-1)*mV+7-1,&
                      (iel-1)*mV+2-1,&
                      (iel-1)*mV+6-1,&
                      (iel-1)*mV+8-1,&
                      (iel-1)*mV+4-1,&
                      (iel-1)*mV+5-1
      end do
   else
      stop 'output_mesh: pb1'
   end if
case default
   stop 'output_mesh: spaceV unknown'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
select case(spaceV)
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
case('_Q1+')
   write(123,*) (iel*4,iel=1,nel)
case('Q1++')
   write(123,*) (iel*8,iel=1,nel)
case default
   stop 'pb in output_solution: cell offset unknown'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
select case(spaceV)
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
case('_Q1+')
   write(123,*) (9,iel=1,nel)
case('Q1++')
   write(123,*) (12,iel=1,nel)
case default
   stop 'pb in output_solution: cell type unknown'
end select
write(123,*) '</DataArray>'
!-----
write(123,*) '</Cells>'
!=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)





write(*,'(a)') shift//'produced OUTPUT/meshV.vtu' 

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_mesh (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
