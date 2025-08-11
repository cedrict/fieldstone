subroutine output_for_paraview (np,nel,x,y,Vx,Vy,press,p,T,density,viscosity,exx,eyy,exy,mat,vpmode,icon,istep)
implicit none
integer np,nel
real(8), dimension(np)    :: x,y,Vx,Vy,T,p
real(8), dimension(nel)   :: press,exx,eyy,exy,density,viscosity,vpmode
integer, dimension(4,nel) :: icon
integer, dimension(nel)   :: mat 
integer istep
character(len=6) cistep

real(8), dimension(nel) :: sxx,sxy,syy
integer i,iel

!=======================================

sxx=2.d0*viscosity*exx
syy=2.d0*viscosity*eyy
sxy=2.d0*viscosity*exy

call int_to_char(cistep,6,istep)

write(*,'(a)') '--> OUT/solution_'//cistep//'.vtu'

open(unit=123,file='OUT/solution_'//cistep//'.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',np,'" NumberOfCells="',nel,'">'
!.............................
write(123,*) '<PointData Scalars="scalars">'

write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="Velocity" Format="ascii">'
do i=1,np
write(123,*) Vx(i),Vy(i),0
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" Name="Temperature" Format="ascii">'
do i=1,np
write(123,*) T(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="p" Format="ascii">'
do i=1,np
write(123,*) p(i)
end do
write(123,*) '</DataArray>'

write(123,*) '</PointData>'
write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,np
write(123,*) x(i),y(i),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<CellData Scalars="scalars">'

write(123,*) '<DataArray type="Float32" Name="press" Format="ascii">'
do iel=1,nel
write(123,*) press(iel)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="material" Format="ascii">'
do iel=1,nel
write(123,*) mat(iel)
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" Name="exx" Format="ascii">'
do iel=1,nel
write(123,*) exx(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="eyy" Format="ascii">'
do iel=1,nel
write(123,*) eyy(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="exy" Format="ascii">'
do iel=1,nel
write(123,*) exy(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="strainrate 2nd inv." Format="ascii">'
do iel=1,nel
write(123,*) sqrt(0.5d0*(exx(iel)**2+eyy(iel)**2)+exy(iel)**2)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="sxx" Format="ascii">'
do iel=1,nel
write(123,*) sxx(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="syy" Format="ascii">'
do iel=1,nel
write(123,*) syy(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="sxy" Format="ascii">'
do iel=1,nel
write(123,*) sxy(iel)
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Float32" Name="dev. stress 2nd inv." Format="ascii">'
do iel=1,nel
write(123,*) sqrt(0.5d0*(sxx(iel)**2+syy(iel)**2)+sxy(iel)**2)
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" Name="rheology" Format="ascii">'
do i=1,nel
write(123,*) vpmode(i)
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" Name="density" Format="ascii">'
do i=1,nel
write(123,*) density(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="viscosity" Format="ascii">'
do i=1,nel
write(123,*) viscosity(i)
end do
write(123,*) '</DataArray>'

write(123,*) '</CellData>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
write(123,*) icon(1,iel)-1,icon(2,iel)-1,icon(3,iel)-1,icon(4,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*4,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (9,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)


end subroutine
