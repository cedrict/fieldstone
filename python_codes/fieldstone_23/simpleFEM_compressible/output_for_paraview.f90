subroutine output_for_paraview (np,nel,x,y,Vx,Vy,p,icon,ibench,phi,density,Rv,Rp,dudx_nodal,dvdy_nodal)
implicit none
integer np,nel
real(8), dimension(np)    :: x,y,Vx,Vy,phi,density,dudx_nodal,dvdy_nodal
real(8), dimension(nel)   :: p,Rp
integer, dimension(4,nel) :: icon
real(8), dimension(2*np)    :: Rv 
real(8) xc,yc
integer ibench
integer i,iel

real(8), external :: uth,vth,pth      
real(8), external :: rho,drhodx,drhody,gx,gy  
real(8), external :: dudxth,dvdyth
!=======================================

open(unit=123,file='OUT/visu.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',np,'" NumberOfCells="',nel,'">'
!.............................
write(123,*) '<PointData Scalars="scalars">'

write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
do i=1,np
write(123,*) Vx(i),Vy(i),0.
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity res." Format="ascii">'
do i=1,np
write(123,*) Rv(2*(i-1)+1),Rv(2*(i-1)+2),0.d0
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="rho(analytical)" Format="ascii">'
do i=1,np
write(123,*) rho(x(i),y(i),ibench)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="phi" Format="ascii">'
do i=1,np
write(123,*) phi(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="density" Format="ascii">'
do i=1,np
write(123,*) density(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="dudx (nodal)" Format="ascii">'
do i=1,np
write(123,*) dudx_nodal(i)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="dvdy (nodal)" Format="ascii">'
do i=1,np
write(123,*) dvdy_nodal(i)
end do
write(123,*) '</DataArray>'


write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity(analytical)" Format="ascii">'
do i=1,np
write(123,*) uth(x(i),y(i),ibench),vth(x(i),y(i),ibench),0. 
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity error" Format="ascii">'
do i=1,np
write(123,*) Vx(i)-uth(x(i),y(i),ibench),Vy(i)-vth(x(i),y(i),ibench),0. 
end do
write(123,*) '</DataArray>'

!write(123,*) '<DataArray type="Float32" Name="mueff (nodal)" Format="ascii">'
!do i=1,np
!write(123,*) mueff(i)
!end do
!write(123,*) '</DataArray>'

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
write(123,*) '<DataArray type="Float32" Name="pressure" Format="ascii">'
do iel=1,nel
write(123,*) p(iel)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="pressure res" Format="ascii">'
do iel=1,nel
write(123,*) Rp(iel)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="pressure(analytical)" Format="ascii">'
do iel=1,nel
xc=sum(x(icon(:,iel)))*0.25d0
yc=sum(y(icon(:,iel)))*0.25d0
write(123,*) pth(xc,yc,ibench)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="pressure error" Format="ascii">'
do iel=1,nel
xc=sum(x(icon(:,iel)))*0.25d0
yc=sum(y(icon(:,iel)))*0.25d0
write(123,*) p(iel)-pth(xc,yc,ibench)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="dudx (elemental)" Format="ascii">'
do iel=1,nel
xc=sum(x(icon(:,iel)))*0.25d0
yc=sum(y(icon(:,iel)))*0.25d0
write(123,*) dudxth(xc,yc,ibench)
end do
write(123,*) '</DataArray>'

write(123,*) '<DataArray type="Float32" Name="dvdy (elemental)" Format="ascii">'
do iel=1,nel
xc=sum(x(icon(:,iel)))*0.25d0
yc=sum(y(icon(:,iel)))*0.25d0
write(123,*) dvdyth(xc,yc,ibench)
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
