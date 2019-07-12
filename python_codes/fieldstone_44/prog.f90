program opla
implicit none
!integer, parameter :: nnp=193785
!integer, parameter :: nel=63590
integer, parameter :: nnp=1615231
integer, parameter :: nel=534363
real(8) :: x(nnp),y(nnp)
real(8) :: iconR(7,nel)
integer :: icon(7,nel)
integer i,iel

open(unit=123,file='GCOORD.txt',action='read')
read(123,*) x(1:nnp)
read(123,*) y(1:nnp)
close(123)

print *,minval(x),maxval(x)
print *,minval(y),maxval(y)


open(unit=123,file='ELEM2NODE.txt',action='read')
read(123,*) iconR(1,1:nel) 
read(123,*) iconR(2,1:nel) 
read(123,*) iconR(3,1:nel) 
read(123,*) iconR(4,1:nel) 
read(123,*) iconR(5,1:nel) 
read(123,*) iconR(6,1:nel) 
read(123,*) iconR(7,1:nel) 
close(123)

print *,minval(iconR(1,:)),maxval(iconR(1,:))
print *,minval(iconR(2,:)),maxval(iconR(2,:))
print *,minval(iconR(3,:)),maxval(iconR(3,:))
print *,minval(iconR(4,:)),maxval(iconR(4,:))
print *,minval(iconR(5,:)),maxval(iconR(5,:))
print *,minval(iconR(6,:)),maxval(iconR(6,:))
print *,minval(iconR(7,:)),maxval(iconR(7,:))

icon=iconR

print *,minval(icon(1,:)),maxval(icon(1,:))
print *,minval(icon(2,:)),maxval(icon(2,:))
print *,minval(icon(3,:)),maxval(icon(3,:))
print *,minval(icon(4,:)),maxval(icon(4,:))
print *,minval(icon(5,:)),maxval(icon(5,:))
print *,minval(icon(6,:)),maxval(icon(6,:))
print *,minval(icon(7,:)),maxval(icon(7,:))

open(unit=123,file='visu.vtu',status='replace',form='formatted')
write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
write(123,*) '<UnstructuredGrid>'
write(123,*) '<Piece NumberOfPoints="',nnp,'" NumberOfCells="',nel,'">'
!.............................
!write(123,*) '<PointData Scalars="scalars">'

!write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Name="velocity" Format="ascii">'
!do i=1,np
!write(123,'(3f30.15)') Vx(i),Vy(i),0.d0
!end do
!write(123,*) '</DataArray>'

!write(123,*) '</PointData>'

write(123,*) '<Points>'
write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do i=1,nnp
write(123,*) x(i),y(i),0.d0
end do
write(123,*) '</DataArray>'
write(123,*) '</Points>'
!.............................
write(123,*) '<Cells>'
write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
do iel=1,nel
write(123,*) icon(1,iel)-1,icon(2,iel)-1,icon(3,iel)-1,icon(6,iel)-1,icon(4,iel)-1,icon(5,iel)-1
end do
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
write(123,*) (iel*6,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(123,*) (22,iel=1,nel)
write(123,*) '</DataArray>'
write(123,*) '</Cells>'
write(123,*) '</Piece>'
write(123,*) '</UnstructuredGrid>'
write(123,*) '</VTKFile>'
close(123)






end program

