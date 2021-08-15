!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine write_field_dp(np,array,fieldname,iunit)

implicit none

integer, intent(in) :: np,iunit
real(8), dimension(np), intent(in) :: array
character(*) fieldname

integer ip

write(iunit,*) '<DataArray type="Float32" Name="'//trim(fieldname)//'" Format="ascii">'
do ip=1,np
write(iunit,'(es35.12)') array(ip)
end do
write(iunit,*) '</DataArray>'

end subroutine

!==================================================================================================!

subroutine write_field_dp2(np,array,fieldname,iunit)

implicit none

integer, intent(in) :: np,iunit
real(8), dimension(np), intent(in) :: array
character(*) fieldname

integer ip

write(iunit,*) '<DataArray type="Float32" Name="'//trim(fieldname)//'" Format="ascii">'
do ip=1,np
write(iunit,'(f35.12)') array(ip)
end do
write(iunit,*) '</DataArray>'

end subroutine


!==================================================================================================!
!==================================================================================================!

subroutine write_field_int(np,array,fieldname,iunit)

implicit none

integer, intent(in) :: np,iunit
integer, dimension(np), intent(in) :: array
character(*) fieldname

integer ip

write(iunit,*) '<DataArray type="Float32" Name="'//trim(fieldname)//'" Format="ascii">'
do ip=1,np
write(iunit,*) array(ip)
end do
write(iunit,*) '</DataArray>'

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine write_field_bool(np,array,fieldname,iunit)

implicit none

integer, intent(in) :: np,iunit
logical(1), dimension(np), intent(in) :: array
character(*) fieldname

integer ip

write(iunit,*) '<DataArray type="Float32" Name="'//trim(fieldname)//'" Format="ascii">'
do ip=1,np
if (array(ip)) then
write(iunit,*) 1
else
write(iunit,*) 0
end if
end do
write(iunit,*) '</DataArray>'

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine write_positions(np,x,y,z,iunit,scalex,scaley,scalez)

implicit none

integer, intent(in) :: np,iunit
real(8), dimension(np), intent(in) :: x,y,z
real(8), intent(in) :: scalex,scaley,scalez

integer ip

write(iunit,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
do ip=1,np
write(iunit,'(3es30.15)') x(ip)/scalex,y(ip)/scaley,z(ip)/scalez
end do
write(iunit,*) '</DataArray>'

end subroutine

!==================================================================================================!
!==================================================================================================!
subroutine write_icon(mpe,nel,icon,iunit,discretisation)

implicit none

integer, intent(in) :: mpe,nel,iunit
integer, dimension(mpe,nel) :: icon
character(len=12) discretisation

integer iel

select case(trim(discretisation))
case('Q1_2D')
   write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
   do iel=1,nel
   write(iunit,*) icon(1:4,iel)-1
   end do
   write(iunit,*) '</DataArray>'
case('Q2_2D')
   write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
   do iel=1,nel
   write(iunit,*) icon(1,iel)-1,icon(3,iel)-1,icon(9,iel)-1,icon(7,iel)-1
   end do
   write(iunit,*) '</DataArray>'
case('Q3_2D')
   write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
   do iel=1,nel
   write(iunit,*) icon(1,iel)-1,icon(4,iel)-1,icon(16,iel)-1,icon(13,iel)-1
   end do
   write(iunit,*) '</DataArray>'
case('Q1_3D','Q2_3D')
   write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
   do iel=1,nel
   write(iunit,*) icon(1:8,iel)-1
   end do
   write(iunit,*) '</DataArray>'
case('P1_2D','P1+_2D') 
   write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
   do iel=1,nel
   write(iunit,*) icon(1:3,iel)-1
   end do
   write(iunit,*) '</DataArray>'
case default 
   stop 'write_icon: discretisation is unknown'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine write_offsets(mpe,nel,iunit,discretisation)

implicit none

integer, intent(in) :: nel, iunit,mpe
character(len=12) discretisation
integer iel

select case(trim(discretisation))
case('Q1_2D','Q2_2D','Q3_2D')
   write(iunit,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
   write(iunit,*) (iel*4,iel=1,nel) 
   write(iunit,*) '</DataArray>'
case('Q1_3D','Q2_3D')
   write(iunit,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
   write(iunit,*) (iel*8,iel=1,nel) 
   write(iunit,*) '</DataArray>'
case('P1_2D','P1+_2D') 
   write(iunit,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
   write(iunit,*) (iel*3,iel=1,nel) 
   write(iunit,*) '</DataArray>'
case default 
   stop 'write_icon: discretisation is unknown'
end select

end subroutine

!==================================================================================================!
!==================================================================================================!

subroutine write_types(mpe,nel,iunit,discretisation)

implicit none

integer, intent(in) :: nel, iunit,mpe
character(len=12), intent(in) :: discretisation
integer iel,celltype

select case(trim(discretisation))
case('Q1_2D','Q2_2D','Q3_2D')
celltype=9
case('Q1_3D','Q2_3D')
celltype=12
case('P1_2D','P1+_2D') 
celltype=5
case default 
   stop 'write_icon: discretisation is unknown'
end select

write(iunit,*) '<DataArray type="Int32" Name="types" Format="ascii">'
write(iunit,*) (celltype,iel=1,nel)
write(iunit,*) '</DataArray>'

end subroutine

!==================================================================================================!
!==================================================================================================!











