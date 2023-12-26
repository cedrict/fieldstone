module module_export_vtu
implicit none

contains 


!----------------------------------------------------------

subroutine write_elemental_field(iunit,naam)
use module_mesh
use module_parameters, only: iel,nel
integer, intent(in) :: iunit
character(*), intent(in) :: naam

select case(trim(naam))

!----------
case('vol')
   write(iunit,'(a)') '<DataArray type="Float32" Name="vol" Format="ascii">'
   do iel=1,nel
   write(iunit,'(es12.4)') mesh(iel)%vol
   end do
   write(iunit,'(a)') '</DataArray>'

!----------
case('a_rho')
   write(iunit,'(a)') '<DataArray type="Float32" Name="a_rho" Format="ascii">'
   do iel=1,nel
   write(iunit,'(es12.4)') mesh(iel)%a_rho
   end do
   write(iunit,'(a)') '</DataArray>'


end select


end subroutine







subroutine write_int_elemental_field_to_vtu(iunit,nel,field,naam)
implicit none
integer, intent(in) :: iunit
integer, intent(in) :: nel
integer, dimension(nel) :: field
character(*) naam
integer iel
write(iunit,*) '<DataArray type="Int32" Name="'//trim(naam)//'" Format="ascii">'
do iel=1,nel
   write(iunit,'(i5)') field(iel)
end do
write(iunit,*) '</DataArray>'
end subroutine

!----------------------------------------------------------

subroutine write_dp_elemental_field_to_vtu(iunit,nel,field,naam)
implicit none
integer, intent(in) :: iunit
integer, intent(in) :: nel
real(8), dimension(nel) :: field
character(*) naam
integer iel
write(iunit,*) '<DataArray type="Float32" Name="'//trim(naam)//'" Format="ascii">'
do iel=1,nel
   write(iunit,'(es12.4)') field(iel)
end do
write(iunit,*) '</DataArray>'
end subroutine

!----------------------------------------------------------

subroutine write_logical1_elemental_field_to_vtu(iunit,nel,field,naam)
implicit none
integer, intent(in) :: iunit
integer, intent(in) :: nel
logical(1), dimension(nel) :: field
character(*) naam
integer iel
write(iunit,*) '<DataArray type="Float32" Name="'//trim(naam)//'" Format="ascii">'
do iel=1,nel
   if (field(iel)) then
      write(iunit,'(i1)') 1
   else
      write(iunit,'(i1)') 0 
   end if
end do
write(iunit,*) '</DataArray>'
end subroutine

!----------------------------------------------------------

subroutine write_connectivity_array_to_vtu(iunit,nel,m,icon,space,ndim)
implicit none
integer, intent(in) :: iunit
integer, intent(in) :: nel
integer, intent(in) :: m,ndim
integer, dimension(m,nel) :: icon
character(*) space

integer iel,k

write(iunit,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
select case(space)
case('__Q1','__P1','__P2')
   if (ndim==2) then 
      do iel=1,nel
         write(iunit,*) ( (iel-1)*m+k-1,k=1,m) 
      end do
   else
      stop 'write_connectivity_array_to_vtu: ndim=3 not supported yet'
   end if
case('__Q2')
   if (ndim==2) then 
      do iel=1,nel
         write(iunit,*) (iel-1)*m+1-1,(iel-1)*m+3-1,(iel-1)*m+9-1,&
                        (iel-1)*m+7-1,(iel-1)*m+2-1,(iel-1)*m+6-1,&
                        (iel-1)*m+8-1,(iel-1)*m+4-1,(iel-1)*m+5-1
      end do
   else
      stop 'write_connectivity_array_to_vtu: ndim=3 not supported yet'
   end if
case default
   stop 'write_connectivity_array_to_vtu: space unknown'
end select
write(iunit,*) '</DataArray>'

end subroutine

!----------------------------------------------------------





end module
