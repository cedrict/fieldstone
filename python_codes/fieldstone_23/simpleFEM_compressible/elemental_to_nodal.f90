
subroutine elemental_to_nodal(field_elemental,field_nodal,icon,nel,np)
integer np,nel
real(8), dimension(np), intent(out) :: field_nodal
real(8),dimension(nel), intent(in) :: field_elemental
integer, dimension(4,nel), intent(in) :: icon
integer, dimension(:), allocatable :: countnode
integer, parameter :: m=4
integer ip

allocate(countnode(np))
field_nodal=0.d0
countnode=0
do ic=1,nel
   do kk=1,m
      j=icon(kk,ic)
      field_nodal(j)   = field_nodal(j)   + field_elemental(ic)
      countnode(j) = countnode(j) + 1
   end do
end do
    
do ip=1,np
   field_nodal(ip) = field_nodal(ip) / dble(countnode(ip))
end do
      
deallocate(countnode)


!do ip=1,np
!if (inv_icon(2,ip) .eq. 0) then !if node only adjacent to one element
!nodal(ip) = elemental(inv_icon(1,ip)) !nodal value equal to value at that element
!else if (inv_icon(3,ip) .eq. 0) then !if adjacent to two
!nodal(ip) = (elemental(inv_icon(1,ip)) + elemental(inv_icon(2,ip)))/2.d0 !equal to average value of the two
!else if (inv_icon(4,ip) .eq. 0) then !etc.
!nodal(ip) = (elemental(inv_icon(1,ip)) + elemental(inv_icon(2,ip)) + elemental(inv_icon(3,ip)))/3.d0
!else 
!nodal(ip) = (elemental(inv_icon(1,ip)) + elemental(inv_icon(2,ip)) + elemental(inv_icon(3,ip)) + elemental(inv_icon(4,ip)))/4.d0
!end if
!end do

end subroutine
