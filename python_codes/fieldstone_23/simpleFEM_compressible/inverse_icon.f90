!Subroutine to generate an "inverse" connectivity matrix
!Defines which elements are adjacent to each node
subroutine inverse_icon(icon,icon_inv,nel,np,m)

integer nel,np,m,counter
integer, dimension(m,nel) :: icon
integer, dimension(m,np)  :: icon_inv
integer ip,iel

icon_inv=0
do ip=1,np
   counter=1
   do iel=1,nel
      do k=1,m
      if (ip .eq. icon(k,iel)) then
      icon_inv(counter,ip) = iel
      counter=counter + 1
      end if 
      end do 
   end do 
end do 

end subroutine
