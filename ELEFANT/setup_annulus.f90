!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_annulus

use module_parameters, only: iproc,debug
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_annulus}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


!mesh(iel)%inner_elt=.false.
!mesh(iel)%outer_elt=

!========================
! generate box-like grid
!========================

sx = Louter / dble (ncellt)
sz = Lr     / dble (ncellr)
sy = sz
    
counter=0    
do j=0,ncellr    
do i=1,ncellt    
   counter=counter+1  
   grid%x(counter)=dble(i-1)*sx   
   grid%z(counter)=dble(j)*sz  
   if (j==0)      grid%node_inner(counter)=.true.
   if (j==ncellr) grid%node_outer(counter)=.true.
end do    
end do    

!=====================




if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_annulus (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
