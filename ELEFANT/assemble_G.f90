!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assemble_G(Gel)

!use module_parameters
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsection{template}
!@@
!==================================================================================================!

if (use_penalty) exit

select case(G_storage)

case(matrix_CSR)

      select case(spacePressure)

      case('__Q0','__P0')
      csrGT%mat(csrGT%ia(iel):csrGT%ia(iel+1)-1)=G_el(:,1)
      rhs_h(iel)=rhs_h(iel)+h_el(1)

      case default

         do k1=1,mV
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1 ! local coordinate of velocity dof
            m1=ndofV*(ik-1)+i1  ! global coordinate of velocity dof                             
            do k2=1,mP
               jkk=k2                 ! local coordinate of pressure dof
               m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
               do k=csrGT%ia(m2),csrGT%ia(m2+1)-1    
                  if (csrGT%ja(k)==m1) then  
                     csrGT%mat(k)=csrGT%mat(k)+G_el(ikk,jkk)  
                  end if    
               end do    
            end do
         end do
         end do

      end select

case(blocks_CSR)

case default
   stop 'assemble_G: unknown G_storage'
end select


end subroutine

!==================================================================================================!
!==================================================================================================!
