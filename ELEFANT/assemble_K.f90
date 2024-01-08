!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assemble_K(K_el)

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
!@@ \subsection{assemble\_K}
!@@
!==================================================================================================!

select case(K_storage)

case('matrix_MUMPS')

      do k1=1,mV
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1
            m1=ndofV*(ik-1)+i1
            Kdiag(m1)=Kdiag(m1)+K_el(ikk,ikk)
            do k2=1,mV
               do i2=1,ndofV
                  jkk=ndofV*(k2-1)+i2
                  if (jkk>=ikk) then
                     counter_mumps=counter_mumps+1
                     idV%A_ELT(counter_mumps)=K_el(ikk,jkk)
                  end if
               end do
            end do
            rhs_f(m1)=rhs_f(m1)+f_el(ikk)
         end do
      end do



case('blocks_MUMPS')

case('matrix_CSR')

      do k1=1,mV    
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV    
            ikk=ndofV*(k1-1)+i1    
            m1=ndofV*(ik-1)+i1    
            !Kdiag(m1)=Kdiag(m1)+K_el(ikk,ikk)
            do k2=1,mV    
               jk=mesh(iel)%iconV(k2)
               do i2=1,ndofV    
                  jkk=ndofV*(k2-1)+i2    
                  m2=ndofV*(jk-1)+i2    
                  ! ikk,jkk local integer coordinates in the elemental matrix
                  do k=csrK%ia(m1),csrK%ia(m1+1)-1    
                     if (csrK%ja(k)==m2) then  
                        csrK%mat(k)=csrK%mat(k)+K_el(ikk,jkk)  
                     end if    
                  end do    
               end do    
            end do    
         end do    
      end do   

case('blocks_CSR')

case default
   stop 'assemble_K: unknown K_storage'
end select


end subroutine

!==================================================================================================!
!==================================================================================================!
