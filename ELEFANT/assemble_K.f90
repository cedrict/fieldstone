!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assemble_K(K_el)

use module_parameters, only: iel,K_storage,mVel 
use module_mesh 
use module_MUMPS
use module_sparse, only: csrK,cooK
use module_arrays, only: K_matrix 
use module_timing

implicit none

real(8), intent(in) :: K_el(mVel,mVel)

integer :: k,kV1,kV2,kkV1,kkV2,counter_mumps

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{assemble\_K}
!@@ This subroutine receives as argument $\K_e$ and assembles it into the global matrix $\K$.
!==================================================================================================!

call cpu_time(t3) 

select case(K_storage)

!__________________
case('matrix_FULL')

   do kV1=1,mVel
      kkV1=mesh(iel)%iconVel(kV1)
      do kV2=1,mVel
         kkV2=mesh(iel)%iconVel(kV2)
         K_matrix(kkV1,kkV2)=K_matrix(kkV1,kkV2)+K_el(kV1,kV2)
      end do
   end do

!__________________
case('matrix_MUMPS') ! symmetric storage

   counter_mumps=0
   do kV1=1,mVel
      kkV1=mesh(iel)%iconVel(kV1)
      do kV2=1,mVel
         kkV2=mesh(iel)%iconVel(kV2)
         if (kkV2>=kkV1) then
            counter_mumps=counter_mumps+1
            idV%A_ELT(counter_mumps)=K_el(kV1,kV2)
         end if
      end do
   end do

!__________________
case('matrix_CSR')

   do kV1=1,mVel
      kkV1=mesh(iel)%iconVel(kV1)
      do kV2=1,mVel
         kkV2=mesh(iel)%iconVel(kV2)
         do k=csrK%ia(kkV1),csrK%ia(kkV1+1)-1    
            if (csrK%ja(k)==kkV2) then  
               csrK%mat(k)=csrK%mat(k)+K_el(kV1,kV2)  
            end if    
         end do    
      end do
   end do

!__________________
case('matrix_COO')

   do kV1=1,mVel
      kkV1=mesh(iel)%iconVel(kV1)
      do kV2=1,mVel
         kkV2=mesh(iel)%iconVel(kV2)
         do k=csrK%ia(kkV1),csrK%ia(kkV1+1)-1    
            if (csrK%ja(k)==kkV2) then  
               cooK%mat(k)=cooK%mat(k)+K_el(kV1,kV2)  
            end if    
         end do
      end do
   end do

!__________________
case('blocks_MUMPS')

   stop 'assemble_K: blocks_MUMPS not available yet'

!__________________
case('blocks_CSR')

   stop 'assemble_K: blocks_CSR not available yet'

!___________
case default

   stop 'assemble_K: unknown K_storage'

end select

call cpu_time(t4) ; time_assemble_K=time_assemble_K+t4-t3

end subroutine

!==================================================================================================!
!==================================================================================================!
