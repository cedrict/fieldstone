!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine assemble_GT(G_el)

use module_parameters, only: GT_storage,spacePressure,mP,iel,use_penalty,mVel
use module_mesh 
use module_arrays, only: GT_matrix
use module_sparse, only: csrGT
use module_timing

implicit none

real(8), intent(in) :: G_el(mVel,mP)

integer :: k,kV,kP,kkV,kkP

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{assemble_GT}
!@@ {\tt G\_el}is of size mVel*mP
!==================================================================================================!

if (use_penalty) return

select case(GT_storage)

!------------------
case('matrix_FULL')

   do kV=1,mVel
      kkV=mesh(iel)%iconVel(kV)
      do kP=1,mP
         kkP=mesh(iel)%iconP(kP) 
         GT_matrix(kkP,kkV)=GT_matrix(kkP,kkV)+G_el(kV,kP)  
      end do
   end do

!------------------
case('matrix_CSR')

   select case(spacePressure)

   !------------------
   case('__Q0','__P0')
      csrGT%mat(csrGT%ia(iel):csrGT%ia(iel+1)-1)=G_el(:,1)

   !-----------
   case default

      do kV=1,mVel
         kkV=mesh(iel)%iconVel(kV)
         do kP=1,mP
            kkP=mesh(iel)%iconP(kP) 
            do k=csrGT%ia(kkP),csrGT%ia(kkP+1)-1    
               if (csrGT%ja(k)==kkV) then  
                  csrGT%mat(k)=csrGT%mat(k)+G_el(kV,kP)  
               end if    
            end do    
         end do
      end do

   end select

!------------------
case('blocks_CSR')

   stop 'assemble_GT: blocks_CSR not supported'

!-----------
case default

   stop 'assemble_GT: unknown GT_storage'

end select

end subroutine

!==================================================================================================!
!==================================================================================================!
