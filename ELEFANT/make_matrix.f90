!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine make_matrix

use global_parameters
use structures
!use constants

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsection{make\_matrix.f90}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!


do iel=1,nel

   call compute_elemental_matrices(Kel,Gel,fel,hel)

   call impose_boundary_conditions(Kel,fel,Gel,hel)

   !--------------------
   !assemble GT, f and h
   !--------------------

   if (pair=='q1p0') then
      csrGT%mat(csrGT%ia(iel):csrGT%ia(iel+1)-1)=Gel(:,1)
      rhs_h(iel,1)=rhs_h(iel,1)+hel(1)
   end if

   if (pair=='q1q1') then

      do k1=1,mV
         ik=grid%icon(k1,iel)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1 ! local coordinate of velocity dof
            m1=ndofV*(ik-1)+i1  ! global coordinate of velocity dof                             
            do k2=1,mP
               jkk=k2               ! local coordinate of pressure dof
               m2=gridP%icon(k2,iel) ! global coordinate of pressure dof
               do k=csrGT%ia(m2),csrGT%ia(m2+1)-1    
                  if (csrGT%ja(k)==m1) then  
                     csrGT%mat(k)=csrGT%mat(k)+Gel(ikk,jkk)  
                  end if    
               end do    
            end do
         end do
      end do

      do k2=1,mP
         m2=gridP%icon(k2,iel) ! global coordinate of pressure dof
         rhs_h(m2,1)=rhs_h(m2,1)+hel(k2)
      end do

   end if

   !--------------------
   ! assemble K
   !--------------------

   do k1=1,mV
         ik=grid%icon(k1,iel)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1
            m1=ndofV*(ik-1)+i1
            do k2=1,mV
               do i2=1,ndofV
                  jkk=ndofV*(k2-1)+i2
                  if (jkk>=ikk) then
                     counter_mumps=counter_mumps+1
                     idV%A_ELT(counter_mumps)=Kel(ikk,jkk)
                  end if
               end do
            end do
            rhs_f(m1,1)=rhs_f(m1,1)+fel(ikk)
         end do
   end do


end do

csrGT%mat=csrGT%mat*block_scaling_coeff
rhs_h=rhs_h*block_scaling_coeff

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> make_matrix ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
