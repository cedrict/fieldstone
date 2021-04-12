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
use timing

implicit none

integer counter_mumps,k1,k2,ikk,jkk,i1,i2,m1,m2,ik ! <- too many ?!
real(8) :: hel(mP)
real(8) :: fel(mV*ndofV)
real(8) :: Kel(mV*ndofV,mV*ndofV)
real(8) :: Gel(mV*ndofV,mP)
real(8) :: Cel(mP,mP)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{make\_matrix.f90}
!@@ This subroutine loops over all elements, build their elemental matrices and rhs, 
!@@ apply the boundary conditions ont these elemental matrices, and then 
!@@ assembles them in the global matrix, either in CSR or in MUMPS format.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

Cel=0.d0

counter_mumps=0
!idV%RHS=0.d0
!idV%A_ELT=0.d0
!csrGT%mat=0

do iel=1,nel

   call compute_elemental_matrices(Kel,Gel,fel,hel)

!   call impose_boundary_conditions(Kel,Gel,fel,hel)

   !--------------------
   !assemble GT, f and h
   !--------------------

   if (pair=='q1p0') then
!      csrGT%mat(csrGT%ia(iel):csrGT%ia(iel+1)-1)=Gel(:,1)
!      rhs_h(iel,1)=rhs_h(iel,1)+hel(1)
   end if

   if (pair=='q1q1') then

      do k1=1,mV
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1 ! local coordinate of velocity dof
            m1=ndofV*(ik-1)+i1  ! global coordinate of velocity dof                             
            do k2=1,mP
               jkk=k2                 ! local coordinate of pressure dof
               m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
!               do k=csrGT%ia(m2),csrGT%ia(m2+1)-1    
!                  if (csrGT%ja(k)==m1) then  
!                     csrGT%mat(k)=csrGT%mat(k)+Gel(ikk,jkk)  
!                  end if    
!               end do    
            end do
         end do
      end do

      do k2=1,mP
         m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
!         rhs_h(m2,1)=rhs_h(m2,1)+hel(k2)
      end do

   end if

   !--------------------
   ! assemble K
   !--------------------

   do k1=1,mV
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV
            ikk=ndofV*(k1-1)+i1
            m1=ndofV*(ik-1)+i1
            do k2=1,mV
               do i2=1,ndofV
                  jkk=ndofV*(k2-1)+i2
                  if (jkk>=ikk) then
                     counter_mumps=counter_mumps+1
!                     idV%A_ELT(counter_mumps)=Kel(ikk,jkk)
                  end if
               end do
            end do
!            rhs_f(m1,1)=rhs_f(m1,1)+fel(ikk)
         end do
   end do


end do

!csrGT%mat=csrGT%mat*block_scaling_coeff
!rhs_h=rhs_h*block_scaling_coeff

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> make_matrix ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
