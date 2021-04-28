!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine make_matrix_stokes

use global_parameters
use global_arrays
use structures
use timing
use matrices, only : csrGT,csrK,csrMP

implicit none

integer counter_mumps,k1,k2,ikk,jkk,i1,i2,m1,m2,ik,k,jk ! <- too many ?!
real(8) :: h_el(mP)
real(8) :: f_el(mV*ndofV)
real(8) :: K_el(mV*ndofV,mV*ndofV)
real(8) :: G_el(mV*ndofV,mP)
real(8) :: C_el(mP,mP)
real(8) :: S_el(mP,mP)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{make\_matrix\_stokes.f90}
!@@ This subroutine loops over all elements, build their elemental matrices and rhs, 
!@@ apply the boundary conditions ont these elemental matrices, and then 
!@@ assembles them in the global matrix, either in CSR or in MUMPS format.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

C_el=0.d0

counter_mumps=0
!idV%RHS=0.d0
!idV%A_ELT=0.d0
csrGT%mat=0d0
Kdiag=0d0
rhs_f=0.
rhs_h=0.
if (allocated(csrK%mat)) csrK%mat=0d0 

do iel=1,nel

   call compute_elemental_matrix_stokes(K_el,G_el,f_el,h_el)

   call impose_boundary_conditions_stokes(K_el,G_el,f_el,h_el)

   !--------------------
   !assemble GT, f and h
   !--------------------

   if (pair=='q1p0') then
      csrGT%mat(csrGT%ia(iel):csrGT%ia(iel+1)-1)=G_el(:,1)
      rhs_h(iel)=rhs_h(iel)+h_el(1)
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
!                     csrGT%mat(k)=csrGT%mat(k)+G_el(ikk,jkk)  
!                  end if    
!               end do    
            end do
         end do
      end do

      do k2=1,mP
         m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
         rhs_h(m2)=rhs_h(m2)+h_el(k2)
      end do

   end if

   !--------------------
   ! assemble K
   !--------------------

   if (use_MUMPS) then
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
!                     idV%A_ELT(counter_mumps)=K_el(ikk,jkk)
                  end if
               end do
            end do
            rhs_f(m1)=rhs_f(m1)+f_el(ikk)
         end do
      end do
   else
      do k1=1,mV    
         ik=mesh(iel)%iconV(k1)
         do i1=1,ndofV    
            ikk=ndofV*(k1-1)+i1    
            m1=ndofV*(ik-1)+i1    
            Kdiag(m1)=Kdiag(m1)+K_el(ikk,ikk)
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
            rhs_f(m1)=rhs_f(m1)+f_el(ikk)    
         end do    
      end do   

   end if

   ! build elemental approximate Schur complement
   ! only keep diagonal of K
   ! should this happen before bc are applied?
   ! add C_el ?
   do k1=1,mV
   do k2=1,mV
      if (k1/=k2) K_el(k1,k2)=0d0
      if (k1==k2) K_el(k1,k2)=1d0/K_el(k1,k2)
   end do
   end do
   S_el=matmul(transpose(G_el),matmul(K_el,G_el))

   !assemble approx Schur complement

   do k1=1,mP
      m1=mesh(iel)%iconP(k1) ! global coordinate of pressure dof
      do k2=1,mP
         m2=mesh(iel)%iconP(k2) ! global coordinate of pressure dof
         do k=csrMP%ia(m1),csrMP%ia(m1+1)-1    
            if (csrMP%ja(k)==m2) then  
               csrMP%mat(k)=csrMP%mat(k)+S_el(k1,k2)  
            end if    
         end do
      end do
   end do

end do

!csrGT%mat=csrGT%mat*block_scaling_coeff
!rhs_h=rhs_h*block_scaling_coeff

                         write(*,'(a,2es12.4)') '        rhs_f (m/M):   ',minval(rhs_f),maxval(rhs_f)
if (allocated(csrK%mat)) write(*,'(a,2es12.4)') '        csrK%mat (m/M):',minval(csrK%mat),maxval(csrK%mat)
                         write(*,'(a,2es12.4)') '        Kdiag (m/M):   ',minval(Kdiag),maxval(Kdiag)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> make_matrix\_stokes              ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
