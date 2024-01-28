!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine make_matrix_stokes

use module_parameters, only: mU,mV,mW,mP,iproc,iel,nel,K_storage,GT_storage,mVel
use module_arrays
use module_mesh 
use module_timing
use module_sparse, only : csrGT,csrK,csrGxT,csrGyT,csrGzT,csrKxx,csrKxy,csrKxz,&
                          csrKyx,csrKyy,csrKyz,csrKzx,csrKzy,csrKzz,cooK
use module_MUMPS

implicit none

real(8) :: h_el(mP)
real(8) :: f_el(mVel)
real(8) :: K_el(mVel,mVel)
real(8) :: G_el(mVel,mP)
real(8) :: S_el(mP,mP)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{make\_matrix\_stokes.f90}
!@@ This subroutine loops over all elements, build their elemental matrices and rhs, 
!@@ apply the boundary conditions ont these elemental matrices, and then 
!@@ assembles them in the global matrix, either in CSR or in MUMPS format.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a,4i3)') shift//'mU,mV,mW,mP=',mU,mV,mW,mP
write(*,'(a,a)') shift//'K_storage: ',K_storage
write(*,'(a,a)') shift//'GT_storage: ',GT_storage

!----------------------------------------------------------
! zero matrix arrays
!----------------------------------------------------------

rhs_f=0.
rhs_h=0.

select case(GT_storage)
case('none')
case('matrix_FULL')
   GT_matrix=0.d0
case('matrix_CSR')
   csrGT%mat=0d0
case('blocks_CSR')
   csrGxT%mat=0d0
   csrGyT%mat=0d0
   csrGzT%mat=0d0
case default
   stop 'make_matrix_stokes: unknown GT_storage'
end select

select case(K_storage)
case('matrix_FULL')
   K_matrix=0.d0
case('matrix_MUMPS')
   idV%RHS=0.d0
   idV%A_ELT=0.d0
case('matrix_CSR')
   csrK%mat=0d0 
case('matrix_COO')
   cooK%mat=0d0 
case('blocks_CSR')
   csrKxx%mat=0d0 ; csrKxy%mat=0d0 ; csrKxz%mat=0d0 
   csrKyx%mat=0d0 ; csrKyy%mat=0d0 ; csrKyz%mat=0d0 
   csrKzx%mat=0d0 ; csrKzy%mat=0d0 ; csrKzz%mat=0d0 
!case('blocks_MUMPS')
case default
   stop 'make_matrix_stokes: unknown K_storage'
end select

!----------------------------------------------------------
!----------------------------------------------------------

do iel=1,nel

   !write(*,'(a,i4)') shift//'build eltal matrix for',iel

   call compute_elemental_matrix_stokes(K_el,G_el,f_el,h_el)
   call impose_boundary_conditions_stokes(K_el,G_el,f_el,h_el)
   call assemble_GT(G_el)
   call assemble_K(K_el)
   call assemble_RHS(f_el,h_el)
   !call compute_elemental_schur_complement(K_el,G_el,S_el_el)
   !call assemble_S(S_el)  not needed
   !call assemble_MP  not needed

end do

!----------------------------------------------------------

select case(K_storage)
case('matrix_FULL')
   write(*,'(a,2es12.5)') shift//'K (m,M):',minval(K_matrix),maxval(K_matrix)
case('matrix_MUMPS')
   write(*,'(a,2es12.5)') shift//'K (m,M):',minval(idV%A_ELT),maxval(idV%A_ELT)
   idV%A_ELT=0.d0
case('blocks_MUMPS')
case('matrix_CSR')
   write(*,'(a,2es12.5)') shift//'K (m,M):',minval(csrK%mat),maxval(csrK%mat)
case('blocks_CSR')
   write(*,'(a,2es12.5)') shift//'Kxx (m,M):',minval(csrKxx%mat),maxval(csrKxx%mat)
   write(*,'(a,2es12.5)') shift//'Kxy (m,M):',minval(csrKxy%mat),maxval(csrKxy%mat)
   write(*,'(a,2es12.5)') shift//'Kxz (m,M):',minval(csrKxz%mat),maxval(csrKxz%mat)
   write(*,'(a,2es12.5)') shift//'Kyx (m,M):',minval(csrKyx%mat),maxval(csrKyx%mat)
   write(*,'(a,2es12.5)') shift//'Kyy (m,M):',minval(csrKyy%mat),maxval(csrKyy%mat)
   write(*,'(a,2es12.5)') shift//'Kyz (m,M):',minval(csrKyz%mat),maxval(csrKyz%mat)
   write(*,'(a,2es12.5)') shift//'Kzx (m,M):',minval(csrKzx%mat),maxval(csrKzx%mat)
   write(*,'(a,2es12.5)') shift//'Kzy (m,M):',minval(csrKzy%mat),maxval(csrKzy%mat)
   write(*,'(a,2es12.5)') shift//'Kzz (m,M):',minval(csrKzz%mat),maxval(csrKzz%mat)
case default
   stop 'make_matrix_stokes: unknown K_storage'
end select

select case(GT_storage)
case('none')
case('matrix_FULL')
   write(*,'(a,2es12.5)') shift//'GT (m,M):',minval(GT_matrix),maxval(GT_matrix)
case('matrix_CSR')
case('blocks_CSR')
case default
   stop 'make_matrix_stokes: unknown GT_storage'
end select

write(*,'(a,2es12.5)') shift//'f (m,M):',minval(rhs_f),maxval(rhs_f)
write(*,'(a,2es12.5)') shift//'h (m,M):',minval(rhs_h),maxval(rhs_h)

write(*,'(a,f6.3,a)') shift//'assemble_K   :',time_assemble_K,'s'
write(*,'(a,f6.3,a)') shift//'assemble_S   :',time_assemble_S,'s'
write(*,'(a,f6.3,a)') shift//'assemble_GT  :',time_assemble_GT,'s'
write(*,'(a,f6.3,a)') shift//'assemble_RHS :',time_assemble_RHS,'s'

!----------------------------------------------------------
!----------------------------------------------------------
!csrGT%mat=csrGT%mat*block_scaling_coeff
!rhs_h=rhs_h*block_scaling_coeff

!                         write(*,'(a,2es12.4)') shift//'rhs_f (m/M):    ',minval(rhs_f),maxval(rhs_f)
!if (allocated(csrK%mat)) write(*,'(a,2es12.4)') shift//'csrK%mat (m/M): ',minval(csrK%mat),maxval(csrK%mat)
!if () write(*,'(a,2es15.4)') shift//'idV%A_ELT (m/M):',minval(idV%A_ELT),maxval(idV%A_ELT)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'make_matrix_stokes:',elapsed,' s             |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
