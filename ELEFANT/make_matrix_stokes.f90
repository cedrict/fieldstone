!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine make_matrix_stokes

use module_parameters, only: mU,mV,mW,mP,iproc,iel,ndofV,spacePressure,ndim2,inner_solver_type,&
                             ndim,nel,use_penalty
use module_arrays
use module_mesh 
use module_timing
use module_sparse, only : csrGT,csrK,csrMP
use module_MUMPS

implicit none

integer counter_mumps,k1,k2,ikk,jkk,i1,i2,m1,m2,ik,k,jk ! <- too many ?!
real(8) :: h_el(mP)
real(8) :: f_el(mU+mV+mW)
real(8) :: K_el(mU+mV+mW,mU+mV+mW)
real(8) :: G_el(mU+mV+mW,mP)
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

write(*,'(a,i3)') shift//'mU=',mU
write(*,'(a,i3)') shift//'mV=',mV
write(*,'(a,i3)') shift//'mW=',mW
write(*,'(a,i3)') shift//'mP=',mP

!----------------------------------------------------------

rhs_f=0.
rhs_h=0.

select case(G_storage)
case('matrix_CSR')
   csrGT%mat=0d0
case('blocks_CSR')
   csrGxT%mat=0d0
   csrGyT%mat=0d0
   csrGzT%mat=0d0
case default
   stop 'make_matrix_stokes: unknown G_storage'
end select

select case(K_storage)
case('matrix_MUMPS')
   counter_mumps=0
   idV%RHS=0.d0
   idV%A_ELT=0.d0
case('blocks_MUMPS')
case('matrix_CSR')
   csrK%mat=0d0 
case('blocks_CSR')
case default
   stop 'make_matrix_stokes: unknown K_storage'
end select

!----------------------------------------------------------

do iel=1,nel

   print *,'building elemental matrix for',iel

   call compute_elemental_matrix_stokes(K_el,G_el,f_el,h_el)
   call compute_elemental_schur_complement(K_el,G_el,S_el)
   call impose_boundary_conditions_stokes(K_el,G_el,f_el,h_el)
   call assemble_G(G_el)
   call assemble_K(K_el)
   call assemble_RHS(f_el,h_el)
   call assemble_S(S_el)

end do

!csrGT%mat=csrGT%mat*block_scaling_coeff
!rhs_h=rhs_h*block_scaling_coeff

!----------------------------------------------------------

!                         write(*,'(a,2es12.4)') shift//'rhs_f (m/M):    ',minval(rhs_f),maxval(rhs_f)
!if (allocated(csrK%mat)) write(*,'(a,2es12.4)') shift//'csrK%mat (m/M): ',minval(csrK%mat),maxval(csrK%mat)
!if () write(*,'(a,2es15.4)') shift//'idV%A_ELT (m/M):',minval(idV%A_ELT),maxval(idV%A_ELT)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'make_matrix_stokes (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
