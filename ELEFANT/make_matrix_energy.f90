!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine make_matrix_energy

use module_parameters
use module_arrays, only: rhs_b
use module_sparse, only: csrA
use module_mesh 
use module_timing

implicit none

integer inode,jnode,k,k1,k2
real(8) :: Ael(mT,mT),bel(mT)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{make\_matrix\_energy}
!@@ This subroutine builds the linear system for the energy equation. 
!@@ It loops over each element, builds its elemental matrix ${\bm A}_{el}$
!@@ and right hand side $\vec{b}_{el}$, applies boundary conditions, 
!@@ and assembles these into the global matrix csrA and the corresponding 
!@@ right hand side rhs\_b. 
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

csrA%mat=0d0
rhs_b=0d0

do iel=1,nel

   call compute_elemental_matrix_energy(Ael,bel)
   call impose_boundary_conditions_energy(Ael,bel)

   do k1=1,mT
      inode=mesh(iel)%iconT(k1)
      do k2=1,mT
         jnode=mesh(iel)%iconT(k2)
         do k=csrA%ia(inode),csrA%ia(inode+1)-1
            if (csrA%ja(k)==jnode) then
               csrA%mat(k)=csrA%mat(k)+Ael(k1,k2)
               exit
            end if
         end do
      end do
      rhs_b(inode)=rhs_b(inode)+bel(k1)
   end do

end do

write(*,'(a,2es12.4)') '        mat (m/M)',minval(csrA%mat),maxval(csrA%mat)
write(*,'(a,2es12.4)') '        rhs (m/M)',minval(rhs_b),maxval(rhs_b)

write(1235,'(4es12.4)') minval(csrA%mat),maxval(csrA%mat),minval(rhs_b),maxval(rhs_b)
call flush(1235)

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'make_matrix_energy (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
