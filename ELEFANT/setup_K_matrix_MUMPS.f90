!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_K_matrix_MUMPS

use module_parameters, only: mVel,NfemVel,iproc,mV,ndofV,debug,nel
use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing
use module_MUMPS

implicit none

integer :: inode,iel,k,idof,LELTVAR,NA_ELT,counter

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_K\_matrix\_MUMPS}
!@@
! see matrix_setup_K_MUMPS.f90 in old ELEFANT
! see matrix_setup_K_SPARSKIT.f90 in old ELEFANT
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

idV%N=NfemVel

idV%NELT=nel
LELTVAR=nel*mVel           ! nb of elts X size of elemental matrix
NA_ELT=nel*mVel*(mVel+1)/2  ! nb of elts X nb of nbs in elemental matrix

allocate(idV%A_ELT (NA_ELT)) 
allocate(idV%RHS   (idV%N))  

   if (iproc==0) then

      allocate(idV%ELTPTR(idV%NELT+1)) 
      allocate(idV%ELTVAR(LELTVAR))    

      !=====[building ELTPTR]=====

      do iel=1,nel
         idV%ELTPTR(iel)=1+(iel-1)*(mVel)
      end do
      idV%ELTPTR(iel)=1+nel*(mVel)

      !=====[building ELTVAR]=====

      counter=0
      do iel=1,nel
         do k=1,mV
            inode=mesh(iel)%iconV(k)
            do idof=1,ndofV
               counter=counter+1
               idV%ELTVAR(counter)=(inode-1)*ndofV+idof
            end do
         end do
      end do

   end if



if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_K_matrix_MUMPS:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
