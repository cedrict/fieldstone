!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine matrix_setup_K

use global_parameters
use structures
use timing

implicit none

integer :: k

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{matrix\_setup\_K}
!@@
!==================================================================================================!

if (iproc==0) call system_clock(counti,count_rate)

!==============================================================================!

Nel=ndofV*mV          ! size of an elemental matrix

idV%N=NfemV

idV%NELT=nel
LELTVAR=nel*Nel           ! nb of elts X size of elemental matrix
NA_ELT=nel*Nel*(Nel+1)/2  ! nb of elts X nb of nbs in elemental matrix

allocate(idV%A_ELT (NA_ELT)) 
allocate(idV%RHS   (idV%N))  

if (iproc==0) then

   allocate(idV%ELTPTR(idV%NELT+1)) 
   allocate(idV%ELTVAR(LELTVAR))    

   !=====[building ELTPTR]=====

   do iel=1,nel
      idV%ELTPTR(iel)=1+(iel-1)*(ndofV*mV)
   end do
   idV%ELTPTR(iel)=1+nel*(ndofV*mV)

   !=====[building ELTVAR]=====

   counter=0
   do iel=1,nel
      do k=1,mV
         inode=mesh(iel)%iconV(k)
         do idof=1,ndofV
            iii=(inode-1)*ndofV+idof
            counter=counter+1
            idV%ELTVAR(counter)=iii
         end do
      end do
   end do

end if ! iproc=0

!==============================================================================!

if (iproc==0) then 

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,*) '     -> matrix_setup_K ',elapsed

end if

end subroutine

!==================================================================================================!
!==================================================================================================!
