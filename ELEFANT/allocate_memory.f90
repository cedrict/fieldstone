!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine allocate_memory

use module_parameters, only: NV,NP,NT,ndofV,NfemV,NfemP,NfemT,iproc,nmat,ndim,debug
use module_arrays, only: solV,solP,rhs_f,rhs_h,Kdiag
use module_materials
use module_timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsection{allocate\_memory}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

ndofV=ndim
NfemV=NV*ndofV
NfemP=NP
NfemT=NT

allocate(solV(NfemV))
allocate(solP(NfemP))
allocate(rhs_f(NfemV))
allocate(rhs_h(NfemP))
allocate(materials(nmat))
allocate(Kdiag(NfemV))

if (debug) then
write(2345,*) limit//'allocate_memory'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'allocate_memory (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
