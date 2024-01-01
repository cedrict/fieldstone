!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine allocate_memory

use module_parameters, only: NU,NV,NW,NP,NT,NfemV,NfemP,NfemT,iproc,nmat,debug,mU,mV,mW,mP,mT
use module_arrays, only: dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz,&
                         NNNU,NNNV,NNNW,NNNP,NNNT,dNNNTdx,dNNNTdy,dNNNTdz,solV,solP,rhs_f,rhs_h,Kdiag
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

NfemV=NU+NV+NW
NfemP=NP
NfemT=NT

write(*,'(a,i7)') shift//'NfemV=',NfemV
write(*,'(a,i7)') shift//'NfemP=',NfemP
write(*,'(a,i7)') shift//'NfemT=',NfemT

allocate(solV(NfemV))
allocate(solP(NfemP))
allocate(rhs_f(NfemV))
allocate(rhs_h(NfemP))
allocate(materials(nmat))
allocate(Kdiag(NfemV))

allocate(NNNU(mU))
allocate(NNNV(mV))
allocate(NNNW(mW))
allocate(NNNT(mT))
allocate(NNNP(mP))

allocate(dNNNUdx(mU))
allocate(dNNNVdx(mV))
allocate(dNNNWdx(mW))
allocate(dNNNTdx(mT))

allocate(dNNNUdy(mU))
allocate(dNNNVdy(mV))
allocate(dNNNWdy(mW))
allocate(dNNNTdy(mT))

allocate(dNNNUdz(mU))
allocate(dNNNVdz(mV))
allocate(dNNNWdz(mW))
allocate(dNNNTdz(mT))

if (debug) then
write(2345,*) limit//'allocate_memory'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'allocate_memory:',elapsed,' s                |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
