!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine allocate_memory

use module_parameters, only: NU,NV,NW,NP,NT,NfemVel,NfemP,NfemT,iproc,nmat,mU,mV,mW,mP,mT,ndim,ndim2,use_T
use module_arrays, only: dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz,&
                         NNNU,NNNV,NNNW,NNNP,NNNT,dNNNTdx,dNNNTdy,dNNNTdz,solVel,solP,rhs_f,rhs_h,&
                         Kdiag,Cmat,Kmat,rhs_b,SolU,SolV,SolW,Kxxdiag,Kyydiag,Kzzdiag
use module_materials
use module_timing

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{allocate\_memory}
!@@ This subroutine essentially allocates all arrays in the {\tt module\_arrays}, i.e.
!@@ {\tt NNNU,NNNV,NNNW,NNNP,NNNT} and all space derivatives.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

NfemVel=NU+NV+NW
NfemP=NP
NfemT=NT

write(*,'(a,i7)') shift//'NfemVel=',NfemVel
write(*,'(a,i7)') shift//'NfemP=',NfemP
write(*,'(a,i7)') shift//'NfemT=',NfemT

allocate(solU(NU))        ; solU=0d0
allocate(solV(NV))        ; solV=0d0
allocate(solW(NW))        ; solW=0d0
allocate(solVel(NfemVel)) ; SolVel=0d0
allocate(solP(NfemP))     ; SolP=0d0
allocate(rhs_f(NfemVel))  ; rhs_f=0d0
allocate(rhs_h(NfemP))    ; rhs_h=0d0
allocate(materials(nmat))
allocate(Kdiag(NfemVel))
allocate(Kxxdiag(NU))
allocate(Kyydiag(NV))
allocate(Kzzdiag(NW))

if (use_T) allocate(rhs_b(NfemT))  

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

!----------------------------------------------------------

if (ndim==2) ndim2=3
if (ndim==3) ndim2=6
allocate(Cmat(ndim2,ndim2)) ; Cmat=0d0
allocate(Kmat(ndim2,ndim2)) ; Kmat=0d0
if (ndim==2) then
Cmat(1,1)=2d0 ; Cmat(2,2)=2d0 ; Cmat(3,3)=1d0
Kmat(1,1)=1d0 ; Kmat(1,2)=1d0 ; Kmat(2,1)=1d0 ; Kmat(2,2)=1d0
end if
if (ndim==3) then
Cmat(1,1)=2d0 ; Cmat(2,2)=2d0 ; Cmat(3,3)=2d0 
Cmat(4,4)=1d0 ; Cmat(5,5)=1d0 ; Cmat(6,6)=1d0 
Kmat(1,1)=1d0 ; Kmat(1,2)=1d0 ; Kmat(1,3)=1d0 
Kmat(2,1)=1d0 ; Kmat(2,2)=1d0 ; Kmat(2,3)=1d0 
Kmat(3,1)=1d0 ; Kmat(3,2)=1d0 ; Kmat(3,3)=1d0
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'allocate_memory:',elapsed,' s                |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
