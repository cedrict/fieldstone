!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_pair

use module_parameters
use module_arrays, only: rV,sV,tV,rT,sT,tT

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_global\_parameters\_pair}
!@@ This subroutine computes {\tt mV,mP,mT,mL}.
!==================================================================================================!

if (iproc==0) then

!==============================================================================!


if (pair=='q1p0') then 
   mV=2**ndim
   mP=1
   mT=2**ndim
   mL=2**ndim
   allocate(rV(mV))
   allocate(sV(mV))
   allocate(tV(mV))
   allocate(rT(mT))
   allocate(sT(mT))
   allocate(tT(mT))
   if (ndim==2) then
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)
      NT=(nelx+1)*(nely+1)
      rV=(/-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0/)
      tV=0d0
      rT=(/-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0/)
      tT=0d0
   else
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)
      NT=(nelx+1)*(nely+1)*(nelz+1)
      rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
      rT=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tT=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   end if
   NP=nel
end if

!----------------------------------------------------------

if (pair=='q1q1') then
   mP=2**ndim
   mT=2**ndim
   mL=2**ndim
   allocate(rT(mT))
   allocate(sT(mT))
   allocate(tT(mT))
   if (ndim==2) then
      mV=2**ndim+1
      nel=nelx*nely
      NV=(nelx+1)*(nely+1)+nel
      NT=(nelx+1)*(nely+1)
      NP=(nelx+1)*(nely+1)
      allocate(rV(mV))
      allocate(sV(mV))
      allocate(tV(mV))
      rV=(/-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0/)
      rT=(/-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0/)
   else
      mV=2**ndim+2
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NT=(nelx+1)*(nely+1)*(nelz+1)
      NP=(nelx+1)*(nely+1)*(nelz+1)
      allocate(rV(mV))
      allocate(sV(mV))
      allocate(tV(mV))
      rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
      rT=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tT=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   end if
end if

!----------------------------------------------------------

if (pair=='q2q1') then
   mV=3**ndim
   mP=2**ndim
   mT=3**ndim
   mL=2**ndim
   allocate(rV(mV))
   allocate(sV(mV))
   allocate(tV(mV))
   allocate(rT(mT))
   allocate(sT(mT))
   allocate(tT(mT))
   if (ndim==2) then
      nel=nelx*nely
      NV=(2*nelx+1)*(2*nely+1)
      NT=(2*nelx+1)*(2*nely+1)
      NP=(nelx+1)*(nely+1)
      rV=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sV=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   else
      nel=nelx*nely*nelz
      NV=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      NT=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      NP=(nelx+1)*(nely+1)*(nelz+1)
      rV=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0,&
           -1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0,&
           -1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sV=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0,&
           -1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0,&
           -1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,&
            0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0, 0d0,&
           +1d0,+1d0,+1d0,+1d0,+1d0,+1d0,+1d0,+1d0,+1d0/)
   end if
end if

!==============================================================================!

write(*,'(a)') 'set_global_parameters_pair '

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
