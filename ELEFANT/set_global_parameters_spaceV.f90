!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceV

use module_parameters, only: spaceV,debug,mV,NV,nelx,nely,nelz,iproc,ndim,nel,geometry,nelr,nelphi
use module_timing
use module_arrays, only: rV,sV,tV

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{set\_global\_parameters\_spaceV}
!@@ This subroutine computes mV,nel,NV and assigns rV,sV,tV
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: Q1,Q2,Q1+,Q3,P1,P2
!@@ \item supported spaces in 3D: Q1,Q2,Q1++
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   select case(spaceV)
   !-----------
   case('__Q1')
      mV=2**ndim
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      rV=(/-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0/)
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NV=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NV=(nelr+1)*nelphi
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
   !-----------
   case('__Q2')
      mV=3**ndim
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NV=(2*nelx+1)*(2*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NV=(2*nelr+1)*(2*nelphi)
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      rV=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sV=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   !-----------
   case('_Q1+')
      mV=2**ndim+1
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NV=(nelx+1)*(nely+1)+nel
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      rV=(/-1d0,+1d0,+1d0,-1d0,0d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,0d0/)
   !-----------
   case('__Q3')
      mV=2**ndim+1
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NV=(3*nelx+1)*(3*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NV=(3*nelr+1)*(3*nelphi)
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      rV=(/-1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0,&
           -1d0,-1d0/3d0,1d0/3d0,1d0/)
      sV=(/-1d0,-1d0,-1d0,-1d0,&
           -1d0/3d0,-1d0/3d0,-1d0/3d0,-1d0/3d0,&
           1d0/3d0,1d0/3d0,1d0/3d0,1d0/3d0,&
           1d0,1d0,1d0,1d0/)
   !-----------
   case('__P1')
      mV=3
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=2*nelx*nely
         NV=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=2*nelr*nelphi
         NV=(nelr+1)*nelphi
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      rV=(/0d0,1d0,0d0/)
      sV=(/0d0,0d0,1d0/)
   !-----------
   case('__P2')
      mV=6
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=2*nelx*nely
         NV=(2*nelx+1)*(2*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=2*nelr*nelphi
         NV=(2*nelr+1)*(2*nelphi)
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      rV=(/0d0,1d0,0d0,0.5d0,0.5d0,0d0/)
      sV=(/0d0,0d0,1d0,0d0,0.5d0,0.5d0/)
   !-----------
   case default
      stop 'spaceV not supported in set_global_parameters_spaceV'
   end select

else

   select case(spaceV)
   !-----------
   case('__Q1')
      mV=2**ndim
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)
      rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   !-----------
   case('__Q2')
      mV=3**ndim
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      nel=nelx*nely*nelz
      NV=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      !missing rV
      !missing sV
      !missing tV
   !-----------
   case('Q1++')
      mV=2**ndim+2
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      nel=nelx*nely*nelz
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0,-1d0/3d0,1d0/3d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/3d0,1d0/3d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0,-1d0/3d0,1d0/3d0/)
   case default
      stop 'spaceV not supported in set_global_parameters_spaceV'
   end select

end if

if (nel==0) stop 'set_global_parameters_spaceV: nel=0'
if (NV==0)  stop 'set_global_parameters_spaceV: NV=0'

write(*,'(a,a)') shift//'spaceV=',spaceV
write(*,'(a,i5)') shift//'nel=',nel
write(*,'(a,i5)') shift//'NV=',NV

!----------------------------------------------------------
if (debug) then
write(2345,*) limit//'set_global_parameters_spaceV'//limit
write(2345,*) 'mV=',mV
write(2345,*) 'nel=',nel
write(2345,*) 'NV=',NV
write(2345,*) allocated(rV)
write(2345,*) allocated(sV)
write(2345,*) allocated(tV)
write(2345,*) 'rV=',rV
write(2345,*) 'sV=',sV
write(2345,*) 'tV=',tV
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_params_spaceV:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
