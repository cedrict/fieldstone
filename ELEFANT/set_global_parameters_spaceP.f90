!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceP

use module_parameters, only: iproc,debug,ndim,mP,nelx,nely,nelz,NP,spaceP,nelr,nelphi,geometry,nel
use module_timing
use module_arrays, only: rP,sP,tP

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{set\_global\_parameters\_spaceP}
!@@ This subroutine computes mP,NP and assigns rP,sP,tP
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: Q0,Q1,Q2,P1
!@@ \item supported spaces in 3D: Q0,Q1,Q2
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   select case(spaceP)
   !------------
   case('__Q0')
      mP=1
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=nel

   !------------
   case('__Q1')
      mP=2**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceP: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceP: nely=0'
         NP=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceP: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceP: nelphi=0'
         NP=(nelr+1)*nelphi
      case('john')
         stop 'set_global_parameters_spaceP: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceP: unknown geometry'
      end select
      rP=(/-1d0,+1d0,+1d0,-1d0/)
      sP=(/-1d0,-1d0,+1d0,+1d0/)

   !------------
   case('__Q2')
      mP=3**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceP: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceP: nely=0'
         NP=(2*nelx+1)*(2*nely+1)
      case('john')
         stop 'set_global_parameters_spaceP: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceP: unknown geometry'
      end select
      rP=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sP=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)

   !------------
   case('__P0')
      mP=1
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=nel
      rP=(/ 1d0/3d0 /)
      sP=(/ 1d0/3d0 /)

   !------------
   case('__P1')
      mP=3
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceP: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceP: nely=0'
         NP=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceP: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceP: nelphi=0'
         NP=(nelr+1)*nelphi
      case('john')
         NP=8 
      case default
         stop 'set_global_parameters_spaceP: unknown geometry'
      end select
      rP=(/0d0,1d0,0d0/)
      sP=(/0d0,0d0,1d0/)

   !------------
   case default
      stop 'spaceP not supported in set_global_parameters_spaceP'
   end select

else

   select case(spaceP)

   !------------
   case('__Q0')
      mP=1
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      NP=nel

   !------------
   case('__Q1')
      mP=2**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceP: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceP: nely=0'
         NP=(nelx+1)*(nely+1)*(nelz+1)
      case default
         stop 'set_global_parameters_spaceP: unknown geometry'
      end select
      rP=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sP=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tP=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)

   !------------
   case('__Q2')
      mP=3**ndim
      allocate(rP(mP)) ; rP=0.d0
      allocate(sP(mP)) ; sP=0.d0
      allocate(tP(mP)) ; tP=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceP: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceP: nely=0'
         NP=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      case default
         stop 'set_global_parameters_spaceP: unknown geometry'
      end select
      !missing rP
      !missing sP
      !missing tP

   case default
      stop 'spaceP not supported in set_global_parameters_spaceP'
   end select

end if

if (NP==0)  stop 'set_global_parameters_spaceP: NP=0'

write(*,'(a,a)') shift//'spaceP=',spaceP
write(*,'(a,i5)') shift//'NP=',NP

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'set_global_parameters_spaceP'//limit
write(2345,*) 'mP=',mP
write(2345,*) 'NP=',NP
write(2345,*) allocated(rP)
write(2345,*) allocated(sP)
write(2345,*) allocated(tP)
write(2345,*) 'rP=',rP
write(2345,*) 'sP=',sP
write(2345,*) 'tP=',tP
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_params_spaceP:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
