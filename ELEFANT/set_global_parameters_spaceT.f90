!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceT

use module_parameters, only: iproc,debug,ndim,mT,nelx,nely,nelz,NT,spaceTemperature,&
                             geometry,nelr,nelphi,spaceVelocity,use_T
use module_timing
use module_arrays, only: rT,sT,tT

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{set\_global\_parameters\_spaceT}
!@@ This subroutine computes {\tt mT}, {\tt NT} and assigns {\tt rT,sT,tT}.
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: $Q_1$, $Q_2$
!@@ \item supported spaces in 3D: $Q_1$, $Q_2$
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

select case(spaceVelocity)
case('__Q1','__Q2','__Q3','__P1','__P2','__P3')
   spaceTemperature=spaceVelocity
case('_Q1+','_Q1F')
   spaceTemperature='__Q1'
case('_P2+')
   spaceTemperature='__P2'
case default
   stop 'set_global_parameters_spaceT: spaceVelocity/spaceTemperature pb'
end select

!----------------------------------------------------------

if (.not.use_T) return

if (ndim==2) then

   select case(spaceTemperature)
   !-----------
   case('__Q1')
      mT=2**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceT: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceT: nely=0'
         NT=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceT: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceT: nelphi=0'
         NT=(nelr+1)*nelphi
      case('john')
         stop 'set_global_parameters_spaceT: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      rT=(/-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0/)
   !-----------
   case('__Q2')
      mT=3**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceT: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceT: nely=0'
         NT=(2*nelx+1)*(2*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceT: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceT: nelphi=0'
         NT=(2*nelr+1)*(2*nelphi)
      case('john')
         stop 'set_global_parameters_spaceT: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      rT=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      sT=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
   !-----------
   case('__P1')
      mT=3
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceT: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceT: nely=0'
         NT=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceT: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceT: nelphi=0'
         NT=(nelr+1)*nelphi
      case('john')
         stop 'set_global_parameters_spaceT: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      rT=(/0d0,1d0,0d0/)
      sT=(/0d0,0d0,1d0/)
   !-----------
   case('__P2')
      mT=6
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         NT=(2*nelx+1)*(2*nely+1)
      case('spherical')
         NT=(2*nelr+1)*(2*nelphi)
      case('john')
         stop 'set_global_parameters_spaceT: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      rT=(/0d0,1d0,0d0,0.5d0,0.5d0,0d0/)
      sT=(/0d0,0d0,1d0,0d0,0.5d0,0.5d0/)
   !-----------
   case default
      stop 'spaceT not supported in set_global_parameters_spaceT'
   end select

else

   select case(spaceTemperature)
   !-----------
   case('__Q1')
      mT=2**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceT: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceT: nely=0'
         if (nelz==0) stop 'set_global_parameters_spaceT: nelz=0'
         NT=(nelx+1)*(nely+1)*(nelz+1)
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      rT=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      sT=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      tT=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
   !-----------
   case('__Q2')
      mT=3**ndim
      allocate(rT(mT)) ; rT=0.d0
      allocate(sT(mT)) ; sT=0.d0
      allocate(tT(mT)) ; tT=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceT: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceT: nely=0'
         if (nelz==0) stop 'set_global_parameters_spaceT: nelz=0'
         NT=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      case default
         stop 'set_global_parameters_spaceT: unknown geometry'
      end select
      !missing rT
      !missing sT
      !missing tT
   !-----------
   case default
      stop 'spaceT not supported in set_global_parameters_spaceT'
   end select

end if

write(*,'(a,a)') shift//'spaceTemperature=',spaceTemperature
write(*,'(a,i5)') shift//'NT=',NT

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'set_global_parameters_spaceT'//limit
write(2345,*) 'mT=',mT
write(2345,*) 'NT=',NT
write(2345,*) allocated(rT)
write(2345,*) allocated(sT)
write(2345,*) allocated(tT)
write(2345,*) 'rT=',rT
write(2345,*) 'sT=',sT
write(2345,*) 'tT=',tT
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_parameters_spaceT (',elapsed,' s)     |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
