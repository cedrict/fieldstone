!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine set_global_parameters_spaceV

use module_parameters, only: spaceVelocity,debug,mU,mV,mW,NU,NV,NW,nelx,nely,nelz,iproc,ndim,&
                             nel,geometry,nelr,nelphi,spaceU,spaceV,spaceW,ndofV,mVel
use module_timing
use module_arrays, only: rU,sU,tU,rV,sV,tV,rW,sW,tW

implicit none

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{set\_global\_parameters\_spaceV}
!@@ This subroutine computes {\tt mU,mV,mW,nel,NU,NV,NW} and assigns 
!@@ {\tt rU,sU,tU,rV,sV,tV,rW,sW,tW}.
!@@ \begin{itemize}
!@@ \item supported spaces in 2D: $Q_1$, $Q_2$, $Q_1^+$, $Q_3$, $P_1$, $P_2$
!@@ \item supported spaces in 3D: $Q_1$, $Q_2$, $Q_1^{++}$
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a,a)') shift//'spaceVelocity=',spaceVelocity

if (ndim==2) then

   mW=0 ; NW=0

   select case(spaceVelocity)
   !-----------
   case('__Q1')
      mU=4
      mV=4
      allocate(rU(mU)) ; rU=(/-1d0,+1d0,+1d0,-1d0/)
      allocate(sU(mU)) ; sU=(/-1d0,-1d0,+1d0,+1d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/-1d0,+1d0,+1d0,-1d0/) 
      allocate(sV(mV)) ; sV=(/-1d0,-1d0,+1d0,+1d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NU=(nelx+1)*(nely+1)
         NV=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NU=(nelr+1)*nelphi
         NV=(nelr+1)*nelphi
      case('john')
         stop 'set_global_parameters_spaceV: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='__Q1'
      spaceV='__Q1'
      spaceW='____'
   !-----------
   case('_Q1F')
      mU=6
      mV=6
      allocate(rU(mU)) ; rU=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0/)
      allocate(sU(mU)) ; sU=(/-1d0,-1d0,+1d0,+1d0,0.d0,0.d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/-1d0,+1d0,+1d0,-1d0,0.d0,0.d0/)
      allocate(sV(mV)) ; sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,+1d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NU=(nelx+1)*(nely+1)+(nelx+1)*nely
         NV=(nelx+1)*(nely+1)+(nely+1)*nelx
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='Q1Fu'
      spaceV='Q1Fv'
      spaceW='____'
   !-----------
   case('__Q2')
      mU=9
      mV=9
      allocate(rU(mU)) ; rU=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      allocate(sU(mU)) ; sU=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/-1d0,0d0,+1d0,-1d0,0d0,+1d0,-1d0,0d0,+1d0/)
      allocate(sV(mV)) ; sV=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,+1d0,+1d0,+1d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NU=(2*nelx+1)*(2*nely+1)
         NV=(2*nelx+1)*(2*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NU=(2*nelr+1)*(2*nelphi)
         NV=(2*nelr+1)*(2*nelphi)
      case('john')
         stop 'set_global_parameters_spaceV: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='__Q2'
      spaceV='__Q2'
      spaceW='____'
   !-----------
   case('_Q1+')
      mU=5
      mV=5
      allocate(rU(mU)) ; rU=(/-1d0,+1d0,+1d0,-1d0,0d0/)
      allocate(sU(mU)) ; sU=(/-1d0,-1d0,+1d0,+1d0,0d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/-1d0,+1d0,+1d0,-1d0,0d0/)
      allocate(sV(mV)) ; sV=(/-1d0,-1d0,+1d0,+1d0,0d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NU=(nelx+1)*(nely+1)+nel
         NV=(nelx+1)*(nely+1)+nel
      case('john')
         stop 'set_global_parameters_spaceV: john geometry not supported'
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='_Q1+'
      spaceV='_Q1+'
      spaceW='____'
   !-----------
   case('__Q3')
      mu=16
      mV=16
      allocate(rU(mU)) ; rU=0.d0
      allocate(sU(mU)) ; sU=0.d0
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=nelx*nely
         NU=(3*nelx+1)*(3*nely+1)
         NV=(3*nelx+1)*(3*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=nelr*nelphi
         NU=(3*nelr+1)*(3*nelphi)
         NV=(3*nelr+1)*(3*nelphi)
      case('john')
         stop 'set_global_parameters_spaceV: john geometry not supported'
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
      rU=rV
      sU=sV
      spaceU='__Q3'
      spaceV='__Q3'
      spaceW='____'
   !-----------
   case('__P1')
      mu=3
      mV=3
      allocate(rU(mU)) ; rU=(/0d0,1d0,0d0/)
      allocate(sU(mU)) ; sU=(/0d0,0d0,1d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/0d0,1d0,0d0/)
      allocate(sV(mV)) ; sV=(/0d0,0d0,1d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=2*nelx*nely
         NU=(nelx+1)*(nely+1)
         NV=(nelx+1)*(nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=2*nelr*nelphi
         NU=(nelr+1)*nelphi
         NV=(nelr+1)*nelphi
      case('john')
         nel=9
         NV=8
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='__P1'
      spaceV='__P1'
      spaceW='____'
   !-----------
   case('__P2')
      mu=6
      mV=6
      allocate(rU(mU)) ; rU=(/0d0,1d0,0d0,0.5d0,0.5d0,0d0/)
      allocate(sU(mU)) ; sU=(/0d0,0d0,1d0,0d0,0.5d0,0.5d0/)
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=(/0d0,1d0,0d0,0.5d0,0.5d0,0d0/)
      allocate(sV(mV)) ; sV=(/0d0,0d0,1d0,0d0,0.5d0,0.5d0/)
      allocate(tV(mV)) ; tV=0.d0
      select case(geometry)
      case('cartesian')
         if (nelx==0) stop 'set_global_parameters_spaceV: nelx=0'
         if (nely==0) stop 'set_global_parameters_spaceV: nely=0'
         nel=2*nelx*nely
         NU=(2*nelx+1)*(2*nely+1)
         NV=(2*nelx+1)*(2*nely+1)
      case('spherical')
         if (nelr==0) stop 'set_global_parameters_spaceV: nelr=0'
         if (nelphi==0) stop 'set_global_parameters_spaceV: nelphi=0'
         nel=2*nelr*nelphi
         NU=(2*nelr+1)*(2*nelphi)
         NV=(2*nelr+1)*(2*nelphi)
      case('john')
         nel=9
         NU=24
         NV=24
      case default
         stop 'set_global_parameters_spaceV: unknown geometry'
      end select
      spaceU='__P2'
      spaceV='__P2'
      spaceW='____'
   !-----------
   case default
      stop 'spaceVelocity not supported in set_global_parameters_spaceV'
   end select

   if (.not.allocated(rU)) stop 'set_global_parameters_spaceV: rU not allocated' 
   if (.not.allocated(sU)) stop 'set_global_parameters_spaceV: sU not allocated' 
   if (.not.allocated(tU)) stop 'set_global_parameters_spaceV: tU not allocated' 
   if (.not.allocated(rV)) stop 'set_global_parameters_spaceV: rV not allocated' 
   if (.not.allocated(sV)) stop 'set_global_parameters_spaceV: sV not allocated' 
   if (.not.allocated(tV)) stop 'set_global_parameters_spaceV: tV not allocated' 

else

   select case(spaceVelocity)
   !-----------
   case('__Q1')
      mu=8
      mV=8
      mW=8
      allocate(rU(mU)) ; rU=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      allocate(sU(mU)) ; sU=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      allocate(tU(mU)) ; tU=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
      allocate(rV(mV)) ; rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      allocate(sV(mV)) ; sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      allocate(tV(mV)) ; tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
      allocate(rW(mW)) ; rW=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/)
      allocate(sW(mW)) ; sW=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0/)
      allocate(tW(mW)) ; tW=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0/)
      nel=nelx*nely*nelz
      NU=(nelx+1)*(nely+1)*(nelz+1)
      NV=(nelx+1)*(nely+1)*(nelz+1)
      NW=(nelx+1)*(nely+1)*(nelz+1)
      spaceU='__Q1'
      spaceV='__Q1'
      spaceW='__Q1'
   !-----------
   case('__Q2')
      mu=27
      mV=27
      mW=27
      allocate(rU(mU)) ; rU=0.d0
      allocate(sU(mU)) ; sU=0.d0
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      allocate(rW(mW)) ; rW=0.d0
      allocate(sW(mW)) ; sW=0.d0
      allocate(tW(mW)) ; tW=0.d0
      nel=nelx*nely*nelz
      NU=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      NV=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      NW=(2*nelx+1)*(2*nely+1)*(2*nelz+1)
      rV=(/-1d0,0d0,1d0,-1d0,0d0,1d0,-1d0,0d0,1d0,\
           -1d0,0d0,1d0,-1d0,0d0,1d0,-1d0,0d0,1d0,\
           -1d0,0d0,1d0,-1d0,0d0,1d0,-1d0,0d0,1d0/)
      sV=(/-1d0,-1d0,-1d0,0d0,0d0,0d0,1d0,1d0,1d0,\
           -1d0,-1d0,-1d0,0d0,0d0,0d0,1d0,1d0,1d0,\
           -1d0,-1d0,-1d0,0d0,0d0,0d0,1d0,1d0,1d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,-1d0,\
           0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,\
           1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0/)
      rU=rV ; sU=sV ; tU=tV
      rW=rV ; sW=sV ; tW=tV
      spaceU='__Q2'
      spaceV='__Q2'
      spaceW='__Q2'
   !-----------
   case('Q1++')
      mu=10
      mV=10
      mW=10
      allocate(rU(mU)) ; rU=0.d0
      allocate(sU(mU)) ; sU=0.d0
      allocate(tU(mU)) ; tU=0.d0
      allocate(rV(mV)) ; rV=0.d0
      allocate(sV(mV)) ; sV=0.d0
      allocate(tV(mV)) ; tV=0.d0
      allocate(rW(mW)) ; rW=0.d0
      allocate(sW(mW)) ; sW=0.d0
      allocate(tW(mW)) ; tW=0.d0
      nel=nelx*nely*nelz
      NU=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NV=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      NW=(nelx+1)*(nely+1)*(nelz+1)+2*nel
      rV=(/-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0,-1d0/3d0,1d0/3d0/)
      sV=(/-1d0,-1d0,+1d0,+1d0,-1d0,-1d0,+1d0,+1d0,-1d0/3d0,1d0/3d0/)
      tV=(/-1d0,-1d0,-1d0,-1d0,+1d0,+1d0,+1d0,+1d0,-1d0/3d0,1d0/3d0/)
      !spaceU='__Q2'
      !spaceV='__Q2'
      !spaceW='__Q2'
   !-----------
   case default
      stop 'spaceV not supported in set_global_parameters_spaceV'
   end select

   if (.not.allocated(rU)) stop 'set_global_parameters_spaceV: rU not allocated' 
   if (.not.allocated(sU)) stop 'set_global_parameters_spaceV: sU not allocated' 
   if (.not.allocated(tU)) stop 'set_global_parameters_spaceV: tU not allocated' 
   if (.not.allocated(rV)) stop 'set_global_parameters_spaceV: rV not allocated' 
   if (.not.allocated(sV)) stop 'set_global_parameters_spaceV: sV not allocated' 
   if (.not.allocated(tV)) stop 'set_global_parameters_spaceV: tV not allocated' 
   if (.not.allocated(rW)) stop 'set_global_parameters_spaceV: rW not allocated' 
   if (.not.allocated(sW)) stop 'set_global_parameters_spaceV: sW not allocated' 
   if (.not.allocated(tW)) stop 'set_global_parameters_spaceV: tW not allocated' 

end if

if (nel==0) stop 'set_global_parameters_spaceV: nel=0'
if (NU==0)  stop 'set_global_parameters_spaceV: NU=0'
if (NV==0)  stop 'set_global_parameters_spaceV: NV=0'
if (ndim==2 .and. NW.ne.0) stop 'set_global_parameters_spaceV: NW=0'
if (ndim==2 .and. mW.ne.0) stop 'set_global_parameters_spaceV: pb with mW'

write(*,'(a,3i5)') shift//'nelx,y,z=',nelx,nely,nelz
write(*,'(a,i5)') shift//'nel=',nel
write(*,'(a,3i5)') shift//'NU,NV,NW=',NU,NV,NW

ndofV = ndim

mVel=mU+mV+mW

!----------------------------------------------------------
if (debug) then
write(2345,*) limit//'set_global_parameters_spaceV'//limit
write(2345,*) 'mU,mV,mW,mVel=',mU,mV,mW,mVel
write(2345,*) 'nel=',nel
write(2345,*) 'NU,NV,NW=',NU,NV,NW
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'set_global_params_spaceV:',elapsed,' s       |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
