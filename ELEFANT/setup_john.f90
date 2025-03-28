!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_john

use module_parameters, only: ndim,iproc,debug,nel,use_T,iel,mU,mV,spaceVelocity,&
                             spacePressure,mP,Lx,Ly,nelx,nely,nelz
use module_mesh 
use module_constants, only: eps 
use module_timing

implicit none

integer :: k
integer, parameter :: ncell=9
integer, dimension(6,ncell) :: icon
real(8), dimension(24) :: xpts,ypts

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_john}
!@@ Supported velocity space:
!@@ \begin{itemize}
!@@ \item $P_1$, $P_2$
!@@ \end{itemize}
!@@ Supported pressure space:
!@@ \begin{itemize}
!@@ \item $P_0$, $P_1$
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==3) stop 'setup_john: ndim=3'

nelx=0
nely=0
nelz=0

xpts( 1)=0d0    ; ypts(1)=0d0
xpts( 2)=1d0    ; ypts(2)=0d0
xpts( 3)=1d0    ; ypts(3)=1d0
xpts( 4)=0d0    ; ypts(4)=1d0
xpts( 5)=0.6d0  ; ypts(5)=0.4d0
xpts( 6)=0.8d0  ; ypts(6)=0.6d0
xpts( 7)=0.3d0  ; ypts(7)=0.8d0
xpts( 8)=0.5d0  ; ypts(8)=1d0

xpts( 9)=0.5d0  ; ypts( 9)=0d0
xpts(10)=0.3d0  ; ypts(10)=0.2d0
xpts(11)=0.8d0  ; ypts(11)=0.2d0
xpts(12)=0.9d0  ; ypts(12)=0.3d0
xpts(13)=1d0    ; ypts(13)=0.5d0
xpts(14)=0.9d0  ; ypts(14)=0.8d0
xpts(15)=0.75d0 ; ypts(15)=1d0
xpts(16)=0.65d0 ; ypts(16)=0.8d0
xpts(17)=0.55d0 ; ypts(17)=0.7d0
xpts(18)=0.7d0  ; ypts(18)=0.5d0
xpts(19)=0.45d0 ; ypts(19)=0.6d0
xpts(20)=0.4d0  ; ypts(20)=0.9d0
xpts(21)=0.25d0 ; ypts(21)=1d0
xpts(22)=0.15d0 ; ypts(22)=0.9d0
xpts(23)=0d0    ; ypts(23)=0.5d0
xpts(24)=0.15d0 ; ypts(24)=0.4d0

icon(1:6,1)=(/1,2,5, 9,11,10/)
icon(1:6,2)=(/1,5,7, 10,19,24/)
icon(1:6,3)=(/5,6,7, 18,17,19/)
icon(1:6,4)=(/5,2,6, 11,12,18/)
icon(1:6,5)=(/6,2,3, 12,13,14/)
icon(1:6,6)=(/8,6,3, 16,14,15/)
icon(1:6,7)=(/7,6,8, 17,16,20/)
icon(1:6,8)=(/4,7,8, 22,20,21/)
icon(1:6,9)=(/1,7,4, 24,22,23/)

!----------------------------------------------------------
!velocity 
!----------------------------------------------------------

select case(spaceVelocity)

!-----------
case('__P1')
   do iel=1,nel
      mesh(iel)%iconV=icon(1:mV,iel)
      do k=1,mV
         mesh(iel)%xV(k)=xpts(icon(k,iel))
         mesh(iel)%yV(k)=ypts(icon(k,iel))
      end do
      mesh(iel)%xc=sum(mesh(iel)%xV)/mV
      mesh(iel)%yc=sum(mesh(iel)%yV)/mV
      mesh(iel)%xU=mesh(iel)%xV
      mesh(iel)%yU=mesh(iel)%yV
      mesh(iel)%iconU=mesh(iel)%iconV
   end do

!-----------
case('__P2')
   do iel=1,nel
      mesh(iel)%iconV=icon(1:mV,iel)
      do k=1,mV
         mesh(iel)%xV(k)=xpts(icon(k,iel))
         mesh(iel)%yV(k)=ypts(icon(k,iel))
      end do
      mesh(iel)%xc=sum(mesh(iel)%xV)/mV
      mesh(iel)%yc=sum(mesh(iel)%yV)/mV
      mesh(iel)%xU=mesh(iel)%xV
      mesh(iel)%yU=mesh(iel)%yV
      mesh(iel)%iconU=mesh(iel)%iconV
   end do

case default
   stop 'setup_john: spaceVelocity not supported yet'

end select

!----------------------------------------------------------
!pressure
!----------------------------------------------------------

select case(spacePressure)

!-----------
case('__P0')
   do iel=1,nel
      mesh(iel)%iconP(1)=iel
      mesh(iel)%xP(1)=mesh(iel)%xc
      mesh(iel)%yP(1)=mesh(iel)%yc
   end do

!-----------
case('__P1')
   do iel=1,nel
      mesh(iel)%iconP=icon(1:mP,iel)
      do k=1,mP
         mesh(iel)%xP(k)=xpts(icon(k,iel))
         mesh(iel)%yP(k)=ypts(icon(k,iel))
      end do
   end do

case default
   stop 'setup_john: spacePressure not supported yet'

end select

!----------------------------------------------------------
!temperature 
!----------------------------------------------------------

if (use_T) stop 'setup_john: temperature not supported yet'

!----------------------------------------------------------
! flag nodes on boundaries

do iel=1,nel
   do k=1,mU
      mesh(iel)%bnd1_Unode(k)=(abs(mesh(iel)%xU(k)-0 )<eps*Lx)
      mesh(iel)%bnd2_Unode(k)=(abs(mesh(iel)%xU(k)-Lx)<eps*Lx)
      mesh(iel)%bnd3_Unode(k)=(abs(mesh(iel)%yU(k)-0 )<eps*Ly)
      mesh(iel)%bnd4_Unode(k)=(abs(mesh(iel)%yU(k)-Ly)<eps*Ly)
   end do
   do k=1,mV
      mesh(iel)%bnd1_Vnode(k)=(abs(mesh(iel)%xV(k)-0 )<eps*Lx)
      mesh(iel)%bnd2_Vnode(k)=(abs(mesh(iel)%xV(k)-Lx)<eps*Lx)
      mesh(iel)%bnd3_Vnode(k)=(abs(mesh(iel)%yV(k)-0 )<eps*Ly)
      mesh(iel)%bnd4_Vnode(k)=(abs(mesh(iel)%yV(k)-Ly)<eps*Ly)
   end do
end do

!----------------------------------------------------------
! initialise boundary arrays

do iel=1,nel
   mesh(iel)%fix_u=.false.
   mesh(iel)%fix_v=.false.
   mesh(iel)%fix_w=.false.
   mesh(iel)%fix_T=.false.
end do

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'setup_john'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_john:',elapsed,' s                     |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
