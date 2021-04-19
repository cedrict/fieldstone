!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_cartesian2D

use global_parameters
use structures
use constants 
use timing

implicit none

integer counter,ielx,iely,i
real(8) hx,hy

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_cartesian2D.f90}
!@@ This subroutine assigns to every element the coordinates of the its velocity, pressure,
!@@ and temperature nodes, the velocity, pressure and temperature connectivity arrays,
!@@ the coordinates of its center (xc,yc), its integer coordinates (ielx, iely),
!@@ and its dimensions (hx,hy).
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely

allocate(mesh(nel))
do iel=1,nel
   mesh(iel)%u=0.d0
   mesh(iel)%v=0.d0
   mesh(iel)%w=0.d0
   mesh(iel)%T=0.d0
   mesh(iel)%p=0.d0
   mesh(iel)%q=0.d0
end do

!==========================================================
!velocity 

if (pair=='q1p0' .or. pair=='q1q1') then
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely
         mesh(counter)%iconV(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconV(4)=ielx+iely*(nelx+1)    
         mesh(counter)%xV(1)=(ielx-1)*hx
         mesh(counter)%xV(2)=(ielx-1)*hx+hx
         mesh(counter)%xV(3)=(ielx-1)*hx+hx
         mesh(counter)%xV(4)=(ielx-1)*hx
         mesh(counter)%yV(1)=(iely-1)*hy
         mesh(counter)%yV(2)=(iely-1)*hy
         mesh(counter)%yV(3)=(iely-1)*hy+hy
         mesh(counter)%yV(4)=(iely-1)*hy+hy
         mesh(counter)%xc=(ielx-1)*hx+hx/2
         mesh(counter)%yc=(iely-1)*hy+hy/2
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         mesh(counter)%vol=hx*hy
         if (ielx==1)    mesh(counter)%bnd1=.true.
         if (ielx==nelx) mesh(counter)%bnd2=.true.
         if (iely==1)    mesh(counter)%bnd3=.true.
         if (iely==nely) mesh(counter)%bnd4=.true.
      end do    
   end do    
end if

if (pair=='q1q1') then ! add bubble node
   do iel=1,nel
      mesh(iel)%xV(5)=mesh(iel)%xc
      mesh(iel)%yV(5)=mesh(iel)%yc
      mesh(iel)%iconV(5)=(nelx+1)*(nely+1)+iel
   end do
end if

!==========================================================
! pressure 

if (pair=='q1p0') then
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%iconP(1)=counter
         mesh(counter)%xP(1)=mesh(counter)%xC
         mesh(counter)%yP(1)=mesh(counter)%yC
      end do    
   end do    
end if

if (pair=='q1q1') then
   do i=1,4
      mesh(1:nel)%xP(i)=mesh(1:nel)%xV(i)
      mesh(1:nel)%yP(i)=mesh(1:nel)%yV(i)
      mesh(1:nel)%iconP(i)=mesh(1:nel)%iconV(i)
   end do
end if

!==========================================================
! temperature 

do i=1,4
   mesh(1:nel)%xT(i)=mesh(1:nel)%xV(i)
   mesh(1:nel)%yT(i)=mesh(1:nel)%yV(i)
   mesh(1:nel)%iconT(i)=mesh(1:nel)%iconV(i)
end do

!==========================================================
! flag nodes on boundaries

do iel=1,nel
   do i=1,4
      mesh(iel)%bnd1_node(i)=(abs(mesh(iel)%xV(i)-0 )<eps*Lx)
      mesh(iel)%bnd2_node(i)=(abs(mesh(iel)%xV(i)-Lx)<eps*Lx)
      mesh(iel)%bnd3_node(i)=(abs(mesh(iel)%yV(i)-0 )<eps*Ly)
      mesh(iel)%bnd4_node(i)=(abs(mesh(iel)%yV(i)-Ly)<eps*Ly)
   end do
end do

!==========================================================
! initialise boundary arrays

do iel=1,nel
   mesh(iel)%fix_u=.false.
   mesh(iel)%fix_v=.false.
   mesh(iel)%fix_w=.false.
   mesh(iel)%fix_T=.false.
end do

!==========================================================

if (debug) then
   do iel=1,nel
   print *,'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV),'iconP',mesh(iel)%iconP(1:mP)
   end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> setup_cartesian2D                ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
