!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_cartesian3D

use global_parameters
use structures
use constants 
use timing

implicit none

integer counter,ielx,iely,ielz,i
real(8) hx,hy,hz

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_cartesian3D.f90}
!@@ 
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

allocate(mesh(nel))

!==========================================================
!velocity 

if (pair=='q1p0' .or. pair=='q1q1') then
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    
            mesh(counter)%ielx=ielx
            mesh(counter)%iely=iely
            mesh(counter)%ielz=ielz
            mesh(counter)%iconV(1)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconV(2)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconV(3)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconV(4)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx
            mesh(counter)%iconV(5)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconV(6)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconV(7)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconV(8)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx
            mesh(counter)%xV(1)=(ielx-1)*hx
            mesh(counter)%xV(2)=(ielx-1)*hx+hx
            mesh(counter)%xV(3)=(ielx-1)*hx+hx
            mesh(counter)%xV(4)=(ielx-1)*hx
            mesh(counter)%xV(5)=(ielx-1)*hx
            mesh(counter)%xV(6)=(ielx-1)*hx+hx
            mesh(counter)%xV(7)=(ielx-1)*hx+hx
            mesh(counter)%xV(8)=(ielx-1)*hx
            mesh(counter)%yV(1)=(iely-1)*hy
            mesh(counter)%yV(2)=(iely-1)*hy
            mesh(counter)%yV(3)=(iely-1)*hy+hy
            mesh(counter)%yV(4)=(iely-1)*hy+hy
            mesh(counter)%yV(5)=(iely-1)*hy
            mesh(counter)%yV(6)=(iely-1)*hy
            mesh(counter)%yV(7)=(iely-1)*hy+hy
            mesh(counter)%yV(8)=(iely-1)*hy+hy
            mesh(counter)%zV(1)=(ielz-1)*hz
            mesh(counter)%zV(2)=(ielz-1)*hz
            mesh(counter)%zV(3)=(ielz-1)*hz
            mesh(counter)%zV(4)=(ielz-1)*hz
            mesh(counter)%zV(5)=(ielz-1)*hz+hz
            mesh(counter)%zV(6)=(ielz-1)*hz+hz
            mesh(counter)%zV(7)=(ielz-1)*hz+hz
            mesh(counter)%zV(8)=(ielz-1)*hz+hz
            mesh(counter)%xc=(ielx-1)*hx+hx/2
            mesh(counter)%yc=(iely-1)*hy+hy/2
            mesh(counter)%zc=(ielz-1)*hz+hz/2
            mesh(counter)%hx=hx
            mesh(counter)%hy=hy
            mesh(counter)%hz=hz
            if (ielx==1)    mesh(counter)%back=.true.
            if (ielx==nelx) mesh(counter)%front=.true.
            if (iely==1)    mesh(counter)%left=.true.
            if (iely==nely) mesh(counter)%right=.true.
            if (ielz==1)    mesh(counter)%bottom=.true.
            if (ielz==nelz) mesh(counter)%top=.true.
         end do    
      end do    
   end do    
end if

if (pair=='q1q1') then ! add bubble node
   stop 'setup_cartesian3D: xyz'
   do iel=1,nel
      mesh(iel)%xV(4)=mesh(iel)%xc
      mesh(iel)%yV(4)=mesh(iel)%yc
      mesh(counter)%iconV(5)=nel+iel
   end do
end if

!==========================================================
! pressure 

if (pair=='q1p0') then
   counter=0    
   do ielz=1,nelz
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    
            mesh(counter)%iconP(1)=counter
            mesh(counter)%xP(1)=mesh(counter)%xC
            mesh(counter)%yP(1)=mesh(counter)%yC
            mesh(counter)%zP(1)=mesh(counter)%zC
         end do    
      end do    
   end do    
end if

if (pair=='q1q1') then
   do i=1,ncorners
      mesh(1:nel)%xP(i)=mesh(1:nel)%xV(i)
      mesh(1:nel)%yP(i)=mesh(1:nel)%yV(i)
      mesh(1:nel)%zP(i)=mesh(1:nel)%zV(i)
      mesh(1:nel)%iconP(i)=mesh(1:nel)%iconV(i)
   end do
end if

!==========================================================
! temperature 

do i=1,ncorners
   mesh(1:nel)%xT(i)=mesh(1:nel)%xV(i)
   mesh(1:nel)%yT(i)=mesh(1:nel)%yV(i)
   mesh(1:nel)%zT(i)=mesh(1:nel)%zV(i)
   mesh(1:nel)%iconT(i)=mesh(1:nel)%iconV(i)
end do

!==========================================================
! flag nodes on boundaries

do iel=1,nel
   do i=1,ncorners
      mesh(iel)%back_node(i)  =(abs(mesh(iel)%xV(i)-0 )<eps*Lx)
      mesh(iel)%front_node(i) =(abs(mesh(iel)%xV(i)-Lx)<eps*Lx)
      mesh(iel)%left_node(i)  =(abs(mesh(iel)%yV(i)-0 )<eps*Ly)
      mesh(iel)%right_node(i) =(abs(mesh(iel)%yV(i)-Ly)<eps*Ly)
      mesh(iel)%bottom_node(i)=(abs(mesh(iel)%zV(i)-0 )<eps*Lz)
      mesh(iel)%top_node(i)   =(abs(mesh(iel)%zV(i)-Lz)<eps*Lz)
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

if (debug) then
   do iel=1,nel
   print *,'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV),mesh(iel)%iconP(1:mP)
   end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

if (iproc==0) write(*,*) '     -> setup_cartesian2D ',elapsed

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
