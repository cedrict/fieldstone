!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_cartesian2D

use module_parameters
use module_mesh 
use module_constants, only: eps 
use module_timing

implicit none

integer counter,ielx,iely,i,nnx,nny,k
integer :: node1,node2,node3,node4,node5,node6,node7,node8,node9
real(8) hx,hy,x1,x2,x3,x4,x5,x6,x7,x8,x9
real(8) y1,y2,y3,y4,y5,y6,y7,y8,y9

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_cartesian2D.f90}
!@@ This subroutine assigns to every element the coordinates of the its velocity, pressure,
!@@ and temperature nodes, the velocity, pressure and temperature connectivity arrays,
!@@ the coordinates of its center (xc,yc), its integer coordinates (ielx, iely),
!@@ and its dimensions (hx,hy).
!@@ Supported velocity space:
!@@ \begin{itemize}
!@@ \item $Q_1$, $Q_1^+$, $Q_2$, $Q_3$
!@@ \item $P_1$, $P_1^+$, $P_2$, $P_2^+$
!@@ \end{itemize}
!@@ Supported pressure space:
!@@ \begin{itemize}
!@@ \item $Q_0$, $Q_1$
!@@ \item $P_0$, $P_1$
!@@ \end{itemize}
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely

!----------------------------------------------------------
!velocity 
!----------------------------------------------------------

select case(spaceVelocity)

!------------------
case('__Q1','_Q1+')
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
         mesh(counter)%hz=0
         mesh(counter)%vol=hx*hy
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
      end do    
   end do    

   if (spaceVelocity=='_Q1+') then ! add bubble node
   do iel=1,nel
      mesh(iel)%xV(5)=mesh(iel)%xc
      mesh(iel)%yV(5)=mesh(iel)%yc
      mesh(iel)%iconV(5)=(nelx+1)*(nely+1)+iel
   end do
   end if

!-----------
case('_Q1F')

   !    u        v
   ! 4-----3  4--6--3
   ! |     |  |     |
   ! 5     6  |     |
   ! |     |  |     |
   ! 1-----2  1--5--2

   nnx=nelx+1
   nny=nely+1
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely

         mesh(counter)%xU(1)=(ielx-1)*hx
         mesh(counter)%xU(2)=(ielx-1)*hx+hx
         mesh(counter)%xU(3)=(ielx-1)*hx+hx
         mesh(counter)%xU(4)=(ielx-1)*hx
         mesh(counter)%xU(5)=(ielx-1)*hx
         mesh(counter)%xU(6)=(ielx-1)*hx+hx

         mesh(counter)%yU(1)=(iely-1)*hy
         mesh(counter)%yU(2)=(iely-1)*hy
         mesh(counter)%yU(3)=(iely-1)*hy+hy
         mesh(counter)%yU(4)=(iely-1)*hy+hy
         mesh(counter)%yU(5)=(iely-1)*hy+0.5*hy
         mesh(counter)%yU(6)=(iely-1)*hy+0.5*hy

         mesh(counter)%xV(1)=(ielx-1)*hx
         mesh(counter)%xV(2)=(ielx-1)*hx+hx
         mesh(counter)%xV(3)=(ielx-1)*hx+hx
         mesh(counter)%xV(4)=(ielx-1)*hx
         mesh(counter)%xV(5)=(ielx-1)*hx+0.5*hx
         mesh(counter)%xV(6)=(ielx-1)*hx+0.5*hx

         mesh(counter)%yV(1)=(iely-1)*hy
         mesh(counter)%yV(2)=(iely-1)*hy
         mesh(counter)%yV(3)=(iely-1)*hy+hy
         mesh(counter)%yV(4)=(iely-1)*hy+hy
         mesh(counter)%yV(5)=(iely-1)*hy
         mesh(counter)%yV(6)=(iely-1)*hy+hy
        
         mesh(counter)%iconU(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconU(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconU(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconU(4)=ielx+iely*(nelx+1)    
         mesh(counter)%iconU(5)=nnx*nny + (ielx-1) + (iely-1)*nnx +1 
         mesh(counter)%iconU(6)=nnx*nny + (ielx-1) +1+ (iely-1)*nnx +1 

         !print *,mesh(counter)%iconU-1
      
         mesh(counter)%iconV(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconV(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconV(4)=ielx+iely*(nelx+1)    
         mesh(counter)%iconV(5)=nnx*nny + (ielx-1)  + (iely-1)*nelx +1
         mesh(counter)%iconV(6)=nnx*nny + nelx + (ielx-1)+ (iely-1)*nelx +1

         print *,mesh(counter)%iconV-1
      
         mesh(counter)%xc=(ielx-1)*hx+hx/2
         mesh(counter)%yc=(iely-1)*hy+hy/2
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         mesh(counter)%hz=0
         mesh(counter)%vol=hx*hy
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
      end do    
   end do    

!-----------
case('__Q2')
   nnx=2*nelx+1
   nny=2*nely+1
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely
         mesh(counter)%iconV(1)=(ielx-1)*2+1+(iely-1)*2*nnx
         mesh(counter)%iconV(2)=(ielx-1)*2+2+(iely-1)*2*nnx
         mesh(counter)%iconV(3)=(ielx-1)*2+3+(iely-1)*2*nnx
         mesh(counter)%iconV(4)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx
         mesh(counter)%iconV(5)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx
         mesh(counter)%iconV(6)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx
         mesh(counter)%iconV(7)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2
         mesh(counter)%iconV(8)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2
         mesh(counter)%iconV(9)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2
         mesh(counter)%xV(1)=(ielx-1)*hx
         mesh(counter)%xV(2)=(ielx-1)*hx+hx/2
         mesh(counter)%xV(3)=(ielx-1)*hx+hx
         mesh(counter)%xV(4)=(ielx-1)*hx
         mesh(counter)%xV(5)=(ielx-1)*hx+hx/2
         mesh(counter)%xV(6)=(ielx-1)*hx+hx
         mesh(counter)%xV(7)=(ielx-1)*hx
         mesh(counter)%xV(8)=(ielx-1)*hx+hx/2
         mesh(counter)%xV(9)=(ielx-1)*hx+hx
         mesh(counter)%yV(1)=(iely-1)*hy
         mesh(counter)%yV(2)=(iely-1)*hy
         mesh(counter)%yV(3)=(iely-1)*hy
         mesh(counter)%yV(4)=(iely-1)*hy+hy/2
         mesh(counter)%yV(5)=(iely-1)*hy+hy/2
         mesh(counter)%yV(6)=(iely-1)*hy+hy/2
         mesh(counter)%yV(7)=(iely-1)*hy+hy
         mesh(counter)%yV(8)=(iely-1)*hy+hy
         mesh(counter)%yV(9)=(iely-1)*hy+hy

         mesh(counter)%xc=(ielx-1)*hx+hx/2
         mesh(counter)%yc=(iely-1)*hy+hy/2
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         mesh(counter)%vol=hx*hy
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
      end do    
   end do    

!-----------
case('__Q3')
   nnx=3*nelx+1
   nny=3*nely+1
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely
         mesh(counter)%iconV(01)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*0
         mesh(counter)%iconV(02)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*0
         mesh(counter)%iconV(03)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*0
         mesh(counter)%iconV(04)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*0
         mesh(counter)%iconV(05)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*1
         mesh(counter)%iconV(06)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*1
         mesh(counter)%iconV(07)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*1
         mesh(counter)%iconV(08)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*1
         mesh(counter)%iconV(09)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*2
         mesh(counter)%iconV(10)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*2
         mesh(counter)%iconV(11)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*2
         mesh(counter)%iconV(12)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*2
         mesh(counter)%iconV(13)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*3
         mesh(counter)%iconV(14)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*3
         mesh(counter)%iconV(15)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*3
         mesh(counter)%iconV(16)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*3

         mesh(counter)%xV(01)=(ielx-1)*hx
         mesh(counter)%xV(02)=(ielx-1)*hx+hx/3
         mesh(counter)%xV(03)=(ielx-1)*hx+hx/3*2
         mesh(counter)%xV(04)=(ielx-1)*hx+hx
         mesh(counter)%xV(05)=(ielx-1)*hx
         mesh(counter)%xV(06)=(ielx-1)*hx+hx/3
         mesh(counter)%xV(07)=(ielx-1)*hx+hx/3*2
         mesh(counter)%xV(08)=(ielx-1)*hx+hx
         mesh(counter)%xV(09)=(ielx-1)*hx
         mesh(counter)%xV(10)=(ielx-1)*hx+hx/3
         mesh(counter)%xV(11)=(ielx-1)*hx+hx/3*2
         mesh(counter)%xV(12)=(ielx-1)*hx+hx
         mesh(counter)%xV(13)=(ielx-1)*hx
         mesh(counter)%xV(14)=(ielx-1)*hx+hx/3
         mesh(counter)%xV(15)=(ielx-1)*hx+hx/3*2
         mesh(counter)%xV(16)=(ielx-1)*hx+hx
         mesh(counter)%yV(01)=(iely-1)*hy
         mesh(counter)%yV(02)=(iely-1)*hy
         mesh(counter)%yV(03)=(iely-1)*hy
         mesh(counter)%yV(04)=(iely-1)*hy
         mesh(counter)%yV(05)=(iely-1)*hy+hy/3
         mesh(counter)%yV(06)=(iely-1)*hy+hy/3
         mesh(counter)%yV(07)=(iely-1)*hy+hy/3
         mesh(counter)%yV(08)=(iely-1)*hy+hy/3
         mesh(counter)%yV(09)=(iely-1)*hy+hy/3*2
         mesh(counter)%yV(10)=(iely-1)*hy+hy/3*2
         mesh(counter)%yV(11)=(iely-1)*hy+hy/3*2
         mesh(counter)%yV(12)=(iely-1)*hy+hy/3*2
         mesh(counter)%yV(13)=(iely-1)*hy+hy
         mesh(counter)%yV(14)=(iely-1)*hy+hy
         mesh(counter)%yV(15)=(iely-1)*hy+hy
         mesh(counter)%yV(16)=(iely-1)*hy+hy

         mesh(counter)%xc=(ielx-1)*hx+hx/2
         mesh(counter)%yc=(iely-1)*hy+hy/2
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         mesh(counter)%vol=hx*hy
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
      end do    
   end do    

!------------------
case('__P1','_P1+')

   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         node1=ielx+(iely-1)*(nelx+1)    
         node2=ielx+1+(iely-1)*(nelx+1)    
         node3=ielx+1+iely*(nelx+1)    
         node4=ielx+iely*(nelx+1)    
         x1=(ielx-1)*hx ; x2=(ielx-1)*hx+hx ; x3=(ielx-1)*hx+hx ; x4=(ielx-1)*hx
         y1=(iely-1)*hy ; y2=(iely-1)*hy    ; y3=(iely-1)*hy+hy ; y4=(iely-1)*hy+hy
         if ((ielx<=nelx/2 .and. iely<=nely/2) .or. (ielx>nelx/2 .and. iely>nely/2)) then
            !C
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node2 ; mesh(counter)%xV(1)=x2 ; mesh(counter)%yV(1)=y2
            mesh(counter)%iconV(2)=node3 ; mesh(counter)%xV(2)=x3 ; mesh(counter)%yV(2)=y3
            mesh(counter)%iconV(3)=node1 ; mesh(counter)%xV(3)=x1 ; mesh(counter)%yV(3)=y1
            mesh(counter)%xc=sum(mesh(counter)%xV)/3
            mesh(counter)%yc=sum(mesh(counter)%yV)/3
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
            !D
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node4 ; mesh(counter)%xV(1)=x4 ; mesh(counter)%yV(1)=y4
            mesh(counter)%iconV(2)=node1 ; mesh(counter)%xV(2)=x1 ; mesh(counter)%yV(2)=y1
            mesh(counter)%iconV(3)=node3 ; mesh(counter)%xV(3)=x3 ; mesh(counter)%yV(3)=y3
            mesh(counter)%xc=sum(mesh(counter)%xV)/3
            mesh(counter)%yc=sum(mesh(counter)%yV)/3
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         else
            !A
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node1 ; mesh(counter)%xV(1)=x1 ; mesh(counter)%yV(1)=y1
            mesh(counter)%iconV(2)=node2 ; mesh(counter)%xV(2)=x2 ; mesh(counter)%yV(2)=y2
            mesh(counter)%iconV(3)=node4 ; mesh(counter)%xV(3)=x4 ; mesh(counter)%yV(3)=y4
            mesh(counter)%xc=sum(mesh(counter)%xV)/3
            mesh(counter)%yc=sum(mesh(counter)%yV)/3
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
            !B
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node3 ; mesh(counter)%xV(1)=x3 ; mesh(counter)%yV(1)=y3
            mesh(counter)%iconV(2)=node4 ; mesh(counter)%xV(2)=x4 ; mesh(counter)%yV(2)=y4
            mesh(counter)%iconV(3)=node2 ; mesh(counter)%xV(3)=x2 ; mesh(counter)%yV(3)=y2
            mesh(counter)%xc=sum(mesh(counter)%xV)/3
            mesh(counter)%yc=sum(mesh(counter)%yV)/3
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         end if
      end do    
   end do    

   if (spaceVelocity=='_P1+') then
      do iel=1,nel
         mesh(iel)%iconV(4)=(nelx+1)*(nely+1)+iel
         mesh(iel)%xV(4)=mesh(iel)%xc
         mesh(iel)%yV(4)=mesh(iel)%yc
         mesh(iel)%iconU(:)=mesh(iel)%iconV(:)
         mesh(iel)%xU(:)=mesh(iel)%xV(:)
         mesh(iel)%yU(:)=mesh(iel)%yV(:)
      end do
   end if

!-----------
case('__P2')
   nnx=2*nelx+1
   nny=2*nely+1
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         node1=(ielx-1)*2+1+(iely-1)*2*nnx
         node2=(ielx-1)*2+2+(iely-1)*2*nnx
         node3=(ielx-1)*2+3+(iely-1)*2*nnx
         node4=(ielx-1)*2+1+(iely-1)*2*nnx+nnx
         node5=(ielx-1)*2+2+(iely-1)*2*nnx+nnx
         node6=(ielx-1)*2+3+(iely-1)*2*nnx+nnx
         node7=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2
         node8=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2
         node9=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2
         x1=(ielx-1)*hx
         x2=(ielx-1)*hx+hx/2
         x3=(ielx-1)*hx+hx
         x4=(ielx-1)*hx
         x5=(ielx-1)*hx+hx/2
         x6=(ielx-1)*hx+hx
         x7=(ielx-1)*hx
         x8=(ielx-1)*hx+hx/2
         x9=(ielx-1)*hx+hx

         y1=(iely-1)*hy
         y2=(iely-1)*hy
         y3=(iely-1)*hy
         y4=(iely-1)*hy+hy/2
         y5=(iely-1)*hy+hy/2
         y6=(iely-1)*hy+hy/2
         y7=(iely-1)*hy+hy
         y8=(iely-1)*hy+hy
         y9=(iely-1)*hy+hy

         if ((ielx<=nelx/2 .and. iely<=nely/2) .or. (ielx>nelx/2 .and. iely>nely/2)) then
            !C
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node3 ; mesh(counter)%xV(1)=x3 ; mesh(counter)%yV(1)=y3
            mesh(counter)%iconV(2)=node9 ; mesh(counter)%xV(2)=x9 ; mesh(counter)%yV(2)=y9
            mesh(counter)%iconV(3)=node1 ; mesh(counter)%xV(3)=x1 ; mesh(counter)%yV(3)=y1
            mesh(counter)%iconV(4)=node6 ; mesh(counter)%xV(4)=x6 ; mesh(counter)%yV(4)=y6
            mesh(counter)%iconV(5)=node5 ; mesh(counter)%xV(5)=x5 ; mesh(counter)%yv(5)=y5
            mesh(counter)%iconV(6)=node2 ; mesh(counter)%xV(6)=x2 ; mesh(counter)%yV(6)=y2
            mesh(counter)%xc=sum(mesh(counter)%xV)/6
            mesh(counter)%yc=sum(mesh(counter)%yV)/6
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
            !D
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node7 ; mesh(counter)%xV(1)=x7 ; mesh(counter)%yV(1)=y7
            mesh(counter)%iconV(2)=node1 ; mesh(counter)%xV(2)=x1 ; mesh(counter)%yV(2)=y1
            mesh(counter)%iconV(3)=node9 ; mesh(counter)%xV(3)=x9 ; mesh(counter)%yV(3)=y9
            mesh(counter)%iconV(4)=node4 ; mesh(counter)%xV(4)=x4 ; mesh(counter)%yV(4)=y4
            mesh(counter)%iconV(5)=node5 ; mesh(counter)%xV(5)=x5 ; mesh(counter)%yV(5)=y5
            mesh(counter)%iconV(6)=node8 ; mesh(counter)%xV(6)=x8 ; mesh(counter)%yV(6)=y8
            mesh(counter)%xc=sum(mesh(counter)%xV)/6
            mesh(counter)%yc=sum(mesh(counter)%yV)/6
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         else
            !A
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node1 ; mesh(counter)%xV(1)=x1 ; mesh(counter)%yV(1)=y1
            mesh(counter)%iconV(2)=node3 ; mesh(counter)%xV(2)=x3 ; mesh(counter)%yV(2)=y3
            mesh(counter)%iconV(3)=node7 ; mesh(counter)%xV(3)=x7 ; mesh(counter)%yV(3)=y7
            mesh(counter)%iconV(4)=node2 ; mesh(counter)%xV(4)=x2 ; mesh(counter)%yV(4)=y2
            mesh(counter)%iconV(5)=node5 ; mesh(counter)%xV(5)=x5 ; mesh(counter)%yV(5)=y5
            mesh(counter)%iconV(6)=node4 ; mesh(counter)%xV(6)=x4 ; mesh(counter)%yV(6)=y4
            mesh(counter)%xc=sum(mesh(counter)%xV)/6
            mesh(counter)%yc=sum(mesh(counter)%yV)/6
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
            !B
            counter=counter+1    
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node9 ; mesh(counter)%xV(1)=x9 ; mesh(counter)%yV(1)=y9
            mesh(counter)%iconV(2)=node7 ; mesh(counter)%xV(2)=x7 ; mesh(counter)%yV(2)=y7
            mesh(counter)%iconV(3)=node3 ; mesh(counter)%xV(3)=x3 ; mesh(counter)%yV(3)=y3
            mesh(counter)%iconV(4)=node8 ; mesh(counter)%xV(4)=x8 ; mesh(counter)%yV(4)=y8
            mesh(counter)%iconV(5)=node5 ; mesh(counter)%xV(5)=x5 ; mesh(counter)%yV(5)=y5
            mesh(counter)%iconV(6)=node6 ; mesh(counter)%xV(6)=x6 ; mesh(counter)%yV(6)=y6
            mesh(counter)%xc=sum(mesh(counter)%xV)/6
            mesh(counter)%yc=sum(mesh(counter)%yV)/6
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
         mesh(counter)%iconU(:)=mesh(counter)%iconV(:)
         mesh(counter)%xU(:)=mesh(counter)%xV(:)
         mesh(counter)%yU(:)=mesh(counter)%yV(:)
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         end if
      end do    
   end do    

   if (spaceVelocity=='_P2+') then
      do iel=1,nel
         mesh(iel)%iconV(7)=nnx*nny+iel
         mesh(iel)%xV(7)=mesh(iel)%xc
         mesh(iel)%yV(7)=mesh(iel)%yc
         mesh(iel)%iconU(:)=mesh(iel)%iconV(:)
         mesh(iel)%xU(:)=mesh(iel)%xV(:)
         mesh(iel)%yU(:)=mesh(iel)%yV(:)
      end do
   end if

case default
   stop 'setup_cartesian2D: unknown spaceVelocity'

end select

!----------------------------------------------------------
! pressure 
!----------------------------------------------------------

select case(spacePressure)

!------------------
case('__Q0','__P0')
   do iel=1,nel
      mesh(iel)%iconP(1)=iel
      mesh(iel)%xP(1)=mesh(iel)%xC
      mesh(iel)%yP(1)=mesh(iel)%yC
   end do

!-----------
case('__Q1')
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%iconP(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconP(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconP(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconP(4)=ielx+iely*(nelx+1)    
         mesh(counter)%xP(1)=(ielx-1)*hx
         mesh(counter)%xP(2)=(ielx-1)*hx+hx
         mesh(counter)%xP(3)=(ielx-1)*hx+hx
         mesh(counter)%xP(4)=(ielx-1)*hx
         mesh(counter)%yP(1)=(iely-1)*hy
         mesh(counter)%yP(2)=(iely-1)*hy
         mesh(counter)%yP(3)=(iely-1)*hy+hy
         mesh(counter)%yP(4)=(iely-1)*hy+hy
      end do
   end do

!-----------
case('__P1')
   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         node1=ielx+(iely-1)*(nelx+1)    
         node2=ielx+1+(iely-1)*(nelx+1)    
         node3=ielx+1+iely*(nelx+1)    
         node4=ielx+iely*(nelx+1)    
         x1=(ielx-1)*hx ; x2=(ielx-1)*hx+hx ; x3=(ielx-1)*hx+hx ; x4=(ielx-1)*hx
         y1=(iely-1)*hy ; y2=(iely-1)*hy    ; y3=(iely-1)*hy+hy ; y4=(iely-1)*hy+hy
         if ((ielx<=nelx/2 .and. iely<=nely/2) .or. (ielx>nelx/2 .and. iely>nely/2)) then
            !C
            counter=counter+1    
            mesh(counter)%iconP(1)=node2 ; mesh(counter)%xP(1)=x2 ; mesh(counter)%yP(1)=y2
            mesh(counter)%iconP(2)=node3 ; mesh(counter)%xP(2)=x3 ; mesh(counter)%yP(2)=y3
            mesh(counter)%iconP(3)=node1 ; mesh(counter)%xP(3)=x1 ; mesh(counter)%yP(3)=y1
            !D
            counter=counter+1    
            mesh(counter)%iconP(1)=node4 ; mesh(counter)%xP(1)=x4 ; mesh(counter)%yP(1)=y4
            mesh(counter)%iconP(2)=node1 ; mesh(counter)%xP(2)=x1 ; mesh(counter)%yP(2)=y1
            mesh(counter)%iconP(3)=node3 ; mesh(counter)%xP(3)=x3 ; mesh(counter)%yP(3)=y3
         else
            !A
            counter=counter+1    
            mesh(counter)%iconP(1)=node1 ; mesh(counter)%xP(1)=x1 ; mesh(counter)%yP(1)=y1
            mesh(counter)%iconP(2)=node2 ; mesh(counter)%xP(2)=x2 ; mesh(counter)%yP(2)=y2
            mesh(counter)%iconP(3)=node4 ; mesh(counter)%xP(3)=x4 ; mesh(counter)%yP(3)=y4
            !B
            counter=counter+1    
            mesh(counter)%iconP(1)=node3 ; mesh(counter)%xP(1)=x3 ; mesh(counter)%yP(1)=y3
            mesh(counter)%iconP(2)=node4 ; mesh(counter)%xP(2)=x4 ; mesh(counter)%yP(2)=y4
            mesh(counter)%iconP(3)=node2 ; mesh(counter)%xP(3)=x2 ; mesh(counter)%yP(3)=y2
         end if
      end do    
   end do    

!-----------
case default
   stop 'setup_cartesian2D: spacePressure unknwown'
end select

!----------------------------------------------------------
! temperature (assumption: spaceT~spaceVelocity) 
!----------------------------------------------------------

if (use_T) then

select case(spaceVelocity)
case('__Q1','__Q2','__Q3','__P1','__P2','__P3')
   do iel=1,nel
      mesh(iel)%xT=mesh(iel)%xV
      mesh(iel)%yT=mesh(iel)%yV
      mesh(iel)%zT=mesh(iel)%zV
      mesh(iel)%iconT=mesh(iel)%iconV
   end do
case('_Q1+')
   do iel=1,nel
      mesh(iel)%xT=mesh(iel)%xV(1:4)
      mesh(iel)%yT=mesh(iel)%yV(1:4)
      mesh(iel)%zT=mesh(iel)%zV(1:4)
      mesh(iel)%iconT=mesh(iel)%iconV(1:4)
   end do
case('_P2+')
   do iel=1,nel
      mesh(iel)%xT=mesh(iel)%xV(1:6)
      mesh(iel)%yT=mesh(iel)%yV(1:6)
      mesh(iel)%zT=mesh(iel)%zV(1:6)
      mesh(iel)%iconT=mesh(iel)%iconV(1:6)
   end do
case default
   stop 'setup_cartesian2D: spaceT/spaceVelocity problem'
end select

end if ! use_T

!----------------------------------------------------------
! flag nodes on boundaries

do iel=1,nel
   do i=1,mU
      mesh(iel)%bnd1_Unode(i)=(abs(mesh(iel)%xU(i)-0 )<eps*Lx)
      mesh(iel)%bnd2_Unode(i)=(abs(mesh(iel)%xU(i)-Lx)<eps*Lx)
      mesh(iel)%bnd3_Unode(i)=(abs(mesh(iel)%yU(i)-0 )<eps*Ly)
      mesh(iel)%bnd4_Unode(i)=(abs(mesh(iel)%yU(i)-Ly)<eps*Ly)
   end do
   do i=1,mV
      mesh(iel)%bnd1_Vnode(i)=(abs(mesh(iel)%xV(i)-0 )<eps*Lx)
      mesh(iel)%bnd2_Vnode(i)=(abs(mesh(iel)%xV(i)-Lx)<eps*Lx)
      mesh(iel)%bnd3_Vnode(i)=(abs(mesh(iel)%yV(i)-0 )<eps*Ly)
      mesh(iel)%bnd4_Vnode(i)=(abs(mesh(iel)%yV(i)-Ly)<eps*Ly)
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
write(2345,*) limit//'setup_cartesian2D'//limit
do iel=1,nel
write(2345,*) 'elt:',iel,' | iconU',mesh(iel)%iconU(1:mU)
write(2345,*) 'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV)
write(2345,*) 'elt:',iel,' | iconP',mesh(iel)%iconP(1:mP)
do k=1,mU
write(2345,*) mesh(iel)%xU(k),mesh(iel)%yU(k),mesh(iel)%zU(k)
end do
do k=1,mV
write(2345,*) mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
end do
end do
do iel=1,nel
write(2345,*) 'iel,hx,hy,',iel,mesh(iel)%hx,mesh(iel)%hy
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_cartesian2D (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
