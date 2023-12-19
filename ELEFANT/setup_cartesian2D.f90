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
use module_constants 
use module_timing

implicit none

integer counter,ielx,iely,i,nnx,nny,k
integer :: node1,node2,node3,node4,node5,node6,node7,node8,node9
real(8) hx,hy,x1,x2,x3,x4,x5,x6,x7,x8,x9
real(8) y1,y2,y3,y4,y5,y6,y7,y8,y9

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_cartesian2D.f90}
!@@ This subroutine assigns to every element the coordinates of the its velocity, pressure,
!@@ and temperature nodes, the velocity, pressure and temperature connectivity arrays,
!@@ the coordinates of its center (xc,yc), its integer coordinates (ielx, iely),
!@@ and its dimensions (hx,hy).
!@@ \begin{center}
!@@ \input{tikz/tikz_3x2_q1}
!@@ \end{center}
!@@ \begin{verbatim}
!@@ elt:  1  | iconV  1  2  6   5  iconP  1
!@@ elt:  2  | iconV  2  3  7   6  iconP  2
!@@ elt:  3  | iconV  3  4  8   7  iconP  3
!@@ elt:  4  | iconV  5  6  10  9  iconP  4
!@@ elt:  5  | iconV  6  7  11  10 iconP  5
!@@ elt:  6  | iconV  7  8  12  11 iconP  6
!@@ \end{verbatim}
!@@ \begin{center}
!@@ \input{tikz/tikz_3x2_mini}
!@@ \end{center}
!@@ \begin{verbatim}
!@@ elt:  1  | iconV  1  2  6   5   13 iconP  1  2  6   5
!@@ elt:  2  | iconV  2  3  7   6   14 iconP  2  3  7   6
!@@ elt:  3  | iconV  3  4  8   7   15 iconP  3  4  8   7
!@@ elt:  4  | iconV  5  6  10  9   16 iconP  5  6  10  9
!@@ elt:  5  | iconV  6  7  11  10  17 iconP  6  7  11  10
!@@ elt:  6  | iconV  7  8  12  11  18 iconP  7  8  12  11
!@@ \end{verbatim}
!@@ \begin{center}
!@@ \input{tikz/tikz_3x2_q2}
!@@ \end{center}
!@@ \begin{verbatim}
!@@ elt:  1  | iconV  1   2   3   8   9   10  15  16  17 iconP 1  2  6  5
!@@ elt:  2  | iconV  3   4   5   10  11  12  17  18  19 iconP 2  3  7  6
!@@ elt:  3  | iconV  5   6   7   12  13  14  19  20  21 iconP 3  4  8  7
!@@ elt:  4  | iconV  15  16  17  22  23  24  29  30  31 iconP 5  6 10  9
!@@ elt:  5  | iconV  17  18  19  24  25  26  31  32  33 iconP 6  7 11 10
!@@ elt:  6  | iconV  19  20  21  26  27  28  33  34  35 iconP 7  8 12 11
!@@ \end{verbatim}
!@@ When adding a space: set iconV,xV,yV,xc,yc,vol + boundary nodes, element info
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely

!----------------------------------------------------------
!velocity 
!----------------------------------------------------------

select case(spaceV)

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
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
      end do    
   end do    

   if (spaceV=='_Q1+') then ! add bubble node
   do iel=1,nel
      mesh(iel)%xV(5)=mesh(iel)%xc
      mesh(iel)%yV(5)=mesh(iel)%yc
      mesh(iel)%iconV(5)=(nelx+1)*(nely+1)+iel
   end do
   end if

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
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
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
            mesh(counter)%ielx=ielx ; mesh(counter)%iely=iely
            mesh(counter)%iconV(1)=node2 ; mesh(counter)%xV(1)=x2 ; mesh(counter)%yV(1)=y2
            mesh(counter)%iconV(2)=node3 ; mesh(counter)%xV(2)=x3 ; mesh(counter)%yV(2)=y3
            mesh(counter)%iconV(3)=node1 ; mesh(counter)%xV(3)=x1 ; mesh(counter)%yV(3)=y1
            mesh(counter)%xc=sum(mesh(counter)%xV)/3
            mesh(counter)%yc=sum(mesh(counter)%yV)/3
            mesh(counter)%hx=hx ; mesh(counter)%hy=hy ; mesh(counter)%hz=0
            mesh(counter)%vol=hx*hy/2
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
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         end if
      end do    
   end do    

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
            if (ielx==1)    mesh(counter)%bnd1_elt=.true.
            if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
            if (iely==1)    mesh(counter)%bnd3_elt=.true.
            if (iely==nely) mesh(counter)%bnd4_elt=.true.
         end if
      end do    
   end do    

case default
   stop 'setup_cartesian2D: unknown spaceV'

end select

!----------------------------------------------------------
! pressure 
!----------------------------------------------------------

select case(spaceP)

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

case default
   stop 'setup_cartesian2D: spaceP unknwown'
end select

!----------------------------------------------------------
! temperature 
!----------------------------------------------------------

if (use_T) then

select case(spaceT)

!-----------
case('__Q1')

   counter=0    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%iconT(1)=ielx+(iely-1)*(nelx+1)    
         mesh(counter)%iconT(2)=ielx+1+(iely-1)*(nelx+1)    
         mesh(counter)%iconT(3)=ielx+1+iely*(nelx+1)    
         mesh(counter)%iconT(4)=ielx+iely*(nelx+1)    
         mesh(counter)%xT(1)=(ielx-1)*hx
         mesh(counter)%xT(2)=(ielx-1)*hx+hx
         mesh(counter)%xT(3)=(ielx-1)*hx+hx
         mesh(counter)%xT(4)=(ielx-1)*hx
         mesh(counter)%yT(1)=(iely-1)*hy
         mesh(counter)%yT(2)=(iely-1)*hy
         mesh(counter)%yT(3)=(iely-1)*hy+hy
         mesh(counter)%yT(4)=(iely-1)*hy+hy
      end do
   end do









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
            mesh(counter)%iconT(1)=node3 ; mesh(counter)%xT(1)=x3 ; mesh(counter)%yT(1)=y3
            mesh(counter)%iconT(2)=node9 ; mesh(counter)%xT(2)=x9 ; mesh(counter)%yT(2)=y9
            mesh(counter)%iconT(3)=node1 ; mesh(counter)%xT(3)=x1 ; mesh(counter)%yT(3)=y1
            mesh(counter)%iconT(4)=node6 ; mesh(counter)%xT(4)=x6 ; mesh(counter)%yT(4)=y6
            mesh(counter)%iconT(5)=node5 ; mesh(counter)%xT(5)=x5 ; mesh(counter)%yT(5)=y5
            mesh(counter)%iconT(6)=node2 ; mesh(counter)%xT(6)=x2 ; mesh(counter)%yT(6)=y2
            !D
            counter=counter+1    
            mesh(counter)%iconT(1)=node7 ; mesh(counter)%xT(1)=x7 ; mesh(counter)%yT(1)=y7
            mesh(counter)%iconT(2)=node1 ; mesh(counter)%xT(2)=x1 ; mesh(counter)%yT(2)=y1
            mesh(counter)%iconT(3)=node9 ; mesh(counter)%xT(3)=x9 ; mesh(counter)%yT(3)=y9
            mesh(counter)%iconT(4)=node4 ; mesh(counter)%xT(4)=x4 ; mesh(counter)%yT(4)=y4
            mesh(counter)%iconT(5)=node5 ; mesh(counter)%xT(5)=x5 ; mesh(counter)%yT(5)=y5
            mesh(counter)%iconT(6)=node8 ; mesh(counter)%xT(6)=x8 ; mesh(counter)%yT(6)=y8
         else
            !A
            counter=counter+1    
            mesh(counter)%iconT(1)=node1 ; mesh(counter)%xT(1)=x1 ; mesh(counter)%yT(1)=y1
            mesh(counter)%iconT(2)=node3 ; mesh(counter)%xT(2)=x3 ; mesh(counter)%yT(2)=y3
            mesh(counter)%iconT(3)=node7 ; mesh(counter)%xT(3)=x7 ; mesh(counter)%yT(3)=y7
            mesh(counter)%iconT(4)=node2 ; mesh(counter)%xT(4)=x2 ; mesh(counter)%yT(4)=y2
            mesh(counter)%iconT(5)=node5 ; mesh(counter)%xT(5)=x5 ; mesh(counter)%yT(5)=y5
            mesh(counter)%iconT(6)=node4 ; mesh(counter)%xT(6)=x4 ; mesh(counter)%yT(6)=y4
            !B
            counter=counter+1    
            mesh(counter)%iconT(1)=node9 ; mesh(counter)%xT(1)=x9 ; mesh(counter)%yT(1)=y9
            mesh(counter)%iconT(2)=node7 ; mesh(counter)%xT(2)=x7 ; mesh(counter)%yT(2)=y7
            mesh(counter)%iconT(3)=node3 ; mesh(counter)%xT(3)=x3 ; mesh(counter)%yT(3)=y3
            mesh(counter)%iconT(4)=node8 ; mesh(counter)%xT(4)=x8 ; mesh(counter)%yT(4)=y8
            mesh(counter)%iconT(5)=node5 ; mesh(counter)%xT(5)=x5 ; mesh(counter)%yT(5)=y5
            mesh(counter)%iconT(6)=node6 ; mesh(counter)%xT(6)=x6 ; mesh(counter)%yT(6)=y6
         end if
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
         mesh(counter)%iconT(1)=(ielx-1)*2+1+(iely-1)*2*nnx
         mesh(counter)%iconT(2)=(ielx-1)*2+2+(iely-1)*2*nnx
         mesh(counter)%iconT(3)=(ielx-1)*2+3+(iely-1)*2*nnx
         mesh(counter)%iconT(4)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx
         mesh(counter)%iconT(5)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx
         mesh(counter)%iconT(6)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx
         mesh(counter)%iconT(7)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2
         mesh(counter)%iconT(8)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2
         mesh(counter)%iconT(9)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2
         mesh(counter)%xT(1)=(ielx-1)*hx
         mesh(counter)%xT(2)=(ielx-1)*hx+hx/2
         mesh(counter)%xT(3)=(ielx-1)*hx+hx
         mesh(counter)%xT(4)=(ielx-1)*hx
         mesh(counter)%xT(5)=(ielx-1)*hx+hx/2
         mesh(counter)%xT(6)=(ielx-1)*hx+hx
         mesh(counter)%xT(7)=(ielx-1)*hx
         mesh(counter)%xT(8)=(ielx-1)*hx+hx/2
         mesh(counter)%xT(9)=(ielx-1)*hx+hx
         mesh(counter)%yT(1)=(iely-1)*hy
         mesh(counter)%yT(2)=(iely-1)*hy
         mesh(counter)%yT(3)=(iely-1)*hy
         mesh(counter)%yT(4)=(iely-1)*hy+hy/2
         mesh(counter)%yT(5)=(iely-1)*hy+hy/2
         mesh(counter)%yT(6)=(iely-1)*hy+hy/2
         mesh(counter)%yT(7)=(iely-1)*hy+hy
         mesh(counter)%yT(8)=(iely-1)*hy+hy
         mesh(counter)%yT(9)=(iely-1)*hy+hy
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
            mesh(counter)%iconT(1)=node2 ; mesh(counter)%xT(1)=x2 ; mesh(counter)%yT(1)=y2
            mesh(counter)%iconT(2)=node3 ; mesh(counter)%xT(2)=x3 ; mesh(counter)%yT(2)=y3
            mesh(counter)%iconT(3)=node1 ; mesh(counter)%xT(3)=x1 ; mesh(counter)%yT(3)=y1
            !D
            counter=counter+1    
            mesh(counter)%iconT(1)=node4 ; mesh(counter)%xT(1)=x4 ; mesh(counter)%yT(1)=y4
            mesh(counter)%iconT(2)=node1 ; mesh(counter)%xT(2)=x1 ; mesh(counter)%yT(2)=y1
            mesh(counter)%iconT(3)=node3 ; mesh(counter)%xT(3)=x3 ; mesh(counter)%yT(3)=y3
         else
            !A
            counter=counter+1    
            mesh(counter)%iconT(1)=node1 ; mesh(counter)%xT(1)=x1 ; mesh(counter)%yT(1)=y1
            mesh(counter)%iconT(2)=node2 ; mesh(counter)%xT(2)=x2 ; mesh(counter)%yT(2)=y2
            mesh(counter)%iconT(3)=node4 ; mesh(counter)%xT(3)=x4 ; mesh(counter)%yT(3)=y4
            !B
            counter=counter+1    
            mesh(counter)%iconT(1)=node3 ; mesh(counter)%xT(1)=x3 ; mesh(counter)%yT(1)=y3
            mesh(counter)%iconT(2)=node4 ; mesh(counter)%xT(2)=x4 ; mesh(counter)%yT(2)=y4
            mesh(counter)%iconT(3)=node2 ; mesh(counter)%xT(3)=x2 ; mesh(counter)%yT(3)=y2
         end if
      end do    
   end do    

!------------
case default
   stop 'setup_cartesian2D: unknown spaceT'
end select

end if ! use_T

!----------------------------------------------------------
! flag nodes on boundaries

do iel=1,nel
   do i=1,mV
      mesh(iel)%bnd1_node(i)=(abs(mesh(iel)%xV(i)-0 )<eps*Lx)
      mesh(iel)%bnd2_node(i)=(abs(mesh(iel)%xV(i)-Lx)<eps*Lx)
      mesh(iel)%bnd3_node(i)=(abs(mesh(iel)%yV(i)-0 )<eps*Ly)
      mesh(iel)%bnd4_node(i)=(abs(mesh(iel)%yV(i)-Ly)<eps*Ly)
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
write(2345,*) 'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV),'iconP',mesh(iel)%iconP(1:mP)
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
