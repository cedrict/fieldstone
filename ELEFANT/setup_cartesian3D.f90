!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_cartesian3D

use module_parameters
use module_mesh
use module_constants 
use module_timing

implicit none

integer counter,ielx,iely,ielz,i,k,nnx,nny,nnz
real(8) hx,hy,hz

call system_clock(counti,count_rate)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_cartesian3D.f90}
!@@ This subroutine assigns to every element the coordinates of the its velocity, pressure,
!@@ and temperature nodes, the velocity, pressure and temperature connectivity arrays,
!@@ the coordinates of its center (xc,yc,zc), its integer coordinates (ielx,iely,ielz),
!@@ and its dimensions (hx,hy,hz).
!==================================================================================================!

if (iproc==0) then

hx=Lx/nelx
hy=Ly/nely
hz=Lz/nelz

counter=0    
do ielz=1,nelz    
   do iely=1,nely    
      do ielx=1,nelx    
         counter=counter+1    
         mesh(counter)%ielx=ielx
         mesh(counter)%iely=iely
         mesh(counter)%ielz=ielz
         mesh(counter)%hx=hx
         mesh(counter)%hy=hy
         mesh(counter)%hz=hz
         mesh(counter)%vol=hx*hy*hz
         if (ielx==1)    mesh(counter)%bnd1_elt=.true.
         if (ielx==nelx) mesh(counter)%bnd2_elt=.true.
         if (iely==1)    mesh(counter)%bnd3_elt=.true.
         if (iely==nely) mesh(counter)%bnd4_elt=.true.
         if (ielz==1)    mesh(counter)%bnd5_elt=.true.
         if (ielz==nelz) mesh(counter)%bnd6_elt=.true.
      end do    
   end do    
end do

!----------------------------------------------------------
!velocity 

if (spaceV=='__Q1' .or. spaceV=='Q1++') then
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    
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
         end do    
      end do    
   end do    
end if

if (spaceV=='Q1++') then ! add bubble node
   do iel=1,nel
      mesh(iel)%xV(9)=mesh(iel)%xV(1)+hx/3
      mesh(iel)%yV(9)=mesh(iel)%yV(1)+hy/3
      mesh(iel)%zV(9)=mesh(iel)%zV(1)+hz/3
      mesh(counter)%iconV(9)=(nelx+1)*(nely+1)*(nelz+1)+2*(iel-1)+1
      mesh(iel)%xV(10)=mesh(iel)%xV(1)+2*hx/3
      mesh(iel)%yV(10)=mesh(iel)%yV(1)+2*hy/3
      mesh(iel)%zV(10)=mesh(iel)%zV(1)+2*hz/3
      mesh(counter)%iconV(10)=(nelx+1)*(nely+1)*(nelz+1)+2*(iel-1)+2
      !write(888,*) mesh(iel)%xV(9),mesh(iel)%yV(9),mesh(iel)%zV(9)
   end do
end if

if (spaceV=='__Q2') then
   nnx=2*nelx+1
   nny=2*nely+1
   nnz=2*nelz+1
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    

            mesh(counter)%iconV(1)=(ielx-1)*2+1+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(2)=(ielx-1)*2+2+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(3)=(ielx-1)*2+3+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(4)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(5)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(6)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(7)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(8)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(9)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
 
            mesh(counter)%iconV(10)=(ielx-1)*2+1+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(11)=(ielx-1)*2+2+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(12)=(ielx-1)*2+3+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(13)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(14)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(15)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(16)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(17)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(18)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)

            mesh(counter)%iconV(19)=(ielx-1)*2+1+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(20)=(ielx-1)*2+2+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(21)=(ielx-1)*2+3+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(22)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(23)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(24)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(25)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(26)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconV(27)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)

            mesh(counter)%xV(01)=(ielx-1)*hx
            mesh(counter)%xV(02)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(03)=(ielx-1)*hx+hx
            mesh(counter)%xV(04)=(ielx-1)*hx
            mesh(counter)%xV(05)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(06)=(ielx-1)*hx+hx
            mesh(counter)%xV(07)=(ielx-1)*hx
            mesh(counter)%xV(08)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(09)=(ielx-1)*hx+hx

            mesh(counter)%xV(10)=(ielx-1)*hx
            mesh(counter)%xV(11)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(12)=(ielx-1)*hx+hx
            mesh(counter)%xV(13)=(ielx-1)*hx
            mesh(counter)%xV(14)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(15)=(ielx-1)*hx+hx
            mesh(counter)%xV(16)=(ielx-1)*hx
            mesh(counter)%xV(17)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(18)=(ielx-1)*hx+hx

            mesh(counter)%xV(19)=(ielx-1)*hx
            mesh(counter)%xV(20)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(21)=(ielx-1)*hx+hx
            mesh(counter)%xV(22)=(ielx-1)*hx
            mesh(counter)%xV(23)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(24)=(ielx-1)*hx+hx
            mesh(counter)%xV(25)=(ielx-1)*hx
            mesh(counter)%xV(26)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xV(27)=(ielx-1)*hx+hx

            mesh(counter)%yV(01)=(iely-1)*hy
            mesh(counter)%yV(02)=(iely-1)*hy
            mesh(counter)%yV(03)=(iely-1)*hy
            mesh(counter)%yV(04)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(05)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(06)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(07)=(iely-1)*hy+hy
            mesh(counter)%yV(08)=(iely-1)*hy+hy
            mesh(counter)%yV(09)=(iely-1)*hy+hy

            mesh(counter)%yV(10)=(iely-1)*hy
            mesh(counter)%yV(11)=(iely-1)*hy
            mesh(counter)%yV(12)=(iely-1)*hy
            mesh(counter)%yV(13)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(14)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(15)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(16)=(iely-1)*hy+hy
            mesh(counter)%yV(17)=(iely-1)*hy+hy
            mesh(counter)%yV(18)=(iely-1)*hy+hy

            mesh(counter)%yV(19)=(iely-1)*hy
            mesh(counter)%yV(20)=(iely-1)*hy
            mesh(counter)%yV(21)=(iely-1)*hy
            mesh(counter)%yV(22)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(23)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(24)=(iely-1)*hy+hy/2d0
            mesh(counter)%yV(25)=(iely-1)*hy+hy
            mesh(counter)%yV(26)=(iely-1)*hy+hy
            mesh(counter)%yV(27)=(iely-1)*hy+hy

            mesh(counter)%zV(01)=(ielz-1)*hz
            mesh(counter)%zV(02)=(ielz-1)*hz
            mesh(counter)%zV(03)=(ielz-1)*hz
            mesh(counter)%zV(04)=(ielz-1)*hz
            mesh(counter)%zV(05)=(ielz-1)*hz
            mesh(counter)%zV(06)=(ielz-1)*hz
            mesh(counter)%zV(07)=(ielz-1)*hz
            mesh(counter)%zV(08)=(ielz-1)*hz
            mesh(counter)%zV(09)=(ielz-1)*hz

            mesh(counter)%zV(10)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(11)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(12)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(13)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(14)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(15)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(16)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(17)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zV(18)=(ielz-1)*hz+hz/2d0

            mesh(counter)%zV(19)=(ielz-1)*hz+hz
            mesh(counter)%zV(20)=(ielz-1)*hz+hz
            mesh(counter)%zV(21)=(ielz-1)*hz+hz
            mesh(counter)%zV(22)=(ielz-1)*hz+hz
            mesh(counter)%zV(23)=(ielz-1)*hz+hz
            mesh(counter)%zV(24)=(ielz-1)*hz+hz
            mesh(counter)%zV(25)=(ielz-1)*hz+hz
            mesh(counter)%zV(26)=(ielz-1)*hz+hz
            mesh(counter)%zV(27)=(ielz-1)*hz+hz

         end do
      end do
   end do
end if

!----------------------------------------------------------
! pressure 

if (spaceP=='__Q0') then
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

if (spaceP=='__Q1') then
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    

            mesh(counter)%iconP(1)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconP(2)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconP(3)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconP(4)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx
            mesh(counter)%iconP(5)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconP(6)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconP(7)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconP(8)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx

            mesh(counter)%xP(1)=(ielx-1)*hx
            mesh(counter)%xP(2)=(ielx-1)*hx+hx
            mesh(counter)%xP(3)=(ielx-1)*hx+hx
            mesh(counter)%xP(4)=(ielx-1)*hx
            mesh(counter)%xP(5)=(ielx-1)*hx
            mesh(counter)%xP(6)=(ielx-1)*hx+hx
            mesh(counter)%xP(7)=(ielx-1)*hx+hx
            mesh(counter)%xP(8)=(ielx-1)*hx

            mesh(counter)%yP(1)=(iely-1)*hy
            mesh(counter)%yP(2)=(iely-1)*hy
            mesh(counter)%yP(3)=(iely-1)*hy+hy
            mesh(counter)%yP(4)=(iely-1)*hy+hy
            mesh(counter)%yP(5)=(iely-1)*hy
            mesh(counter)%yP(6)=(iely-1)*hy
            mesh(counter)%yP(7)=(iely-1)*hy+hy
            mesh(counter)%yP(8)=(iely-1)*hy+hy

            mesh(counter)%zP(1)=(ielz-1)*hz
            mesh(counter)%zP(2)=(ielz-1)*hz
            mesh(counter)%zP(3)=(ielz-1)*hz
            mesh(counter)%zP(4)=(ielz-1)*hz
            mesh(counter)%zP(5)=(ielz-1)*hz+hz
            mesh(counter)%zP(6)=(ielz-1)*hz+hz
            mesh(counter)%zP(7)=(ielz-1)*hz+hz
            mesh(counter)%zP(8)=(ielz-1)*hz+hz
         end do
      end do
   end do
end if

!----------------------------------------------------------
! temperature 

if (spaceT=='__Q1') then
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    

            mesh(counter)%iconT(1)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconT(2)=(nelx+1)*(nely+1)*(ielz-1)+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconT(3)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconT(4)=(nelx+1)*(nely+1)*(ielz-1)+ (iely  )*(nelx+1) + ielx
            mesh(counter)%iconT(5)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx
            mesh(counter)%iconT(6)=(nelx+1)*(nely+1)*(ielz  )+ (iely-1)*(nelx+1) + ielx+1
            mesh(counter)%iconT(7)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx+1
            mesh(counter)%iconT(8)=(nelx+1)*(nely+1)*(ielz  )+ (iely  )*(nelx+1) + ielx

            mesh(counter)%xT(1)=(ielx-1)*hx
            mesh(counter)%xT(2)=(ielx-1)*hx+hx
            mesh(counter)%xT(3)=(ielx-1)*hx+hx
            mesh(counter)%xT(4)=(ielx-1)*hx
            mesh(counter)%xT(5)=(ielx-1)*hx
            mesh(counter)%xT(6)=(ielx-1)*hx+hx
            mesh(counter)%xT(7)=(ielx-1)*hx+hx
            mesh(counter)%xT(8)=(ielx-1)*hx

            mesh(counter)%yT(1)=(iely-1)*hy
            mesh(counter)%yT(2)=(iely-1)*hy
            mesh(counter)%yT(3)=(iely-1)*hy+hy
            mesh(counter)%yT(4)=(iely-1)*hy+hy
            mesh(counter)%yT(5)=(iely-1)*hy
            mesh(counter)%yT(6)=(iely-1)*hy
            mesh(counter)%yT(7)=(iely-1)*hy+hy
            mesh(counter)%yT(8)=(iely-1)*hy+hy

            mesh(counter)%zT(1)=(ielz-1)*hz
            mesh(counter)%zT(2)=(ielz-1)*hz
            mesh(counter)%zT(3)=(ielz-1)*hz
            mesh(counter)%zT(4)=(ielz-1)*hz
            mesh(counter)%zT(5)=(ielz-1)*hz+hz
            mesh(counter)%zt(6)=(ielz-1)*hz+hz
            mesh(counter)%zt(7)=(ielz-1)*hz+hz
            mesh(counter)%zT(8)=(ielz-1)*hz+hz
         end do
      end do
   end do
end if

if (spaceT=='__Q2') then
   nnx=2*nelx+1
   nny=2*nely+1
   nnz=2*nelz+1
   counter=0    
   do ielz=1,nelz    
      do iely=1,nely    
         do ielx=1,nelx    
            counter=counter+1    

            mesh(counter)%iconT(1)=(ielx-1)*2+1+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(2)=(ielx-1)*2+2+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(3)=(ielx-1)*2+3+(iely-1)*2*nnx                   + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(4)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(5)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(6)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx               + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(7)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(8)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(9)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2             + 2*nnx*nny*(ielz-1)
 
            mesh(counter)%iconT(10)=(ielx-1)*2+1+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(11)=(ielx-1)*2+2+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(12)=(ielx-1)*2+3+(iely-1)*2*nnx        + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(13)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(14)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(15)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx    + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(16)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(17)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(18)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2  + nnx*nny + 2*nnx*nny*(ielz-1)

            mesh(counter)%iconT(19)=(ielx-1)*2+1+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(20)=(ielx-1)*2+2+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(21)=(ielx-1)*2+3+(iely-1)*2*nnx        + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(22)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(23)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(24)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx    + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(25)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(26)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)
            mesh(counter)%iconT(27)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2  + 2*nnx*nny + 2*nnx*nny*(ielz-1)

            mesh(counter)%xT(01)=(ielx-1)*hx
            mesh(counter)%xT(02)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(03)=(ielx-1)*hx+hx
            mesh(counter)%xT(04)=(ielx-1)*hx
            mesh(counter)%xT(05)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(06)=(ielx-1)*hx+hx
            mesh(counter)%xT(07)=(ielx-1)*hx
            mesh(counter)%xT(08)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(09)=(ielx-1)*hx+hx

            mesh(counter)%xT(10)=(ielx-1)*hx
            mesh(counter)%xT(11)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(12)=(ielx-1)*hx+hx
            mesh(counter)%xT(13)=(ielx-1)*hx
            mesh(counter)%xT(14)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(15)=(ielx-1)*hx+hx
            mesh(counter)%xT(16)=(ielx-1)*hx
            mesh(counter)%xT(17)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(18)=(ielx-1)*hx+hx

            mesh(counter)%xT(19)=(ielx-1)*hx
            mesh(counter)%xT(20)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(21)=(ielx-1)*hx+hx
            mesh(counter)%xT(22)=(ielx-1)*hx
            mesh(counter)%xT(23)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(24)=(ielx-1)*hx+hx
            mesh(counter)%xT(25)=(ielx-1)*hx
            mesh(counter)%xT(26)=(ielx-1)*hx+hx/2d0
            mesh(counter)%xT(27)=(ielx-1)*hx+hx

            mesh(counter)%yT(01)=(iely-1)*hy
            mesh(counter)%yT(02)=(iely-1)*hy
            mesh(counter)%yT(03)=(iely-1)*hy
            mesh(counter)%yT(04)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(05)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(06)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(07)=(iely-1)*hy+hy
            mesh(counter)%yT(08)=(iely-1)*hy+hy
            mesh(counter)%yT(09)=(iely-1)*hy+hy

            mesh(counter)%yT(10)=(iely-1)*hy
            mesh(counter)%yT(11)=(iely-1)*hy
            mesh(counter)%yT(12)=(iely-1)*hy
            mesh(counter)%yT(13)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(14)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(15)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(16)=(iely-1)*hy+hy
            mesh(counter)%yT(17)=(iely-1)*hy+hy
            mesh(counter)%yT(18)=(iely-1)*hy+hy

            mesh(counter)%yT(19)=(iely-1)*hy
            mesh(counter)%yT(20)=(iely-1)*hy
            mesh(counter)%yT(21)=(iely-1)*hy
            mesh(counter)%yT(22)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(23)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(24)=(iely-1)*hy+hy/2d0
            mesh(counter)%yT(25)=(iely-1)*hy+hy
            mesh(counter)%yT(26)=(iely-1)*hy+hy
            mesh(counter)%yT(27)=(iely-1)*hy+hy

            mesh(counter)%zT(01)=(ielz-1)*hz
            mesh(counter)%zT(02)=(ielz-1)*hz
            mesh(counter)%zT(03)=(ielz-1)*hz
            mesh(counter)%zT(04)=(ielz-1)*hz
            mesh(counter)%zT(05)=(ielz-1)*hz
            mesh(counter)%zT(06)=(ielz-1)*hz
            mesh(counter)%zT(07)=(ielz-1)*hz
            mesh(counter)%zT(08)=(ielz-1)*hz
            mesh(counter)%zT(09)=(ielz-1)*hz

            mesh(counter)%zT(10)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(11)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(12)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(13)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(14)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(15)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(16)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(17)=(ielz-1)*hz+hz/2d0
            mesh(counter)%zT(18)=(ielz-1)*hz+hz/2d0

            mesh(counter)%zT(19)=(ielz-1)*hz+hz
            mesh(counter)%zT(20)=(ielz-1)*hz+hz
            mesh(counter)%zT(21)=(ielz-1)*hz+hz
            mesh(counter)%zT(22)=(ielz-1)*hz+hz
            mesh(counter)%zT(23)=(ielz-1)*hz+hz
            mesh(counter)%zT(24)=(ielz-1)*hz+hz
            mesh(counter)%zT(25)=(ielz-1)*hz+hz
            mesh(counter)%zT(26)=(ielz-1)*hz+hz
            mesh(counter)%zT(27)=(ielz-1)*hz+hz

         end do
      end do
   end do
end if

!----------------------------------------------------------
! flag nodes on boundaries

do iel=1,nel
   do i=1,mV
      mesh(iel)%bnd1_node(i)=(abs(mesh(iel)%xV(i)-0 )<eps*Lx)
      mesh(iel)%bnd2_node(i)=(abs(mesh(iel)%xV(i)-Lx)<eps*Lx)
      mesh(iel)%bnd3_node(i)=(abs(mesh(iel)%yV(i)-0 )<eps*Ly)
      mesh(iel)%bnd4_node(i)=(abs(mesh(iel)%yV(i)-Ly)<eps*Ly)
      mesh(iel)%bnd5_node(i)=(abs(mesh(iel)%zV(i)-0 )<eps*Lz)
      mesh(iel)%bnd6_node(i)=(abs(mesh(iel)%zV(i)-Lz)<eps*Lz)
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

if (debug) then
write(2345,*) limit//'setup_cartesian3D'//limit
do iel=1,nel
print *,'--------------------------------------------------'
print *,'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV)
do k=1,mV
write(2345,*) mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
end do
end do
print *,'--------------------------------------------------'
print *,'--------------------------------------------------'
do iel=1,nel
print *,'--------------------------------------------------'
print *,'elt:',iel,' | iconP',mesh(iel)%iconP(1:mP)
do k=1,mP
write(2345,*) mesh(iel)%xP(k),mesh(iel)%yP(k),mesh(iel)%zP(k)
end do
end do
print *,'--------------------------------------------------'
print *,'--------------------------------------------------'
do iel=1,nel
print *,'--------------------------------------------------'
print *,'elt:',iel,' | iconT',mesh(iel)%iconT(1:mT)
do k=1,mT
write(2345,*) mesh(iel)%xT(k),mesh(iel)%yT(k),mesh(iel)%zT(k)
end do
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') '     >> setup_cartesian3D                ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
