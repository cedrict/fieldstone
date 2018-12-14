!==============================================!
!                                              !
! C. thieulot ; May 2011                       !
!                                              !
!==============================================!
                                               !
program fcubed                                 !
                                               !
implicit none                                  !
                                               !
integer, parameter :: m=8                      ! number of nodes which constitute an element
integer, parameter :: ndof=3                   ! number of dofs per node
integer nnx                                    ! number of grid points in the x direction
integer nny                                    ! number of grid points in the y direction
integer nnz                                    ! number of grid points in the z direction
integer np                                     ! number of grid points
integer nelx                                   ! number of elements in the x direction
integer nely                                   ! number of elements in the y direction
integer nelz                                   ! number of elements in the z direction
integer nel                                    ! number of elements
integer Nfem                                   ! size of the FEM matrix 
integer, dimension(:,:), allocatable :: icon   ! connectivity array
integer, dimension(:), allocatable :: ipvt     ! work array needed by the solver 
                                               !
integer i1,i2,i3,i,j,k,iel,counter,iq,jq,kq          !
integer ik,jk,ikk,jkk,m1,m2,k1,k2,job          !
                                               !  
real(8) Lx,Ly,Lz                               ! size of the numerical domain
real(8) viscosity                              ! dynamic viscosity $\mu$ of the material
real(8) density                                ! mass density $\rho$ of the material
real(8) gx,gy,gz                               ! gravity acceleration
real(8) penalty                                ! penalty parameter lambda
real(8), dimension(:),   allocatable :: x,y,z  ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v,w  ! node velocity arrays
real(8), dimension(:),   allocatable :: press  ! pressure 
real(8), dimension(:),   allocatable :: rho    !  
real(8), dimension(:),   allocatable :: B      ! right hand side
real(8), dimension(:,:), allocatable :: A      ! FEM matrix
real(8), dimension(:),   allocatable :: work   ! work array needed by the solver
real(8), dimension(:),   allocatable :: bc_val ! array containing bc values
                                               !
real(8), external :: rhofct,uth,vth,pth         ! body force and analytical solution
real(8) rq,sq,tq,weightq                              ! local coordinate and weight of qpoint
real(8) xq,yq,zq                               ! global coordinate of qpoint
real(8) uq,vq,wq                               ! velocity at qpoint
real(8) exxq,eyyq,ezzq,exyq,exzq,eyzq          ! strain-rate components at qpoint  
real(8) Ael(m*ndof,m*ndof)                     ! elemental FEM matrix
real(8) Bel(m*ndof)                            ! elemental right hand side
real(8) N(m),dNdx(m),dNdy(m),dNdz(m),dNdr(m),dNds(m),dNdt(m)  ! shape fcts and derivatives
real(8) jcob                                   ! determinant of jacobian matrix
real(8) jcb(3,3)                               ! jacobian matrix
real(8) jcbi(3,3)                              ! inverse of jacobian matrix
real(8) Bmat(6,ndof*m)                         ! B matrix
real(8), dimension(6,6) :: Kmat                ! K matrix 
real(8), dimension(6,6) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) eps                                    !
real(8) rcond                                  !
                                               !
logical, dimension(:), allocatable :: bc_fix   ! prescribed b.c. array
logical, dimension(:,:), allocatable :: C      ! non-zero terms in FEM matrix
                                               !
!==============================================!
!=====[setup]==================================!
!==============================================!

Lx=1.d0
Ly=1.d0
Lz=1.d0

nnx=17
nny=17
nnz=17

np=nnx*nny*nnz

nelx=nnx-1
nely=nny-1
nelz=nnz-1

nel=nelx*nely*nelz

penalty=1.d7

viscosity=1.d0
density=1.d0

Nfem=np*ndof

eps=1.d-10

Kmat=0
Kmat(1,1:3)=1.d0
Kmat(2,1:3)=1.d0
Kmat(3,1:3)=1.d0

Cmat=0
Cmat(1,1)=2.d0
Cmat(2,2)=2.d0
Cmat(3,3)=2.d0
Cmat(4,4)=1.d0
Cmat(5,5)=1.d0
Cmat(6,6)=1.d0

gx=0
gy=0
gz=-1

!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(z(np))
allocate(u(np))
allocate(v(np))
allocate(w(np))
allocate(icon(m,nel))
allocate(A(Nfem,Nfem))
allocate(B(Nfem))
allocate(C(Nfem,Nfem))
allocate(bc_fix(Nfem))
allocate(bc_val(Nfem))
allocate(press(nel))
allocate(rho(nel))

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do i=0,nelx
do j=0,nely
do k=0,nelz
   counter=counter+1
   x(counter)=dble(i)*Lx/dble(nelx)
   y(counter)=dble(j)*Ly/dble(nely)
   z(counter)=dble(k)*Lz/dble(nelz)
end do
end do
end do

open(unit=123,file='OUT/gridnodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,np
   write(123,'(3f10.5,i8)') x(i),y(i),z(i),i
end do
close(123)


!==============================================!
!===[connectivity]=============================!
!==============================================!

counter=0
do i=1,nelx
do j=1,nely
do k=1,nelz
counter=counter+1
icon(1,counter)=nny*nnz*(i-1)+nnz*(j-1)+k
icon(2,counter)=nny*nnz*(i  )+nnz*(j-1)+k
icon(3,counter)=nny*nnz*(i  )+nnz*(j  )+k
icon(4,counter)=nny*nnz*(i-1)+nnz*(j  )+k
icon(5,counter)=nny*nnz*(i-1)+nnz*(j-1)+k+1
icon(6,counter)=nny*nnz*(i  )+nnz*(j-1)+k+1
icon(7,counter)=nny*nnz*(i  )+nnz*(j  )+k+1
icon(8,counter)=nny*nnz*(i-1)+nnz*(j  )+k+1
end do
end do
end do

open(unit=123,file='OUT/icon.dat',status='replace')
do iel=1,nel
   write(123,'(a)') '----------------------------'
   write(123,'(a,i4,a)') '---element #',iel,' -----------'
   write(123,'(a)') '----------------------------'
   write(123,'(a,i8,a,2f20.10)') '  node 1 ', icon(1,iel),' at pos. ',x(icon(1,iel)),y(icon(1,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 2 ', icon(2,iel),' at pos. ',x(icon(2,iel)),y(icon(2,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 3 ', icon(3,iel),' at pos. ',x(icon(3,iel)),y(icon(3,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 4 ', icon(4,iel),' at pos. ',x(icon(4,iel)),y(icon(4,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 5 ', icon(5,iel),' at pos. ',x(icon(5,iel)),y(icon(5,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 6 ', icon(6,iel),' at pos. ',x(icon(6,iel)),y(icon(6,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 7 ', icon(7,iel),' at pos. ',x(icon(7,iel)),y(icon(7,iel))
   write(123,'(a,i8,a,2f20.10)') '  node 8 ', icon(8,iel),' at pos. ',x(icon(8,iel)),y(icon(8,iel))
end do
close(123)


!==============================================!
!=====[define bc]==============================!
!==============================================!

bc_fix=.false.

do i=1,np
   if (x(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      !bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
   if (x(i).gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      !bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
   if (y(i).lt.eps) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      !bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
   if (y(i).gt.(Ly-eps) ) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      !bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
   if (z(i).lt.eps) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
   if (z(i).gt.(Lz-eps) ) then
      !bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      !bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      bc_fix((i-1)*ndof+3)=.true. ; bc_val((i-1)*ndof+3)=0.d0
   endif
end do

open(unit=123,file='OUT/bc_u.dat',status='replace')
open(unit=234,file='OUT/bc_v.dat',status='replace')
open(unit=345,file='OUT/bc_w.dat',status='replace')
do i=1,np
   if (bc_fix((i-1)*ndof+1)) write(123,'(4f20.10)') x(i),y(i),z(i),bc_val((i-1)*ndof+1) 
   if (bc_fix((i-1)*ndof+2)) write(234,'(4f20.10)') x(i),y(i),z(i),bc_val((i-1)*ndof+2) 
   if (bc_fix((i-1)*ndof+3)) write(345,'(4f20.10)') x(i),y(i),z(i),bc_val((i-1)*ndof+2) 
end do
close(123)
close(234)
close(345)


!==============================================!
!=====[build FE matrix]========================!
!==============================================!

A=0.d0
B=0.d0
C=.false.

do iel=1,nel

   Ael=0.d0
   Bel=0.d0

   do iq=-1,1,2
   do jq=-1,1,2
   do kq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      tq=kq/sqrt(3.d0)
      weightq=1.d0*1.d0*1.d0

      N(1)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0-tq)
      N(2)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0-tq)
      N(3)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0-tq)
      N(4)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0-tq)
      N(5)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0+tq)
      N(6)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0+tq)
      N(7)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0+tq)
      N(8)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0+tq)

      dNdr(1)= - 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(2)= + 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(3)= + 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(4)= - 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(5)= - 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(6)= + 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(7)= + 0.125d0*(1.d0+sq)*(1.d0+tq)    
      dNdr(8)= - 0.125d0*(1.d0+sq)*(1.d0+tq)    

      dNds(1)= - 0.125d0*(1.d0-rq)*(1.d0-tq)    
      dNds(2)= - 0.125d0*(1.d0+rq)*(1.d0-tq)    
      dNds(3)= + 0.125d0*(1.d0+rq)*(1.d0-tq)  
      dNds(4)= + 0.125d0*(1.d0-rq)*(1.d0-tq)
      dNds(5)= - 0.125d0*(1.d0-rq)*(1.d0+tq)    
      dNds(6)= - 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(7)= + 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(8)= + 0.125d0*(1.d0-rq)*(1.d0+tq)    

      dNdt(1)= - 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(2)= - 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(3)= - 0.125d0*(1.d0+rq)*(1.d0+sq)    
      dNdt(4)= - 0.125d0*(1.d0-rq)*(1.d0+sq)    
      dNdt(5)= + 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(6)= + 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(7)= + 0.125d0*(1.d0+rq)*(1.d0+sq)  
      dNdt(8)= + 0.125d0*(1.d0-rq)*(1.d0+sq) 

      jcb=0.d0    
      do k=1,8    
      jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
      jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
      jcb(1,3)=jcb(1,3)+dNdr(k)*z(icon(k,iel))
      jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
      jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      jcb(2,3)=jcb(2,3)+dNds(k)*z(icon(k,iel))
      jcb(3,1)=jcb(3,1)+dNdt(k)*x(icon(k,iel))
      jcb(3,2)=jcb(3,2)+dNdt(k)*y(icon(k,iel))
      jcb(3,3)=jcb(3,3)+dNdt(k)*z(icon(k,iel))
      enddo    
    
      jcob=jcb(1,1)*jcb(2,2)*jcb(3,3) &    
          +jcb(1,2)*jcb(2,3)*jcb(3,1) &    
          +jcb(2,1)*jcb(3,2)*jcb(1,3) &    
          -jcb(1,3)*jcb(2,2)*jcb(3,1) &    
          -jcb(1,2)*jcb(2,1)*jcb(3,3) &    
          -jcb(2,3)*jcb(3,2)*jcb(1,1)    

      jcbi(1,1)=(jcb(2,2)*jcb(3,3)-jcb(2,3)*jcb(3,2))/jcob    
      jcbi(2,1)=(jcb(2,3)*jcb(3,1)-jcb(2,1)*jcb(3,3))/jcob    
      jcbi(3,1)=(jcb(2,1)*jcb(3,2)-jcb(2,2)*jcb(3,1))/jcob  
      jcbi(1,2)=(jcb(1,3)*jcb(3,2)-jcb(1,2)*jcb(3,3))/jcob
      jcbi(2,2)=(jcb(1,1)*jcb(3,3)-jcb(1,3)*jcb(3,1))/jcob    
      jcbi(3,2)=(jcb(1,2)*jcb(3,1)-jcb(1,1)*jcb(3,2))/jcob    
      jcbi(1,3)=(jcb(1,2)*jcb(2,3)-jcb(1,3)*jcb(2,2))/jcob    
      jcbi(2,3)=(jcb(1,3)*jcb(2,1)-jcb(1,1)*jcb(2,3))/jcob    
      jcbi(3,3)=(jcb(1,1)*jcb(2,2)-jcb(1,2)*jcb(2,1))/jcob    



      xq=0.d0
      yq=0.d0
      zq=0.d0
      uq=0.d0
      vq=0.d0
      wq=0.d0
      exxq=0.d0
      eyyq=0.d0
      ezzq=0.d0
      exyq=0.d0
      exzq=0.d0
      eyzq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         zq=zq+N(k)*z(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         wq=wq+N(k)*w(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)&    
                +jcbi(1,2)*dNds(k)&    
                +jcbi(1,3)*dNdt(k)    
         dNdy(k)=jcbi(2,1)*dNdr(k)&    
                +jcbi(2,2)*dNds(k)&    
                +jcbi(2,3)*dNdt(k)    
         dNdz(k)=jcbi(3,1)*dNdr(k)&    
                +jcbi(3,2)*dNds(k)&    
                +jcbi(3,3)*dNdt(k)  
         exxq=exxq+ dNdx(k)*u(icon(k,iel))
         eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
         ezzq=ezzq+ dNdz(k)*w(icon(k,iel))
         exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
                  + dNdy(k)*u(icon(k,iel)) *0.5d0
         exzq=exzq+ dNdx(k)*w(icon(k,iel)) *0.5d0 &
                  + dNdz(k)*u(icon(k,iel)) *0.5d0
         eyzq=eyzq+ dNdy(k)*w(icon(k,iel)) *0.5d0 &
                  + dNdz(k)*v(icon(k,iel)) *0.5d0
      end do

      !write(999,*) xq,yq,zq,uq,vq,wq,exxq,eyyq,ezzq,exyq,exzq,eyzq

      Bmat=0.d0
      do i=1,m
         i1=ndof*i-2
         i2=ndof*i-1
         i3=ndof*i
         Bmat(1,i1)=dNdx(i)
         Bmat(2,i2)=dNdy(i)
         Bmat(3,i3)=dNdz(i)
         Bmat(4,i1)=dNdy(i) ; Bmat(4,i2)=dNdx(i)
         Bmat(5,i1)=dNdz(i) ; Bmat(5,i3)=dNdx(i)
         Bmat(6,i2)=dNdz(i) ; Bmat(6,i3)=dNdy(i)
      end do

      Ael=Ael + matmul(transpose(Bmat),matmul(viscosity*Cmat,Bmat))*weightq*jcob

      do i=1,m
         i1=ndof*i-2
         i2=ndof*i-1
         i3=ndof*i
         density=rhofct(xq,yq,zq)
         Bel(i1)=Bel(i1)-N(i)*jcob*weightq*density*gx
         Bel(i2)=Bel(i2)-N(i)*jcob*weightq*density*gy
         Bel(i3)=Bel(i3)-N(i)*jcob*weightq*density*gz
      end do

   end do
   end do
   end do

   ! 1 point integration

      rq=0.d0
      sq=0.d0
      tq=0.d0
      weightq=2.d0*2.d0*2.d0

      N(1)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0-tq)
      N(2)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0-tq)
      N(3)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0-tq)
      N(4)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0-tq)
      N(5)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0+tq)
      N(6)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0+tq)
      N(7)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0+tq)
      N(8)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0+tq)

      dNdr(1)= - 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(2)= + 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(3)= + 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(4)= - 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(5)= - 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(6)= + 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(7)= + 0.125d0*(1.d0+sq)*(1.d0+tq)    
      dNdr(8)= - 0.125d0*(1.d0+sq)*(1.d0+tq)    

      dNds(1)= - 0.125d0*(1.d0-rq)*(1.d0-tq)    
      dNds(2)= - 0.125d0*(1.d0+rq)*(1.d0-tq)    
      dNds(3)= + 0.125d0*(1.d0+rq)*(1.d0-tq)  
      dNds(4)= + 0.125d0*(1.d0-rq)*(1.d0-tq)
      dNds(5)= - 0.125d0*(1.d0-rq)*(1.d0+tq)    
      dNds(6)= - 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(7)= + 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(8)= + 0.125d0*(1.d0-rq)*(1.d0+tq)    

      dNdt(1)= - 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(2)= - 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(3)= - 0.125d0*(1.d0+rq)*(1.d0+sq)    
      dNdt(4)= - 0.125d0*(1.d0-rq)*(1.d0+sq)    
      dNdt(5)= + 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(6)= + 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(7)= + 0.125d0*(1.d0+rq)*(1.d0+sq)  
      dNdt(8)= + 0.125d0*(1.d0-rq)*(1.d0+sq) 

      jcb=0.d0    
      do k=1,8    
      jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
      jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
      jcb(1,3)=jcb(1,3)+dNdr(k)*z(icon(k,iel))
      jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
      jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      jcb(2,3)=jcb(2,3)+dNds(k)*z(icon(k,iel))
      jcb(3,1)=jcb(3,1)+dNdt(k)*x(icon(k,iel))
      jcb(3,2)=jcb(3,2)+dNdt(k)*y(icon(k,iel))
      jcb(3,3)=jcb(3,3)+dNdt(k)*z(icon(k,iel))
      enddo    
    
      jcob=jcb(1,1)*jcb(2,2)*jcb(3,3) &    
          +jcb(1,2)*jcb(2,3)*jcb(3,1) &    
          +jcb(2,1)*jcb(3,2)*jcb(1,3) &    
          -jcb(1,3)*jcb(2,2)*jcb(3,1) &    
          -jcb(1,2)*jcb(2,1)*jcb(3,3) &    
          -jcb(2,3)*jcb(3,2)*jcb(1,1)    

      jcbi(1,1)=(jcb(2,2)*jcb(3,3)-jcb(2,3)*jcb(3,2))/jcob    
      jcbi(2,1)=(jcb(2,3)*jcb(3,1)-jcb(2,1)*jcb(3,3))/jcob    
      jcbi(3,1)=(jcb(2,1)*jcb(3,2)-jcb(2,2)*jcb(3,1))/jcob  
      jcbi(1,2)=(jcb(1,3)*jcb(3,2)-jcb(1,2)*jcb(3,3))/jcob
      jcbi(2,2)=(jcb(1,1)*jcb(3,3)-jcb(1,3)*jcb(3,1))/jcob    
      jcbi(3,2)=(jcb(1,2)*jcb(3,1)-jcb(1,1)*jcb(3,2))/jcob    
      jcbi(1,3)=(jcb(1,2)*jcb(2,3)-jcb(1,3)*jcb(2,2))/jcob    
      jcbi(2,3)=(jcb(1,3)*jcb(2,1)-jcb(1,1)*jcb(2,3))/jcob    
      jcbi(3,3)=(jcb(1,1)*jcb(2,2)-jcb(1,2)*jcb(2,1))/jcob    

      do k=1,m
         dNdx(k)=jcbi(1,1)*dNdr(k)&    
                +jcbi(1,2)*dNds(k)&    
                +jcbi(1,3)*dNdt(k)    
         dNdy(k)=jcbi(2,1)*dNdr(k)&    
                +jcbi(2,2)*dNds(k)&    
                +jcbi(2,3)*dNdt(k)    
         dNdz(k)=jcbi(3,1)*dNdr(k)&    
                +jcbi(3,2)*dNds(k)&    
                +jcbi(3,3)*dNdt(k)  
      end do

      Bmat=0.d0
      do i=1,m
         i1=ndof*i-2
         i2=ndof*i-1
         i3=ndof*i
         Bmat(1,i1)=dNdx(i)
         Bmat(2,i2)=dNdy(i)
         Bmat(3,i3)=dNdz(i)
         Bmat(4,i1)=dNdy(i) ; Bmat(4,i2)=dNdx(i)
         Bmat(5,i1)=dNdz(i) ; Bmat(5,i3)=dNdx(i)
         Bmat(6,i2)=dNdz(i) ; Bmat(6,i3)=dNdy(i)
      end do

      Ael=Ael + matmul(transpose(Bmat),matmul(penalty*Kmat,Bmat))*weightq*jcob

      !=====================
      !=====[assemble]======
      !=====================

      do k1=1,m
         ik=icon(k1,iel)
         do i1=1,ndof
            ikk=ndof*(k1-1)+i1
            m1=ndof*(ik-1)+i1
            do k2=1,m
               jk=icon(k2,iel)
               do i2=1,ndof
                  jkk=ndof*(k2-1)+i2
                  m2=ndof*(jk-1)+i2
                  A(m1,m2)=A(m1,m2)+Ael(ikk,jkk)
                  C(m1,m2)=.true.
               end do
            end do
            B(m1)=B(m1)+Bel(ikk)
         end do
      end do

end do

!==============================================!
!=====[impose b.c.]============================!
!==============================================!

do i=1,Nfem
    if (bc_fix(i)) then
      Aref=A(i,i)
      do j=1,Nfem
         B(j)=B(j)-A(i,j)*bc_val(i)
         A(i,j)=0.d0
         A(j,i)=0.d0
      enddo
      A(i,i)=Aref
      B(i)=Aref*bc_val(i)
   endif
enddo


open(unit=123,file='OUT/matrix.dat',status='replace')
open(unit=234,file='OUT/rhs.dat',status='replace')
do i=1,Nfem
   do j=1,Nfem
      if (C(i,j)) write(123,'(2i6,f20.10)') i,Nfem-j,A(i,j)
   end do
   write(234,'(i6,f20.10)') i,B(i)
end do
close(123)
close(234)

!==============================================!
!=====[solve system]===========================!
!==============================================!

print *,'A        :',minval(A),maxval(A)
print *,'B        :',minval(B),maxval(B)

job=0
allocate(work(Nfem))
allocate(ipvt(Nfem))
call DGECO (A, Nfem, Nfem, ipvt, rcond, work)
call DGESL (A, Nfem, Nfem, ipvt, B, job)
deallocate(ipvt)
deallocate(work)

do i=1,np
   u(i)=B((i-1)*ndof+1)
   v(i)=B((i-1)*ndof+2)
   w(i)=B((i-1)*ndof+3)
end do

print *,'vel x    :',minval(u),maxval(u)
print *,'vel y    :',minval(v),maxval(v)
print *,'vel z    :',minval(w),maxval(w)

open(unit=123,file='OUT/solution_u.dat',status='replace')
open(unit=234,file='OUT/solution_v.dat',status='replace')
open(unit=345,file='OUT/solution_w.dat',status='replace')
do i=1,np
   write(123,'(4f20.10)') x(i),y(i),z(i),u(i)
   write(234,'(4f20.10)') x(i),y(i),z(i),v(i)
   write(345,'(4f20.10)') x(i),y(i),z(i),v(i)
end do
close(123)
close(234)
close(345)

!==============================================!
!=====[retrieve pressure]======================!
!==============================================!

open(unit=123,file='OUT/solution_p.dat',status='replace')

do iel=1,nel

   rq=0.d0
   sq=0.d0
   tq=0.d0

      N(1)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0-tq)
      N(2)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0-tq)
      N(3)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0-tq)
      N(4)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0-tq)
      N(5)=0.125d0*(1.d0-rq)*(1.d0-sq)*(1.d0+tq)
      N(6)=0.125d0*(1.d0+rq)*(1.d0-sq)*(1.d0+tq)
      N(7)=0.125d0*(1.d0+rq)*(1.d0+sq)*(1.d0+tq)
      N(8)=0.125d0*(1.d0-rq)*(1.d0+sq)*(1.d0+tq)

      dNdr(1)= - 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(2)= + 0.125d0*(1.d0-sq)*(1.d0-tq)    
      dNdr(3)= + 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(4)= - 0.125d0*(1.d0+sq)*(1.d0-tq)    
      dNdr(5)= - 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(6)= + 0.125d0*(1.d0-sq)*(1.d0+tq)    
      dNdr(7)= + 0.125d0*(1.d0+sq)*(1.d0+tq)    
      dNdr(8)= - 0.125d0*(1.d0+sq)*(1.d0+tq)    

      dNds(1)= - 0.125d0*(1.d0-rq)*(1.d0-tq)    
      dNds(2)= - 0.125d0*(1.d0+rq)*(1.d0-tq)    
      dNds(3)= + 0.125d0*(1.d0+rq)*(1.d0-tq)  
      dNds(4)= + 0.125d0*(1.d0-rq)*(1.d0-tq)
      dNds(5)= - 0.125d0*(1.d0-rq)*(1.d0+tq)    
      dNds(6)= - 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(7)= + 0.125d0*(1.d0+rq)*(1.d0+tq)    
      dNds(8)= + 0.125d0*(1.d0-rq)*(1.d0+tq)    

      dNdt(1)= - 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(2)= - 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(3)= - 0.125d0*(1.d0+rq)*(1.d0+sq)    
      dNdt(4)= - 0.125d0*(1.d0-rq)*(1.d0+sq)    
      dNdt(5)= + 0.125d0*(1.d0-rq)*(1.d0-sq)    
      dNdt(6)= + 0.125d0*(1.d0+rq)*(1.d0-sq)    
      dNdt(7)= + 0.125d0*(1.d0+rq)*(1.d0+sq)  
      dNdt(8)= + 0.125d0*(1.d0-rq)*(1.d0+sq) 

      jcb=0.d0    
      do k=1,8    
      jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
      jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
      jcb(1,3)=jcb(1,3)+dNdr(k)*z(icon(k,iel))
      jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
      jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      jcb(2,3)=jcb(2,3)+dNds(k)*z(icon(k,iel))
      jcb(3,1)=jcb(3,1)+dNdt(k)*x(icon(k,iel))
      jcb(3,2)=jcb(3,2)+dNdt(k)*y(icon(k,iel))
      jcb(3,3)=jcb(3,3)+dNdt(k)*z(icon(k,iel))
      enddo    
    
      jcob=jcb(1,1)*jcb(2,2)*jcb(3,3) &    
          +jcb(1,2)*jcb(2,3)*jcb(3,1) &    
          +jcb(2,1)*jcb(3,2)*jcb(1,3) &    
          -jcb(1,3)*jcb(2,2)*jcb(3,1) &    
          -jcb(1,2)*jcb(2,1)*jcb(3,3) &    
          -jcb(2,3)*jcb(3,2)*jcb(1,1)    

      jcbi(1,1)=(jcb(2,2)*jcb(3,3)-jcb(2,3)*jcb(3,2))/jcob    
      jcbi(2,1)=(jcb(2,3)*jcb(3,1)-jcb(2,1)*jcb(3,3))/jcob    
      jcbi(3,1)=(jcb(2,1)*jcb(3,2)-jcb(2,2)*jcb(3,1))/jcob  
      jcbi(1,2)=(jcb(1,3)*jcb(3,2)-jcb(1,2)*jcb(3,3))/jcob
      jcbi(2,2)=(jcb(1,1)*jcb(3,3)-jcb(1,3)*jcb(3,1))/jcob    
      jcbi(3,2)=(jcb(1,2)*jcb(3,1)-jcb(1,1)*jcb(3,2))/jcob    
      jcbi(1,3)=(jcb(1,2)*jcb(2,3)-jcb(1,3)*jcb(2,2))/jcob    
      jcbi(2,3)=(jcb(1,3)*jcb(2,1)-jcb(1,1)*jcb(2,3))/jcob    
      jcbi(3,3)=(jcb(1,1)*jcb(2,2)-jcb(1,2)*jcb(2,1))/jcob    

      do k=1,m
         dNdx(k)=jcbi(1,1)*dNdr(k)&    
                +jcbi(1,2)*dNds(k)&    
                +jcbi(1,3)*dNdt(k)    
         dNdy(k)=jcbi(2,1)*dNdr(k)&    
                +jcbi(2,2)*dNds(k)&    
                +jcbi(2,3)*dNdt(k)    
         dNdz(k)=jcbi(3,1)*dNdr(k)&    
                +jcbi(3,2)*dNds(k)&    
                +jcbi(3,3)*dNdt(k)  
      end do

   xq=0.d0
   yq=0.d0
   zq=0.d0
   exxq=0.d0
   eyyq=0.d0
   ezzq=0.d0
   do k=1,m
      xq=xq+N(k)*x(icon(k,iel))
      yq=yq+N(k)*y(icon(k,iel))
      zq=zq+N(k)*z(icon(k,iel))
      exxq=exxq+ dNdx(k)*u(icon(k,iel))
      eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
      ezzq=ezzq+ dNdz(k)*w(icon(k,iel))
   end do

   press(iel)=-penalty*(exxq+eyyq+ezzq)
  
   rho(iel)=rhofct(xq,yq,zq)
 
   write(123,*) xq,yq,zq,press(iel)

end do

close(123)

print *,'pressure :',minval(press),maxval(press)

call output_for_paraview (np,nel,x,y,z,u,v,w,press,icon,rho)

end program

!==============================================!
!==============================================!
!==============================================!

function rhofct (x,y,z)
implicit none
real(8) rhofct,x,y,z

if (abs(x-0.5)<2./16. .and. & 
    abs(y-0.5)<2./16. .and. & 
    abs(z-0.5)<2./16. ) then  
   rhofct=-2.
else
   rhofct=-1
end if

end function





