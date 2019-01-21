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
integer, parameter :: m=4                      ! number of nodes which constitute an element
integer, parameter :: ndof=2                   ! number of dofs per node
integer nnx                                    ! number of grid points in the x direction
integer nny                                    ! number of grid points in the y direction
integer np                                     ! number of grid points
integer nelx                                   ! number of elements in the x direction
integer nely                                   ! number of elements in the y direction
integer nel                                    ! number of elements
integer Nfem                                   ! size of the FEM matrix 
integer, dimension(:,:), allocatable :: icon   ! connectivity array
integer, dimension(:), allocatable :: ipvt     ! work array needed by the solver 
                                               !
integer i1,i2,i,j,k,iel,counter,iq,jq          !
integer ik,jk,ikk,jkk,m1,m2,k1,k2,job          !
                                               !  
real(8) Lx,Ly                                  ! size of the numerical domain
real(8) gx,gy                                  ! gravity acceleration
real(8) penalty                                ! penalty parameter lambda
real(8), dimension(:),   allocatable :: x,y    ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v    ! node velocity arrays
real(8), dimension(:),   allocatable :: uth,vth
real(8), dimension(:),   allocatable :: press  ! pressure 
real(8), dimension(:),   allocatable :: B      ! right hand side
real(8), dimension(:,:), allocatable :: A      ! FEM matrix
real(8), dimension(:),   allocatable :: work   ! work array needed by the solver
real(8), dimension(:),   allocatable :: bc_val ! array containing bc values
                                               !
real(8), external :: b1,b2,viscosity    
real(8) rq,sq,wq                               ! local coordinate and weight of qpoint
real(8) xq,yq                                  ! global coordinate of qpoint
real(8) uq,vq                                  ! velocity at qpoint
real(8) exxq,eyyq,exyq                         ! strain-rate components at qpoint  
real(8) Ael(m*ndof,m*ndof)                     ! elemental FEM matrix
real(8) Bel(m*ndof)                            ! elemental right hand side
real(8) N(m),dNdx(m),dNdy(m),dNdr(m),dNds(m)   ! shape fcts and derivatives
real(8) jcob                                   ! determinant of jacobian matrix
real(8) jcb(2,2)                               ! jacobian matrix
real(8) jcbi(2,2)                              ! inverse of jacobian matrix
real(8) Bmat(3,ndof*m)                         ! B matrix
real(8), dimension(3,3) :: Kmat                ! K matrix 
real(8), dimension(3,3) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) eps                                    !
real(8) rcond                                  !
                                               !
logical, dimension(:), allocatable :: bc_fix   ! prescribed b.c. array

integer nmarker,icx,icy,ip
real(8) xi,yi,xmin,xmax,ymin,ymax,r,s
real(8) exxqth,eyyqth,uqth,vqth
                                               !
!==============================================!
!=====[setup]==================================!
!==============================================!

Lx=1.d0
Ly=1.d0

nnx=33
nny=33

np=nnx*nny

nelx=nnx-1
nely=nny-1

nel=nelx*nely

penalty=1.d11

Nfem=np*ndof

eps=1.d-10

Kmat(1,1)=1.d0 ; Kmat(1,2)=1.d0 ; Kmat(1,3)=0.d0  
Kmat(2,1)=1.d0 ; Kmat(2,2)=1.d0 ; Kmat(2,3)=0.d0  
Kmat(3,1)=0.d0 ; Kmat(3,2)=0.d0 ; Kmat(3,3)=0.d0  

Cmat(1,1)=2.d0 ; Cmat(1,2)=0.d0 ; Cmat(1,3)=0.d0  
Cmat(2,1)=0.d0 ; Cmat(2,2)=2.d0 ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0  


!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(u(np))
allocate(v(np))
allocate(icon(m,nel))
allocate(A(Nfem,Nfem))
allocate(B(Nfem))
allocate(bc_fix(Nfem))
allocate(bc_val(Nfem))
allocate(press(nel))
allocate(uth(np))
allocate(vth(np))

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do j=0,nely
   do i=0,nelx
      counter=counter+1
      x(counter)=dble(i)*Lx/dble(nelx)
      y(counter)=dble(j)*Ly/dble(nely)
   end do
end do

open(unit=123,file='OUT/gridnodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,np
   write(123,'(2f10.5,i8)') x(i),y(i),i
end do
close(123)

!==============================================!
!===[connectivity]=============================!
!==============================================!

counter=0
do j=1,nely
   do i=1,nelx
      counter=counter+1
      icon(1,counter)=i+(j-1)*(nelx+1)
      icon(2,counter)=i+1+(j-1)*(nelx+1)
      icon(3,counter)=i+1+j*(nelx+1)
      icon(4,counter)=i+j*(nelx+1)
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
end do
close(123)

!==============================================!
!=====[define bc]==============================!
!==============================================!

bc_fix=.false.

do i=1,np
   if (x(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
   endif
   if (x(i).gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
   endif
   if (y(i).lt.eps) then
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).gt.(Ly-eps) ) then
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
end do

open(unit=123,file='OUT/bc_u.dat',status='replace')
open(unit=234,file='OUT/bc_v.dat',status='replace')
do i=1,np
   if (bc_fix((i-1)*ndof+1)) write(123,'(3f20.10)') x(i),y(i),bc_val((i-1)*ndof+1) 
   if (bc_fix((i-1)*ndof+2)) write(234,'(3f20.10)') x(i),y(i),bc_val((i-1)*ndof+2) 
end do
close(123)
close(234)

!==============================================!
!=====[build FE matrix]========================!
!==============================================!

A=0.d0
B=0.d0

do iel=1,nel

   Ael=0.d0
   Bel=0.d0

   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

      jcbi(1,1)=    jcb(2,2) /jcob
      jcbi(1,2)=  - jcb(1,2) /jcob
      jcbi(2,1)=  - jcb(2,1) /jcob
      jcbi(2,2)=    jcb(1,1) /jcob

      xq=0.d0
      yq=0.d0
      uq=0.d0
      vq=0.d0
      exxq=0.d0
      eyyq=0.d0
      exyq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
         exxq=exxq+ dNdx(k)*u(icon(k,iel))
         eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
         exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
                  + dNdy(k)*u(icon(k,iel)) *0.5d0
      end do

      !write(999,*) xq,yq,uq,vq,exxq,eyyq,exyq

      do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat(1,i1)=dNdx(i) ; Bmat(1,i2)=0.d0
         Bmat(2,i1)=0.d0    ; Bmat(2,i2)=dNdy(i)
         Bmat(3,i1)=dNdy(i) ; Bmat(3,i2)=dNdx(i)
      end do

      Ael=Ael + matmul(transpose(Bmat),matmul(viscosity(xq,yq)*Cmat,Bmat))*wq*jcob

      do i=1,m
      i1=2*i-1
      i2=2*i
      Bel(i1)=Bel(i1)+N(i)*jcob*wq*b1(xq,yq)
      Bel(i2)=Bel(i2)+N(i)*jcob*wq*b2(xq,yq)
      end do

   end do
   end do

   ! 1 point integration

      rq=0.d0
      sq=0.d0
      wq=2.d0*2.d0

      N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
      N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
      N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
      N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

      dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
      dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
      dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
      dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

      jcb=0.d0
      do k=1,m
         jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
         jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
         jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
         jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
      enddo

      jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

      jcbi(1,1)=    jcb(2,2) /jcob
      jcbi(1,2)=  - jcb(1,2) /jcob
      jcbi(2,1)=  - jcb(2,1) /jcob
      jcbi(2,2)=    jcb(1,1) /jcob

      do k=1,m
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
      end do

      do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat(1,i1)=dNdx(i) ; Bmat(1,i2)=0.d0
         Bmat(2,i1)=0.d0    ; Bmat(2,i2)=dNdy(i)
         Bmat(3,i1)=dNdy(i) ; Bmat(3,i2)=dNdx(i)
      end do

      Ael=Ael + matmul(transpose(Bmat),matmul(penalty*Kmat,Bmat))*wq*jcob

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

!==============================================!
!=====[solve system]===========================!
!==============================================!

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
end do

print *,minval(u),maxval(u)
print *,minval(v),maxval(v)

open(unit=123,file='OUT/solution_u.dat',status='replace')
open(unit=234,file='OUT/solution_v.dat',status='replace')
open(unit=345,file='OUT/solution_vel.dat',status='replace')
do i=1,np
   write(123,'(5f20.10)') x(i),y(i),u(i)
   write(234,'(5f20.10)') x(i),y(i),v(i)
   write(345,'(5f20.10)') x(i),y(i),u(i),v(i)
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
      
   N(1)=0.25d0*(1.d0-rq)*(1.d0-sq)
   N(2)=0.25d0*(1.d0+rq)*(1.d0-sq)
   N(3)=0.25d0*(1.d0+rq)*(1.d0+sq)
   N(4)=0.25d0*(1.d0-rq)*(1.d0+sq)

   dNdr(1)= - 0.25d0*(1.d0-sq)   ;   dNds(1)= - 0.25d0*(1.d0-rq)
   dNdr(2)= + 0.25d0*(1.d0-sq)   ;   dNds(2)= - 0.25d0*(1.d0+rq)
   dNdr(3)= + 0.25d0*(1.d0+sq)   ;   dNds(3)= + 0.25d0*(1.d0+rq)
   dNdr(4)= - 0.25d0*(1.d0+sq)   ;   dNds(4)= + 0.25d0*(1.d0-rq)

   jcb=0.d0
   do k=1,m
      jcb(1,1)=jcb(1,1)+dNdr(k)*x(icon(k,iel))
      jcb(1,2)=jcb(1,2)+dNdr(k)*y(icon(k,iel))
      jcb(2,1)=jcb(2,1)+dNds(k)*x(icon(k,iel))
      jcb(2,2)=jcb(2,2)+dNds(k)*y(icon(k,iel))
   enddo

   jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)

   jcbi(1,1)=    jcb(2,2) /jcob
   jcbi(1,2)=  - jcb(1,2) /jcob
   jcbi(2,1)=  - jcb(2,1) /jcob
   jcbi(2,2)=    jcb(1,1) /jcob

   do k=1,m
      dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
      dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
   end do

   xq=0.d0
   yq=0.d0
   exxq=0.d0
   eyyq=0.d0
   do k=1,m
      xq=xq+N(k)*x(icon(k,iel))
      yq=yq+N(k)*y(icon(k,iel))
      exxq=exxq+ dNdx(k)*u(icon(k,iel))
      eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
   end do

   press(iel)=-penalty*(exxq+eyyq)
   
   write(123,*) xq,yq,press(iel)

end do

close(123)

!=========================================================

do ip=1,np
   call analytical_solution(x(ip),y(ip),uth(ip),vth(ip))
end do

open(unit=345,file='OUT/solution_vel_th.dat',status='replace')
do i=1,np
   write(345,'(5f20.10)') x(i),y(i),uth(i),vth(i)
end do
close(345)

!=========================================================

nmarker=100000

do i=1,nmarker

   call random_number(xi)
   call random_number(yi)

   icx=xi/Lx*nelx+1
   icy=yi/Ly*nely+1
   iel=(icy-1)*nelx+icx

   xmin=x(icon(1,iel))
   xmax=x(icon(3,iel))
   r=((xi-xmin)/(xmax-xmin)-0.5d0)*2.d0

   ymin=y(icon(1,iel))
   ymax=y(icon(3,iel))
   s=((yi-ymin)/(ymax-ymin)-0.5d0)*2.d0

   write(789,*) xi,yi,icx,icy,iel,r,s

   N(1)=0.25d0*(1.d0-r)*(1.d0-s)
   N(2)=0.25d0*(1.d0+r)*(1.d0-s)
   N(3)=0.25d0*(1.d0+r)*(1.d0+s)
   N(4)=0.25d0*(1.d0-r)*(1.d0+s)

   dNdr(1)= - 0.25d0*(1.d0-s)   ;   dNds(1)= - 0.25d0*(1.d0-r)
   dNdr(2)= + 0.25d0*(1.d0-s)   ;   dNds(2)= - 0.25d0*(1.d0+r)
   dNdr(3)= + 0.25d0*(1.d0+s)   ;   dNds(3)= + 0.25d0*(1.d0+r)
   dNdr(4)= - 0.25d0*(1.d0+s)   ;   dNds(4)= + 0.25d0*(1.d0-r)

   uq=0.d0
   vq=0.d0
   uqth=0.d0
   vqth=0.d0
   exxq=0.d0
   eyyq=0.d0
   exxqth=0.d0
   eyyqth=0.d0
   do k=1,4
      uq=uq+N(k)*u(icon(k,iel))
      vq=vq+N(k)*v(icon(k,iel))
      uqth=uqth+N(k)*uth(icon(k,iel))
      vqth=vqth+N(k)*vth(icon(k,iel))
      dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
      dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
      exxq=exxq+ dNdx(k)*u(icon(k,iel))
      eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
      exxqth=exxqth+ dNdx(k)*uth(icon(k,iel))
      eyyqth=eyyqth+ dNdy(k)*vth(icon(k,iel))
   end do

   write(1000,*) xi,yi,exxq+eyyq,uq,vq
   write(1001,*) xi,yi,exxqth+eyyqth,uqth,vqth
   write(1002,*) xi,yi,(uq-uqth),(vq-vqth)

end do


end program

!==============================================!
!==============================================!
!==============================================!

function b1 (x,y)
real(8) b1,x,y
b1=0
end function

function b2 (x,y)
real(8) b2,x,y
real(8),parameter :: pi=3.141592653589793238462643383279
b2=sin(pi*y)*cos(pi*x) 
end function

function viscosity (x,y)
real(8) viscosity,x,y
if (x<0.5d0) then
viscosity=1.d0
else
viscosity=1.d6
end if
end function


