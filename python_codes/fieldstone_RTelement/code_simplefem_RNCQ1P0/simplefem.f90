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
integer np,np2                                 ! number of grid points
integer nelx                                   ! number of elements in the x direction
integer nely                                   ! number of elements in the y direction
integer nel                                    ! number of elements
integer Nfem                                   ! size of the FEM matrix 
integer formulation                            ! 
integer, dimension(:,:), allocatable :: icon   ! connectivity array for FEM grid
integer, dimension(:,:), allocatable :: icon2  ! connectivity array for visu grid
integer, dimension(:), allocatable :: ipvt     ! work array needed by the solver 
integer, dimension(:), allocatable :: counter2 ! work array needed by the solver 
                                               !
integer i1,i2,i,j,k,iel,counter,iq,jq,ibench   !
integer ik,jk,ikk,jkk,m1,m2,k1,k2,job          !
                                               !  
real(8) Lx,Ly                                  ! size of the numerical domain
real(8) gx,gy                                  ! gravity vector
real(8) viscosity                              ! dynamic viscosity $\mu$ of the material
real(8) density                                ! mass density $\rho$ of the material
real(8) penalty                                ! penalty parameter lambda
real(8), dimension(:),   allocatable :: x,y    ! node coordinates arrays of FEM grid
real(8), dimension(:),   allocatable :: x2,y2  ! node coordinates arrays of visu grid
real(8), dimension(:),   allocatable :: u,v    ! node velocity arrays on FEM grid
real(8), dimension(:),   allocatable :: u2,v2  ! node velocity arrays on visu grid
real(8), dimension(:),   allocatable :: press  ! pressure 
real(8), dimension(:),   allocatable :: B      ! right hand side
real(8), dimension(:,:), allocatable :: A      ! FEM matrix
real(8), dimension(:),   allocatable :: work   ! work array needed by the solver
real(8), dimension(:),   allocatable :: bc_val ! array containing bc values
                                               !
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
real(8) Bmat1(3,ndof*m)                        ! B matrix
real(8) Bmat2(4,ndof*m)                        ! B matrix
real(8), dimension(3,3) :: Kmat                ! K matrix 
real(8), dimension(3,3) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) eps                                    !
real(8) rcond                                  !
real(8) sx,sy                                  ! size of element
real(8) uuu,vvv
                                               !
logical, dimension(:), allocatable :: bc_fix   ! prescribed b.c. array
logical noslip
                                               !
!==============================================!
!=====[setup]==================================!
!==============================================!

Lx=1.d0
Ly=1.d0

nelx=32
nely=nelx

np=(nely+1)*nelx + nely*(nelx+1) ! number of points

nel=nelx*nely ! number of elements

penalty=1.d7 ! lambda

viscosity=1.d0

Nfem=np*ndof ! total number of dofs

eps=1.d-10

Kmat(1,1)=1.d0 ; Kmat(1,2)=1.d0 ; Kmat(1,3)=0.d0  ! penalty matrix
Kmat(2,1)=1.d0 ; Kmat(2,2)=1.d0 ; Kmat(2,3)=0.d0  
Kmat(3,1)=0.d0 ; Kmat(3,2)=0.d0 ; Kmat(3,3)=0.d0  

Cmat(1,1)=2.d0 ; Cmat(1,2)=0.d0 ; Cmat(1,3)=0.d0  ! viscous matrix 
Cmat(2,1)=0.d0 ; Cmat(2,2)=2.d0 ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0  

!Cmat(1,1)=4.d0/3 ; Cmat(1,2)=-2.d0/3. ; Cmat(1,3)=0.d0  
!Cmat(2,1)=-2./3 ; Cmat(2,2)=4.d0/3. ; Cmat(2,3)=0.d0  
!Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0  

!1: symm gradient
!2: gradient

formulation=2

!1: aquarium
!2: stokes sphere
!3: polynomial
!4: stokes sphere, reduced density

ibench=1

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

!==============================================!
!===[grid points setup]========================!
!==============================================!

sx=Lx/nelx
sy=Ly/nely

counter=0
do j=1,nely
   !bottom line
   do i=1,nelx
      counter=counter+1
      x(counter)=(i-0.5)*sx
      y(counter)=(j-1)*sy
   end do
   !middle line
   do i=0,nelx
      counter=counter+1
      x(counter)=(i)*sx
      y(counter)=(j-0.5)*sy
   end do
end do
!top line
do i=1,nelx
   counter=counter+1
   x(counter)=(i-0.5)*sx
   y(counter)=Ly
end do

!open(unit=123,file='OUT/gridnodes.dat',status='replace')
!write(123,'(a)') '#     xpos      ypos    node '
!do i=1,np
!   write(123,'(2f10.5,i8)') x(i),y(i),i
!end do
!close(123)

!==============================================!
!===[connectivity]=============================!
!==============================================!
!  3
! / \
!4   2
! \ /
!  1

iel=0
do j=1,nely
   do i=1,nelx
      iel=iel+1
      icon(1,iel)=(j-1)*(2*nelx+1)+i
      icon(2,iel)=icon(1,iel)+nelx+1
      icon(3,iel)=icon(1,iel)+2*nelx+1
      icon(4,iel)=icon(1,iel)+nelx
   end do
end do

!==============================================!
!=====[define bc]==============================!
!==============================================!

if (ibench==1) noslip=.false.
if (ibench==2) noslip=.false.
if (ibench==3) noslip=.true.
if (ibench==4) noslip=.false.

bc_fix=.false.
do i=1,np
   if (x(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      if (noslip) then
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      end if
   endif
   if (x(i).gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      if (noslip) then
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
      end if
   endif
   if (y(i).lt.eps) then
      if (noslip) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      end if
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).gt.(Ly-eps) ) then
      if (noslip) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      end if
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
end do

!==============================================!
!=====[build FE matrix]========================!
!==============================================!
! global coordinates are x,y
! local coordinates are r,s

A=0.d0
B=0.d0

do iel=1,nel

   Ael=0.d0
   Bel=0.d0

   ! looping over 2x2 quad points

   do iq=-1,1,2
   do jq=-1,1,2

      rq=iq/sqrt(3.d0)
      sq=jq/sqrt(3.d0)
      wq=1.d0*1.d0

      N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
      N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
      N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
      N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)

      dNdr(1)=0.5d0*(-rq)      ; dNds(1)=0.5d0*(-1.d0+sq)
      dNdr(2)=0.5d0*(1.d0+rq)  ; dNds(2)=0.5d0*(-sq)
      dNdr(3)=0.5d0*(-rq)      ; dNds(3)=0.5d0*(1.d0+sq)
      dNdr(4)=0.5d0*(-1.d0+rq) ; dNds(4)=0.5d0*(-sq)

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

      if (formulation==1) then
         do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat1(1,i1)=dNdx(i) ; Bmat1(1,i2)=0.d0
         Bmat1(2,i1)=0.d0    ; Bmat1(2,i2)=dNdy(i)
         Bmat1(3,i1)=dNdy(i) ; Bmat1(3,i2)=dNdx(i)
         end do
         Ael=Ael + matmul(transpose(Bmat1),matmul(viscosity*Cmat,Bmat1))*wq*jcob
      else
         do i=1,m
         i1=2*i-1
         i2=2*i
         Bmat2(1,i1)=dNdx(i) ;  Bmat2(1,i2)=0.d0
         Bmat2(2,i1)=dNdy(i) ;  Bmat2(2,i2)=0.d0
         Bmat2(3,i1)=0       ;  Bmat2(3,i2)=dNdx(i)
         Bmat2(4,i1)=0       ;  Bmat2(4,i2)=dNdy(i)
         end do
         Ael=Ael + viscosity*matmul(transpose(Bmat2),Bmat2)*wq*jcob
      end if

      density=rho(xq,yq,ibench)
      gx=gravx(xq,yq,ibench)
      gy=gravy(xq,yq,ibench)

      do i=1,m
      i1=2*i-1
      i2=2*i
      Bel(i1)=Bel(i1)+N(i)*jcob*wq*density*gx
      Bel(i2)=Bel(i2)+N(i)*jcob*wq*density*gy
      end do

   end do
   end do

   ! 1 point integration

      rq=0.d0
      sq=0.d0
      wq=2.d0*2.d0

      dNdr(1)=0.5d0*(-rq)      ; dNds(1)=0.5d0*(-1.d0+sq)
      dNdr(2)=0.5d0*(1.d0+rq)  ; dNds(2)=0.5d0*(-sq)
      dNdr(3)=0.5d0*(-rq)      ; dNds(3)=0.5d0*(1.d0+sq)
      dNdr(4)=0.5d0*(-1.d0+rq) ; dNds(4)=0.5d0*(-sq)

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
         Bmat1(1,i1)=dNdx(i) ; Bmat1(1,i2)=0.d0
         Bmat1(2,i1)=0.d0    ; Bmat1(2,i2)=dNdy(i)
         Bmat1(3,i1)=dNdy(i) ; Bmat1(3,i2)=dNdx(i)
      end do
      Ael=Ael + matmul(transpose(Bmat1),matmul(penalty*Kmat,Bmat1))*wq*jcob

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

print *,'u (m/M): ',minval(u),maxval(u)
print *,'v (m/M): ',minval(v),maxval(v)

open(unit=345,file='OUT/solution_vel.dat',status='replace')
do i=1,np
   write(345,'(6es20.10)') x(i),y(i),u(i),v(i),uth(x(i),y(i)),vth(x(i),y(i))
end do
close(345)

!==============================================!
!=====[retrieve pressure]======================!
!==============================================!
! pressure is computed at middle of element 
! p=-lambda div v

open(unit=123,file='OUT/solution_p.dat',status='replace')

do iel=1,nel

   rq=0.d0
   sq=0.d0

   N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
   N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
   N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
   N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)

   dNdr(1)=0.5d0*(-rq)      ; dNds(1)=0.5d0*(-1.d0+sq)
   dNdr(2)=0.5d0*(1.d0+rq)  ; dNds(2)=0.5d0*(-sq)
   dNdr(3)=0.5d0*(-rq)      ; dNds(3)=0.5d0*(1.d0+sq)
   dNdr(4)=0.5d0*(-1.d0+rq) ; dNds(4)=0.5d0*(-sq)

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
   
   write(123,*) xq,yq,press(iel),pth(xq,yq)

end do

close(123)

call output_for_paraview (np,nel,x,y,u,v,press,icon,'1')

!==============================================!
! build visualisation grid
!==============================================!

np2=(nelx+1)*(nely+1)

allocate(x2(np2))
allocate(y2(np2))
allocate(u2(np2))
allocate(v2(np2))
allocate(icon2(m,nel))
allocate(counter2(np2))

counter=0
do j=0,nely
   do i=0,nelx
      counter=counter+1
      x2(counter)=dble(i)*Lx/dble(nelx)
      y2(counter)=dble(j)*Ly/dble(nely)
   end do
end do

counter=0
do j=1,nely
   do i=1,nelx
      counter=counter+1
      icon2(1,counter)=i+(j-1)*(nelx+1)
      icon2(2,counter)=i+1+(j-1)*(nelx+1)
      icon2(3,counter)=i+1+j*(nelx+1)
      icon2(4,counter)=i+j*(nelx+1)
   end do
end do

!project velocity back onto visu grid

counter2=0
u2=0
v2=0

do iel=1,nel
   !----------------------------------  
   rq=-1 ; sq=-1
   N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
   N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
   N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
   N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)
   uuu=sum(u(icon(1:4,iel))*N(1:4))
   vvv=sum(v(icon(1:4,iel))*N(1:4))
   u2(icon2(1,iel))=u2(icon2(1,iel))+uuu
   v2(icon2(1,iel))=v2(icon2(1,iel))+vvv
   counter2(icon2(1,iel))=counter2(icon2(1,iel))+1

   !----------------------------------  
   rq=+1 ; sq=-1
   N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
   N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
   N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
   N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)
   uuu=sum(u(icon(1:4,iel))*N(1:4))
   vvv=sum(v(icon(1:4,iel))*N(1:4))
   u2(icon2(2,iel))=u2(icon2(2,iel))+uuu
   v2(icon2(2,iel))=v2(icon2(2,iel))+vvv
   counter2(icon2(2,iel))=counter2(icon2(2,iel))+1

   !----------------------------------  
   rq=+1 ; sq=+1
   N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
   N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
   N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
   N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)
   uuu=sum(u(icon(1:4,iel))*N(1:4))
   vvv=sum(v(icon(1:4,iel))*N(1:4))
   u2(icon2(3,iel))=u2(icon2(3,iel))+uuu
   v2(icon2(3,iel))=v2(icon2(3,iel))+vvv
   counter2(icon2(3,iel))=counter2(icon2(3,iel))+1

   !----------------------------------  
   rq=-1 ; sq=+1
   N(1)=0.25d0*(1.d0-rq**2-2*sq+sq**2)
   N(2)=0.25d0*(1.d0+2*rq+rq**2-sq**2)
   N(3)=0.25d0*(1.d0-rq**2+2*sq+sq**2)
   N(4)=0.25d0*(1.d0-2*rq+rq**2-sq**2)
   uuu=sum(u(icon(1:4,iel))*N(1:4))
   vvv=sum(v(icon(1:4,iel))*N(1:4))
   u2(icon2(4,iel))=u2(icon2(4,iel))+uuu
   v2(icon2(4,iel))=v2(icon2(4,iel))+vvv
   counter2(icon2(4,iel))=counter2(icon2(4,iel))+1

   !----------------------------------  
end do

u2=u2/counter2
v2=v2/counter2

call output_for_paraview (np2,nel,x2,y2,u2,v2,press,icon2,'2')

contains


!==============================================!
!==============================================!
!==============================================!

function uth (x,y)
real(8) uth,x,y
uth = x**2 * (1.d0-x)**2 * (2.d0*y - 6.d0*y**2 + 4*y**3)
end function

function vth (x,y)
real(8) vth,x,y
vth = -y**2 * (1.d0-y)**2 * (2.d0*x - 6.d0*x**2 + 4*x**3)
end function

function pth (x,y)
real(8) pth,x,y
pth = x*(1.d0-x)
end function

function gravx (x,y,ibench)
real(8) gravx,x,y
integer ibench
if (ibench==1) then
   gravx=0
elseif (ibench==2 .or. ibench==4) then
   gravx=0
elseif (ibench==3) then
   gravx = ( (12.d0-24.d0*y)*x**4 + (-24.d0+48.d0*y)*x**3 + (-48.d0*y+72.d0*y**2-48.d0*y**3+12.d0)*x**2 &
         + (-2.d0+24.d0*y-72.d0*y**2+48.d0*y**3)*x + 1.d0-4.d0*y+12.d0*y**2-8.d0*y**3 )
end if
end function

function gravy (x,y,ibench)
real(8) gravy,x,y
integer ibench
if (ibench==1) then
   gravy=-1
elseif (ibench==2 .or. ibench==4) then
   gravy=-1
elseif (ibench==3) then
   gravy= ( (8.d0-48.d0*y+48.d0*y**2)*x**3 + (-12.d0+72.d0*y-72*y**2)*x**2 + &
          (4.d0-24.d0*y+48.d0*y**2-48.d0*y**3+24.d0*y**4)*x - 12.d0*y**2 + 24.d0*y**3 -12.d0*y**4)
end if
end function


function rho(xq,yq,ibench)
implicit none
real(8) xq,yq,rho
integer ibench

if (ibench==1) then
   rho=1
elseif (ibench==2) then
   if ((xq-0.5)**2+(yq-0.5)**2 < 0.2**2) then
   rho=1.+1.
   else
   rho=1
   end if
elseif (ibench==3) then
   rho=1
elseif (ibench==4) then
   if ((xq-0.5)**2+(yq-0.5)**2 < 0.2**2) then
   rho=+0.01
   else
   rho=0
   end if
else
   stop 'rho: ibench pb'
end if
end function

end program
