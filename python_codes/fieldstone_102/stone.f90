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
integer, parameter :: m=4                         ! number of nodes which constitute an element
integer, parameter :: ndof=2                      ! number of dofs per node
integer nnx                                       ! number of grid points in the x direction
integer nny                                       ! number of grid points in the y direction
integer np                                        ! number of grid points
integer np_new                                    ! number of grid pts after ref
integer np_real                                   ! number of grid pts after ref
integer nelx                                      ! number of elements in the x direction
integer nely                                      ! number of elements in the y direction
integer nel                                       ! number of elements
integer nel_new                                   ! number of elts after refinement
integer nel_real                                  ! number of elts after refinement, double removed
integer Nfem                                      ! size of the FEM matrix 
integer, dimension(:,:), allocatable :: icon      ! connectivity array
integer, dimension(:,:), allocatable :: icon_new  ! connectivity array
integer, dimension(:,:), allocatable :: icon_real ! connectivity array
integer, dimension(:), allocatable :: ipvt        ! work array needed by the solver 
integer, dimension(:), allocatable :: compact     ! work array needed to renumber nodes
integer, dimension(:), allocatable :: pointto     ! work array needed to renumber nodes
integer, dimension(:), allocatable :: crtype      ! type of conformal ref of element

                                                  !
integer counter_e,counter_n,ip,jp                 !
integer i1,i2,i,j,k,iel,counter,iq,jq             !
integer ik,jk,ikk,jkk,m1,m2,k1,k2,job             !
integer test 
                                                  !  
real(8) Lx,Ly                                     ! size of the numerical domain
real(8) viscosity                                 ! dynamic viscosity $\mu$ of the material
real(8) density                                   ! mass density $\rho$ of the material
real(8) gx,gy                                     ! gravity acceleration
real(8) penalty                                   ! penalty parameter lambda
real(8), dimension(:),   allocatable :: x,y       ! node coordinates arrays
real(8), dimension(:),   allocatable :: xc,yc       ! node coordinates arrays
real(8), dimension(:),   allocatable :: xnew      ! node coordinates arrays
real(8), dimension(:),   allocatable :: ynew      ! node coordinates arrays
real(8), dimension(:),   allocatable :: xreal     ! node coordinates arrays
real(8), dimension(:),   allocatable :: yreal     ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v       ! node velocity arrays
real(8), dimension(:),   allocatable :: press     ! pressure 
real(8), dimension(:),   allocatable :: B         ! right hand side
real(8), dimension(:,:), allocatable :: A         ! FEM matrix
real(8), dimension(:),   allocatable :: work      ! work array needed by the solver
real(8), dimension(:),   allocatable :: bc_val    ! array containing bc values
logical, dimension(:),   allocatable :: crnode    ! nodes belonging to flagged elements
logical, dimension(:),   allocatable :: flag      ! elements flagged for refinement
logical, dimension(:),   allocatable :: doubble   ! array needed for renumbering
                                                  !
real(8), external :: b1,b2,uth,vth,pth            ! body force and analytical solution
real(8) rq,sq,wq                                  ! local coordinate and weight of qpoint
real(8) xq,yq                                     ! global coordinate of qpoint
real(8) uq,vq                                     ! velocity at qpoint
real(8) exxq,eyyq,exyq                            ! strain-rate components at qpoint  
real(8) Ael(m*ndof,m*ndof)                        ! elemental FEM matrix
real(8) Bel(m*ndof)                               ! elemental right hand side
real(8) N(m),dNdx(m),dNdy(m),dNdr(m),dNds(m)      ! shape fcts and derivatives
real(8) jcob                                      ! determinant of jacobian matrix
real(8) jcb(2,2)                                  ! jacobian matrix
real(8) jcbi(2,2)                                 ! inverse of jacobian matrix
real(8) Bmat(3,ndof*m)                            ! B matrix
real(8), dimension(3,3) :: Kmat                   ! K matrix 
real(8), dimension(3,3) :: Cmat                   ! C matrix
real(8) Aref                                      !
real(8) eps                                       !
real(8) rcond                                     !
real(8) distance,xip,yip,dx,dy                    !
                                                  !
logical, dimension(:), allocatable :: bc_fix      ! prescribed b.c. array
logical, dimension(:,:), allocatable :: C         ! non-zero terms in FEM matrix
logical crnode_el(4)                              !
                                                  !
!=================================================!
!=====[setup]=====================================!
!=================================================!

Lx=1.d0
Ly=1.d0

nnx=11
nny=11

penalty=1.d7

viscosity=1.d0
density=1.d0

eps=1.d-10

Kmat(1,1)=1.d0 ; Kmat(1,2)=1.d0 ; Kmat(1,3)=0.d0  
Kmat(2,1)=1.d0 ; Kmat(2,2)=1.d0 ; Kmat(2,3)=0.d0  
Kmat(3,1)=0.d0 ; Kmat(3,2)=0.d0 ; Kmat(3,3)=0.d0  

Cmat(1,1)=2.d0 ; Cmat(1,2)=0.d0 ; Cmat(1,3)=0.d0  
Cmat(2,1)=0.d0 ; Cmat(2,2)=2.d0 ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0  

test=3

if (test==0) then
   Lx=9
   Ly=7.d0
   nnx=10
   nny=8
end if

np=nnx*nny

nelx=nnx-1
nely=nny-1

nel=nelx*nely

!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(icon(m,nel))
allocate(crnode(np))
allocate(crtype(nel))
allocate(flag(nel))

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
! compute element center coordinates
!==============================================!

allocate(xc(nel))
allocate(yc(nel))

do iel=1,nel
   xc(iel)=0.25*sum(x(icon(1:4,iel)))
   yc(iel)=0.25*sum(y(icon(1:4,iel)))
end do


!==============================================!
! flag elements for refinement
!==============================================!
   
if (test==0) then 
   flag=.false.
   flag( 8)=.true.
   flag( 9)=.true.
   flag(17)=.true.
   flag(18)=.true.
   flag(21:23)=.true.
   flag(31:33)=.true.
   flag(40:41)=.true.
end if

if (test==1) then ! left half refined
   do iel=1,nel
      if (xc(iel)<0.5*Lx) flag(iel)=.true.
   end do
end if

if (test==2) then ! upper diagonal refined
   do iel=1,nel
      if (xc(iel)<yc(iel)) flag(iel)=.true.
   end do
end if

if (test==3) then ! modulo 7 
   do iel=1,nel
      if (mod(iel,7)==0) flag(iel)=.true.
   end do
end if





crnode=.false.
do iel=1,nel
   if (flag(iel)) then
      do k=1,m
         crnode(icon(k,iel))=.true.
      end do
   end if
end do

print *,'# flagged elements:',count(flag)
print *,'# flagged nodes   :',count(crnode)

!==============================================!
!==============================================!

crtype=-1

do iel=1,nel
   
   crnode_el(1)=crnode(icon(1,iel))
   crnode_el(2)=crnode(icon(2,iel))
   crnode_el(3)=crnode(icon(3,iel))
   crnode_el(4)=crnode(icon(4,iel))

   if (count(crnode_el)==0) crtype(iel)=0

   if (count(crnode_el)==1) then
      if (crnode_el(1)) crtype(iel)=1
      if (crnode_el(2)) crtype(iel)=2
      if (crnode_el(3)) crtype(iel)=3
      if (crnode_el(4)) crtype(iel)=4
   end if

   if (count(crnode_el)==2) then
      if (crnode_el(1) .and. crnode_el(2)) crtype(iel)=5
      if (crnode_el(2) .and. crnode_el(3)) crtype(iel)=6
      if (crnode_el(3) .and. crnode_el(4)) crtype(iel)=7
      if (crnode_el(4) .and. crnode_el(1)) crtype(iel)=8
      if (crnode_el(1) .and. crnode_el(3)) crtype(iel)=9
      if (crnode_el(2) .and. crnode_el(4)) crtype(iel)=10
   end if

   if (count(crnode_el)==3) then
      if (crnode_el(1) .and. crnode_el(2) .and. crnode_el(3)) crtype(iel)=11
      if (crnode_el(2) .and. crnode_el(3) .and. crnode_el(4)) crtype(iel)=12
      if (crnode_el(3) .and. crnode_el(4) .and. crnode_el(1)) crtype(iel)=13
      if (crnode_el(4) .and. crnode_el(1) .and. crnode_el(2)) crtype(iel)=14
   end if

   if (count(crnode_el)==4) then 
   crtype(iel)=15
   end if

end do

print *,'crtype= 0',count(crtype==0)
print *,'crtype= 1',count(crtype==1)
print *,'crtype= 2',count(crtype==2)
print *,'crtype= 3',count(crtype==3)
print *,'crtype= 4',count(crtype==4)
print *,'crtype= 5',count(crtype==5)
print *,'crtype= 6',count(crtype==6)
print *,'crtype= 7',count(crtype==7)
print *,'crtype= 8',count(crtype==8)
print *,'crtype= 9',count(crtype==9)
print *,'crtype=10',count(crtype==10)
print *,'crtype=11',count(crtype==11)
print *,'crtype=12',count(crtype==12)
print *,'crtype=13',count(crtype==13)
print *,'crtype=14',count(crtype==14)
print *,'crtype=15',count(crtype==15)

call output_for_paraview (np,nel,x,y,icon,crnode,crtype)

!==============================================!
! compute new number of elements
!==============================================!

nel_new=0
np_new=0

do iel=1,nel

   select case(crtype(iel))
   case(0)
   nel_new=nel_new+1
   np_new=np_new+4
   case(1,2,3,4)
   nel_new=nel_new+3
   np_new=np_new+3+4
   case(5,6,7,8,9,10)
   nel_new=nel_new+7
   np_new=np_new+8+4
   case(11,12,13,14)
   nel_new=nel_new+8
   np_new=np_new+10+4
   case(15)
   nel_new=nel_new+9
   np_new=np_new+12+4
   end select

end do

print *,'np ',np ,'->',np_new
print *,'nel',nel,'->',nel_new

!==============================================!
!==============================================!

allocate(xnew(np_new))
allocate(ynew(np_new))
allocate(icon_new(4,nel_new))

xnew=0
ynew=0

counter_e=0 ! elements
counter_n=0 ! nodes

do iel=1,nel

   dx=(x(icon(3,iel))-x(icon(1,iel)))/3
   dy=(y(icon(3,iel))-y(icon(1,iel)))/3

   select case(crtype(iel))

   case(0) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-3
      icon_new(2,counter_e) = counter_n-2
      icon_new(3,counter_e) = counter_n-1
      icon_new(4,counter_e) = counter_n

   case(1) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))  
      ynew(counter_n)=y(icon(1,iel)) + 3*dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 3*dx 
      ynew(counter_n)=y(icon(1,iel)) + 3*dy

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+1
      icon_new(2,counter_e) = counter_n-7+2
      icon_new(3,counter_e) = counter_n-7+5
      icon_new(4,counter_e) = counter_n-7+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+2
      icon_new(2,counter_e) = counter_n-7+3
      icon_new(3,counter_e) = counter_n-7+7
      icon_new(4,counter_e) = counter_n-7+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+4
      icon_new(2,counter_e) = counter_n-7+5
      icon_new(3,counter_e) = counter_n-7+7
      icon_new(4,counter_e) = counter_n-7+6


   case(2) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 3*dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel)) 
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel)) 

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+1
      icon_new(2,counter_e) = counter_n-7+2
      icon_new(3,counter_e) = counter_n-7+4
      icon_new(4,counter_e) = counter_n-7+6
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+2
      icon_new(2,counter_e) = counter_n-7+3
      icon_new(3,counter_e) = counter_n-7+5
      icon_new(4,counter_e) = counter_n-7+4
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+4
      icon_new(2,counter_e) = counter_n-7+5
      icon_new(3,counter_e) = counter_n-7+7
      icon_new(4,counter_e) = counter_n-7+6


   case(3) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) +2*dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+1
      icon_new(2,counter_e) = counter_n-7+2
      icon_new(3,counter_e) = counter_n-7+4
      icon_new(4,counter_e) = counter_n-7+3
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+1
      icon_new(2,counter_e) = counter_n-7+3
      icon_new(3,counter_e) = counter_n-7+6
      icon_new(4,counter_e) = counter_n-7+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+3
      icon_new(2,counter_e) = counter_n-7+4
      icon_new(3,counter_e) = counter_n-7+7
      icon_new(4,counter_e) = counter_n-7+6


   case(4) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 1*dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel)) 
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) + dx  
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel)) 

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+1
      icon_new(2,counter_e) = counter_n-7+2
      icon_new(3,counter_e) = counter_n-7+4
      icon_new(4,counter_e) = counter_n-7+3
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+3
      icon_new(2,counter_e) = counter_n-7+4
      icon_new(3,counter_e) = counter_n-7+6
      icon_new(4,counter_e) = counter_n-7+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-7+4
      icon_new(2,counter_e) = counter_n-7+2
      icon_new(3,counter_e) = counter_n-7+7
      icon_new(4,counter_e) = counter_n-7+6

   case(5) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 3*dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel)) 
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel)) 

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+2
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+7
      icon_new(4,counter_e) = counter_n-12+6
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+3
      icon_new(2,counter_e) = counter_n-12+4
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+9
      icon_new(4,counter_e) = counter_n-12+11
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+6
      icon_new(2,counter_e) = counter_n-12+7
      icon_new(3,counter_e) = counter_n-12+10
      icon_new(4,counter_e) = counter_n-12+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+9
      icon_new(2,counter_e) = counter_n-12+10
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11

   case(6) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +dx
      ynew(counter_n)=y(icon(1,iel)) +dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) + dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) + 3*dx
      ynew(counter_n)=y(icon(1,iel)) + 2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 11
      counter_n=counter_n+1
      xnew(counter_n)=x(icon(1,iel)) + 2*dx
      ynew(counter_n)=y(icon(1,iel)) + 3*dy
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))  
      ynew(counter_n)=y(icon(3,iel))  

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+5
      icon_new(4,counter_e) = counter_n-12+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+2
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+4
      icon_new(3,counter_e) = counter_n-12+7
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+4
      icon_new(2,counter_e) = counter_n-12+5
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+9
      icon_new(4,counter_e) = counter_n-12+8
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+11
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+8
      icon_new(2,counter_e) = counter_n-12+9
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11


   case(7) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) +dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) +2*dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+4
      icon_new(4,counter_e) = counter_n-12+3
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+3
      icon_new(2,counter_e) = counter_n-12+4
      icon_new(3,counter_e) = counter_n-12+7
      icon_new(4,counter_e) = counter_n-12+6
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+4
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+10
      icon_new(4,counter_e) = counter_n-12+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+6
      icon_new(2,counter_e) = counter_n-12+7
      icon_new(3,counter_e) = counter_n-12+11
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11

   case(8) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) +dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+5
      icon_new(4,counter_e) = counter_n-12+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+2
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+4
      icon_new(2,counter_e) = counter_n-12+5
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+9
      icon_new(4,counter_e) = counter_n-12+8
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+6
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+11
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+8
      icon_new(2,counter_e) = counter_n-12+9
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11


   case(9) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx 
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel)) 
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))+3*dy
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel)) 
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+5
      icon_new(4,counter_e) = counter_n-12+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+2
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+4
      icon_new(2,counter_e) = counter_n-12+5
      icon_new(3,counter_e) = counter_n-12+7
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+6
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+9
      icon_new(4,counter_e) = counter_n-12+8
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+11
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+8
      icon_new(2,counter_e) = counter_n-12+9
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11

      
   case(10) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel))
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+2
      icon_new(3,counter_e) = counter_n-12+5
      icon_new(4,counter_e) = counter_n-12+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+2
      icon_new(2,counter_e) = counter_n-12+3
      icon_new(3,counter_e) = counter_n-12+6
      icon_new(4,counter_e) = counter_n-12+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+1
      icon_new(2,counter_e) = counter_n-12+4
      icon_new(3,counter_e) = counter_n-12+8
      icon_new(4,counter_e) = counter_n-12+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+4
      icon_new(2,counter_e) = counter_n-12+5
      icon_new(3,counter_e) = counter_n-12+9
      icon_new(4,counter_e) = counter_n-12+8
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+5
      icon_new(2,counter_e) = counter_n-12+6
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+7
      icon_new(2,counter_e) = counter_n-12+8
      icon_new(3,counter_e) = counter_n-12+11
      icon_new(4,counter_e) = counter_n-12+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-12+8
      icon_new(2,counter_e) = counter_n-12+9
      icon_new(3,counter_e) = counter_n-12+12
      icon_new(4,counter_e) = counter_n-12+11



     

   case(11) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel))
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 13
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+2*dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 14
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+1
      icon_new(2,counter_e) = counter_n-14+2
      icon_new(3,counter_e) = counter_n-14+6
      icon_new(4,counter_e) = counter_n-14+5
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+2
      icon_new(2,counter_e) = counter_n-14+3
      icon_new(3,counter_e) = counter_n-14+7
      icon_new(4,counter_e) = counter_n-14+6
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+3
      icon_new(2,counter_e) = counter_n-14+4
      icon_new(3,counter_e) = counter_n-14+8
      icon_new(4,counter_e) = counter_n-14+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+5
      icon_new(2,counter_e) = counter_n-14+6
      icon_new(3,counter_e) = counter_n-14+9
      icon_new(4,counter_e) = counter_n-14+12
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+6
      icon_new(2,counter_e) = counter_n-14+7
      icon_new(3,counter_e) = counter_n-14+10
      icon_new(4,counter_e) = counter_n-14+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+7
      icon_new(2,counter_e) = counter_n-14+8
      icon_new(3,counter_e) = counter_n-14+11
      icon_new(4,counter_e) = counter_n-14+10
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+9
      icon_new(2,counter_e) = counter_n-14+10
      icon_new(3,counter_e) = counter_n-14+13
      icon_new(4,counter_e) = counter_n-14+12
      ! sub elt 8
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+10
      icon_new(2,counter_e) = counter_n-14+11
      icon_new(3,counter_e) = counter_n-14+14
      icon_new(4,counter_e) = counter_n-14+13






   case(12) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 13
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+2*dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 14
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+1
      icon_new(2,counter_e) = counter_n-14+4
      icon_new(3,counter_e) = counter_n-14+8
      icon_new(4,counter_e) = counter_n-14+7
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+1
      icon_new(2,counter_e) = counter_n-14+2
      icon_new(3,counter_e) = counter_n-14+5
      icon_new(4,counter_e) = counter_n-14+4
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+2
      icon_new(2,counter_e) = counter_n-14+3
      icon_new(3,counter_e) = counter_n-14+6
      icon_new(4,counter_e) = counter_n-14+5
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+4
      icon_new(2,counter_e) = counter_n-14+5
      icon_new(3,counter_e) = counter_n-14+9
      icon_new(4,counter_e) = counter_n-14+8
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+5
      icon_new(2,counter_e) = counter_n-14+6
      icon_new(3,counter_e) = counter_n-14+10
      icon_new(4,counter_e) = counter_n-14+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+7
      icon_new(2,counter_e) = counter_n-14+8
      icon_new(3,counter_e) = counter_n-14+12
      icon_new(4,counter_e) = counter_n-14+11
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+8
      icon_new(2,counter_e) = counter_n-14+9
      icon_new(3,counter_e) = counter_n-14+13
      icon_new(4,counter_e) = counter_n-14+12
      ! sub elt 8
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+9
      icon_new(2,counter_e) = counter_n-14+10
      icon_new(3,counter_e) = counter_n-14+14
      icon_new(4,counter_e) = counter_n-14+13

   case(13) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel)) 
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 13
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+2*dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 14
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+1
      icon_new(2,counter_e) = counter_n-14+2
      icon_new(3,counter_e) = counter_n-14+5
      icon_new(4,counter_e) = counter_n-14+4
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+2
      icon_new(2,counter_e) = counter_n-14+3
      icon_new(3,counter_e) = counter_n-14+6
      icon_new(4,counter_e) = counter_n-14+5
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+4
      icon_new(2,counter_e) = counter_n-14+5
      icon_new(3,counter_e) = counter_n-14+8
      icon_new(4,counter_e) = counter_n-14+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+5
      icon_new(2,counter_e) = counter_n-14+6
      icon_new(3,counter_e) = counter_n-14+9
      icon_new(4,counter_e) = counter_n-14+8
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+6
      icon_new(2,counter_e) = counter_n-14+3
      icon_new(3,counter_e) = counter_n-14+10
      icon_new(4,counter_e) = counter_n-14+9
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+7
      icon_new(2,counter_e) = counter_n-14+8
      icon_new(3,counter_e) = counter_n-14+12
      icon_new(4,counter_e) = counter_n-14+11
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+8
      icon_new(2,counter_e) = counter_n-14+9
      icon_new(3,counter_e) = counter_n-14+13
      icon_new(4,counter_e) = counter_n-14+12
      ! sub elt 8
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+9
      icon_new(2,counter_e) = counter_n-14+10
      icon_new(3,counter_e) = counter_n-14+14
      icon_new(4,counter_e) = counter_n-14+13

   case(14) !-------------------------------------

      ! sub pt 1
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel)) 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 2
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 3
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx 
      ynew(counter_n)=y(icon(1,iel))
      ! sub pt 4
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(2,iel))
      ynew(counter_n)=y(icon(2,iel))
      ! sub pt 5
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 6
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 7
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 8
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+3*dx
      ynew(counter_n)=y(icon(1,iel))+dy
      ! sub pt 9
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 10
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 11
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(1,iel))+2*dx
      ynew(counter_n)=y(icon(1,iel))+2*dy
      ! sub pt 12
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 13
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(4,iel))+dx
      ynew(counter_n)=y(icon(4,iel))
      ! sub pt 14
      counter_n=counter_n+1 
      xnew(counter_n)=x(icon(3,iel))
      ynew(counter_n)=y(icon(3,iel))

      ! sub elt 1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+1
      icon_new(2,counter_e) = counter_n-14+2
      icon_new(3,counter_e) = counter_n-14+6
      icon_new(4,counter_e) = counter_n-14+5
      ! sub elt 2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+2
      icon_new(2,counter_e) = counter_n-14+3
      icon_new(3,counter_e) = counter_n-14+7
      icon_new(4,counter_e) = counter_n-14+6
      ! sub elt 3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+3
      icon_new(2,counter_e) = counter_n-14+4
      icon_new(3,counter_e) = counter_n-14+8
      icon_new(4,counter_e) = counter_n-14+7
      ! sub elt 4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+5
      icon_new(2,counter_e) = counter_n-14+6
      icon_new(3,counter_e) = counter_n-14+10
      icon_new(4,counter_e) = counter_n-14+9
      ! sub elt 5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+6
      icon_new(2,counter_e) = counter_n-14+7
      icon_new(3,counter_e) = counter_n-14+11
      icon_new(4,counter_e) = counter_n-14+10
      ! sub elt 6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+7
      icon_new(2,counter_e) = counter_n-14+8
      icon_new(3,counter_e) = counter_n-14+14
      icon_new(4,counter_e) = counter_n-14+11
      ! sub elt 7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+9
      icon_new(2,counter_e) = counter_n-14+10
      icon_new(3,counter_e) = counter_n-14+13
      icon_new(4,counter_e) = counter_n-14+12
      ! sub elt 8
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-14+10
      icon_new(2,counter_e) = counter_n-14+11
      icon_new(3,counter_e) = counter_n-14+14
      icon_new(4,counter_e) = counter_n-14+13

   case(15) !-------------------------------------
      ! sub pt 1
      counter_n=counter_n+1 ! 1
      xnew(counter_n)=x(icon(1,iel)) +0*dx
      ynew(counter_n)=y(icon(1,iel)) +0*dy
      ! sub pt 2
      counter_n=counter_n+1 ! 2
      xnew(counter_n)=x(icon(1,iel)) +1*dx
      ynew(counter_n)=y(icon(1,iel)) +0*dy
      ! sub pt 3
      counter_n=counter_n+1 ! 3
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +0*dy
      ! sub pt 4
      counter_n=counter_n+1 ! 4
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) +0*dy
      ! sub pt 5
      counter_n=counter_n+1 ! 1
      xnew(counter_n)=x(icon(1,iel)) +0*dx
      ynew(counter_n)=y(icon(1,iel)) +1*dy
      ! sub pt 6
      counter_n=counter_n+1 ! 2
      xnew(counter_n)=x(icon(1,iel)) +1*dx
      ynew(counter_n)=y(icon(1,iel)) +1*dy
      ! sub pt 7
      counter_n=counter_n+1 ! 3
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +1*dy
      ! sub pt 8
      counter_n=counter_n+1 ! 4
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) +1*dy
      ! sub pt 9
      counter_n=counter_n+1 ! 1
      xnew(counter_n)=x(icon(1,iel)) +0*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 10
      counter_n=counter_n+1 ! 2
      xnew(counter_n)=x(icon(1,iel)) +1*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 11
      counter_n=counter_n+1 ! 3
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 12
      counter_n=counter_n+1 ! 4
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) +2*dy
      ! sub pt 13
      counter_n=counter_n+1 ! 1
      xnew(counter_n)=x(icon(1,iel)) +0*dx
      ynew(counter_n)=y(icon(1,iel)) +3*dy
      ! sub pt 14
      counter_n=counter_n+1 ! 2
      xnew(counter_n)=x(icon(1,iel)) +1*dx
      ynew(counter_n)=y(icon(1,iel)) +3*dy
      ! sub pt 15
      counter_n=counter_n+1 ! 3
      xnew(counter_n)=x(icon(1,iel)) +2*dx
      ynew(counter_n)=y(icon(1,iel)) +3*dy
      ! sub pt 16
      counter_n=counter_n+1 ! 4
      xnew(counter_n)=x(icon(1,iel)) +3*dx
      ynew(counter_n)=y(icon(1,iel)) +3*dy

      !sub elt1
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+1
      icon_new(2,counter_e) = counter_n-16+2
      icon_new(3,counter_e) = counter_n-16+6
      icon_new(4,counter_e) = counter_n-16+5
      !sub elt2
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+2
      icon_new(2,counter_e) = counter_n-16+3
      icon_new(3,counter_e) = counter_n-16+7
      icon_new(4,counter_e) = counter_n-16+6
      !sub elt3
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+3
      icon_new(2,counter_e) = counter_n-16+4
      icon_new(3,counter_e) = counter_n-16+8
      icon_new(4,counter_e) = counter_n-16+7
      !sub elt4
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+5
      icon_new(2,counter_e) = counter_n-16+6
      icon_new(3,counter_e) = counter_n-16+10
      icon_new(4,counter_e) = counter_n-16+9
      !sub elt5
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+6
      icon_new(2,counter_e) = counter_n-16+7
      icon_new(3,counter_e) = counter_n-16+11
      icon_new(4,counter_e) = counter_n-16+10
      !sub elt6
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+7
      icon_new(2,counter_e) = counter_n-16+8
      icon_new(3,counter_e) = counter_n-16+12
      icon_new(4,counter_e) = counter_n-16+11
      !sub elt7
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+5+4
      icon_new(2,counter_e) = counter_n-16+6+4
      icon_new(3,counter_e) = counter_n-16+10+4
      icon_new(4,counter_e) = counter_n-16+9+4
      !sub elt9
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+6+4
      icon_new(2,counter_e) = counter_n-16+7+4
      icon_new(3,counter_e) = counter_n-16+11+4
      icon_new(4,counter_e) = counter_n-16+10+4
      !sub elt9
      counter_e=counter_e+1
      icon_new(1,counter_e) = counter_n-16+7+4
      icon_new(2,counter_e) = counter_n-16+8+4
      icon_new(3,counter_e) = counter_n-16+12+4
      icon_new(4,counter_e) = counter_n-16+11+4



   end select
end do

open(unit=123,file='OUT/gridnodes_new.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,np_new
   write(123,'(2f10.5,i8)') xnew(i),ynew(i),i
end do
close(123)

print *,counter_n,counter_e

!==============================================!
! compute real number of points
!==============================================!

allocate(doubble(np_new)) ; doubble=.false.
allocate(pointto(np_new))

do ip=1,np_new
   pointto(ip)=ip
end do

distance=1.d-8

counter=0
do ip=2,np_new
   xip=xnew(ip)
   yip=ynew(ip)
   do jp=1,ip-1
      if (abs(xip-xnew(jp))<distance .and. &
          abs(yip-ynew(jp))<distance ) then 
          doubble(ip)=.true.
          pointto(ip)=jp
          exit 
      end if
   end do
end do

np_real=np_new-count(doubble)

nel_real=nel_new

print *,'real new nb of nodes, np_real =',np_real

print *,'real new nb of elts , nel_real=',nel_real

print *,'pointto :',minval(pointto),maxval(pointto)

!==============================================!
!==============================================!

allocate(xreal(np_real))
allocate(yreal(np_real))
allocate(icon_real(4,nel_real))

counter=0
do ip=1,np_new
   if (.not.doubble(ip)) then
      counter=counter+1
      xreal(counter)=xnew(ip)
      yreal(counter)=ynew(ip)
   end if
end do

icon_real=icon_new

do iel=1,nel_real
   do i=1,4
      icon_real(i,iel)=pointto(icon_real(i,iel))
   end do
end do

!print *,'bef compaction:',minval(icon_real),maxval(icon_real)

allocate(compact(np_new))

counter=0
do ip=1,np_new
   if (.not.doubble(ip)) then
      counter=counter+1
      compact(ip)=counter
   end if
end do

!print *,'compact :',minval(compact),maxval(compact)

do iel=1,nel_real
   do i=1,4
      icon_real(i,iel)=compact(icon_real(i,iel))
   end do
end do

!print *,'aft compaction:',minval(icon_real),maxval(icon_real)

open(unit=123,file='OUT/gridnodes_real.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,np_real
   write(123,'(2f10.5,i8)') xreal(i),yreal(i),i
end do
close(123)

!==============================================!
!==============================================!


call output_for_paraview2 (np_real,nel_real,xreal,yreal,icon_real)

!==============================================!

deallocate(x,y,icon)
deallocate(crnode)
deallocate(crtype)

np=np_real

nel=nel_real

Nfem=np*ndof

allocate(x(np))
allocate(y(np))
allocate(icon(m,nel))
allocate(u(np))
allocate(v(np))
allocate(bc_fix(Nfem))
allocate(bc_val(Nfem))
allocate(press(nel))
allocate(A(Nfem,Nfem))
allocate(B(Nfem))
allocate(C(Nfem,Nfem))

icon=icon_real

x=xreal

y=yreal

!==============================================!
!=====[define bc]==============================!
!==============================================!
print *,'define_bc'

bc_fix=.false.

do i=1,np
   if (x(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (x(i).gt.(Lx-eps)) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).lt.eps) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
      bc_fix((i-1)*ndof+2)=.true. ; bc_val((i-1)*ndof+2)=0.d0
   endif
   if (y(i).gt.(Ly-eps) ) then
      bc_fix((i-1)*ndof+1)=.true. ; bc_val((i-1)*ndof+1)=0.d0
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
print *,'building matrix'

A=0.d0
B=0.d0
C=.false.

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

      Ael=Ael + matmul(transpose(Bmat),matmul(viscosity*Cmat,Bmat))*wq*jcob

      do i=1,m
      i1=2*i-1
      i2=2*i
      !Bel(i1)=Bel(i1)+N(i)*jcob*wq*density*gx
      !Bel(i2)=Bel(i2)+N(i)*jcob*wq*density*gy
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
print *,'impose bc'

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

open(unit=123,file='OUT/solution_u.dat',status='replace')
open(unit=234,file='OUT/solution_v.dat',status='replace')
do i=1,np
   write(123,'(5f20.10)') x(i),y(i),u(i),uth(x(i),y(i)),u(i)-uth(x(i),y(i))
   write(234,'(5f20.10)') x(i),y(i),v(i),vth(x(i),y(i)),v(i)-vth(x(i),y(i))
end do
close(123)
close(234)

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
   
   write(123,*) xq,yq,press(iel),pth(xq,yq)

end do

close(123)

call output_for_paraview3 (np,nel,x,y,u,v,press,icon)

end program

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

function b1 (x,y)
real(8) b1,x,y
b1 = ( (12.d0-24.d0*y)*x**4 + (-24.d0+48.d0*y)*x**3 + (-48.d0*y+72.d0*y**2-48.d0*y**3+12.d0)*x**2 &
   + (-2.d0+24.d0*y-72.d0*y**2+48.d0*y**3)*x + 1.d0-4.d0*y+12.d0*y**2-8.d0*y**3 )
end function

function b2 (x,y)
real(8) b2,x,y
b2= ( (8.d0-48.d0*y+48.d0*y**2)*x**3 + (-12.d0+72.d0*y-72*y**2)*x**2 + &
    (4.d0-24.d0*y+48.d0*y**2-48.d0*y**3+24.d0*y**4)*x - 12.d0*y**2 + 24.d0*y**3 -12.d0*y**4)
end function












