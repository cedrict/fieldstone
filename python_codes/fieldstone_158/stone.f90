!==============================================
!==============================================

program essai

use structures

implicit none

type(grid) gridb ! background
type(grid) gridu
type(grid) gridv
type(grid) gridp
type(grid) grids ! visualisation

integer, parameter :: m=4 

integer counter
integer i,j,ip
integer N,N_v,N_p
integer ku,kv
integer iel,which
integer ii,jj,job

integer kp_w,kp_e,kp_n,kp_s
integer kv_s,kv_n,kv_e,kv_w
integer ku_w,ku_e,ku_n,ku_s

real(8) Lx,Ly
real(8) hx,hy
real(8) gx,gy
real(8) rho_e,rho_w,rho_n,rho_s,rho
real(8) mu
real(8) t1,t2

real(8), dimension(:,:), allocatable :: K
real(8), dimension(:), allocatable :: rhs
real(8), dimension(:), allocatable :: sol

integer, dimension(:), allocatable :: ipvt     ! work array needed by the solver 
real(8), dimension(:),   allocatable :: work   ! work array needed by the solver

!==============================================!
! viscosity has to remain constant in this program

Lx=10
Ly=10

gridb%nnx=32
gridb%nny=32

mu=1.23456789

gx=0
gy=-1

!==============================================!
!initialise background grid
!==============================================!

gridb%np=gridb%nnx*gridb%nny
gridb%nelx=gridb%nnx-1
gridb%nely=gridb%nny-1
gridb%nel=gridb%nelx*gridb%nely

!==============================================!
!initialise u grid
!==============================================!

gridu%nnx=gridb%nnx
gridu%nny=gridb%nny-1
gridu%np=gridu%nnx*gridu%nny
gridu%nelx=gridu%nnx-1
gridu%nely=gridu%nny-1
gridu%nel=gridu%nelx*gridu%nely

!==============================================!
!initialise v grid
!==============================================!

gridv%nnx=gridb%nnx-1
gridv%nny=gridb%nny
gridv%np=gridv%nnx*gridv%nny
gridv%nelx=gridv%nnx-1
gridv%nely=gridv%nny-1
gridv%nel=gridv%nelx*gridv%nely

!==============================================!
!initialise p grid
!==============================================!

gridp%nnx=gridb%nnx-1
gridp%nny=gridb%nny-1
gridp%np=gridp%nnx*gridp%nny
gridp%nelx=gridp%nnx-1
gridp%nely=gridp%nny-1
gridp%nel=gridp%nelx*gridp%nely

!==============================================!

grids%nnx=gridb%nelx
grids%nny=gridb%nely
grids%np=grids%nnx*grids%nny
grids%nelx=grids%nnx-1
grids%nely=grids%nny-1
grids%nel=grids%nelx*grids%nely

!==============================================!

hx=Lx/(gridb%nnx-1)
hy=Ly/(gridb%nny-1)

N_v=gridu%np+gridv%np
N_p=gridp%np
N=N_v+N_p

!==============================================!

print *,'u: nnx,nny ',gridu%nnx,gridu%nny
print *,'v: nnx,nny ',gridv%nnx,gridv%nny
print *,'p: nnx,nny ',gridp%nnx,gridp%nny
print *,'nb of u nodes:',gridu%np
print *,'nb of v nodes:',gridv%np
print *,'nb of p nodes:',gridp%np
print *,'nb of dofs:',N
print *,'nb of dofs vel  :',N_v
print *,'nb of dofs press:',N_p

!==============================================!

allocate(gridb%x(gridb%np))
allocate(gridb%y(gridb%np))
allocate(gridb%rho(gridb%np))
allocate(gridb%icon(m,gridb%nel))

allocate(gridu%x(gridu%np))
allocate(gridu%y(gridu%np))
allocate(gridu%bc(gridu%np))
allocate(gridu%icon(m,gridu%nel))
allocate(gridu%field(gridu%np))

allocate(gridv%x(gridv%np))
allocate(gridv%y(gridv%np))
allocate(gridv%bc(gridv%np))
allocate(gridv%icon(m,gridv%nel))
allocate(gridv%field(gridv%np))

allocate(gridp%x(gridp%np))
allocate(gridp%y(gridp%np))
allocate(gridp%icon(m,gridp%nel))
allocate(gridp%field(gridp%np))

allocate(grids%x(grids%np))
allocate(grids%y(grids%np))
allocate(grids%rho(grids%np))
allocate(grids%icon(m,grids%nel))
allocate(grids%u(grids%np))
allocate(grids%v(grids%np))
allocate(grids%p(grids%np))
allocate(grids%divv(grids%np))

gridu%field=0
gridv%field=0
gridp%field=0

allocate(K(N,N)) 
allocate(rhs(N))   
allocate(sol(N))   

K=0
rhs=0
sol=0

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do j=1,gridb%nny
do i=1,gridb%nnx
   counter=counter+1
   gridb%x(counter)=dble(i-1)*hx
   gridb%y(counter)=dble(j-1)*hy
end do
end do

counter=0
do j=1,gridu%nny
do i=1,gridu%nnx
   counter=counter+1
   gridu%x(counter)=(i-1)*hx
   gridu%y(counter)=(j-1)*hy+hy/2
end do
end do

counter=0
do j=1,gridv%nny
do i=1,gridv%nnx
   counter=counter+1
   gridv%x(counter)=(i-1)*hx+hx/2
   gridv%y(counter)=(j-1)*hy
end do
end do

counter=0
do j=1,gridp%nny
do i=1,gridp%nnx
   counter=counter+1
   gridp%x(counter)=(i-1)*hx+hx/2
   gridp%y(counter)=(j-1)*hy+hy/2
end do
end do

counter=0
do j=1,grids%nny
do i=1,grids%nnx
   counter=counter+1
   grids%x(counter)=(i-1)*hx+hx/2
   grids%y(counter)=(j-1)*hy+hy/2
end do
end do

open(unit=123,file='OUT/grid_nodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridb%np
   write(123,'(2f10.5,i8)') gridb%x(i),gridb%y(i),i
end do
close(123)

open(unit=123,file='OUT/gridu_nodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridu%np
   write(123,'(2f10.5,i8)') gridu%x(i),gridu%y(i),i
end do
close(123)

open(unit=123,file='OUT/gridv_nodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridv%np
   write(123,'(2f10.5,i8)') gridv%x(i),gridv%y(i),i
end do
close(123)

open(unit=123,file='OUT/gridp%nodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridp%np
   write(123,'(2f10.5,i8)') gridp%x(i),gridp%y(i),i
end do
close(123)

open(unit=123,file='OUT/grids_nodes.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,grids%np
   write(123,'(2f10.5,i8)') grids%x(i),grids%y(i),i
end do
close(123)


!==============================================!
!===[grid connectivity]========================! 
!==============================================!

counter=0
do j=1,gridb%nely
   do i=1,gridb%nelx
      counter=counter+1
      gridb%icon(1,counter)=i+(j-1)*(gridb%nelx+1)
      gridb%icon(2,counter)=i+1+(j-1)*(gridb%nelx+1)
      gridb%icon(3,counter)=i+1+j*(gridb%nelx+1)
      gridb%icon(4,counter)=i+j*(gridb%nelx+1)
   end do
end do

counter=0
do j=1,gridu%nely
   do i=1,gridu%nelx
      counter=counter+1
      gridu%icon(1,counter)=i+(j-1)*(gridu%nelx+1)
      gridu%icon(2,counter)=i+1+(j-1)*(gridu%nelx+1)
      gridu%icon(3,counter)=i+1+j*(gridu%nelx+1)
      gridu%icon(4,counter)=i+j*(gridu%nelx+1)
   end do
end do

counter=0
do j=1,gridv%nely
   do i=1,gridv%nelx
      counter=counter+1
      gridv%icon(1,counter)=i+(j-1)*(gridv%nelx+1)
      gridv%icon(2,counter)=i+1+(j-1)*(gridv%nelx+1)
      gridv%icon(3,counter)=i+1+j*(gridv%nelx+1)
      gridv%icon(4,counter)=i+j*(gridv%nelx+1)
   end do
end do

counter=0
do j=1,gridp%nely
   do i=1,gridp%nelx
      counter=counter+1
      gridp%icon(1,counter)=i+(j-1)*(gridp%nelx+1)
      gridp%icon(2,counter)=i+1+(j-1)*(gridp%nelx+1)
      gridp%icon(3,counter)=i+1+j*(gridp%nelx+1)
      gridp%icon(4,counter)=i+j*(gridp%nelx+1)
   end do
end do

counter=0
do j=1,grids%nely
   do i=1,grids%nelx
      counter=counter+1
      grids%icon(1,counter)=i+(j-1)*(grids%nelx+1)
      grids%icon(2,counter)=i+1+(j-1)*(grids%nelx+1)
      grids%icon(3,counter)=i+1+j*(grids%nelx+1)
      grids%icon(4,counter)=i+j*(grids%nelx+1)
   end do
end do

!==============================================!
! flag boundary u and v nodes
!==============================================!

gridu%bc=.false.
counter=0
do j=1,gridu%nny
do i=1,gridu%nnx
   counter=counter+1
   if (j==1        ) gridu%bc(counter)=.true.
   if (j==gridu%nny) gridu%bc(counter)=.true.
   if (i==1        ) gridu%bc(counter)=.true.
   if (i==gridu%nnx) gridu%bc(counter)=.true.
end do
end do

gridv%bc=.false.
counter=0
do j=1,gridv%nny
do i=1,gridv%nnx
   counter=counter+1
   if (j==1        ) gridv%bc(counter)=.true.
   if (j==gridv%nny) gridv%bc(counter)=.true.
   if (i==1        ) gridv%bc(counter)=.true.
   if (i==gridv%nnx) gridv%bc(counter)=.true.
end do
end do

open(unit=123,file='OUT/gridu_nodes_bc.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridu%np
   if (gridu%bc(i)) &
   write(123,'(2f10.5,i8)') gridu%x(i),gridu%y(i),i
end do
close(123)

open(unit=123,file='OUT/gridv_nodes_bc.dat',status='replace')
write(123,'(a)') '#     xpos      ypos    node '
do i=1,gridv%np
   if (gridv%bc(i)) &
   write(123,'(2f10.5,i8)') gridv%x(i),gridv%y(i),i
end do
close(123)

!==============================================!
! assigning density and viscosity to nodes
! of background grid
!==============================================!

do ip=1,gridb%np
   if (abs(gridb%x(ip)-Lx/2)< 0.1*Lx .and.  &
       abs(gridb%y(ip)-Ly/2)< 0.1*Ly)  then
      gridb%rho(ip)=2
   else
      gridb%rho(ip)=1
   end if
end do

!==============================================!
! assigning density and viscosity to nodes
! of background grid
!==============================================!

!iel=0
!do j=1,gridb%nely
!do i=1,gridb%nelx
!   iel=iel+1
   !pressure grid
   !gridp%rho  (iel)=sum(gridb%rho  (gridb%icon(:,iel)))/m
   !u grid
   !gridu%rho(iel + j-1 )  =(gridb%rho(gridb%icon(1,iel))+gridb%rho(gridb%icon(4,iel)))/2.d0
   !gridu%rho(iel + j   )  =(gridb%rho(gridb%icon(2,iel))+gridb%rho(gridb%icon(3,iel)))/2.d0
   !v grid   
   !gridv%rho(iel           )  =(gridb%rho(gridb%icon(1,iel))+gridb%rho(gridb%icon(2,iel)))/2.d0
   !gridv%rho(iel+gridb%nelx)  =(gridb%rho(gridb%icon(3,iel))+gridb%rho(gridb%icon(4,iel)))/2.d0
!end do
!end do

!==============================================!
! looping over 'elements' for continuity eq
!==============================================!

do iel=1,gridb%nel

   ku_w=gridb%icon(1,iel)
   ku_e=gridb%icon(2,iel)

   kv_s=iel
   kv_n=iel+gridb%nelx

   !print *,iel,ku_w,ku_e,kv_s,kv_n

   K(N_v+iel,         ku_e)=+1.d0/hx   
   K(N_v+iel,         ku_w)=-1.d0/hx   
   K(N_v+iel,gridu%np+kv_n)=+1.d0/hy   
   K(N_v+iel,gridu%np+kv_s)=-1.d0/hy   

end do

!==============================================!
! looping over u nodes
!==============================================!

do ku=1,gridu%np

   if (.not.gridu%bc(ku)) then

      !-------------------------------------------------
      ! find which two horizontal elements it belongs to
      ! for pressure gradient
      !-------------------------------------------------

      ! west
      ii=mod(ku,gridu%nnx)-1
      jj=(ku-mod(ku,gridu%nnx))/gridu%nnx+1
      kp_w=gridb%nelx*(jj-1)+ii

      ! east 
      ii=mod(ku,gridu%nnx)
      jj=(ku-mod(ku,gridu%nnx))/gridu%nnx+1
      kp_e=gridb%nelx*(jj-1)+ii
      !print *,'node',ku,'|element east/west ',kp_w,kp_e

      !-------------------------------------------------
      ! find west, east, north and south neighbours to compute  
      ! d s_{xx} / dx and d s_{xy} / dy
      !-------------------------------------------------

      ku_w=ku-1
      ku_e=ku+1
      ku_n=ku+gridb%nnx
      ku_s=ku-gridb%nnx

      !print *,'node',ku,'|ku_w',ku_w,'|ku_e',ku_e,'|ku_s',ku_s,'|ku_n',ku_n  

      !---------------------
      ! compute density 
      !---------------------
      ! average of north and south

      rho_n=gridb%rho(ku+gridu%nnx)
      rho_s=gridb%rho(ku)
      rho=(rho_s+rho_n)*0.5d0

      !write(300,*) gridu_x(ku),gridu_y(ku),rho

      !----------------
      ! fill K matrix 
      !----------------

      ! pressure grad

      K(ku,N_v+kp_e)=-1.d0/hx
      K(ku,N_v+kp_w)=+1.d0/hx

      ! d^2u/dx^2

      K(ku,ku_e)=K(ku,ku_e)+  mu/hx**2
      K(ku,ku  )=K(ku,ku  )-2*mu/hx**2
      K(ku,ku_w)=K(ku,ku_w)+  mu/hx**2

      ! d^2u/dy^2

      K(ku,ku_n)=K(ku,ku_n)+  mu/hy**2
      K(ku,ku  )=K(ku,ku  )-2*mu/hy**2
      K(ku,ku_s)=K(ku,ku_s)+  mu/hy**2

      rhs(ku)=-rho*gx

   end if

end do

!==============================================!
! looping over v nodes
!==============================================!

do kv=1,gridv%np

   if (.not.gridv%bc(kv)) then

      !-------------------------------------------------
      ! find which two vertical elements it belongs to
      ! for pressure gradient
      !-------------------------------------------------

      kp_n=kv
      kp_s=kv-gridb%nelx

      !print *,'node',kv,'| kp_s=',kp_s,' kp_n=',kp_n

      !-------------------------------------------------
      ! find west, east, north and south neighbours to compute  
      ! d s_{xy} / dx and d s_{yy} / dy
      !-------------------------------------------------

      kv_n=kv+gridv%nnx
      kv_s=kv-gridv%nnx
      kv_e=kv+1
      kv_w=kv-1

      !print *,'node',kv,'| w,e,s,n :',kv_w,kv_e,kv_s,kv_n

      !---------------------
      ! compute density 
      !---------------------
      ! average of east and west

      rho_w=gridb%rho(gridb%icon(1,kv))
      rho_e=gridb%rho(gridb%icon(2,kv))
      rho=(rho_w+rho_e)*0.5d0

      !write(*,*) 'node',kv,'| w,e',grid_icon(1,kv),grid_icon(2,kv) 

      !write(400,*) gridv_x(kv),gridv_y(kv),rho

      !----------------
      ! fill K matrix 
      !----------------

      ! pressure grad

      K(gridu%np+kv,N_v+kp_n)=-1.d0/hy
      K(gridu%np+kv,N_v+kp_s)=+1.d0/hy

      ! d^2v/dx^2

      K(kv+gridu%np,kv_e+gridu%np)=K(kv+gridu%np,kv_e+gridu%np)+  mu/hx**2
      K(kv+gridu%np,kv  +gridu%np)=K(kv+gridu%np,kv  +gridu%np)-2*mu/hx**2
      K(kv+gridu%np,kv_w+gridu%np)=K(kv+gridu%np,kv_w+gridu%np)+  mu/hx**2

      ! d^2v/dy^2

      K(kv+gridu%np,kv_n+gridu%np)=K(kv+gridu%np,kv_n+gridu%np)+  mu/hy**2
      K(kv+gridu%np,kv  +gridu%np)=K(kv+gridu%np,kv  +gridu%np)-2*mu/hy**2
      K(kv+gridu%np,kv_s+gridu%np)=K(kv+gridu%np,kv_s+gridu%np)+  mu/hy**2

      rhs(gridu%np+kv)=-rho*gy

   end if

end do

!==============================================!
! taking care of pressure corners
!==============================================!

iel=0
do j=1,gridb%nely
do i=1,gridb%nelx
   iel=iel+1

   if (i==1 .and. j==1) then
      kp_e=(j-1)*gridb%nelx+i + 1
      !print *,i,j,iel,kp_e
      K(N_v+iel,N_v+iel)=1
      K(N_v+iel,N_v+kp_e)=-1
   end if

   if (i==1 .and. j==gridb%nely) then
      kp_e=(j-1)*gridb%nelx+i + 1
      !print *,i,j,iel,kp_e
      K(N_v+iel,N_v+iel)=1
      K(N_v+iel,N_v+kp_e)=-1
   end if

   if (i==gridb%nelx .and. j==1) then
      kp_w=(j-1)*gridb%nelx+i - 1 
      !print *,i,j,iel,kp_w
      K(N_v+iel,N_v+iel)=1
      K(N_v+iel,N_v+kp_w)=-1
   end if

   if (i==gridb%nelx .and. j==gridb%nely) then
      kp_w=(j-1)*gridb%nelx+i - 1 
      !print *,i,j,iel,kp_w
      K(N_v+iel,N_v+iel)=1
      K(N_v+iel,N_v+kp_w)=-1
   end if

end do
end do

!==============================================!
! apply boundary conditions
!==============================================!
! on u nodes

do ku=1,gridu%np
   if (gridu%bc(ku)) then
      K(ku,:)=0
      K(:,ku)=0
      K(ku,ku)=1
      rhs(ku)=0
   end if
end do

! on v nodes

do kv=1,gridv%np
   if (gridv%bc(kv)) then
      K(gridu%np+kv,:)=0
      K(:,gridu%np+kv)=0
      K(gridu%np+kv,gridu%np+kv)=1
      rhs(gridu%np+kv)=0
   end if
end do

!==============================================!
! test for symmetricity
!==============================================!

!do i=1,N
!   do j=1,N
!      if (abs(K(i,j)-K(j,i))>1.d-10) print *,i,j
!   end do
!end do

!==============================================!
!==============================================!

open(unit=123,file='OUT/K_matrix.dat',status='replace')
do i=1,N
   do j=1,N
      if (abs(K(i,j))>1.d-10) &
      write(123,*)  j,N-i+1,K(i,j)
   end do
end do
close(123)

open(unit=123,file='OUT/rhs_matrix.dat',status='replace')
do i=1,N
   write(123,*)  i,rhs(i)
end do
close(123)

!==============================================!
! solve system
!==============================================!
call cpu_time(t1)
job=0
allocate(work(N))
allocate(ipvt(N))
call DGEFA (K, N, N, ipvt, work)
call DGESL (K, N, N, ipvt, rhs, job)
deallocate(ipvt)
deallocate(work)
call cpu_time(t2)
print *,'solve time',t2-t1

gridu%field=rhs(1:gridu%np)
gridv%field=rhs(gridu%np+1:gridu%np+gridv%np)
gridp%field=rhs(N_v+1:N_v+gridp%np)

print *,'u',minval(gridu%field),maxval(gridu%field) 
print *,'v',minval(gridv%field),maxval(gridv%field) 
print *,'p',minval(gridp%field),maxval(gridp%field) 

!==============================================!
! transfer solution to background grid
!==============================================!

! pressure is easy

grids%p=gridp%field

do iel=1,gridb%nel

   grids%rho(iel)=sum(gridb%rho(gridb%icon(:,iel)))/m

   ! u velocity

   ku_w=gridb%icon(1,iel)
   ku_e=gridb%icon(2,iel)
   !print *,iel,ku_w,ku_e
   grids%u(iel)=0.5d0*(gridu%field(ku_w)+gridu%field(ku_e))

   ! v velocity

   kv_s=iel
   kv_n=iel+gridb%nelx
   !print *,iel,kv_s,kv_n
   grids%v(iel)=0.5d0*(gridv%field(kv_s)+gridv%field(kv_n))

   grids%divv(iel)= (gridu%field(ku_e)-gridu%field(ku_w))/hx &
                  + (gridv%field(kv_n)-gridv%field(kv_s))/hy 

   !write(234,*) grids_x(iel),grids_y(iel),grids_u(iel),grids_v(iel),grids_p(iel)
   !write(500,*) grids_x(iel),grids_y(iel),grids_rho(iel)

end do

!==============================================!

which = 0
call output_for_paraview (gridb,which) 

which = 1
call output_for_paraview (gridu,which) 

which = 2
call output_for_paraview (gridv,which) 

which = 3
call output_for_paraview (gridp,which) 

call output_for_paraview_visu (grids)  !np_s,nel_s,grids_x,grids_y,grids_icon,grids_rho,grids_u,grids_v,grids_p,grids_divv)

end program



