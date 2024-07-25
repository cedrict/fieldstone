program opla

implicit none

integer i,nnx,nny,np,nelx,nely,nel,counter,j,ielx,iely,k
integer, parameter :: m=4 ! nb of nodes per mesh cell
integer, parameter :: swarm_n=4000000 ! nb of markers
real(8), dimension(swarm_n) :: swarm_x,swarm_y,swarm_u,swarm_v
real(8), parameter :: Lx=1 ! domain size
real(8), parameter :: Ly=1 ! domain size
real(8), dimension(4):: N ! array of basis functions
real(8), dimension(:),   allocatable :: x,y    ! node coordinates arrays
real(8), dimension(:),   allocatable :: u,v    ! node coordinates arrays
integer, dimension(:,:), allocatable :: icon   ! connectivity array
integer, dimension(swarm_n):: iel   
real(8) r,s,xmin,xmax,ymin,ymax,dt
real(8) chi,eta,start,finish

!==============================================!
! mesh parameters
!==============================================!

nnx=128
nny=nnx
np=nnx*nny
nelx=nnx-1
nely=nny-1
nel=nelx*nely

dt=1e-3

!==============================================!
!===[allocate memory for mesh]=================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(icon(m,nel))
allocate(u(np))
allocate(v(np))

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

!==============================================!
!===[swarm setup]========================!
!==============================================!

do i=1,swarm_n
   call random_number(eta)
   call random_number(chi)
   swarm_x(i)=eta*Lx
   swarm_y(i)=chi*Ly
   !write(123,*) swarm_x(i),swarm_y(i)
end do

swarm_x=min(swarm_x,0.999999)
swarm_y=min(swarm_y,0.999999)
swarm_x=max(swarm_x,0.000001)
swarm_y=max(swarm_y,0.000001)

!==============================================!
!===[assign velocity to nodes]=================!
!==============================================!

do i=1,np
   u(i)=-(0.5-y(i))/10
   v(i)=(0.5-x(i))/10
   !write(124,*) x(i),y(i),u(i),v(i)
end do

!==============================================!
!===[localise & advect swarm]==================!
!==============================================!

call cpu_time(start)

!$omp parallel
!$omp do

do i=1,swarm_n

      !ielx=swarm_x(i)/Lx*nelx+1
      !iely=swarm_y(i)/Ly*nely+1

      do k=1,nel
         xmin=x(icon(1,k))
         xmax=x(icon(3,k))
         ymin=y(icon(1,k))
         ymax=y(icon(3,k))
         if (xmin<swarm_x(i) .and. &
             ymin<swarm_y(i) .and. &
             xmax>swarm_x(i) .and. &
             ymax>swarm_y(i) ) then

            iel(i)=k
            r=((swarm_x(i)-xmin)/(xmax-xmin)-0.5d0)*2.d0
            s=((swarm_y(i)-ymin)/(ymax-ymin)-0.5d0)*2.d0
         end if
      end do

      ! evaluate Q1 basis functions
      N(1)=0.25*(1-r)*(1-s)
      N(2)=0.25*(1+r)*(1-s)
      N(3)=0.25*(1+r)*(1+s)
      N(4)=0.25*(1-r)*(1+s)

      ! compute velocity
      swarm_u(i)=sum(N*u(icon(:,iel(i))))
      swarm_v(i)=sum(N*v(icon(:,iel(i))))

      ! advect marker
      swarm_x(i)=swarm_x(i)+swarm_u(i)*dt
      swarm_y(i)=swarm_y(i)+swarm_v(i)*dt

end do

!$omp end do
!$omp end parallel

call cpu_time(finish)
print '("Time = ",f6.3," seconds.")',finish-start

end program 
