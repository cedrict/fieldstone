program opla
use mpi

implicit none

integer i,nnx,nny,np,nelx,nely,nel,counter,j,ielx,iely,ierr,iproc,nproc,k
integer, parameter :: m=4 ! nb of nodes per mesh cell
integer, parameter :: swarm_n=10000000 ! nb of markers
real(8), dimension(swarm_n) :: swarm_x,swarm_y,swarm_u,swarm_v,xtemp,ytemp
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

call mpi_init(ierr)
call mpi_comm_size (mpi_comm_world,nproc,ierr)
call mpi_comm_rank (mpi_comm_world,iproc,ierr)

!==============================================!
! mesh parameters
!==============================================!

nnx=17
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
call cpu_time(start)

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

!==============================================!
!===[assign velocity to nodes]=================!
!==============================================!

do i=1,np
   u(i)=-(0.5-y(i))/10
   v(i)=(0.5-x(i))/10
   write(124,*) x(i),y(i),u(i),v(i)
end do

!==============================================!
!===[localise & advect swarm]==================!
!==============================================!

start=MPI_Wtime()

xtemp=0.d0
ytemp=0.d0

do i=1+iproc,swarm_n,nproc

   !print *,'iproc=',iproc,'takes care of marker',i

   do k=1,4 ! to mimic cost of RK4

      ! find cell
      ielx=swarm_x(i)/Lx*nelx+1
      iely=swarm_y(i)/Ly*nely+1
      iel(i)=nelx*(iely-1)+ielx 

      ! find local coordinates in element
      xmin=x(icon(1,iel(i)))
      xmax=x(icon(3,iel(i)))
      ymin=y(icon(1,iel(i)))
      ymax=y(icon(3,iel(i)))
      r=((swarm_x(i)-xmin)/(xmax-xmin)-0.5d0)*2.d0
      s=((swarm_y(i)-ymin)/(ymax-ymin)-0.5d0)*2.d0

      ! evaluate Q1 basis functions
      N(1)=0.25*(1-r)*(1-s)
      N(2)=0.25*(1+r)*(1-s)
      N(3)=0.25*(1+r)*(1+s)
      N(4)=0.25*(1-r)*(1+s)

      ! compute velocity
      swarm_u(i)=sum(N*u(icon(:,iel(i))))
      swarm_v(i)=sum(N*v(icon(:,iel(i))))

      ! advect marker
      xtemp(i)=swarm_x(i)+swarm_u(i)*dt
      ytemp(i)=swarm_y(i)+swarm_v(i)*dt

   end do
end do

finish=MPI_Wtime()

print *,'advect:',finish-start,iproc
call mpi_barrier(mpi_comm_world,ierr)

start  = MPI_Wtime()

call mpi_allreduce(xtemp,swarm_x,swarm_n,mpi_double_precision,mpi_sum,mpi_comm_world,ierr)
call mpi_allreduce(ytemp,swarm_y,swarm_n,mpi_double_precision,mpi_sum,mpi_comm_world,ierr)

finish  = MPI_Wtime()
   
!write(123,*) swarm_x(i),swarm_y(i),ielx,iely,iel(i),r,s,swarm_u(i),swarm_v(i)

print *,'allreduce:',finish-start,iproc

!print *,iproc,'|',xtemp
call mpi_barrier(mpi_comm_world,ierr)
!print *,xm

call MPI_Finalize(ierr)

end program 
