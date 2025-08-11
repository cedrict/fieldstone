!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!
                                               !
program fcubed                                 !
                                               !
implicit none                                  !

include 'mpif.h'                               !   
include 'dmumps_struc.h'                       ! 
                                               !
integer, parameter :: m=4                      ! number of nodes which constitute an element
integer, parameter :: ndofV=2                  ! number of velocity dofs per node
integer, parameter :: ndofT=1                  ! number of temperature dofs per node
integer nnx                                    ! number of grid points in the x direction
integer nny                                    ! number of grid points in the y direction
integer np                                     ! number of grid points
integer nelx                                   ! number of elements in the x direction
integer nely                                   ! number of elements in the y direction
integer nel                                    ! number of elements
integer NfemV                                  ! size of the FEM Stokes matrix 
integer NfemT                                  ! size of the FEM temperature matrix 
integer nstep                                  ! number of timesteps 
integer niter                                  ! number of nonlinear iterations
integer,dimension(:,:),allocatable :: icon     ! connectivity array
integer,dimension(:),  allocatable :: countnode!
integer,dimension(:),  allocatable :: mat      ! material id
                                               !
integer i1,i2,i,j,k,iel,counter,iq,jq,istep    ! integer variables needed 
integer idof,ik,ikk,jkk,m1,k1,k2               ! by the code
integer ierr,iii,inode,iproc,nproc,iter        !
integer LELTVAR,NA_ELT,counter_mumps,ii,ij     !
                                               !  
real(8) Lx,Ly                                  ! size of the numerical domain
real(8) mueff                                  ! effective dynamic viscosity 
real(8) gx,gy                                  ! gravity acceleration
real(8) penalty                                ! penalty parameter lambda
real(8), dimension(:),allocatable :: x,y       ! node coordinates arrays
real(8), dimension(:),allocatable :: u,v       ! node velocity arrays
real(8), dimension(:),allocatable :: press     ! pressure 
real(8), dimension(:),allocatable :: exx       ! strainrate tensor xx term 
real(8), dimension(:),allocatable :: eyy       ! strainrate tensor yy term 
real(8), dimension(:),allocatable :: exy       ! strainrate tensor xy term 
real(8), dimension(:),allocatable :: bc_valV   ! array containing bc values for velocity
real(8), dimension(:),allocatable :: bc_valT   ! array containing bc values for temperature
real(8), dimension(:),allocatable :: T         ! node temperature array
real(8), dimension(:),allocatable :: p         ! node pressure array
real(8), dimension(:),allocatable :: density   ! element density array
real(8), dimension(:),allocatable :: qx,qy     ! heat flux vector 
real(8), dimension(:),allocatable :: Tavrg     !
real(8), dimension(:),allocatable :: viscosity ! element density array
real(8), dimension(:),allocatable :: vpmode    ! 
real(8), dimension(:),allocatable :: Res       ! residual vector 
                                               !
real(8) rq,sq,wq                               ! local coordinate and weight of qpoint
real(8) xq,yq                                  ! global coordinate of qpoint
real(8) uq,vq                                  ! velocity at qpoint
real(8) pq,Tq                                  ! pressure and temperature at qpoint
real(8) exxq,eyyq,exyq                         ! strain-rate components at qpoint  
real(8) AelV(m*ndofV,m*ndofV)                  ! elemental Stokes FEM matrix
real(8) BelV(m*ndofV)                          ! elemental Stokes right hand side
real(8) AelT(m*ndofT,m*ndofT)                  ! elemental temperature FEM matrix
real(8) BelT(m*ndofT)                          ! elemental temperature right hand side
real(8) N(m),dNdx(m),dNdy(m),dNdr(m),dNds(m)   ! shape fcts and derivatives
real(8) jcob                                   ! determinant of jacobian matrix
real(8) jcb(2,2)                               ! jacobian matrix
real(8) jcbi(2,2)                              ! inverse of jacobian matrix
real(8) BmatV(3,ndofV*m)                       ! B matrix
real(8), dimension(3,3) :: Kmat                ! K matrix 
real(8), dimension(3,3) :: Cmat                ! C matrix
real(8) Aref                                   !
real(8) hcapa                                  ! heat capacity
real(8) hcond                                  ! heat conductivity
real(8) time                                   ! real time
real(8) dt                                     ! timestep
real(8) CFL_nb                                 ! Courant number for CFL criterion
real(8) rtol                                   ! relative convergence tolerance
real(8) atol                                   ! absolute convergence tolerance
real(8), parameter :: theta = 0.5              ! mid-point timestepping parameter
real(8), parameter :: eps=1.d-10               !
real(8), parameter :: pi = 3.14159265359d0     !
real(8), parameter :: year=31536000.d0         !
real(8), parameter :: cm=1.d-2                 !
real(8) M_T(4,4),Ka(4,4),Kc(4,4),KK(4,4)       ! various FEM arrays
real(8) Nvect(1,4),NvectT(4,1)                 !
real(8) velnorm,vel2D(1,2),temp(4)
real(8) Vel(m*ndofV),ResEl(m*ndofV)
real(8) BmatT(2,4),BmatTT(4,2)                 !
real(8) hx,hy                                  ! grid spacing
real(8) umax,vmax,rho                          !
real(8) fixt,mode                              !
real(8) Rknorm,R0norm
                                               !
logical, dimension(:), allocatable :: bc_fixV  ! prescribed b.c. array for velocities
logical, dimension(:), allocatable :: bc_fixT  ! prescribed b.c. array for temperature
                                               !
type(dmumps_struc) idV                         !
type(dmumps_struc) idT                         !
                                               !
!==============================================!
                                               !
CALL mpi_init(ierr)                            !  
call mpi_comm_size (mpi_comm_world,nproc,ierr) !
call mpi_comm_rank (mpi_comm_world,iproc,ierr) !
                                               !
!==============================================!
! initialise MUMPS                             !
!==============================================!

idV%COMM = MPI_COMM_WORLD                      ! Define a communicator for the package 
idV%SYM = 1                                    ! Ask for symmetric matrix storage 
idV%par=1                                      ! Host working 
idV%JOB = -1                                   ! Initialize an instance of the package 
call DMUMPS(idV)                               ! MUMPS initialisation

idT%COMM = MPI_COMM_WORLD                      ! Define a communicator for the package 
idT%SYM = 0                                    ! Ask for symmetric matrix storage 
idT%par=1                                      ! Host working 
idT%JOB = -1                                   ! Initialize an instance of the package 
call DMUMPS(idT)                               ! MUMPS initialisation

!==============================================!
!=====[Basic parameters]=======================!
!==============================================!

Lx=400.d3
Ly=100.d3

nelx=400
nely=nelx/4

gx=0
gy=-9.81d0

rtol=1.d-12
atol=1.d-20
niter=200

hcapa=750 ! heat capacity of all materials 
hcond=2.5 ! heat conductivity of all materials

!==============================================!
!==============================================!

penalty=1.d30 ! see section 3.2 of thieulot, PEPI, 2011

nstep=1

CFL_nb=0.0

nnx=nelx+1
nny=nely+1

time=0.d0

nel=nelx*nely ! total number of elements

np=nnx*nny ! total number of nodes

NfemV=np*ndofV ! size of Stokes matrix
NfemT=np*ndofT ! size of temperature matrix

hx=Lx/(nnx-1) ! grid spacing in x direction
hy=Ly/(nny-1) ! grid spacing in y direction

!==============================================!

Kmat(1,1)=1.d0 ; Kmat(1,2)=1.d0 ; Kmat(1,3)=0.d0  
Kmat(2,1)=1.d0 ; Kmat(2,2)=1.d0 ; Kmat(2,3)=0.d0  
Kmat(3,1)=0.d0 ; Kmat(3,2)=0.d0 ; Kmat(3,3)=0.d0  

Cmat(1,1)=2.d0 ; Cmat(1,2)=0.d0 ; Cmat(1,3)=0.d0  
Cmat(2,1)=0.d0 ; Cmat(2,2)=2.d0 ; Cmat(2,3)=0.d0  
Cmat(3,1)=0.d0 ; Cmat(3,2)=0.d0 ; Cmat(3,3)=1.d0

!==============================================!

print *,'========================================='
print *,'Lx=',Lx
print *,'Ly=',Ly
print *,'hx=',hx
print *,'hy=',hy
print *,'nelx',nelx
print *,'nely',nely
print *,'nnx',nnx
print *,'nny',nny
print *,'gx=',gx
print *,'gy=',gy
print *,'NfemV=',NfemV
print *,'NfemT=',NfemT
print *,'========================================='

!==============================================!
!===[open files]===============================!
!==============================================!

open(unit=1000,file='OUT/velocity_stats.dat')
open(unit=1001,file='OUT/pressure_stats.dat')
open(unit=1002,file='OUT/temperature_stats.dat')
open(unit=1003,file='OUT/convergence_nl.dat')
open(unit=1004,file='OUT/viscosity_stats.dat')
open(unit=1005,file='OUT/residual_stats.dat')
open(unit=1020,file='OUT/mumps_info.dat')

!==============================================!
!===[allocate memory]==========================!
!==============================================!

allocate(x(np))
allocate(y(np))
allocate(T(np))
allocate(u(np)) ; u=0.d0
allocate(v(np)) ; v=0.d0
allocate(p(np)) ; p=0.d0
allocate(icon(m,nel))
allocate(bc_fixV(NfemV))
allocate(bc_fixT(NfemT))
allocate(bc_valV(NfemV))
allocate(bc_valT(NfemT))
allocate(press(nel))
allocate(mat(nel))
allocate(vpmode(nel))
allocate(exx(nel))
allocate(eyy(nel))
allocate(exy(nel))
allocate(density(nel))
allocate(viscosity(nel))
allocate(qx(nel),qy(nel))
allocate(Tavrg(nny))
allocate(Res(np*ndofV))

!==============================================!
!===[grid points setup]========================!
!==============================================!

counter=0
do j=0,nely
   do i=0,nelx
      counter=counter+1
      x(counter)=dble(i)*hx
      y(counter)=dble(j)*hy
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
! MUMPS arrays for idV
!==============================================!

idV%N=NfemV                                    ! total number of degrees of freedom, size of FEM matrix
idV%NELT=nel                                   ! number of elements
LELTVAR=nel*(m*ndofV)                          ! nb of elts X size of elemental matrix
NA_ELT=nel*(m*ndofV)*(m*ndofV+1)/2             ! nb of elts X nb of nbs in elemental matrix !NEW

allocate(idV%A_ELT (NA_ELT))
allocate(idV%RHS   (idV%N))

if (iproc==0) then

allocate(idV%ELTPTR(idV%NELT+1))
allocate(idV%ELTVAR(LELTVAR))

do i=1,nel                                     !
   idV%ELTPTR(i)=1+(i-1)*(ndofV*m)             ! building ELTPTR array
end do                                         !
idV%ELTPTR(i)=1+nel*(ndofV*m)                  !

counter=0                                      !
do iel=1,nel                                   !
   do k=1,m                                    !
      inode=icon(k,iel)                        !
      do idof=1,ndofV                          !
         iii=(inode-1)*ndofV+idof              !
         counter=counter+1                     !
         idV%ELTVAR(counter)=iii               ! building ELTVAR
      end do                                   !
   end do                                      !
end do                                         !

end if ! iproc

idV%ICNTL(3) = 6                               !
idV%ICNTL(4) = 1                               !

!==============================================!
! MUMPS arrays for idT
!==============================================!

idT%N=NfemT                                    ! total number of degrees of freedom, size of FEM matrix
idT%NELT=nel                                   ! number of elements
LELTVAR=nel*m                                  ! nb of elts X size of elemental matrix
NA_ELT=nel*m*m                                 ! nb of elts X nb of nbs in elemental matrix !NEW

allocate(idT%A_ELT (NA_ELT))
allocate(idT%RHS   (idT%N))

if (iproc==0) then

allocate(idT%ELTPTR(idT%NELT+1))
allocate(idT%ELTVAR(LELTVAR))

do i=1,nel                                     !
   idT%ELTPTR(i)=1+(i-1)*m                     ! building ELTPTR array
end do                                         !
idT%ELTPTR(nel+1)=1+nel*m                      !

counter=0                                      !
do iel=1,nel                                   !
   do k=1,m                                    !
      inode=icon(k,iel)                        !
      counter=counter+1                        !
      idT%ELTVAR(counter)=inode                ! building ELTVAR
   end do                                      !
end do                                         !

end if ! iproc

idT%ICNTL(3) = 1020                            !
idT%ICNTL(4) = 0                               !

!==============================================!
!=====[material geometry]======================!
!==============================================!

call material_layout(np,nel,x,y,icon,mat)

!==============================================!
!=====[define bc for velocity]=================!
!==============================================!

call define_bc(x,y,np,bc_fixV,bc_valV,Lx,Ly)

!==============================================!
!=====[Initial temperature field]==============!
!==============================================!

call temperature_layout(np,x,y,T)

!********************************************************************************************
!********************************************************************************************

do istep=1,nstep

!==============================================!
!=====[build FE matrix]========================!
!==============================================!

do iter=1,niter ! nonlinear iterations loop

Res=0.d0
density=0.d0
vpmode=0.d0
viscosity=0.d0
idV%RHS=0.d0
idV%A_ELT=0.d0
counter_mumps=0

do iel=1,nel

   AelV=0.d0
   BelV=0.d0
   Vel=(/u(icon(1,iel)),v(icon(1,iel)),&
         u(icon(2,iel)),v(icon(2,iel)),&
         u(icon(3,iel)),v(icon(3,iel)),&
         u(icon(4,iel)),v(icon(4,iel))/)

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
      pq=0.d0
      Tq=0.d0
      exxq=0.d0
      eyyq=0.d0
      exyq=0.d0
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         Tq=Tq+N(k)*T(icon(k,iel))
         pq=pq+N(k)*p(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
         exxq=exxq+ dNdx(k)*u(icon(k,iel))
         eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
         exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
                  + dNdy(k)*u(icon(k,iel)) *0.5d0
      end do

      do i=1,m
         i1=2*i-1
         i2=2*i
         BmatV(1,i1)=dNdx(i) ; BmatV(1,i2)=0.d0
         BmatV(2,i1)=0.d0    ; BmatV(2,i2)=dNdy(i)
         BmatV(3,i1)=dNdy(i) ; BmatV(3,i2)=dNdx(i)
      end do

      call material_model(xq,yq,pq,Tq,exxq,eyyq,exyq,mat(iel),mueff,rho,mode) !; print *,viscosity,rho

      density(iel)=density(iel)+rho/4.d0
      viscosity(iel)=viscosity(iel)+mueff/4.d0
      vpmode(iel)=vpmode(iel)+mode/4.d0

      AelV=AelV + matmul(transpose(BmatV),matmul(mueff*Cmat,BmatV))*wq*jcob

      do i=1,m
      i1=2*i-1
      i2=2*i
      BelV(i1)=BelV(i1)+N(i)*jcob*wq*rho*gx
      BelV(i2)=BelV(i2)+N(i)*jcob*wq*rho*gy
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
         BmatV(1,i1)=dNdx(i) ; BmatV(1,i2)=0.d0
         BmatV(2,i1)=0.d0    ; BmatV(2,i2)=dNdy(i)
         BmatV(3,i1)=dNdy(i) ; BmatV(3,i2)=dNdx(i)
      end do

      AelV=AelV + matmul(transpose(BmatV),matmul(penalty*Kmat,BmatV))*wq*jcob


      !======================================
      !=====[impose boundary conditions]=====
      !======================================

      do ii=1,m
         inode=icon(ii,iel)
         do k=1,ndofV
            ij=(inode-1)*ndofV+k
            if (bc_fixV(ij)) then
            fixt=bc_valV(ij)
            i=(ii-1)*ndofV+k
            Aref=AelV(i,i)
            do j=1,m*ndofV
               BelV(j)=BelV(j)-AelV(j,i)*fixt
               AelV(i,j)=0.d0
               AelV(j,i)=0.d0
            enddo
            AelV(i,i)=Aref
            BelV(i)=Aref*fixt
            endif
         enddo
      enddo

      !======================================
      !=====[compute elemental residual]=====
      !======================================

      ResEl=Belv-matmul(AelV,Vel)

      !=====================
      !=====[assemble]======
      !=====================

      do k1=1,m   
         ik=icon(k1,iel) 
         do i1=1,ndofV  
            ikk=ndofV*(k1-1)+i1    
            m1=ndofV*(ik-1)+i1    
            do k2=1,m    
               do i2=1,ndofV    
                  jkk=ndofV*(k2-1)+i2  
                  if (jkk>=ikk) then 
                     counter_mumps=counter_mumps+1   
                     idV%A_ELT(counter_mumps)=AelV(ikk,jkk) 
                  end if    
               end do    
            end do    
            idV%RHS(m1)=idV%RHS(m1)+BelV(ikk)    
            Res(m1)=Res(m1)+ResEl(ikk)
         end do    
      end do    

end do

Rknorm=sqrt(sum(Res**2))
R0norm=sqrt(sum(idV%RHS**2))

write(*,'(a,i4,a,es12.5,a,es12.5)') 'iteration:',iter,' | Res(k)/Res(0):',Rknorm/R0norm,' | tol:',rtol

write(1003,*) iter,Rknorm/R0norm,rtol
write(1004,*) iter,minval(viscosity),maxval(viscosity)
write(1005,*) iter,minval(Res),maxval(Res),Rknorm

!idV%RHS=Res ! defect correction

!==============================================!
!=====[solve system]===========================!
!==============================================!

idV%ICNTL(5) = 1                               ! elemental format

idV%JOB = 1                                    ! analysis phase 
CALL DMUMPS(idV)

idV%JOB = 2                                    ! factorisation phase 
CALL DMUMPS(idV)

idV%JOB = 3                                    ! solve phase
CALL DMUMPS(idV)

!==============================================!
!=====[transfer solution]======================!
!==============================================!

do i=1,np
   !u(i)=u(i)+idV%RHS((i-1)*ndofV+1) ! defect correction
   !v(i)=v(i)+idV%RHS((i-1)*ndofV+2)
   u(i)=idV%RHS((i-1)*ndofV+1)
   v(i)=idV%RHS((i-1)*ndofV+2)
end do

!write(*,*) 'min/max u',minval(u)*year/cm,maxval(u)*year/cm
!write(*,*) 'min/max v',minval(v)*year/cm,maxval(v)*year/cm

write(1000,*) time,minval(u),maxval(u),minval(v),maxval(v) ; call flush(1000)

!==============================================!
!=====[retrieve pressure]======================!
!==============================================!

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
   exyq=0.d0
   do k=1,m
      xq=xq+N(k)*x(icon(k,iel))
      yq=yq+N(k)*y(icon(k,iel))
      exxq=exxq+ dNdx(k)*u(icon(k,iel))
      eyyq=eyyq+ dNdy(k)*v(icon(k,iel))
      exyq=exyq+ dNdx(k)*v(icon(k,iel)) *0.5d0 &
               + dNdy(k)*u(icon(k,iel)) *0.5d0
   end do
   
   exx(iel)=exxq
   eyy(iel)=eyyq
   exy(iel)=exyq
   press(iel)=-penalty*(exxq+eyyq)

end do

!project elemental pressure onto nodes

allocate(countnode(np))
countnode=0
p=0
do iel=1,nel
do k=1,m
inode=icon(k,iel)
p(inode) = p(inode) + press(iel)
countnode(inode) = countnode(inode) + 1
end do
end do
p=p/countnode
deallocate(countnode)
write(1001,*) time,minval(press),maxval(press),minval(p),maxval(p)

!==============================================!
!=====[assess convergence]=====================!
!==============================================!

if (Rknorm < rtol*R0norm + atol) then
   write(*,*) 'Convergence reached'
   exit
end if

end do ! iter

!==============================================!
!=====[Compute timestep]=======================!
!==============================================!

umax=maxval(abs(u))
vmax=maxval(abs(v))

dt=CFL_nb*min(hx,hy)/max(umax,vmax)

time=time+dt

!==============================================!
!=====[Build temperature matrix]===============!
!==============================================!

idT%RHS=0.d0
idT%A_ELT=0.d0
counter_mumps=0

do iel=1,nel

   AelT=0.d0
   BelT=0.d0

   temp(1)=T(icon(1,iel))
   temp(2)=T(icon(2,iel))
   temp(3)=T(icon(3,iel))
   temp(4)=T(icon(4,iel))

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
      do k=1,m
         xq=xq+N(k)*x(icon(k,iel))
         yq=yq+N(k)*y(icon(k,iel))
         uq=uq+N(k)*u(icon(k,iel))
         vq=vq+N(k)*v(icon(k,iel))
         dNdx(k)=jcbi(1,1)*dNdr(k)+jcbi(1,2)*dNds(k)
         dNdy(k)=jcbi(2,1)*dNdr(k)+jcbi(2,2)*dNds(k)
      end do

      velnorm=sqrt(uq**2+vq**2)

      vel2D(1,1)=uq
      vel2D(1,2)=vq

      Nvect(1,:)=N(:)
      NvectT(:,1)=N(:)

      BmatT(1,:)=dNdx
      BmatT(2,:)=dNdy

      BmatTT=transpose(BmatT)

      call material_model(xq,yq,pq,Tq,exxq,eyyq,exyq,mat(iel),mueff,rho,mode) !; print *,viscosity,rho

      Ka=matmul(NvectT,matmul(vel2D,BmatT))*wq*jcob*rho*hcapa

      Kc=matmul(BmatTT,BmatT)*wq*jcob*hcond

      KK=(Ka+Kc)

      M_T=matmul(NvectT,Nvect)*wq*jcob*rho*hcapa

      AelT=AelT+(M_T+KK*theta*dt)

      BelT=BelT+matmul(M_T-KK*(1.d0-theta)*dt,temp(1:4))

   end do
   end do

   !=====================
   ! impose boundary conditions
   !=====================

   do i=1,m
      ik=icon(i,iel)
      if (bc_fixT(ik)) then
         Aref=AelT(i,i)
         do j=1,m
         BelT(j)=BelT(j)-AelT(j,i)*bc_valT(ik)
         AelT(i,j)=0.d0
         AelT(j,i)=0.d0
         enddo
         AelT(i,i)=Aref
         BelT(i)=Aref*bc_valT(ik)
      endif
   enddo

   !=====================
   !=====[assemble]======
   !=====================

   do k1=1,m   
      m1=icon(k1,iel) 
      do k2=1,m   
         counter_mumps=counter_mumps+1   
         idT%A_ELT(counter_mumps)=AelT(k2,k1) 
      end do    
      idT%RHS(m1)=idT%RHS(m1)+BelT(k1)    
   end do    

end do ! end of loop over cells

!==============================================!
!=====[solve system]===========================!
!==============================================!

idT%ICNTL(5) = 1                               ! elemental format

idT%JOB = 1                                    ! analysis phase 
CALL DMUMPS(idT)

idT%JOB = 2                                    ! factorisation phase 
CALL DMUMPS(idT)

idT%JOB = 3                                    ! solve phase
CALL DMUMPS(idT)

!==============================================!
!=====[transfer solution]======================!
!==============================================!

T=idT%RHS

!write(*,*) 'min/max T',minval(T),maxval(T)

write(1002,*) time,minval(T),maxval(T)

!==============================================!
!=====[output data in vtu format]==============!
!==============================================!

call output_for_paraview (np,nel,x,y,u*year/cm,v*year/cm,press,p,T,&
                          density,viscosity,exx,eyy,exy,mat,vpmode,icon,istep)

call output_profiles(np,nelx,nely,x,y,T,mat,density,viscosity,exx,eyy,exy,press,vpmode,icon)

call output_surface(np,nel,x,y,viscosity,exx,eyy,exy,press,icon)

end do ! timestepping

!********************************************************************************************
!********************************************************************************************

print *,'***********************************************************'

end program




