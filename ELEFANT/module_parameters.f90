module module_parameters
implicit none

integer :: mmapping            ! number of nodes for mapping 
integer :: mU,mV,mW            ! number of velocity nodes per element
integer :: mVel
integer :: mP                  ! number of pressure nodes per element
integer :: mT                  ! number of temperature nodes per element
integer :: nelx,nely,nelz      ! number of elements in each direction
integer :: nelr                ! number of elements in r direction (annulus)
integer :: neltheta            ! number of elements in theta direction (annulus,shell)
integer :: nelphi              ! number of elements in phi direction (annulus)
integer :: nel                 ! total number of elements
integer :: ndim                ! number of dimensions
integer :: nqpts               ! number of quadrature points (Q: per dim; T: per elt)  
integer :: nqel                ! number of quadrature points per element
integer :: ndofU,ndofV,ndofW   ! number of dofs per velocity node
integer :: NfemV               ! total number of velocity dofs
integer :: NfemP               ! total number of pressure dofs
integer :: NfemT               ! total number of temperature dofs
integer :: NU,NV,NW            ! total number of V velocity nodes
integer :: NP                  ! total number of pressure nodes
integer :: NT                  ! total number of temperature nodes
integer :: NQ                  ! total number of quadrature points
integer :: nmarker_per_dim     ! initial number of markers per dimension
integer :: nmarker             ! total number of markers
integer :: nmat                ! number of materials in the domain
integer :: nstep               ! number of time steps 
integer :: nproc               ! number of threads/processors 
integer :: ndim2               ! size of G_el (3 in 2D, 6 in 3D) 
integer :: nxstripes           ! nb of paint stripes in x direction
integer :: nystripes           ! nb of paint stripes in y direction
integer :: nzstripes           ! nb of paint stripes in z direction
logical :: solve_stokes_system ! whether the Stokes system is solved or not
logical :: use_swarm           ! whether markers are used or not
logical :: init_marker_random  ! whether markers are initally randomised
logical :: use_T               ! whether the code solves the energy equation
logical :: debug               ! triggers lots of additional checks & prints
logical :: use_penalty         ! whether the penalty formulation is used
logical :: use_ALE             ! whether the ALE (free surface) is used
logical :: normalise_pressure  ! 
logical :: isoparametric_mapping ! 



real(8) :: Lx,Ly,Lz            ! cartesian domain size
real(8) :: block_scaling_coeff ! scaling coefficient for the G block
real(8) :: penalty             ! penalty parameter
real(8) :: time                ! real human/model time
real(8) :: dt,dt_prev          ! time step 
real(8) :: CFL_nb
real(8) :: dparam1,dparam2,dparam3 
real(8) :: outer_radius,inner_radius 

character(len=10) :: geometry         ! type of domain geometry
character(len=6) :: cistep            ! istep parameter in string
character(len=4) :: spaceU            ! finite element space for velocity
character(len=4) :: spaceV            ! finite element space for velocity
character(len=4) :: spaceW            ! finite element space for velocity
character(len=4) :: spaceVelocity     ! finite element space for velocity
character(len=4) :: spacePressure     ! finite element space for pressure
character(len=4) :: spaceTemperature  ! finite element space for temperature
character(len=4) :: mapping           ! type of mapping 
character(len=6) :: inner_solver_type ! which type of solver for the inner solve 
character(len=6) :: outer_solver_type ! which type of solver for the outer solve 
character(len=6) :: bnd1_bcV_type     ! type of velocity b.c. on bnd 1
character(len=6) :: bnd2_bcV_type     ! type of velocity b.c. on bnd 2
character(len=6) :: bnd3_bcV_type     ! type of velocity b.c. on bnd 3
character(len=6) :: bnd4_bcV_type     ! type of velocity b.c. on bnd 4
character(len=6) :: bnd5_bcV_type     ! type of velocity b.c. on bnd 5
character(len=6) :: bnd6_bcV_type     ! type of velocity b.c. on bnd 6
character(len=10) :: K_storage,GT_storage,RHS_storage

integer :: iel
integer :: istep           
integer :: iproc
integer :: output_freq

contains

subroutine write_params
implicit none
write(*,'(a,3i10)')    ' ndim,ndim2,ndofV        =',ndim,ndim,ndofV
write(*,'(a,a11)')     ' geometry                =',geometry
write(*,'(a,3a10)')    ' spaceU,spaceV,spaceW    =',spaceU,spaceV,spaceW
write(*,'(a,2a10)')    ' spacePressure           =',spacePressure
write(*,'(a,2a10)')    ' spaceTemperature        =',spaceTemperature
write(*,'(a,a10,i10)') ' mapping,mmaping         =',mapping,mmapping
write(*,'(a,3f10.3)')  ' Lx,Ly,Lz                =',Lx,Ly,Lz
write(*,'(a,4i10)')    ' nelx,nely,nelz,nel      =',nelx,nely,nelz,nel
write(*,'(a,3i10)')    ' nelr,neltheta,nelphi    =',nelr,neltheta,nelphi
write(*,'(a,5i10)')    ' mU,mV,mW,mP,mT,mVel     =',mU,mV,mW,mP,mT,mVel
write(*,'(a,5i10)')    ' NU,NV,NW,NP,NT          =',NU,NV,NW,NP,NT
write(*,'(a,3i10)')    ' NfemV,NfemP,NfemT       =',NfemV,NfemP,NfemT
write(*,'(a,3i10)')    ' nqpts,nqel,NQ           =',nqpts,nqel,NQ
write(*,'(a,a10)')     ' inner_solver_type       =',inner_solver_type
write(*,'(a,a10)')     ' outer_solver_type       =',outer_solver_type
write(*,'(a,i10)')     ' nmat                    =',nmat
write(*,'(a,l10)')     ' use_penalty             =',use_penalty
write(*,'(a,es10.3)')  ' penalty                 =',penalty
write(*,'(a,l10)')     ' use_ALE                 =',use_ALE
write(*,'(a,l10)')     ' use_swarm               =',use_swarm
write(*,'(a,l10)')     ' use_T                   =',use_T
write(*,'(a,l10)')     ' isoparametric_mapping   =',isoparametric_mapping
write(*,'(a,l10)')     ' normalise_pressure      =',normalise_pressure  
write(*,'(a,2i10)')    ' nmarker_per_dim,nmarker =',nmarker_per_dim,nmarker
write(*,'(a,i10)')     ' nstep                   =',nstep
write(*,'(a,i10)')     ' nproc                   =',nproc
write(*,'(a,l10)')     ' debug                   =',debug
write(*,'(a,l10)')     ' solve_stokes_system     =',solve_stokes_system
write(*,'(a,l10)')     ' init_marker_random      =',init_marker_random

end subroutine

end module
