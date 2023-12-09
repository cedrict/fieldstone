module module_parameters
implicit none

integer :: mmapping            ! number of nodes for mapping 
integer :: mV                  ! number of velocity nodes per element
integer :: mP                  ! number of pressure nodes per element
integer :: mT                  ! number of temperature nodes per element
integer :: mL                  ! number of vertices/corners of basic shape
integer :: nelx,nely,nelz      ! number of elements in each direction
integer :: nel                 ! total number of elements
integer :: ndim                ! number of dimensions
integer :: nq_per_dim          ! number of quadrature points per dimension
integer :: nqel                ! number of quadrature points per element
integer :: ndofV               ! number of dofs per velocity node
integer :: NfemV               ! total number of velocity dofs
integer :: NfemP               ! total number of pressure dofs
integer :: NfemT               ! total number of temperature dofs
integer :: NV                  ! total number of velocity nodes
integer :: NP                  ! total number of pressure nodes
integer :: NT                  ! total number of temperature nodes
integer :: Nq                  ! total number of quadrature points
integer :: nmarker_per_dim     ! initial number of markers per dimension
integer :: nmarker             ! total number of markers
integer :: nmat                ! number of materials in the domain
integer :: ncorners            ! number of corners an element has
integer :: nstep               ! number of time steps 
integer :: nproc               ! number of threads/processors 
integer :: ndim2               ! size of G_el (3 in 2D, 6 in 3D) 
integer :: nxstripes           ! nb of paint stripes in x direction
integer :: nystripes           ! nb of paint stripes in y direction
integer :: nzstripes           ! nb of paint stripes in z direction
logical :: solve_stokes_system ! whether the Stokes system is solved or not
logical :: use_swarm           ! whether markers are used or not
logical :: init_marker_random  ! whether markers are initally randomised
logical :: use_MUMPS           ! whether MUMPS is used for inner solve 
logical :: use_T               ! whether the code solves the energy equation
logical :: debug               ! triggers lots of additional checks & prints
logical :: use_penalty         ! whether the penalty formulation is used
logical :: use_ALE             ! whether the ALE (free surface) is used
logical :: normalise_pressure  ! 

real(8) :: Lx,Ly,Lz            ! cartesian domain size
real(8) :: block_scaling_coeff ! scaling coefficient for the G block
real(8) :: penalty             ! penalty parameter
real(8) :: time                ! real human/model time
real(8) :: dt,dt_prev          ! time step 
real(8) :: CFL_nb
real(8) :: dparam1,dparam2,dparam3 

character(len=10) :: geometry  ! type of domain geometry
character(len=4) :: pair       ! type of element pair
character(len=6) :: cistep
character(len=4) :: spaceV     ! finite element space for velocity
character(len=4) :: spaceP     ! finite element space for pressure
character(len=4) :: spaceT     ! finite element space for temperature
character(len=4) :: mapping    ! type of mapping 

integer :: iel
integer :: istep           
integer :: iproc
integer :: output_freq
end module
