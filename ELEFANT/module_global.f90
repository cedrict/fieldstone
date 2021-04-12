module global_parameters
implicit none

integer :: mV                  ! number of velocity nodes per element
integer :: mP                  ! number of pressure nodes per element
integer :: mT                  ! number of temperature nodes per element
integer :: nelx,nely,nelz      ! number of elements in each direction
integer :: nel                 ! total number of elements
real(8) :: Lx,Ly,Lz            ! cartesian domain size
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
logical :: solve_stokes_system ! whether the Stokes system is solved or not
logical :: use_markers         ! whether markers are used or not
logical :: init_marker_random  ! whether markers are initally randomised
real(8) :: block_scaling_coeff ! scaling coefficient for the G block

character(len=12) :: geometry  ! type of domain geometry
character(len=4) :: pair       ! type of element pair

integer :: iel
integer :: istep           
integer :: iproc
end module
