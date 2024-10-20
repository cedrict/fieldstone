module module_mesh
implicit none

type element
  integer, allocatable :: iconU(:)            ! connectivity array for x-velocity nodes
  integer, allocatable :: iconV(:)            ! connectivity array for y-velocity nodes
  integer, allocatable :: iconW(:)            ! connectivity array for z-velocity nodes
  integer, allocatable :: iconT(:)            ! connectivity array for temperature nodes 
  integer, allocatable :: iconP(:)            ! connectivity array for pressure nodes
  integer, allocatable :: iconM(:)            ! connectivity array for mapping nodes
  integer, allocatable :: iconVel(:)          ! connectivity array for all velocity dofs
  integer :: ielx,iely,ielz                   ! integer coords of the elt (Cartesian geom.)
  integer :: nmarker                          ! number of markers in element
  integer :: list_of_markers(200)             ! list of markers inside the element
  real(8), allocatable :: xU(:),yU(:),zU(:)   ! coordinates of x component of velocity nodes
  real(8), allocatable :: xV(:),yV(:),zV(:)   ! coordinates of y component of velocity nodes
  real(8), allocatable :: xW(:),yW(:),zW(:)   ! coordinates of z component of velocity nodes
  real(8), allocatable :: xT(:),yT(:),zT(:)   ! coordinates of temperature nodes
  real(8), allocatable :: xP(:),yP(:),zP(:)   ! coordinates of pressure nodes
  real(8), allocatable :: xM(:),yM(:),zM(:)   ! coordinates of mapping nodes
  real(8), allocatable :: u(:),v(:),w(:)      ! velocity degrees of freedom
  real(8), allocatable :: p(:)                ! pressure dofs
  real(8), allocatable :: T(:)                ! temperature degrees of freedom
  real(8), allocatable :: qx(:),qy(:),qz(:)   ! nodal heat flux vector
  real(8), allocatable :: rV(:),rP(:)         ! 
  real(8), allocatable :: thetaV(:),thetaP(:) !
  real(8), allocatable :: phiV(:),phiP(:)     !
  real(8), allocatable :: xvisu(:),yvisu(:),zvisu(:)   !
  real(8) :: exx,eyy,exy                      ! strain rate components for 2D
  real(8) :: ezz,exz,eyz                      ! additional strain rate components for 3D
  real(8) :: a_eta,b_eta,c_eta,d_eta          ! least square coeffs for viscosity
  real(8) :: a_rho,b_rho,c_rho,d_rho          ! least square coeffs for density
  real(8) :: vol                              ! volume of the element
  real(8) :: rho_avrg                         ! average density inside the element
  real(8) :: eta_avrg                         ! average viscosity inside the element
  real(8) :: xc,yc,zc                         ! coordinates of element center
  real(8) :: hx,hy,hz                         ! element size (Cartesian geom)
  real(8) :: hr,htheta,hphi
  logical(1) :: bnd1_elt                      ! true if element on x=0 boundary 
  logical(1) :: bnd2_elt                      ! true if element on x=Lx boundary 
  logical(1) :: bnd3_elt                      ! true if element on y=0 boundary 
  logical(1) :: bnd4_elt                      ! true if element on y=Ly boundary 
  logical(1) :: bnd5_elt                      ! true if element on z=0 boundary 
  logical(1) :: bnd6_elt                      ! true if element on z=Lz boundary 
  logical(1) :: inner_elt                     ! true if element is on inner annulus/sphere boundary
  logical(1) :: outer_elt                     ! true if element is on outer annulus/sphere boundary
  logical(1), allocatable :: bnd1_Unode(:)     ! flags for nodes on x=0 boundary  
  logical(1), allocatable :: bnd2_Unode(:)     ! flags for nodes on x=Lx boundary  
  logical(1), allocatable :: bnd3_Unode(:)     ! flags for nodes on y=0 boundary  
  logical(1), allocatable :: bnd4_Unode(:)     ! flags for nodes on y=Ly boundary  
  logical(1), allocatable :: bnd5_Unode(:)     ! flags for nodes on z=0 boundary  
  logical(1), allocatable :: bnd6_Unode(:)     ! flags for nodes on z=Lz boundary  
  logical(1), allocatable :: bnd1_Vnode(:)     ! flags for nodes on x=0 boundary  
  logical(1), allocatable :: bnd2_Vnode(:)     ! flags for nodes on x=Lx boundary  
  logical(1), allocatable :: bnd3_Vnode(:)     ! flags for nodes on y=0 boundary  
  logical(1), allocatable :: bnd4_Vnode(:)     ! flags for nodes on y=Ly boundary  
  logical(1), allocatable :: bnd5_Vnode(:)     ! flags for nodes on z=0 boundary  
  logical(1), allocatable :: bnd6_Vnode(:)     ! flags for nodes on z=Lz boundary  
  logical(1), allocatable :: bnd1_Wnode(:)     ! flags for nodes on x=0 boundary  
  logical(1), allocatable :: bnd2_Wnode(:)     ! flags for nodes on x=Lx boundary  
  logical(1), allocatable :: bnd3_Wnode(:)     ! flags for nodes on y=0 boundary  
  logical(1), allocatable :: bnd4_Wnode(:)     ! flags for nodes on y=Ly boundary  
  logical(1), allocatable :: bnd5_Wnode(:)     ! flags for nodes on z=0 boundary  
  logical(1), allocatable :: bnd6_Wnode(:)     ! flags for nodes on z=Lz boundary  
  logical(1), allocatable :: inner_Unode(:)   ! flags for nodes on inner boundary of annulus/shell
  logical(1), allocatable :: outer_Unode(:)   ! flags for nodes on outer boundary of annulus/shell
  logical(1), allocatable :: inner_Vnode(:)   ! flags for nodes on inner boundary of annulus/shell
  logical(1), allocatable :: outer_Vnode(:)   ! flags for nodes on outer boundary of annulus/shell
  logical(1), allocatable :: inner_Wnode(:)   ! flags for nodes on inner boundary of annulus/shell
  logical(1), allocatable :: outer_Wnode(:)   ! flags for nodes on outer boundary of annulus/shell
  logical(1), allocatable :: fix_u(:)         ! whether a given velocity dof is prescribed
  logical(1), allocatable :: fix_v(:)         ! whether a given velocity dof is prescribed
  logical(1), allocatable :: fix_w(:)         ! whether a given velocity dof is prescribed
  logical(1), allocatable :: fix_T(:)         ! whether a given temperature dof is prescribed
  real(8),allocatable :: xq(:),yq(:),zq(:)    ! coordinates of q. points inside elt
  real(8),allocatable :: JxWq(:)              ! jacobian*weight at q. point
  real(8),allocatable :: weightq(:)           ! weight of q. points
  real(8),allocatable :: rq(:),sq(:),tq(:)    ! reduced coordinates of q. points
  real(8),allocatable :: gxq(:),gyq(:),gzq(:) ! gravity vector at q. point
  real(8),allocatable :: pq(:),tempq(:)      ! pressure and temperature at q. points
  real(8),allocatable :: etaq(:),rhoq(:)      ! viscosity and density at q. points
  real(8),allocatable :: hcondq(:)            ! heat conductivity at q. points 
  real(8),allocatable :: hcapaq(:)            ! heat capacity at q. points 
  real(8),allocatable :: hprodq(:)            ! heat productivity at q. points 
end type element

type(element), dimension(:), allocatable :: mesh

end module
