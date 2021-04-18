module structures
implicit none

type element
   integer :: iconV(10)                       ! connectivity array for velocity dofs
   integer :: iconT(8)                        ! connectivity array for temperature dofs
   integer :: iconP(8)                        ! connectivity array for pressure dofs
   integer :: ielx,iely,ielz                  ! integer coords of the elt (Cartesian geom.)
   integer :: nmarker                         ! number of markers in element
   integer :: list_of_markers(100)            ! list of markers inside the element
   real(8) :: xV(10),yV(10),zV(10)            ! coordinates of velocity nodes
   real(8) :: xT(8),yT(8),zT(8)               ! coordinates of temperature nodes
   real(8) :: xP(8),yP(8),zP(8)               ! coordinates of pressure nodes
   real(8) :: xc,yc,zc                        ! coordinates of element center
   real(8) :: hx,hy,hz                        ! element size (Cartesian geom)
   real(8) :: u(10),v(10),w(10)               ! velocity degrees of freedom
   real(8) :: p(8),q(8)                       ! pressure dofs and projected pressure q 
   real(8) :: T(8)                            ! temperature degrees of freedom
   real(8) :: exx(8),eyy(8),exy(8)            ! strain rate components for 2D
   real(8) :: ezz(8),exz(8),eyz(8)            ! additional strain rate components for 3D
   real(8) :: a_eta,b_eta,c_eta,d_eta         ! least square coeffs for viscosity
   real(8) :: a_rho,b_rho,c_rho,d_rho         ! least square coeffs for density
   logical(1) :: left,right                   ! true if element on left or right boundary      
   logical(1) :: top,bottom                   ! true if element on left or right boundary      
   logical(1) :: front,back                   ! true if element on left or right boundary      
   logical(1) :: left_node(8),right_node(8)   ! flags for nodes on left or right boundaries
   logical(1) :: top_node(8),bottom_node(8)   ! flags for nodes on bottom or top boundaries
   logical(1) :: front_node(8),back_node(8)   ! flags for nodes on back or front boundaries
   logical(1) :: fix_u(8),fix_v(8),fix_w(8)   ! whether a given velocity dof is prescribed
   logical(1) :: fix_T(8)                     ! whether a given temperature dof is prescribed
   real(8),allocatable :: xq(:),yq(:),zq(:)   ! coordinates of q. points inside elt
   real(8),allocatable :: JxWq(:)             ! jacobian*weight at q. point
   real(8),allocatable :: weightq(:)          ! weight of q. points
   real(8),allocatable :: rq(:),sq(:),tq(:)   ! reduced coordinates of q. points
   real(8),allocatable :: gxq(:),gyq(:),gzq(:)! gravity vector at q. point
   real(8),allocatable :: pq(:),thetaq(:)     ! pressure and temperature at q. points
   real(8),allocatable :: etaq(:),rhoq(:)     ! viscosity and density at q. points
   real(8),allocatable :: hcondq(:)           ! 
   real(8),allocatable :: hcapaq(:)           !
   real(8),allocatable :: hprodq(:)           !
end type element
type(element), dimension(:), allocatable :: mesh

type marker
     real(8)    :: x,y,z
     real(8)    :: r,s,t 
     real(4)    :: strain
     integer(1) :: mat 
     real(4)    :: paint
     logical(1) :: active
     integer    :: iel 
     real(8)    :: eta
     real(8)    :: rho 
     real(8)    :: hcond 
     real(8)    :: hcapa
     real(8)    :: hprod
end type
type(marker), dimension(:), allocatable :: swarm

type material    
   real(8) rho0
   real(8) eta0 
   real(8) c,c_sw,phi,phi_sw           !
   real(8) alpha,T0                    ! thermal expansion coeff.
   real(8) hcapa                       ! heat capacity
   real(8) hcond                       ! heat conductivity
   real(8) hprod                       ! heat production coefficient
   real(8) A_diff,Q_diff,V_diff,f_diff !
   real(8) n_disl,A_disl,Q_disl,V_disl,f_disl !
   real(8) n_prls,A_prls,Q_prls,V_prls,f_prls
end type material 
type(material), dimension(:), allocatable :: mat
end module



!---------------------------------------------------------------------------
module matrix_storage

type compressedrowstorage_sqr    
   integer n                              ! size of square matrix 
   integer nz                             ! number of nonzeros
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
end type compressedrowstorage_sqr

type compressedrowstorage_rec    
   integer nr                
   integer nc                
   integer nz                             ! number of nonzeros
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
end type compressedrowstorage_rec

end module

!-----------------------------------------
module matrices
use matrix_storage
implicit none
type(compressedrowstorage_sqr) csrK
type(compressedrowstorage_rec) csrGT
type(compressedrowstorage_sqr) csrM
end module




