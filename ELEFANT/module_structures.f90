module structures
implicit none

type element
   integer :: iconV(27)                       ! connectivity array for velocity dofs
   integer :: iconT(27)                        ! connectivity array for temperature dofs
   integer :: iconP(8)                        ! connectivity array for pressure dofs
   integer :: ielx,iely,ielz                  ! integer coords of the elt (Cartesian geom.)
   integer :: nmarker                         ! number of markers in element
   integer :: list_of_markers(200)            ! list of markers inside the element
   real(8) :: xV(27),yV(27),zV(27)            ! coordinates of velocity nodes
   real(8) :: xT(27),yT(27),zT(27)            ! coordinates of temperature nodes
   real(8) :: xP(8),yP(8),zP(8)               ! coordinates of pressure nodes
   real(8) :: xc,yc,zc                        ! coordinates of element center
   real(8) :: hx,hy,hz                        ! element size (Cartesian geom)
   real(8) :: u(27),v(27),w(27)               ! velocity degrees of freedom
   real(8) :: p(8),q(8)                       ! pressure dofs and projected pressure q 
   real(8) :: T(8)                            ! temperature degrees of freedom
   real(8) :: exx(27),eyy(27),exy(27)         ! strain rate components for 2D
   real(8) :: ezz(27),exz(27),eyz(27)         ! additional strain rate components for 3D
   real(8) :: a_eta,b_eta,c_eta,d_eta         ! least square coeffs for viscosity
   real(8) :: a_rho,b_rho,c_rho,d_rho         ! least square coeffs for density
   real(8) :: vol                             ! volume/area of the element
   real(8) :: rho_avrg                        ! average density inside the element
   real(8) :: eta_avrg                        ! average viscosity inside the element
   logical(1) :: bnd1,bnd2                    ! true if element on x=0 or x=Lx boundary 
   logical(1) :: bnd3,bnd4                    ! true if element on y=0 or y=Ly boundary 
   logical(1) :: bnd5,bnd6                    ! true if element on z=0 or z=Lz boundary 
   logical(1) :: bnd1_node(27),bnd2_node(27)    ! flags for nodes on x=0 or x=Lx boundary  
   logical(1) :: bnd3_node(27),bnd4_node(27)    ! flags for nodes on y=0 or y=Ly boundary  
   logical(1) :: bnd5_node(27),bnd6_node(27)    ! flags for nodes on z=0 or z=Lz boundary  
   logical(1) :: fix_u(27),fix_v(27),fix_w(27)   ! whether a given velocity dof is prescribed
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
type(compressedrowstorage_sqr) csrA
end module




