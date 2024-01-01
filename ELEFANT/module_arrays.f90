module module_arrays
implicit none

real(8), dimension(:), allocatable :: rhs_f,rhs_h,rhs_b
real(8), dimension(:), allocatable :: solV,solP
real(8), dimension(:), allocatable :: Kdiag

real(8), dimension(:,:), allocatable :: Cmat
real(8), dimension(:,:), allocatable :: Kmat

real(8), dimension(:), allocatable :: rU,sU,tU
real(8), dimension(:), allocatable :: rV,sV,tV
real(8), dimension(:), allocatable :: rW,sW,tW
real(8), dimension(:), allocatable :: rP,sP,tP
real(8), dimension(:), allocatable :: rT,sT,tT
real(8), dimension(:), allocatable :: rmapping,smapping,tmapping

real(8), dimension(:), allocatable :: NNNU,dNNNUdx,dNNNUdy,dNNNUdz
real(8), dimension(:), allocatable :: NNNV,dNNNVdx,dNNNVdy,dNNNVdz
real(8), dimension(:), allocatable :: NNNW,dNNNWdx,dNNNWdy,dNNNWdz
real(8), dimension(:), allocatable :: NNNT,dNNNTdx,dNNNTdy,dNNNTdz
real(8), dimension(:), allocatable :: NNNP

integer(4), dimension(:,:), allocatable :: Unode_belongs_to
integer(4), dimension(:,:), allocatable :: Vnode_belongs_to
integer(4), dimension(:,:), allocatable :: Wnode_belongs_to
integer(4), dimension(:,:), allocatable :: Pnode_belongs_to

end module
