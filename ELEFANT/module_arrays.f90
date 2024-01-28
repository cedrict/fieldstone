module module_arrays
implicit none

real(8), dimension(:), allocatable :: rhs_f,rhs_h,rhs_b,rhs_fx,rhs_fy,rhs_fz
real(8), dimension(:), allocatable :: solVel,solP,solU,solV,solW
real(8), dimension(:), allocatable :: Kdiag,Kxxdiag,Kyydiag,Kzzdiag

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

real(8), dimension(:,:), allocatable :: K_matrix
real(8), dimension(:,:), allocatable :: Kxx_matrix
real(8), dimension(:,:), allocatable :: Kxy_matrix
real(8), dimension(:,:), allocatable :: Kxz_matrix
real(8), dimension(:,:), allocatable :: Kyx_matrix
real(8), dimension(:,:), allocatable :: Kyy_matrix
real(8), dimension(:,:), allocatable :: Kyz_matrix
real(8), dimension(:,:), allocatable :: Kzx_matrix
real(8), dimension(:,:), allocatable :: Kzy_matrix
real(8), dimension(:,:), allocatable :: Kzz_matrix
real(8), dimension(:,:), allocatable :: GT_matrix

end module
