module module_arrays
implicit none

real(8), dimension(:), allocatable :: rhs_f,rhs_h,rhs_b
real(8), dimension(:), allocatable :: solV,solP
real(8), dimension(:), allocatable :: Kdiag

real(8), dimension(:,:), allocatable :: Cmat
real(8), dimension(:,:), allocatable :: Kmat

real(8), dimension(:), allocatable :: rV,sV,tV
real(8), dimension(:), allocatable :: rP,sP,tP
real(8), dimension(:), allocatable :: rT,sT,tT
real(8), dimension(:), allocatable :: rmapping,smapping,tmapping

integer(4), dimension(:,:), allocatable :: vnode_belongs_to
integer(4), dimension(:,:), allocatable :: pnode_belongs_to

end module
