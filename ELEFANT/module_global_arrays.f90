module module_arrays
implicit none

real(8), dimension(:), allocatable :: rhs_f,rhs_h,rhs_b
real(8), dimension(:), allocatable :: solV,solP
real(8), dimension(:), allocatable :: Kdiag

real(8), dimension(:,:), allocatable :: Cmat
real(8), dimension(:,:), allocatable :: Kmat

real(8), dimension(:), allocatable :: rV,sV,tV

end module
