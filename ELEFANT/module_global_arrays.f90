module global_arrays

implicit none

real(8), dimension(:), allocatable :: rhs_f,rhs_h
real(8), dimension(:), allocatable :: solV,solP,solQ
real(8), dimension(:), allocatable :: Kdiag

real(8), dimension(:,:), allocatable :: Cmat
real(8), dimension(:,:), allocatable :: Kmat

end module
