
!==============================================
!==============================================

module structures

type grid
   integer nnx,nny,np
   integer nelx,nely,nel
   real(8), dimension(:), allocatable :: x,y
   real(8), dimension(:), allocatable :: u,v
   real(8), dimension(:), allocatable :: p
   real(8), dimension(:), allocatable :: rho
   integer, dimension(:,:), allocatable :: icon
   logical, dimension(:), allocatable :: bc
   real(8), dimension(:), allocatable :: field
   real(8), dimension(:), allocatable :: divv
end type

end module

