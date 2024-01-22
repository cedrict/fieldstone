module module_sparse 

type compressedrowstorage_sqr    
   integer n                              ! size of square matrix 
   integer nz                             ! number of nonzeros
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
   logical full_matrix_storage
   integer, dimension(:), allocatable :: snr,rnr
end type compressedrowstorage_sqr

type compressedrowstorage_rec    
   integer nr                
   integer nc                
   integer nz                             ! number of nonzeros
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
end type compressedrowstorage_rec

type(compressedrowstorage_sqr) csrK    ! (1,1) block of Stokes matrix
type(compressedrowstorage_sqr) csrKxx,csrKxy,csrKxz
type(compressedrowstorage_sqr) csrKyx,csrKyy,csrKyz
type(compressedrowstorage_sqr) csrKzx,csrKzy,csrKzz
type(compressedrowstorage_rec) csrGT   ! (2,1) block of Stokes matrix
type(compressedrowstorage_rec) csrGxT 
type(compressedrowstorage_rec) csrGyT 
type(compressedrowstorage_rec) csrGzT 
type(compressedrowstorage_sqr) csrMV   ! velocity mass matrix
type(compressedrowstorage_sqr) csrMP   ! pressure mass matrix 
type(compressedrowstorage_sqr) csrA    ! energy matrix

end module
