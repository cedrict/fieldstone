module module_sparse 

type compressedrowstorage_sqr    
   integer N                              ! size of square matrix 
   integer NZ                             ! number of nonzeros
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
   logical full_matrix_storage
end type compressedrowstorage_sqr

type compressedrowstorage_rec    
   integer nr                
   integer nc                
   integer NZ
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   real(8),dimension(:),allocatable :: mat
end type compressedrowstorage_rec

type coordinatestorage
   integer NR
   integer NC
   integer NZ
   integer,dimension(:),allocatable :: ia  
   integer,dimension(:),allocatable :: ja
   integer,dimension(:),allocatable :: snr 
   integer,dimension(:),allocatable :: rnr
   real(8),dimension(:),allocatable :: mat
end type coordinatestorage

type(compressedrowstorage_sqr) csrK    
type(compressedrowstorage_rec) csrKxx,csrKxy,csrKxz
type(compressedrowstorage_rec) csrKyx,csrKyy,csrKyz
type(compressedrowstorage_rec) csrKzx,csrKzy,csrKzz
type(compressedrowstorage_rec) csrGT   
type(compressedrowstorage_rec) csrGxT 
type(compressedrowstorage_rec) csrGyT 
type(compressedrowstorage_rec) csrGzT 
type(compressedrowstorage_sqr) csrMV   ! velocity mass matrix
type(compressedrowstorage_sqr) csrMP   ! pressure mass matrix 
type(compressedrowstorage_sqr) csrS    ! Schur complement matrix 
type(compressedrowstorage_sqr) csrA    ! energy matrix

type(coordinatestorage) cooK 

end module
