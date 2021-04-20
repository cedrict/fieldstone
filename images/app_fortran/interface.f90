program kwadraat

implicit none

integer, parameter :: IntegerRoot = 6
real,    parameter :: RealRoot = 4.5

Interface Square  
    function RealSquare(root)
        real :: root
        real :: RealSquare
    end function

    function IntegerSquare(root)
        integer :: root
        integer :: IntegerSquare
    end function
end interface


write(*,*) "Integer square: ", Square(IntegerRoot)
write(*,*) "Real square:    ", Square(realRoot)


end program

function IntegerSquare(root)
    implicit none
    integer :: root, IntegerSquare
    IntegerSquare = root**2
end function

function RealSquare(root)
    implicit none
    real :: root, RealSquare
    RealSquare = root*root
end function
