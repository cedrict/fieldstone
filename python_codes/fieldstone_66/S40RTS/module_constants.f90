!==================================================================================================!
!  GGG   RRRR    AAA   PPPP   EEEEE                                                                !
! G      R   R  A   A  P   P  E                                                                    !
! G  G   RRRR   AAAAA  PPPP   EEE                                                                  !
! G   G  R   R  A   A  P      E                                                                    !
!  GGG   R   R  A   A  P      EEEEE                                                    C. Thieulot !
!==================================================================================================!

module constants

integer, parameter :: ndim=3
real(8), parameter :: Ggrav =6.6738480d-11
real(8), parameter :: pi  = 3.14159265358979323846264338327950288d0
real(8), parameter :: pi2  = pi*0.5d0
real(8), parameter :: pi4  = pi*0.25d0
real(8), parameter :: pi8  = pi*0.125d0
real(8), parameter :: fourpi = 4.d0*pi
real(8), parameter :: sqrt2 = 1.414213562373095048801688724209d0
real(8), parameter :: sqrt3 = 1.732050807568877293527446341505d0
real(8), parameter :: rcmb=3480.d3      ! radius of the core-mantle boundary
real(8), parameter :: rmoho=6346.d3     ! radius of the Moho
real(8), parameter :: rearth=6371.d3
real(8), parameter :: eotvos=1d-9       ! s^{-2}
real(8), parameter :: epsiloon=1.d-12 
real(8), parameter :: Mearth=5.9722d24  ! mass of the Eaeth
real(8), parameter :: mu0=fourpi*1.d-7 ! permeability of vacuum (H/m)
real(8), parameter :: mGal=0.01d-3 ! m/s^2 

integer, parameter :: xx=1
integer, parameter :: yy=2
integer, parameter :: zz=3
integer, parameter :: xy=4
integer, parameter :: xz=5
integer, parameter :: yz=6

integer, parameter :: buriedSphere = 1
integer, parameter :: buriedCylinder = 2
integer, parameter :: buriedBlock = 3
integer, parameter :: hollowSphere = 4

integer, parameter :: octree_refinement_surface=1
integer, parameter :: octree_refinement_box=2
integer, parameter :: octree_refinement_sphere=3

end module 


!==================================================================================================!
