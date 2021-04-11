module global_parameters
implicit none

integer :: mV,mP,mT
integer :: nelx,nely,nelz,nel,iel
real(8) :: Lx,Ly,Lz
integer :: ndim
character(len=12) :: geometry
character(len=4) :: pair
integer nq_per_dim,nqel
integer ndofV
integer NfemV,NfemP,NfemT
integer NV, NP, NT, Nq 
logical(1) use_markers
integer nmarker_per_dim
integer nmarker
logical(1) init_marker_random

integer counti,countf,count_rate
integer iproc,idummy
real(8) elapsed
end module

!----------------------------------------------------------

module constants
implicit none

real(8), parameter :: mm=1.d-3
real(8), parameter :: cm=1.d-2
real(8), parameter :: km=1.d+3
real(8), parameter :: hour=3600.d0
real(8), parameter :: day=86400.d0
real(8), parameter :: year=31536000.d0 
real(8), parameter :: Myr=3.1536d13 
real(8), parameter :: TKelvin=273.15d0 
real(8), parameter :: eps=1.d-8
real(8), parameter :: sqrt3 = 1.732050807568877293527446341505d0
real(8), dimension(2), parameter :: qcoords=(/-1.d0/sqrt3,+1.d0/sqrt3/)
real(8), dimension(2), parameter :: qweights=(/1.d0 , 1.d0 /)
integer, parameter :: one=1
integer, parameter :: two=2
integer, parameter :: three=3
integer, parameter :: four=4

end module constants

!----------------------------------------------------------

module structures
implicit none

type element
     integer :: iconV(5),iconT(4),iconP(4)
     integer :: ielx,iely,ielz
     integer :: nmarker
     integer :: list_of_markers(100)
     real(8) :: xV(5),yV(5),zV(5)
     real(8) :: xT(4),yT(4),zT(4)
     real(8) :: xP(4),yP(4),zP(4)
     real(8) :: xc,yc,zc
     real(8) :: hx,hy,hz
     real(8) :: u(5),v(5),w(5),p(4),T(4)
     real(8) :: a_eta,b_eta,c_eta,d_eta
     real(8) :: a_rho,b_rho,c_rho,d_rho
     logical(1) :: left,right,top,bottom
     logical(1) :: left_node(4),right_node(4),top_node(4),bottom_node(4)
     logical(1) :: fix_u(4),fix_v(4),fix_w(4),fix_T(4)
     real(8), dimension(:), allocatable :: xq,yq,zq,weightq,rq,sq,tq,gxq,gyq,gzq
     real(8), dimension(:), allocatable :: etaq,rhoq,hcondq,hcapaq,hprodq
     real(8) :: exx(4),eyy(4),ezz(4),exy(4),exz(4),eyz(4) 
end type element

type(element), dimension(:), allocatable :: mesh


type marker
     real(8)    :: x,y,z
     real(8)    :: r,s,t 
     real(4)    :: strain
     integer(1) :: mat 
     real(4)    :: paint
     logical(1) :: active
     integer    :: iel 
     real(8)    :: eta
     real(8)    :: rho 
     real(8)    :: hcond 
     real(8)    :: hcapa
     real(8)    :: hprod
end type

type(marker), dimension(:), allocatable :: swarm

end module
