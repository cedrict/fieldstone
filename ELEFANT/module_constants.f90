module module_constants
implicit none
integer, parameter :: one=1
integer, parameter :: two=2
integer, parameter :: three=3
integer, parameter :: four=4
real(8), parameter :: mm=1.d-3
real(8), parameter :: cm=1.d-2
real(8), parameter :: km=1.d+3
real(8), parameter :: hour=3600.d0    !seconds
real(8), parameter :: day=86400.d0    !seconds
real(8), parameter :: year=31557600d0 !seconds 
real(8), parameter :: Myr=3.15576d13  !seconds
real(8), parameter :: TKelvin=273.15d0 
real(8), parameter :: eps=1.d-8
real(8), parameter :: epsilon_test=1.d-6
real(8), parameter :: sqrt2 = 1.414213562373095048801688724209d0
real(8), parameter :: sqrt3 = 1.732050807568877293527446341505d0
real(8), parameter :: Ggrav =6.6738480d-11
real(8), parameter :: pi  = 3.14159265358979323846264338327950288d0
real(8), parameter :: pi2  = pi*0.5d0
real(8), parameter :: pi4  = pi*0.25d0
real(8), parameter :: pi8  = pi*0.125d0
real(8), parameter :: twopi = 2d0*pi
real(8), parameter :: fourpi = 4d0*pi
real(8), parameter :: eotvos=1d-9
real(8), parameter :: mGal=0.01d-3 ! m/s^2
end module
