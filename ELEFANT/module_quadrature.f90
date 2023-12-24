module module_quadrature
implicit none
!-------------------------------------------------------------------
real(8), parameter:: qc2a=1.d0/sqrt(3d0)
real(8), parameter:: qw2a=1.d0
!-------------------------------------------------------------------
real(8), parameter:: qc3a=sqrt(3d0/5d0)
real(8), parameter:: qc3b=0d0
real(8), parameter:: qw3a=5d0/9d0
real(8), parameter:: qw3b=8d0/9d0
!-------------------------------------------------------------------
real(8), parameter:: qc4a=sqrt(3./7.+2./7.*sqrt(6./5.))
real(8), parameter:: qc4b=sqrt(3./7.-2./7.*sqrt(6./5.))
real(8), parameter:: qw4a=(18-sqrt(30.))/36.
real(8), parameter:: qw4b=(18+sqrt(30.))/36
!-------------------------------------------------------------------
real(8), parameter :: qc5a=sqrt(5.d0+2.d0*sqrt(10.d0/7.d0))/3.d0  
real(8), parameter :: qc5b=sqrt(5.d0-2.d0*sqrt(10.d0/7.d0))/3.d0  
real(8), parameter :: qc5c=0.d0            
real(8), parameter :: qw5a=(322.d0-13.d0*sqrt(70.d0))/900.d0
real(8), parameter :: qw5b=(322.d0+13.d0*sqrt(70.d0))/900.d0
real(8), parameter :: qw5c=128.d0/225.d0
!-------------------------------------------------------------------
real(8), parameter :: qc6a=0.932469514203152
real(8), parameter :: qc6b=0.661209386466265
real(8), parameter :: qc6c=0.238619186083197
real(8), parameter :: qw6a=0.171324492379170
real(8), parameter :: qw6b=0.360761573048139
real(8), parameter :: qw6c=0.467913934572691
!-------------------------------------------------------------------
real(8), parameter :: qc7a=0.9491079123427585
real(8), parameter :: qc7b=0.7415311855993945
real(8), parameter :: qc7c=0.4058451513773972
real(8), parameter :: qc7d=0.0000000000000000
real(8), parameter :: qw7a=0.1294849661688697
real(8), parameter :: qw7b=0.2797053914892766
real(8), parameter :: qw7c=0.3818300505051189
real(8), parameter :: qw7d=0.4179591836734694
!-------------------------------------------------------------------
real(8), parameter :: qc8a=0.9602898564975363
real(8), parameter :: qc8b=0.7966664774136267
real(8), parameter :: qc8c=0.5255324099163290
real(8), parameter :: qc8d=0.1834346424956498
real(8), parameter :: qw8a=0.1012285362903763
real(8), parameter :: qw8b=0.2223810344533745
real(8), parameter :: qw8c=0.3137066458778873
real(8), parameter :: qw8d=0.3626837833783620
!-------------------------------------------------------------------
real(8), parameter :: qc9a=0.9681602395076261
real(8), parameter :: qc9b=0.8360311073266358
real(8), parameter :: qc9c=0.6133714327005904
real(8), parameter :: qc9d=0.3242534234038089
real(8), parameter :: qc9e=0.0000000000000000
real(8), parameter :: qw9a=0.0812743883615744
real(8), parameter :: qw9b=0.1806481606948574
real(8), parameter :: qw9c=0.2606106964029354
real(8), parameter :: qw9d=0.3123470770400029
real(8), parameter :: qw9e=0.3302393550012598
!-------------------------------------------------------------------
real(8), parameter :: qc10a=0.973906528517172
real(8), parameter :: qc10b=0.865063366688985
real(8), parameter :: qc10c=0.679409568299024
real(8), parameter :: qc10d=0.433395394129247
real(8), parameter :: qc10e=0.148874338981631
real(8), parameter :: qw10a=0.066671344308688
real(8), parameter :: qw10b=0.149451349150581
real(8), parameter :: qw10c=0.219086362515982
real(8), parameter :: qw10d=0.269266719309996
real(8), parameter :: qw10e=0.295524224714753
            
end module
