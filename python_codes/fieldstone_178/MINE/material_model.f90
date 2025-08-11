!==============================================!
!                                              !
! C. thieulot ; October 2018                   !
!                                              !
!==============================================!

subroutine material_model(xq,yq,pq,Tq,exxq,eyyq,exyq,imat,mueffq,rhoq,mode)

implicit none

real(8), intent(in) :: xq,yq,exxq,eyyq,exyq,pq,Tq
integer, intent(in) :: imat
real(8), intent(out) :: mueffq,rhoq,mode

real(8), parameter :: pi =  3.14159265358979323846264338327950288d0
real(8), parameter :: Rgas=8.314d0
real(8) E2,mueff_p,mueff_v,mu_min,mu_max,c,phi,A,Q,V,f,n,alpha,T0,rho0

!==============================================!
! 1 upper crust 
! 2 lower crust 
! 3 mantle
! 4 seed

if (imat<=0) stop 'material_model: imat<=0'

E2= sqrt( 0.5d0*(exxq**2+eyyq**2)+exyq**2 )

mu_min=1.d19
mu_max=1.d26

select case(imat)
case(1)! upper crust
   rho0=2800
   alpha=2.5d-5
   T0=273.15
   phi=20.d0/180.*pi
   c=20d6
   A=8.57d-28
   Q=223d3
   n=4
   V=0
   f=1
case(2)! lower crust
   rho0=2900
   alpha=2.5d-5
   T0=273.15
   phi=20.d0/180.*pi
   c=20d6
   A=7.13e-18
   Q=345d3
   n=3
   V=0
   f=1
case(3)! mantle
   rho0=3300
   alpha=2.5d-5
   T0=500+273.15
   phi=20.d0/180.*pi
   c=20d6
   A=6.52d-16
   Q=530d3
   n=3.5
   V=18e-6
   f=1
case(4)! seed
   rho0=3300
   alpha=2.5d-5
   T0=500+273.15
   phi=20.d0/180.*pi
   c=20d6
   A=7.13e-18
   Q=345d3
   n=3
   V=0
   f=1
end select

!-------------------------------------
! compute effective viscous viscosity

mueff_v= 0.5d0 * f * A**(-1.d0/n) &
       * E2**(1.d0/n-1.d0) & 
       * exp(max(Q+pq*V,Q)/n/Rgas/Tq)

if (E2<1d-20) mueff_v=mu_max

!-------------------------------------
! compute effective plastic viscosity

mueff_p=( max(pq*sin(phi)+c*cos(phi),c*cos(phi)) )/E2 * 0.5d0

if (E2<1d-20) mueff_p=mu_max

!-------------------------------------
! blend the two viscosities

mueffq=2.d0/(1.d0/mueff_p + 1.d0/mueff_v)

!mueffq=min(mueff_p,mueff_v)
!mueffq=exp(0.5*(log(mueff_p)+log(mueff_v)))

mueffq=min(mueffq,mu_max)
mueffq=max(mueffq,mu_min)

if (mueff_p>mueff_v) then
mode=1.
else
mode=2.
end if

!-------------------------------------
! compute rho as a function of temperature

rhoq=rho0*(1.d0-alpha*(Tq-T0))

end subroutine
