!==============================================!
!
!==============================================!

subroutine material_model(xq,yq,pq,Tq,exxq,eyyq,exyq,imat,mueffq,rhoq,mode)

implicit none

real(8), intent(in) :: xq,yq,exxq,eyyq,exyq,pq,Tq
integer, intent(in) :: imat
real(8), intent(out) :: mueffq,rhoq,mode

real(8), parameter :: pi =  3.14159265358979327d0
real(8), parameter :: Rgas=8.314d0
real(8) dotepsilon,mueff_pl,mueff_dl,mu_min,mu_max,c,phi,A,Q,V,f,n,alpha,T0,rho0

!==============================================!
! 1 upper crust 
! 2 lower crust 
! 3 mantle
! 4 seed

dotepsilon=0    ! MODIFY

mu_min=1.d19
mu_max=1.d25

select case(imat)

case(1)! upper crust
   rho0=0
   alpha=0
   T0=0
   phi=0
   c=0
   A=0
   Q=0
   n=0
   V=0
   f=0

case(2)! lower crust

   ! same work here

case(3)! mantle

   ! same work here

case(4)! seed

   ! same work here

end select

!-------------------------------------
! compute effective viscous viscosity

mueff_dl= 0   ! MODIFY

if (dotepsilon<1d-20) mueff_dl=mu_max   ! DO NOT MODIFY

mueff_dl=min(mueff_dl,mu_max)   ! DO NOT MODIFY
mueff_dl=max(mueff_dl,mu_min)   ! DO NOT MODIFY

!-------------------------------------
! compute effective plastic viscosity

mueff_pl=0   ! MODIFY

if (dotepsilon<1d-20) mueff_pl=mu_max   ! DO NOT MODIFY

mueff_pl=min(mueff_pl,mu_max)   ! DO NOT MODIFY
mueff_pl=max(mueff_pl,mu_min)   ! DO NOT MODIFY

!-------------------------------------
! blend the two viscosities

mueffq= 0 ! MODIFY

if (mueff_pl>mueff_dl) then   ! DO NOT MODIFY
mode=1.                       ! DO NOT MODIFY
else                          ! DO NOT MODIFY
mode=2.                       ! DO NOT MODIFY
end if                        ! DO NOT MODIFY

!-------------------------------------
! compute rho as a function of temperature

rhoq=0

end subroutine
