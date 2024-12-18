program opla

implicit none

integer i,iter
integer, parameter :: nnz=512
real(8), parameter :: Rgas=8.31
real(8), parameter :: Tkelvin=273.
real(8) sr_T,rho,gz,Lz
real(8) z(nnz)
real(8) p(nnz)
real(8) T(nnz)
real(8) mueff(nnz)
real(8) A_ds,n_ds,Q_ds,V_ds,mu_ds,sr_ds
real(8) A_df,n_df,Q_df,V_df,grain_size,mu_df,sr_df
real(8) TA,TB,TC,TD,zA,zB,zC,zD,tau,func,funcp
integer n_lim
real(8) mu_lim,sr_lim,mu_pl,sr_pl,tau_lim,tau_lim0

!=================================================

grain_size = 1.d-3
A_df     = 8.7d15/(8.d10)*(0.5d-9/grain_size)**2.5
Q_df     = 300.d3
V_df     = 5d-6

A_ds     = 2.417d-16
n_ds     = 3.5
Q_ds     = 540.d3
V_ds     = 20d-6

Lz=660.d3
sr_T=1.d-15
rho=3300.
gz=9.81

n_lim=3
mu_lim=1.d25
sr_lim=1.d-20
tau_lim=2.d0*sr_lim*mu_lim

print *,tau_lim

!================================
open(unit=123,file='setup.dat')

zA=Lz        ; TA=  20.d0+Tkelvin
zB=Lz-30.d3  ; TB= 550.d0+Tkelvin
zC=Lz-120.d3 ; TC=1330.d0+Tkelvin
zD=0.d0      ; TD=1380.d0+Tkelvin

do i=1,nnz
   z(i)=(i-1)*Lz/(nnz-1)
   p(i)=rho*gz*(Lz-z(i))

   if (z(i) > Lz-30.d3) then 
      T(i)= (TA-TB)/(zA-zB)*(z(i)-zB)+TB
   elseif (z(i) > Lz-120.d3) then 
      T(i)= (TB-TC)/(zB-zC)*(z(i)-zC)+TC
   else
      T(i)= (TC-TD)/(zC-zD)*(z(i)-zD)+TD
   end if 
   write(123,*) (Lz-z(i))/1000,T(i)-273,p(i)
end do


!================================
! old method
!================================

open(unit=100,file='old.dat',action='write')

do i=1,nnz
   mu_df=0.5d0 / A_df * exp((Q_df+p(i)*V_df)/Rgas/T(i)) 
   mu_ds= 0.5d0 * A_ds**(-1.d0/n_ds) * sr_T**(1.d0/n_ds-1.d0) * exp((Q_ds+p(i)*V_ds)/n_ds/Rgas/T(i))
   mu_pl=mu_lim*(sr_T/sr_lim)**n_lim
   mueff(i)=1.d0/(1./mu_ds+1./mu_df+1.d0/mu_pl)
   mueff(i)=min(mueff(i),1.d28)
   mueff(i)=max(mueff(i),1.d18)
   write(100,*) (Lz-z(i))/1000,mu_df,mu_ds,mu_pl,mueff(i)
end do


!================================
! new method
!================================

open(unit=200,file='new.dat',action='write')
open(unit=201,file='NR.dat',action='write')

do i=1,nnz

   ! compute tau
   tau=1.d6
   iter=0
   func=1
   do while (abs(func/sr_T)>1.d-8)
      iter=iter+1 
      sr_ds=A_ds * tau**n_ds * exp(-(Q_ds+p(i)*V_ds)/Rgas/T(i))
      sr_df=A_df * tau       * exp(-(Q_df+p(i)*V_df)/Rgas/T(i))
      sr_pl=sr_lim*(tau/tau_lim)**n_lim

      !sr_pl=max(1e-18,sr_pl)

      func=sr_T-sr_ds-sr_df-sr_pl 

      funcp= - A_ds * n_ds * tau**(n_ds-1) * exp(-(Q_ds+p(i)*V_ds)/Rgas/T(i)) &
             - A_df                        * exp(-(Q_df+p(i)*V_df)/Rgas/T(i)) & 
             - n_lim*sr_lim/tau_lim*(tau/tau_lim)**(n_lim-1)

      tau=tau - func/funcp 
      write(201,*) i,iter,tau
   end do

   write(333,*) tau_lim
 
   ! compute viscosities

   mu_df=0.5d0 / A_df * exp((Q_df+p(i)*V_df)/Rgas/T(i)) 
   mu_df=max(mu_df,1.d18)
   mu_df=min(mu_df,1.d28)

   mu_ds=0.5d0 * A_ds**(-1.d0/n_ds) * sr_ds**(1.d0/n_ds-1.d0) * exp((Q_ds+p(i)*V_ds)/n_ds/Rgas/T(i))
   mu_ds=max(mu_ds,1.d18)
   mu_ds=min(mu_ds,1.d28)

   mu_pl=mu_lim*(sr_pl/sr_lim)**n_lim
   mu_pl=max(mu_pl,1.d18)
   mu_pl=min(mu_pl,1.d28)

   ! compute effective viscosity

   mueff(i)=1.d0/(1./mu_ds+1./mu_df+1.d0/mu_pl)
   mueff(i)=min(mueff(i),1.d28)
   mueff(i)=max(mueff(i),1.d18)
   write(200,*) (Lz-z(i))/1000,mu_df,mu_ds,mu_pl,mueff(i),sr_ds,sr_df,sr_pl,sr_ds+sr_df+sr_pl,tau

end do

end program




