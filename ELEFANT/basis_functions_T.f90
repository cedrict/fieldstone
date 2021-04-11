
subroutine NNT(r,s,t,NT,mT,ndim)

implicit none
integer, intent(in) :: mT,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: NT(mT)

if (ndim==2) then

   NT(1)=0.25*(1-r)*(1-s)
   NT(2)=0.25*(1+r)*(1-s)
   NT(3)=0.25*(1+r)*(1+s)
   NT(4)=0.25*(1-r)*(1+s)

end if

if (ndim==3) then

   NT(1)=0.125*(1-r)*(1-s)*(1-t)
   NT(2)=0.125*(1+r)*(1-s)*(1-t)
   NT(3)=0.125*(1+r)*(1+s)*(1-t)
   NT(4)=0.125*(1-r)*(1+s)*(1-t)
   NT(5)=0.125*(1-r)*(1-s)*(1+t)
   NT(6)=0.125*(1+r)*(1-s)*(1+t)
   NT(7)=0.125*(1+r)*(1+s)*(1+t)
   NT(8)=0.125*(1-r)*(1+s)*(1+t)

end if

end subroutine

!==========================================================

subroutine dNNTdr(r,s,t,dNTdr)
use global_parameters
implicit none
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNTdr(mT)

if (ndim==2) then

   dNTdr(1)=-0.25*(1-s)
   dNTdr(2)=+0.25*(1-s)
   dNTdr(3)=+0.25*(1+s)
   dNTdr(4)=-0.25*(1+s)

end if

end subroutine

!==========================================================

subroutine dNNTds(r,s,t,dNTds)
use global_parameters
implicit none
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNTds(mT)

if (ndim==2) then

   dNTds(1)=-0.25*(1-r)
   dNTds(2)=-0.25*(1+r)
   dNTds(3)=+0.25*(1+r)
   dNTds(4)=+0.25*(1-r)

end if

end subroutine

!==========================================================






