
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

subroutine dNNTdr(r,s,t,dNTdr,mT,ndim)
implicit none
integer, intent(in) :: mT,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNTdr(mT)

if (ndim==2) then
   dNTdr(1)=-0.25*(1-s)
   dNTdr(2)=+0.25*(1-s)
   dNTdr(3)=+0.25*(1+s)
   dNTdr(4)=-0.25*(1+s)
end if

if (ndim==3) then
   dNTdr(1)=-0.125*(1-s)*(1-t)
   dNTdr(2)=+0.125*(1-s)*(1-t)
   dNTdr(3)=+0.125*(1+s)*(1-t)
   dNTdr(4)=-0.125*(1+s)*(1-t)
   dNTdr(5)=-0.125*(1-s)*(1+t)
   dNTdr(6)=+0.125*(1-s)*(1+t)
   dNTdr(7)=+0.125*(1+s)*(1+t)
   dNTdr(8)=-0.125*(1+s)*(1+t)
end if

end subroutine

!==========================================================

subroutine dNNTds(r,s,t,dNTds,mT,ndim)
implicit none
integer, intent(in) :: mT,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNTds(mT)

if (ndim==2) then
   dNTds(1)=-0.25*(1-r)
   dNTds(2)=-0.25*(1+r)
   dNTds(3)=+0.25*(1+r)
   dNTds(4)=+0.25*(1-r)
end if

if (ndim==3) then
   dNTds(1)=-0.125*(1-r)*(1-t)
   dNTds(2)=-0.125*(1+r)*(1-t)
   dNTds(3)=+0.125*(1+r)*(1-t)
   dNTds(4)=+0.125*(1-r)*(1-t)
   dNTds(5)=-0.125*(1-r)*(1+t)
   dNTds(6)=-0.125*(1+r)*(1+t)
   dNTds(7)=+0.125*(1+r)*(1+t)
   dNTds(8)=+0.125*(1-r)*(1+t)
end if

end subroutine

!==========================================================

subroutine dNNTdt(r,s,t,dNTdt,mT,ndim)
implicit none
integer, intent(in) :: mT,ndim
real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNTdt(mT)

dNTdt(1)=-0.125*(1-r)*(1-s)
dNTdt(2)=-0.125*(1+r)*(1-s)
dNTdt(3)=-0.125*(1+r)*(1+s)
dNTdt(4)=-0.125*(1-r)*(1+s)
dNTdt(5)=+0.125*(1-r)*(1-s)
dNTdt(6)=+0.125*(1+r)*(1-s)
dNTdt(7)=+0.125*(1+r)*(1+s)
dNTdt(8)=+0.125*(1-r)*(1+s)

end subroutine

!==========================================================
