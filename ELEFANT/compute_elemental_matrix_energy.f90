!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_matrix_energy(Ael,bel)

use global_parameters
use structures
use timing

implicit none

real(8), dimension(mT,mT), intent(out) :: Ael
real(8), dimension(mT), intent(out) :: bel

integer iq
real(8) Bmat(ndim,mT),rq,sq,tq,jcob,vel(1,ndim)
real(8) Ka(mT,mT),Kd(mT,mT),KK(mT,mT),Nvect(1,mT),NvectT(mT,1)
real(8) NNNV(mV),NNNT(mT),dNdx(mT),dNdy(mT),dNdz(mT),F(mT),MMM(mT,mT),MMMK(mT,mT)
real(8), parameter :: alphaT=0.5d0

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_elemental\_matrix\_energy}
!@@ This subroutine buils the elemental matrix ${\bm A}_{el}$ and its corresponding 
!@@ right hand side $\vec{b}_{el}$. 
!==================================================================================================!

Ael=0d0
bel=0d0

do iq=1,nqel

   rq=mesh(iel)%rq(iq)
   sq=mesh(iel)%sq(iq)
   tq=mesh(iel)%tq(iq)

   call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
   call NNT(rq,sq,tq,NNNT(1:mT),mT,ndim)

   if (ndim==2) then
      call compute_dNTdx_dNTdy(rq,sq,dNdx,dNdy,jcob)
      vel(1,1)=sum(NNNV(1:mV)*mesh(iel)%u(1:mV))
      vel(1,2)=sum(NNNV(1:mV)*mesh(iel)%v(1:mV))
      Bmat(1,1:mT)=dNdx(1:mT)
      Bmat(2,1:mT)=dNdy(1:mT)
   else
      call compute_dNTdx_dNTdy_dNTdz(rq,sq,tq,dNdx(1:mT),dNdy(1:mT),dNdz(1:mT),jcob)
      vel(1,1)=sum(NNNV(1:mV)*mesh(iel)%u(1:mV))
      vel(1,2)=sum(NNNV(1:mV)*mesh(iel)%v(1:mV))
      vel(1,3)=sum(NNNV(1:mV)*mesh(iel)%w(1:mV))
      Bmat(1,1:mT)=dNdx(1:mT)
      Bmat(2,1:mT)=dNdy(1:mT)
      Bmat(3,1:mT)=dNdz(1:mT)
   end if

   Nvect(1,:)=NNNT(1:mT) 
   NvectT(:,1)=NNNT(1:mT)

   Ka=matmul(NvectT,matmul(vel,Bmat))*mesh(iel)%rhoq(iq)*mesh(iel)%hcapaq(iq)*mesh(iel)%JxWq(iq)

   Kd=matmul(transpose(Bmat),Bmat)*mesh(iel)%hcondq(iq)*mesh(iel)%JxWq(iq)

   MMM=matmul(NvectT,Nvect)*mesh(iel)%rhoq(iq)*mesh(iel)%hcapaq(iq)*mesh(iel)%JxWq(iq)

   F=NNNT(1:mT)*mesh(iel)%JxWq(iq)*mesh(iel)%hprodq(iq)

   KK=Ka+Kd

   Ael=Ael+(MMM+KK*alphaT*dt)

   MMMK=MMM-KK*(1.d0-alphaT)*dt

   bel=bel+matmul(MMMK,mesh(iel)%T(1:mT))+F*dt

end do

end subroutine

!==================================================================================================!
!==================================================================================================!
