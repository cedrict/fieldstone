!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_matrices(Kel,Gel,fel,hel)

use global_parameters
use structures
use constants

implicit none

real(8), intent(out) :: Kel(mV*ndofV,mV*ndofV)
real(8), intent(out) :: Gel(mV*ndofV,mP)
real(8), intent(out) :: fel(mV*ndofV)
real(8), intent(out) :: hel(mP)

integer iq,k,i1,i2
real(8) Bmat(3,mV*ndofV)
real(8) NNNmat(3,mP)
real(8) gx,gy,gz,rq,sq,tq
real(8) NNNV(mV),dNNNVdr(mV),dNNNVds(mV),NNNP(mP)
real(8) jcb2D(2,2),jcbi2D(2,2),jcob
real(8) dNdx(mV),dNdy(mV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_elemental\_matrices.f90}
!@@
!==================================================================================================!

Kel=0.d0
Gel=0.d0
fel=0.d0
hel=0.d0

do iq=1,nqel

   rq=mesh(iel)%rq(iq)
   sq=mesh(iel)%rq(iq)
   tq=0.d0

   call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
   call dNNVdr(rq,sq,tq,dNNNVdr(1:mV),mV,ndim,pair)
   call dNNVds(rq,sq,tq,dNNNVds(1:mV),mV,ndim,pair)
   call NNP(rq,sq,tq,NNNP(1:mP),mP,ndim,pair)
   
   jcb2D=0.d0 
   do k=1,mV 
      jcb2D(1,1)=jcb2D(1,1)+dNNNVdr(k)*mesh(iel)%xV(k)
      jcb2D(1,2)=jcb2D(1,2)+dNNNVdr(k)*mesh(iel)%yV(k)
      jcb2D(2,1)=jcb2D(2,1)+dNNNVds(k)*mesh(iel)%xV(k)
      jcb2D(2,2)=jcb2D(2,2)+dNNNVds(k)*mesh(iel)%yV(k)
   enddo    
    
   jcob=jcb2D(1,1)*jcb2D(2,2)-jcb2D(2,1)*jcb2D(1,2)   

   mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)
    
   jcbi2D(1,1)=    jcb2D(2,2) /jcob    
   jcbi2D(1,2)=  - jcb2D(1,2) /jcob    
   jcbi2D(2,1)=  - jcb2D(2,1) /jcob    
   jcbi2D(2,2)=    jcb2D(1,1) /jcob    

   do k=1,mV
      dNdx(k)=jcbi2D(1,1)*dNNNVdr(k)+jcbi2D(1,2)*dNNNVds(k) 
      dNdy(k)=jcbi2D(2,1)*dNNNVdr(k)+jcbi2D(2,2)*dNNNVds(k)  
   end do

   !-------------------------
   ! building gradient matrix
   !-------------------------

   Bmat=0.d0
   do k=1,mV    
         i1=ndofV*k-1    
         i2=ndofV*k    
         Bmat(1,i1)=dNdx(k) ; Bmat(1,i2)=0.d0
         Bmat(2,i1)=0.d0    ; Bmat(2,i2)=dNdy(k)
         Bmat(3,i1)=dNdy(k) ; Bmat(3,i2)=dNdx(k)
   end do 

   !---------------
   ! compute fel
   !---------------

   call gravity_model(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),gx,gy,gz)

   do k=1,mV    
      i1=ndofV*k-1
      i2=ndofV*k
      fel(i1)=fel(i1)+NNNV(k)*gx*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      fel(i2)=fel(i2)+NNNV(k)*gy*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
   end do

   !---------------
   ! compute Gel
   !---------------

   do k=1,mP
      NNNmat(1,k)=NNNp(k)
      NNNmat(2,k)=NNNp(k)
      NNNmat(3,k)=0.d0
   end do

   Gel=Gel-matmul(transpose(Bmat(1:3,1:mV*ndofV)),NNNmat(1:3,1:mP))*mesh(iel)%JxWq(iq)

   Kel=Kel+matmul(transpose(Bmat(1:3,1:mV*ndofV)),&
                  matmul(Cmat2D,Bmat(1:3,1:mV*ndofV)))*mesh(iel)%etaq(iq)*mesh(iel)%JxWq(iq)

end do

end subroutine

!==================================================================================================!
!==================================================================================================!
