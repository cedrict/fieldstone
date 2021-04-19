!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy_dNdz(r,s,t,dNdx,dNdy,dNdz,jcob)

use global_parameters
use structures

implicit none

real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNdx(mV),dNdy(mV),dNdz(mV),jcob

integer k
real(8) dNNNVdr(mV),dNNNVds(mV),dNNNVdt(mV)
real(8) jcb(3,3),jcbi(3,3)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_dNdx\_dNdy\_dNdz.f90}
!@@ This subroutine computes $\partial{\bN^\upnu}/\partial x$, $\partial{\bN^\upnu}/\partial y$ and
!@@ $\partial{\bN^\upnu}/\partial z$ at a location $r,s,t$ passed as argument.
!==================================================================================================!

call dNNVdr(r,s,t,dNNNVdr(1:mV),mV,ndim,pair)
call dNNVds(r,s,t,dNNNVds(1:mV),mV,ndim,pair)
call dNNVdt(r,s,t,dNNNVdt(1:mV),mV,ndim,pair)

jcb=0.d0 
do k=1,mV 
   jcb(1,1)=jcb(1,1)+dNNNVdr(k)*mesh(iel)%xV(k)
   jcb(1,2)=jcb(1,2)+dNNNVdr(k)*mesh(iel)%yV(k)
   jcb(1,3)=jcb(1,3)+dNNNVdr(k)*mesh(iel)%zV(k)
   jcb(2,1)=jcb(2,1)+dNNNVds(k)*mesh(iel)%xV(k)
   jcb(2,2)=jcb(2,2)+dNNNVds(k)*mesh(iel)%yV(k)
   jcb(2,3)=jcb(2,3)+dNNNVds(k)*mesh(iel)%zV(k)
   jcb(3,1)=jcb(3,1)+dNNNVdt(k)*mesh(iel)%xV(k)
   jcb(3,2)=jcb(3,2)+dNNNVdt(k)*mesh(iel)%yV(k)
   jcb(3,3)=jcb(3,3)+dNNNVdt(k)*mesh(iel)%zV(k)
enddo    

jcob=jcb(1,1)*jcb(2,2)*jcb(3,3) &    
    +jcb(1,2)*jcb(2,3)*jcb(3,1) &    
    +jcb(2,1)*jcb(3,2)*jcb(1,3) &    
    -jcb(1,3)*jcb(2,2)*jcb(3,1) &    
    -jcb(1,2)*jcb(2,1)*jcb(3,3) &    
    -jcb(2,3)*jcb(3,2)*jcb(1,1) 

jcbi(1,1)=(jcb(2,2)*jcb(3,3)-jcb(2,3)*jcb(3,2))/jcob    
jcbi(2,1)=(jcb(2,3)*jcb(3,1)-jcb(2,1)*jcb(3,3))/jcob    
jcbi(3,1)=(jcb(2,1)*jcb(3,2)-jcb(2,2)*jcb(3,1))/jcob  
jcbi(1,2)=(jcb(1,3)*jcb(3,2)-jcb(1,2)*jcb(3,3))/jcob
jcbi(2,2)=(jcb(1,1)*jcb(3,3)-jcb(1,3)*jcb(3,1))/jcob    
jcbi(3,2)=(jcb(1,2)*jcb(3,1)-jcb(1,1)*jcb(3,2))/jcob    
jcbi(1,3)=(jcb(1,2)*jcb(2,3)-jcb(1,3)*jcb(2,2))/jcob    
jcbi(2,3)=(jcb(1,3)*jcb(2,1)-jcb(1,1)*jcb(2,3))/jcob    
jcbi(3,3)=(jcb(1,1)*jcb(2,2)-jcb(1,2)*jcb(2,1))/jcob 

do k=1,mV
   dNdx(k)=jcbi(1,1)*dNNNVdr(k)+jcbi(1,2)*dNNNVds(k)+jcbi(1,3)*dNNNVdt(k) 
   dNdy(k)=jcbi(2,1)*dNNNVdr(k)+jcbi(2,2)*dNNNVds(k)+jcbi(2,3)*dNNNVdt(k) 
   dNdz(k)=jcbi(3,1)*dNNNVdr(k)+jcbi(3,2)*dNNNVds(k)+jcbi(3,3)*dNNNVdt(k) 
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
