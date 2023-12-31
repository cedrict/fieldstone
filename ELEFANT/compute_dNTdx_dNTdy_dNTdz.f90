!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNTdx_dNTdy_dNTdz(r,s,t,dNdx,dNdy,dNdz,jcob)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNdx(mT),dNdy(mT),dNdz(mT),jcob

integer k
real(8) dNNNTdr(mT),dNNNTds(mT),dNNNTdt(mT)
real(8) jcb(3,3),jcbi(3,3)
integer, parameter :: caller_id01=401
integer, parameter :: caller_id02=402
integer, parameter :: caller_id03=403

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNdx\_dNdy\_dNdz.f90}
!@@ This subroutine computes $\partial{\bN^\uptheta}/\partial x$, 
!@@ $\partial{\bN^\uptheta}/\partial y$ and
!@@ $\partial{\bN^\uptheta}/\partial z$ at a location $r,s,t$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNTdr(1:mT),mT,ndim,spaceTemperature,caller_id01)
call dNNNds(r,s,t,dNNNTds(1:mT),mT,ndim,spaceTemperature,caller_id02)
call dNNNdt(r,s,t,dNNNTdt(1:mT),mT,ndim,spaceTemperature,caller_id03)

jcb=0.d0 
do k=1,mT 
   jcb(1,1)=jcb(1,1)+dNNNTdr(k)*mesh(iel)%xV(k)
   jcb(1,2)=jcb(1,2)+dNNNTdr(k)*mesh(iel)%yV(k)
   jcb(1,3)=jcb(1,3)+dNNNTdr(k)*mesh(iel)%zV(k)
   jcb(2,1)=jcb(2,1)+dNNNTds(k)*mesh(iel)%xV(k)
   jcb(2,2)=jcb(2,2)+dNNNTds(k)*mesh(iel)%yV(k)
   jcb(2,3)=jcb(2,3)+dNNNTds(k)*mesh(iel)%zV(k)
   jcb(3,1)=jcb(3,1)+dNNNTdt(k)*mesh(iel)%xV(k)
   jcb(3,2)=jcb(3,2)+dNNNTdt(k)*mesh(iel)%yV(k)
   jcb(3,3)=jcb(3,3)+dNNNTdt(k)*mesh(iel)%zV(k)
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

do k=1,mT
   dNdx(k)=jcbi(1,1)*dNNNTdr(k)+jcbi(1,2)*dNNNTds(k)+jcbi(1,3)*dNNNTdt(k) 
   dNdy(k)=jcbi(2,1)*dNNNTdr(k)+jcbi(2,2)*dNNNTds(k)+jcbi(2,3)*dNNNTdt(k) 
   dNdz(k)=jcbi(3,1)*dNNNTdr(k)+jcbi(3,2)*dNNNTds(k)+jcbi(3,3)*dNNNTdt(k) 
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
