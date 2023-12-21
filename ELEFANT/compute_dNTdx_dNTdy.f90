!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNTdx_dNTdy(r,s,dNdx,dNdy,jcob)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: r,s
real(8), intent(out) :: dNdx(mT),dNdy(mT),jcob

integer k
real(8) dNNNTdr(mT),dNNNTds(mT)
real(8) jcb2D(2,2),jcbi2D(2,2)
real(8), parameter :: t=0d0

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNTdx\_dNTdy.f90}
!@@ This subroutine computes $\partial{\bN^\uptheta}/\partial x$ 
!@@ and $\partial{\bN^\uptheta}/\partial y$ !@@ at a location $r,s$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNTdr(1:mT),mT,ndim,spaceT)
call dNNNds(r,s,t,dNNNTds(1:mT),mT,ndim,spaceT)

jcb2D=0.d0 
do k=1,mT 
   jcb2D(1,1)=jcb2D(1,1)+dNNNTdr(k)*mesh(iel)%xT(k)
   jcb2D(1,2)=jcb2D(1,2)+dNNNTdr(k)*mesh(iel)%yT(k)
   jcb2D(2,1)=jcb2D(2,1)+dNNNTds(k)*mesh(iel)%xT(k)
   jcb2D(2,2)=jcb2D(2,2)+dNNNTds(k)*mesh(iel)%yT(k)
enddo    

jcob=jcb2D(1,1)*jcb2D(2,2)-jcb2D(2,1)*jcb2D(1,2)   

jcbi2D(1,1)=   jcb2D(2,2)/jcob    
jcbi2D(1,2)= - jcb2D(1,2)/jcob    
jcbi2D(2,1)= - jcb2D(2,1)/jcob    
jcbi2D(2,2)=   jcb2D(1,1)/jcob    

do k=1,mT
   dNdx(k)=jcbi2D(1,1)*dNNNTdr(k)+jcbi2D(1,2)*dNNNTds(k) 
   dNdy(k)=jcbi2D(2,1)*dNNNTdr(k)+jcbi2D(2,2)*dNNNTds(k)  
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
