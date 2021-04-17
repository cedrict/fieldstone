!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy(r,s,dNdx,dNdy,jcob)

use global_parameters
use structures

implicit none

real(8), intent(in) :: r,s
real(8), intent(out) :: dNdx(mV),dNdy(mV),jcob

integer k
real(8) dNNNVdr(mV),dNNNVds(mV)
real(8) jcb2D(2,2),jcbi2D(2,2)
real(8), parameter :: t=0d0

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_dNdx\_dNdy.f90}
!@@ This subroutine computes $\partial{\bN^\upnu}/\partial x$ and $\partial{\bN^\upnu}/\partial y$
!@@ at a location $r,s$ passed as argument.
!==================================================================================================!

call dNNVdr(r,s,t,dNNNVdr(1:mV),mV,ndim,pair)
call dNNVds(r,s,t,dNNNVds(1:mV),mV,ndim,pair)

jcb2D=0.d0 
do k=1,mV 
   jcb2D(1,1)=jcb2D(1,1)+dNNNVdr(k)*mesh(iel)%xV(k)
   jcb2D(1,2)=jcb2D(1,2)+dNNNVdr(k)*mesh(iel)%yV(k)
   jcb2D(2,1)=jcb2D(2,1)+dNNNVds(k)*mesh(iel)%xV(k)
   jcb2D(2,2)=jcb2D(2,2)+dNNNVds(k)*mesh(iel)%yV(k)
enddo    

jcob=jcb2D(1,1)*jcb2D(2,2)-jcb2D(2,1)*jcb2D(1,2)   

jcbi2D(1,1)=    jcb2D(2,2) /jcob    
jcbi2D(1,2)=  - jcb2D(1,2) /jcob    
jcbi2D(2,1)=  - jcb2D(2,1) /jcob    
jcbi2D(2,2)=    jcb2D(1,1) /jcob    

do k=1,mV
   dNdx(k)=jcbi2D(1,1)*dNNNVdr(k)+jcbi2D(1,2)*dNNNVds(k) 
   dNdy(k)=jcbi2D(2,1)*dNNNVdr(k)+jcbi2D(2,2)*dNNNVds(k)  
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
