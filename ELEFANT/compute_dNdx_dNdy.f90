!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy(r,s,dNdx,dNdy,jcob)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: r,s
real(8), intent(out) :: dNdx(mmapping),dNdy(mmapping),jcob

integer k
real(8) dNNNMdr(mmapping),dNNNMds(mmapping)
real(8) jcb2D(2,2),jcbi2D(2,2)
real(8), parameter :: t=0d0

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNdx\_dNdy.f90}
!@@ This subroutine computes $\partial{\bN^\upnu}/\partial x$ and $\partial{\bN^\upnu}/\partial y$
!@@ at a location $r,s$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNMdr(1:mmapping),mmapping,ndim,mapping)
call dNNNds(r,s,t,dNNNMds(1:mmapping),mmapping,ndim,mapping)

jcb2D=0.d0 
do k=1,mmapping 
   jcb2D(1,1)=jcb2D(1,1)+dNNNMdr(k)*mesh(iel)%xM(k)
   jcb2D(1,2)=jcb2D(1,2)+dNNNMdr(k)*mesh(iel)%yM(k)
   jcb2D(2,1)=jcb2D(2,1)+dNNNMds(k)*mesh(iel)%xM(k)
   jcb2D(2,2)=jcb2D(2,2)+dNNNMds(k)*mesh(iel)%yM(k)
enddo    

jcob=jcb2D(1,1)*jcb2D(2,2)-jcb2D(2,1)*jcb2D(1,2)   

jcbi2D(1,1)=    jcb2D(2,2) /jcob    
jcbi2D(1,2)=  - jcb2D(1,2) /jcob    
jcbi2D(2,1)=  - jcb2D(2,1) /jcob    
jcbi2D(2,2)=    jcb2D(1,1) /jcob    

do k=1,mmapping
   dNdx(k)=jcbi2D(1,1)*dNNNMdr(k)+jcbi2D(1,2)*dNNNMds(k) 
   dNdy(k)=jcbi2D(2,1)*dNNNMdr(k)+jcbi2D(2,2)*dNNNMds(k)  
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
