!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy_dNdz(r,s,t,dNdx,dNdy,dNdz,jcob)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNdx(mmapping),dNdy(mmapping),dNdz(mmapping),jcob

integer k
real(8) dNNNMdr(mmapping),dNNNMds(mmapping),dNNNMdt(mmapping)
real(8) jcb(3,3),jcbi(3,3)

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNdx\_dNdy\_dNdz.f90}
!@@ This subroutine computes $\partial{\bN^\upnu}/\partial x$, $\partial{\bN^\upnu}/\partial y$ and
!@@ $\partial{\bN^\upnu}/\partial z$ at a location $r,s,t$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNMdr(1:mmapping),mmapping,ndim,mapping)
call dNNNds(r,s,t,dNNNMds(1:mmapping),mmapping,ndim,mapping)
call dNNNdt(r,s,t,dNNNMdt(1:mmapping),mmapping,ndim,mapping)

jcb=0.d0 
do k=1,mmapping 
   jcb(1,1)=jcb(1,1)+dNNNMdr(k)*mesh(iel)%xM(k)
   jcb(1,2)=jcb(1,2)+dNNNMdr(k)*mesh(iel)%yM(k)
   jcb(1,3)=jcb(1,3)+dNNNMdr(k)*mesh(iel)%zM(k)
   jcb(2,1)=jcb(2,1)+dNNNMds(k)*mesh(iel)%xM(k)
   jcb(2,2)=jcb(2,2)+dNNNMds(k)*mesh(iel)%yM(k)
   jcb(2,3)=jcb(2,3)+dNNNMds(k)*mesh(iel)%zM(k)
   jcb(3,1)=jcb(3,1)+dNNNMdt(k)*mesh(iel)%xM(k)
   jcb(3,2)=jcb(3,2)+dNNNMdt(k)*mesh(iel)%yM(k)
   jcb(3,3)=jcb(3,3)+dNNNMdt(k)*mesh(iel)%zM(k)
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

do k=1,mmapping
   dNdx(k)=jcbi(1,1)*dNNNMdr(k)+jcbi(1,2)*dNNNMds(k)+jcbi(1,3)*dNNNMdt(k) 
   dNdy(k)=jcbi(2,1)*dNNNMdr(k)+jcbi(2,2)*dNNNMds(k)+jcbi(2,3)*dNNNMdt(k) 
   dNdz(k)=jcbi(3,1)*dNNNMdr(k)+jcbi(3,2)*dNNNMds(k)+jcbi(3,3)*dNNNMdt(k) 
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
