!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy(r,s,dNUdx,dNUdy,dNVdx,dNVdy,jcob)

use module_parameters, only: mU,mV,mmapping,iel,mapping,ndim,spaceU,spaceV
use module_mesh 

implicit none

real(8), intent(in) :: r,s
real(8), intent(out) :: dNUdx(mU),dNUdy(mU)
real(8), intent(out) :: dNVdx(mV),dNVdy(mV)
real(8), intent(out) :: jcob 

integer k
real(8) dNNNMdr(mmapping),dNNNMds(mmapping)
real(8) dNNNUdr(mU),dNNNUds(mU)
real(8) dNNNVdr(mV),dNNNVds(mV)
real(8) jcb(2,2),jcbi(2,2)
real(8), parameter :: t=0d0
integer, parameter :: caller_id01=101
integer, parameter :: caller_id02=102
integer, parameter :: caller_id03=103
integer, parameter :: caller_id04=104
integer, parameter :: caller_id05=105
integer, parameter :: caller_id06=106

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNdx\_dNdy.f90}
!@@ This subroutine computes $\partial{\bN^u}/\partial \{x,y\}$ and 
!@@ $\partial{\bN^v}/\partial \{x,y\}$ at a location $r,s$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNMdr(1:mmapping),mmapping,ndim,mapping,caller_id01)
call dNNNds(r,s,t,dNNNMds(1:mmapping),mmapping,ndim,mapping,caller_id02)

jcb=0.d0 
do k=1,mmapping 
   jcb(1,1)=jcb(1,1)+dNNNMdr(k)*mesh(iel)%xM(k)
   jcb(1,2)=jcb(1,2)+dNNNMdr(k)*mesh(iel)%yM(k)
   jcb(2,1)=jcb(2,1)+dNNNMds(k)*mesh(iel)%xM(k)
   jcb(2,2)=jcb(2,2)+dNNNMds(k)*mesh(iel)%yM(k)
enddo    

jcob=jcb(1,1)*jcb(2,2)-jcb(2,1)*jcb(1,2)   

jcbi(1,1)=  jcb(2,2)/jcob    
jcbi(1,2)= -jcb(1,2)/jcob    
jcbi(2,1)= -jcb(2,1)/jcob    
jcbi(2,2)=  jcb(1,1)/jcob    

call dNNNdr(r,s,t,dNNNUdr(1:mU),mU,ndim,spaceU,caller_id03)
call dNNNds(r,s,t,dNNNUds(1:mU),mU,ndim,spaceU,caller_id04)
call dNNNdr(r,s,t,dNNNVdr(1:mV),mV,ndim,spaceV,caller_id05)
call dNNNds(r,s,t,dNNNVds(1:mV),mV,ndim,spaceV,caller_id06)

do k=1,mU
   dNUdx(k)=jcbi(1,1)*dNNNUdr(k)+jcbi(1,2)*dNNNUds(k) 
   dNUdy(k)=jcbi(2,1)*dNNNUdr(k)+jcbi(2,2)*dNNNUds(k) 
end do

do k=1,mV
   dNVdx(k)=jcbi(1,1)*dNNNVdr(k)+jcbi(1,2)*dNNNVds(k)  
   dNVdy(k)=jcbi(2,1)*dNNNVdr(k)+jcbi(2,2)*dNNNVds(k)  
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
