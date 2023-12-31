!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_dNdx_dNdy_dNdz(r,s,t,dNUdx,dNUdy,dNUdz,dNVdx,dNVdy,dNVdz,dNWdx,dNWdy,dNWdz,jcob)

use module_parameters
use module_mesh 

implicit none

real(8), intent(in) :: r,s,t
real(8), intent(out) :: dNUdx(mU),dNUdy(mU),dNUdz(mU)
real(8), intent(out) :: dNVdx(mV),dNVdy(mV),dNVdz(mV)
real(8), intent(out) :: dNWdx(mW),dNWdy(mW),dNWdz(mW)
real(8), intent(out) :: jcob 

integer k
real(8) dNNNMdr(mmapping),dNNNMds(mmapping),dNNNMdt(mmapping)
real(8) dNNNUdr(mU),dNNNUds(mU),dNNNUdt(mU)
real(8) dNNNVdr(mV),dNNNVds(mV),dNNNVdt(mV)
real(8) dNNNWdr(mW),dNNNWds(mW),dNNNWdt(mW)
real(8) jcb(3,3),jcbi(3,3)
integer, parameter :: caller_id01=201
integer, parameter :: caller_id02=202
integer, parameter :: caller_id03=203
integer, parameter :: caller_id04=204
integer, parameter :: caller_id05=205
integer, parameter :: caller_id06=206
integer, parameter :: caller_id07=207
integer, parameter :: caller_id08=208
integer, parameter :: caller_id09=209
integer, parameter :: caller_id10=210
integer, parameter :: caller_id11=211
integer, parameter :: caller_id12=212

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_dNdx\_dNdy\_dNdz.f90}
!@@ This subroutine computes $\partial{\bN^u}/\partial \{x,y,z\}$, 
!@@ $\partial{\bN^v}/\partial \{x,y,z\}$ and
!@@ $\partial{\bN^w}/\partial \{x,y,z\}$ at a location $r,s,t$ passed as argument.
!==================================================================================================!

call dNNNdr(r,s,t,dNNNMdr(1:mmapping),mmapping,ndim,mapping,caller_id01)
call dNNNds(r,s,t,dNNNMds(1:mmapping),mmapping,ndim,mapping,caller_id02)
call dNNNdt(r,s,t,dNNNMdt(1:mmapping),mmapping,ndim,mapping,caller_id03)

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


call dNNNdr(r,s,t,dNNNUdr(1:mU),mU,ndim,spaceU,caller_id04)
call dNNNds(r,s,t,dNNNUds(1:mU),mU,ndim,spaceU,caller_id05)
call dNNNdt(r,s,t,dNNNUdt(1:mU),mU,ndim,spaceU,caller_id06)

call dNNNdr(r,s,t,dNNNVdr(1:mV),mV,ndim,spaceV,caller_id07)
call dNNNds(r,s,t,dNNNVds(1:mV),mV,ndim,spaceV,caller_id08)
call dNNNdt(r,s,t,dNNNVdt(1:mV),mV,ndim,spaceV,caller_id09)

call dNNNdr(r,s,t,dNNNWdr(1:mW),mW,ndim,spaceW,caller_id10)
call dNNNds(r,s,t,dNNNWds(1:mW),mW,ndim,spaceW,caller_id11)
call dNNNdt(r,s,t,dNNNWdt(1:mW),mW,ndim,spaceW,caller_id12)

do k=1,mU
   dNUdx(k)=jcbi(1,1)*dNNNUdr(k)+jcbi(1,2)*dNNNUds(k)+jcbi(1,3)*dNNNUdt(k) 
   dNUdy(k)=jcbi(2,1)*dNNNUdr(k)+jcbi(2,2)*dNNNUds(k)+jcbi(2,3)*dNNNUdt(k) 
   dNUdz(k)=jcbi(3,1)*dNNNUdr(k)+jcbi(3,2)*dNNNUds(k)+jcbi(3,3)*dNNNUdt(k) 
end do

do k=1,mV
   dNVdx(k)=jcbi(1,1)*dNNNVdr(k)+jcbi(1,2)*dNNNVds(k)+jcbi(1,3)*dNNNVdt(k) 
   dNVdy(k)=jcbi(2,1)*dNNNVdr(k)+jcbi(2,2)*dNNNVds(k)+jcbi(2,3)*dNNNVdt(k) 
   dNVdz(k)=jcbi(3,1)*dNNNVdr(k)+jcbi(3,2)*dNNNVds(k)+jcbi(3,3)*dNNNVdt(k) 
end do

do k=1,mW
   dNVdx(k)=jcbi(1,1)*dNNNWdr(k)+jcbi(1,2)*dNNNWds(k)+jcbi(1,3)*dNNNWdt(k) 
   dNVdy(k)=jcbi(2,1)*dNNNWdr(k)+jcbi(2,2)*dNNNWds(k)+jcbi(2,3)*dNNNWdt(k) 
   dNVdz(k)=jcbi(3,1)*dNNNWdr(k)+jcbi(3,2)*dNNNWds(k)+jcbi(3,3)*dNNNWdt(k) 
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
