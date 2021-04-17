!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine test_basis_functions

use global_parameters
use structures
use timing

implicit none

integer iq
real(8) NNNV(1:mV),rq,sq,tq,uq,vq,wq,exxq,eyyq,ezzq
real(8) dNdx(mV),dNdy(mV),dNdz(mV),jcob

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{test\_basis\_functions}
!@@ This subroutine tests the consistency of the basis functions. 
!@@ An analytical velocity field is prescribed (constant, linear or quadratic) and the 
!@@ corresponding values are computed onto the quadrature points via the 
!@@ (derivatives of the) basis functions.
!@@ It generates three ascii files in the {\foldernamefont OUTPUT} folder.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (debug) then

open(unit=123,file='OUTPUT/test_basis_functions_constant.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=1
   mesh(iel)%v=1
   mesh(iel)%w=1
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNV(1:mV)*mesh(iel)%w(1:mV)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      exxq=sum(dNdx(1:mV)*mesh(iel)%u(1:mV)) 
      eyyq=sum(dNdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNdz(1:mV)*mesh(iel)%w(1:mV)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)

open(unit=123,file='OUTPUT/test_basis_functions_linear.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV
   mesh(iel)%v=mesh(iel)%yV
   mesh(iel)%w=mesh(iel)%zV
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNV(1:mV)*mesh(iel)%w(1:mV)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      exxq=sum(dNdx(1:mV)*mesh(iel)%u(1:mV)) 
      eyyq=sum(dNdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNdz(1:mV)*mesh(iel)%w(1:mV)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)

open(unit=123,file='OUTPUT/test_basis_functions_quadratic.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV**2
   mesh(iel)%v=mesh(iel)%yV**2
   mesh(iel)%w=mesh(iel)%zV**2
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      uq=sum(NNNV(1:mV)*mesh(iel)%u(1:mV)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNV(1:mV)*mesh(iel)%w(1:mV)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      exxq=sum(dNdx(1:mV)*mesh(iel)%u(1:mV)) 
      eyyq=sum(dNdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNdz(1:mV)*mesh(iel)%w(1:mV)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)

end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f4.2,a)') '     >> test_basis_functions             ',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
