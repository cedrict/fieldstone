!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine test_basis_functions

use module_parameters, only: nel,iproc,debug,nqel,mU,mV,mW,mT,mP,ndim,use_T,iel,&
                             spaceU,spaceV,spaceW,spacePressure,spaceTemperature
use module_mesh 
use module_timing

implicit none

integer iq
real(8) NNNu(1:mU),NNNV(1:mV),NNNW(1:mV),rq,sq,tq,uq,vq,wq,exxq,eyyq,ezzq,pq,dTdxq,dTdyq
real(8) dNUdx(mU),dNUdy(mU),dNUdz(mU)
real(8) dNVdx(mV),dNVdy(mV),dNVdz(mV)
real(8) dNWdx(mW),dNWdy(mW),dNWdz(mW)
real(8) dNTdx(mT),dNTdy(mT),dNTdz(mT)
real(8) jcob,NNNP(1:mP),NNNT(1:mT)
integer, parameter :: caller_id01=701
integer, parameter :: caller_id02=702
integer, parameter :: caller_id03=703
integer, parameter :: caller_id04=704
integer, parameter :: caller_id05=705

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{test\_basis\_functions}
!@@ This subroutine tests the consistency of the basis functions. 
!@@ An analytical velocity field is prescribed (constant, linear or quadratic) and the 
!@@ corresponding values are computed onto the quadrature points via the 
!@@ (derivatives of the) basis functions.
!@@ It generates three ascii files in the {\foldernamefont OUTPUT} folder which 
!@@ are to be processed with the gnuplot script present there.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (debug) then

open(unit=123,file='OUTPUT/TEST/test_basis_functions_V_constant.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=1
   mesh(iel)%v=1
   mesh(iel)%w=1
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNU,mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV,mV,ndim,spaceV,caller_id02)
      if (ndim==3) &
      call NNN(rq,sq,tq,NNNW,mW,ndim,spaceW,caller_id03)
      uq=sum(NNNU(1:mU)*mesh(iel)%u(1:mU)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNW(1:mW)*mesh(iel)%w(1:mW)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNUdx,dNUdy,dNVdx,dNVdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNUdx,dNUdy,dNUdz,dNVdx,dNVdy,dNVdz,&
                                               dNWdx,dNWdy,dNWdz,jcob)
      exxq=sum(dNUdx(1:mU)*mesh(iel)%u(1:mU)) 
      eyyq=sum(dNVdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNWdz(1:mW)*mesh(iel)%w(1:mW)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_V_constant.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_V_linear.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV
   mesh(iel)%v=mesh(iel)%yV
   mesh(iel)%w=mesh(iel)%zV
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNU,mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV,mV,ndim,spaceV,caller_id02)
      if (ndim==3) &
      call NNN(rq,sq,tq,NNNW,mW,ndim,spaceW,caller_id03)
      uq=sum(NNNU(1:mU)*mesh(iel)%u(1:mU)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNW(1:mW)*mesh(iel)%w(1:mW)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNUdx,dNUdy,dNVdx,dNVdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNUdx,dNUdy,dNUdz,dNVdx,dNVdy,dNVdz,&
                                               dNWdx,dNWdy,dNWdz,jcob)
      exxq=sum(dNUdx(1:mU)*mesh(iel)%u(1:mU)) 
      eyyq=sum(dNVdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNWdz(1:mW)*mesh(iel)%w(1:mW)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_V_linear.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_V_quadratic.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV**2
   mesh(iel)%v=mesh(iel)%yV**2
   mesh(iel)%w=mesh(iel)%zV**2
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNU,mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV,mV,ndim,spaceV,caller_id02)
      if (ndim==3) &
      call NNN(rq,sq,tq,NNNW,mW,ndim,spaceW,caller_id03)
      uq=sum(NNNU(1:mU)*mesh(iel)%u(1:mU)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNW(1:mW)*mesh(iel)%w(1:mW)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNUdx,dNUdy,dNVdx,dNVdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNUdx,dNUdy,dNUdz,dNVdx,dNVdy,dNVdz,&
                                               dNWdx,dNWdy,dNWdz,jcob)
      exxq=sum(dNUdx(1:mU)*mesh(iel)%u(1:mU)) 
      eyyq=sum(dNVdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNWdz(1:mW)*mesh(iel)%w(1:mW)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_V_quadratic.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_V_cubic.ascii',action='write')
do iel=1,nel
   mesh(iel)%u=mesh(iel)%xV**3
   mesh(iel)%v=mesh(iel)%yV**3
   mesh(iel)%w=mesh(iel)%zV**3
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNU,mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV,mV,ndim,spaceV,caller_id02)
      if (ndim==3) &
      call NNN(rq,sq,tq,NNNW,mW,ndim,spaceW,caller_id03)
      uq=sum(NNNU(1:mU)*mesh(iel)%u(1:mU)) 
      vq=sum(NNNV(1:mV)*mesh(iel)%v(1:mV)) 
      wq=sum(NNNW(1:mW)*mesh(iel)%w(1:mW)) 
      if (ndim==2) call compute_dNdx_dNdy(rq,sq,dNUdx,dNUdy,dNVdx,dNVdy,jcob)
      if (ndim==3) call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNUdx,dNUdy,dNUdz,dNVdx,dNVdy,dNVdz,&
                                               dNWdx,dNWdy,dNWdz,jcob)
      exxq=sum(dNUdx(1:mU)*mesh(iel)%u(1:mU)) 
      eyyq=sum(dNVdy(1:mV)*mesh(iel)%v(1:mV)) 
      ezzq=sum(dNWdz(1:mW)*mesh(iel)%w(1:mW)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                   uq,vq,wq,exxq,eyyq,ezzq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_V_cubic.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_P_constant.ascii',action='write')
do iel=1,nel
   mesh(iel)%p=1
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id04)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),pq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_P_constant.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_P_linear.ascii',action='write')
do iel=1,nel
   mesh(iel)%p=mesh(iel)%xP
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id04)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),pq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_P_linear.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_P_quadratic.ascii',action='write')
do iel=1,nel
   mesh(iel)%p=mesh(iel)%xP**2
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id04)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),pq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_P_quadratic.ascii'

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_P_cubic.ascii',action='write')
do iel=1,nel
   mesh(iel)%p=mesh(iel)%xP**3
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id04)
      pq=sum(NNNP(1:mP)*mesh(iel)%p(1:mP)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),pq
   end do
end do
close(123)
write(*,'(a)') shift//'-> OUTPUT/TEST/test_basis_functions_P_cubic.ascii'

!----------------------------------------------------------

if (use_T) then
open(unit=123,file='OUTPUT/TEST/test_basis_functions_T_constant.ascii',action='write')
do iel=1,nel
   mesh(iel)%T=1
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNT(1:mT),mT,ndim,spaceTemperature,caller_id05)
      Tq=sum(NNNT(1:mT)*mesh(iel)%T(1:mT)) 
      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),Tq
   end do
end do
close(123)

!----------------------------------------------------------

open(unit=123,file='OUTPUT/TEST/test_basis_functions_T_linear.ascii',action='write')
do iel=1,nel
   mesh(iel)%T(1:mT)=mesh(iel)%yT(1:mT)
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNN(rq,sq,tq,NNNT(1:mT),mT,ndim,spaceTemperature,caller_id05)
      Tq=sum(NNNT(1:mT)*mesh(iel)%T(1:mT)) 

      if (ndim==2) call compute_dNTdx_dNTdy(rq,sq,dNTdx(1:mT),dNTdy(1:mT),jcob)
      !if (ndim==3) call compute_dNTdx_dNTdy_dNTdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      dTdxq=sum(dNTdx(1:mT)*mesh(iel)%T(1:mT)) 
      dTdyq=sum(dNTdy(1:mT)*mesh(iel)%T(1:mT)) 

      write(123,*) mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),Tq,dTdxq,dTdyq
   end do
end do
close(123)
end if


end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'test_basis_functions:',elapsed,' s           |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
