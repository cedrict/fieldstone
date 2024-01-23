!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_matrix_stokes(K_el,G_el,f_el,h_el)

use module_parameters, only: mU,mV,mW,mP,spaceU,spaceV,spaceW,spacePressure,&
                             iel,penalty,nqel,ndim,ndim2,mVel,stokes_solve_strategy
use module_arrays
use module_mesh
use module_constants

implicit none

real(8), intent(out) :: K_el(mVel,mVel)
real(8), intent(out) :: G_el(mVel,mP)
real(8), intent(out) :: f_el(mVel)
real(8), intent(out) :: h_el(mP)

integer iq,k,a,b,c,d,e,f
real(8) Bmat(ndim2,mVel),NNNmat(ndim2,mP)
real(8) rq,sq,tq,jcob,weightq
integer, parameter :: caller_id01=801
integer, parameter :: caller_id02=802
integer, parameter :: caller_id03=803
integer, parameter :: caller_id04=804

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{compute\_elemental\_matrix\_stokes.f90}
!@@ Note that when the material model is called directly on the quadrature points and 
!@@ the penalty formulation is used then the viscosity at the reduced quadrature location 
!@@ is obtained by taking the maximum viscosity value carried by the quadrature points of 
!@@ the element. 
!==================================================================================================!

a=1
b=mU
c=mU+1
d=mU+mV
e=mU+mV+1
f=mU+mV+mW

NNNmat=0.d0

K_el=0.d0
G_el=0.d0
f_el=0.d0
h_el=0.d0
Bmat=0.d0

do iq=1,nqel

   rq=mesh(iel)%rq(iq)
   sq=mesh(iel)%sq(iq)
   tq=mesh(iel)%tq(iq)

   call experiment_gravity_model(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                                 mesh(iel)%gxq(iq),mesh(iel)%gyq(iq),mesh(iel)%gzq(iq))

   if (ndim==2) then

      call NNN(rq,sq,tq,NNNU(1:mU),mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV(1:mV),mV,ndim,spaceV,caller_id02)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id03)

      call compute_dNdx_dNdy(rq,sq,dNNNUdx,dNNNUdy,dNNNVdx,dNNNVdy,jcob)

      mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)

      !------------------------------------------
      ! building gradient matrix and compute f_el
      !------------------------------------------

      Bmat(1,a:b)=dNNNUdx(1:mU) 
      Bmat(2,c:d)=dNNNVdy(1:mV) 
      Bmat(3,a:b)=dNNNUdy(1:mU) 
      Bmat(3,c:d)=dNNNVdx(1:mU) 

      f_el(a:b)=f_el(a:b)+NNNU(1:mU)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      f_el(c:d)=f_el(c:d)+NNNV(1:mV)*mesh(iel)%gyq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)

      !do k=1,mV    
      !   i1=ndofV*k-1    
      !   i2=ndofV*k    
      !   Bmat(1,i1)=dNNNUdx(k) 
      !   Bmat(2,i2)=dNNNVdy(k)
      !   Bmat(3,i1)=dNNNUdy(k) 
      !   Bmat(3,i2)=dNNNVdx(k)
      !   f_el(i1)=f_el(i1)+NNNU(k)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      !   f_el(i2)=f_el(i2)+NNNV(k)*mesh(iel)%gyq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      !end do

      !---------------
      ! compute NNNmat 
      !---------------
      do k=1,mP
         NNNmat(1,k)=NNNp(k)
         NNNmat(2,k)=NNNp(k)
      end do

   else

      call NNN(rq,sq,tq,NNNU(1:mU),mU,ndim,spaceU,caller_id01)
      call NNN(rq,sq,tq,NNNV(1:mV),mV,ndim,spaceV,caller_id02)
      call NNN(rq,sq,tq,NNNW(1:mW),mW,ndim,spaceW,caller_id03)
      call NNN(rq,sq,tq,NNNP(1:mP),mP,ndim,spacePressure,caller_id04)

      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz,jcob)

      mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)

      !------------------------------------------
      ! building gradient matrix and compute f_el
      !------------------------------------------

      Bmat(1,a:b)=dNNNUdx(1:mU) 
      Bmat(2,c:d)=dNNNVdy(1:mV) 
      Bmat(3,e:f)=dNNNWdz(1:mW) 
      Bmat(4,a:b)=dNNNUdy(1:mU) ; Bmat(4,c:d)=dNNNVdx(1:mV) 
      Bmat(5,a:b)=dNNNUdz(1:mU) ; Bmat(5,e:f)=dNNNWdx(1:mW) 
      Bmat(6,c:d)=dNNNVdz(1:mV) ; Bmat(5,e:f)=dNNNWdy(1:mW) 

      f_el(a:b)=f_el(a:b)+NNNU(1:mU)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      f_el(c:d)=f_el(c:d)+NNNV(1:mV)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      f_el(e:f)=f_el(e:f)+NNNW(1:mW)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)

      !do k=1,mV    
      !   i1=ndofV*k-2    
      !   i2=ndofV*k-1    
      !   i3=ndofV*k    
         !Bmat(1,i1)=dNNNUdx(k)
         !Bmat(2,i2)=dNNNVdy(k)
         !Bmat(3,i3)=dNNNWdz(k)
         !Bmat(4,i1)=dNNNUdy(k) ; Bmat(4,i2)=dNNNVdx(k)
         !Bmat(5,i1)=dNNNUdz(k) ; Bmat(5,i3)=dNnNWdx(k)
         !Bmat(6,i2)=dNNNVdz(k) ; Bmat(6,i3)=dNNNWdy(k)
      !   f_el(i1)=f_el(i1)+NNNU(k)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      !   f_el(i2)=f_el(i2)+NNNV(k)*mesh(iel)%gyq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      !   f_el(i3)=f_el(i3)+NNNW(k)*mesh(iel)%gzq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      !end do 

      !---------------
      ! compute NNNmat 
      !---------------
      do k=1,mP
         NNNmat(1,k)=NNNp(k)
         NNNmat(2,k)=NNNp(k)
         NNNmat(3,k)=NNNp(k)
      end do

   end if

   K_el=K_el+matmul(transpose(Bmat),matmul(Cmat,Bmat))*mesh(iel)%etaq(iq)*mesh(iel)%JxWq(iq)

   G_el=G_el-matmul(transpose(Bmat),NNNmat)*mesh(iel)%JxWq(iq)

end do ! nqel

!----------------------------------------------------------

if (stokes_solve_strategy=='___penalty') then

   rq=0d0
   sq=0d0
   tq=0d0
   weightq=2**ndim

   if (ndim==2) then

      call compute_dNdx_dNdy(rq,sq,dNNNUdx,dNNNUdy,dNNNVdx,dNNNVdy,jcob)

      Bmat(1,a:b)=dNNNUdx(1:mU) 
      Bmat(2,c:d)=dNNNVdy(1:mV) 
      Bmat(3,a:b)=dNNNUdy(1:mU) 
      Bmat(3,c:d)=dNNNVdx(1:mU) 

      !do k=1,mV    
      !   i1=ndofV*k-1    
      !   i2=ndofV*k    
      !   Bmat(1,i1)=dNNNUdx(k) 
      !   Bmat(2,i2)=dNNNVdy(k)
      !   Bmat(3,i1)=dNNNUdy(k) 
      !   Bmat(3,i2)=dNNNVdx(k)
      !end do 

   else

      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNNNUdx,dNNNUdy,dNNNUdz,dNNNVdx,dNNNVdy,dNNNVdz,dNNNWdx,dNNNWdy,dNNNWdz,jcob)

      Bmat(1,a:b)=dNNNUdx(1:mU) 
      Bmat(2,c:d)=dNNNVdy(1:mV) 
      Bmat(3,e:f)=dNNNWdz(1:mW) 
      Bmat(4,a:b)=dNNNUdy(1:mU) ; Bmat(4,c:d)=dNNNVdx(1:mV) 
      Bmat(5,a:b)=dNNNUdz(1:mU) ; Bmat(5,e:f)=dNNNWdx(1:mW) 
      Bmat(6,c:d)=dNNNVdz(1:mV) ; Bmat(5,e:f)=dNNNWdy(1:mW) 

      !do k=1,mV    
      !   Bmat(1,i1)=dNNNUdx(k)
      !   Bmat(2,i2)=dNNNVdy(k)
      !   Bmat(3,i3)=dNNNWdz(k)
      !   Bmat(4,i1)=dNNNUdy(k) ; Bmat(4,i2)=dNNNVdx(k)
      !   Bmat(5,i1)=dNNNUdz(k) ; Bmat(5,i3)=dNNNWdx(k)
      !   Bmat(6,i2)=dNNNVdz(k) ; Bmat(6,i3)=dNNNWdy(k)
      !   i1=ndofV*k-2    
      !   i2=ndofV*k-1    
      !   i3=ndofV*k    
      !end do 

   end if

   K_el=K_el+matmul(transpose(Bmat),matmul(Kmat,Bmat))*penalty*mesh(iel)%eta_avrg*jcob*weightq

end if ! penalty

!----------------------------------------------------------

!if (debug) then   
!print *,iel,'================================='
!print *,shape(K_el)
!K_el=K_el*72
!K_el=K_el-transpose(K_el)
!print *,(K_el(1,:))
!print *,(K_el(2,:))
!print *,(K_el(3,:))
!print *,(K_el(4,:))
!print *,(K_el(5,:))
!print *,(K_el(6,:))
!print *,(K_el(7,:))
!print *,(K_el(8,:))
!stop
!print *,G_el
!end if

end subroutine

!==================================================================================================!
!==================================================================================================!
