!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine compute_elemental_matrix_stokes(K_el,G_el,f_el,h_el)

use module_parameters
use module_arrays
use module_mesh
use module_constants

implicit none

real(8), intent(out) :: K_el(mV*ndofV,mV*ndofV)
real(8), intent(out) :: G_el(mV*ndofV,mP)
real(8), intent(out) :: f_el(mV*ndofV)
real(8), intent(out) :: h_el(mP)

integer iq,k,i1,i2,i3
real(8) Bmat(ndim2,mV*ndofV),NNNmat(ndim2,mP)
real(8) rq,sq,tq,jcob,weightq,etaq
real(8) NNNV(mV),NNNP(mP),dNdx(mV),dNdy(mV),dNdz(mV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{compute\_elemental\_matrix\_stokes.f90}
!@@ Note that when the material model is called directly on the quadrature points and 
!@@ the penalty formulation is used then the viscosity at the reduced quadrature location 
!@@ is obtained by taking the maximum viscosity value carried by the quadrature points of 
!@@ the element. 
!==================================================================================================!

K_el=0.d0
G_el=0.d0
f_el=0.d0
h_el=0.d0
Bmat=0.d0

do iq=1,nqel

   rq=mesh(iel)%rq(iq)
   sq=mesh(iel)%sq(iq)
   tq=mesh(iel)%tq(iq)

   call gravity_model(mesh(iel)%xq(iq),mesh(iel)%yq(iq),mesh(iel)%zq(iq),&
                      mesh(iel)%gxq(iq),mesh(iel)%gyq(iq),mesh(iel)%gzq(iq))

   call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
   call NNP(rq,sq,tq,NNNP(1:mP),mP,ndim,pair)

   if (ndim==2) then

      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)

      mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)

      !-------------------------
      ! building gradient matrix
      !-------------------------
      do k=1,mV    
         i1=ndofV*k-1    
         i2=ndofV*k    
         Bmat(1,i1)=dNdx(k) 
         Bmat(2,i2)=dNdy(k)
         Bmat(3,i1)=dNdy(k) 
         Bmat(3,i2)=dNdx(k)
      end do 

      !---------------
      ! compute f_el
      !---------------
      do k=1,mV    
         i1=ndofV*k-1
         i2=ndofV*k
         f_el(i1)=f_el(i1)+NNNV(k)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
         f_el(i2)=f_el(i2)+NNNV(k)*mesh(iel)%gyq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      end do

      !---------------
      ! compute NNNmat 
      !---------------
      do k=1,mP
         NNNmat(1,k)=NNNp(k)
         NNNmat(2,k)=NNNp(k)
         NNNmat(3,k)=0.d0
      end do

   else

      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx(1:mV),dNdy(1:mV),dNdz(1:mV),jcob)

      mesh(iel)%JxWq(iq)=jcob*mesh(iel)%weightq(iq)

      !-------------------------
      ! building gradient matrix
      !-------------------------
      Bmat=0
      do k=1,mV    
         i1=ndofV*k-2    
         i2=ndofV*k-1    
         i3=ndofV*k    
         Bmat(1,i1)=dNdx(k)
         Bmat(2,i2)=dNdy(k)
         Bmat(3,i3)=dNdz(k)
         Bmat(4,i1)=dNdy(k) ; Bmat(4,i2)=dNdx(k)
         Bmat(5,i1)=dNdz(k) ; Bmat(5,i3)=dNdx(k)
         Bmat(6,i2)=dNdz(k) ; Bmat(6,i3)=dNdy(k)
      end do 

      !---------------
      ! compute f_el
      !---------------
      do k=1,mV    
         i1=ndofV*k-2
         i2=ndofV*k-1
         i3=ndofV*k
         f_el(i1)=f_el(i1)+NNNV(k)*mesh(iel)%gxq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
         f_el(i2)=f_el(i2)+NNNV(k)*mesh(iel)%gyq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
         f_el(i3)=f_el(i3)+NNNV(k)*mesh(iel)%gzq(iq)*mesh(iel)%rhoq(iq)*mesh(iel)%JxWq(iq)
      end do

      !---------------
      ! compute NNNmat 
      !---------------
      do k=1,mP
         NNNmat(1,k)=NNNp(k)
         NNNmat(2,k)=NNNp(k)
         NNNmat(3,k)=NNNp(k)
         NNNmat(4,k)=0.d0
         NNNmat(5,k)=0.d0
         NNNmat(6,k)=0.d0
      end do

   end if

   K_el=K_el+matmul(transpose(Bmat(1:ndim2,1:mV*ndofV)),&
                    matmul(Cmat,Bmat(1:ndim2,1:mV*ndofV)))&
                    *mesh(iel)%etaq(iq)*mesh(iel)%JxWq(iq)

   G_el=G_el-matmul(transpose(Bmat(1:ndim2,1:mV*ndofV)),NNNmat(1:ndim2,1:mP))*mesh(iel)%JxWq(iq)

end do ! nqel

!----------------------------------------------------------

if (use_penalty) then

   rq=0d0
   sq=0d0
   tq=0d0
   weightq=2**ndim
   etaq=maxval(mesh(iel)%etaq(:))

   if (ndim==2) then

      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)

      do k=1,mV    
         i1=ndofV*k-1    
         i2=ndofV*k    
         Bmat(1,i1)=dNdx(k) 
         Bmat(2,i2)=dNdy(k)
         Bmat(3,i1)=dNdy(k) 
         Bmat(3,i2)=dNdx(k)
      end do 

   else

      call compute_dNdx_dNdy_dNdz(rq,sq,tq,dNdx,dNdy,dNdz,jcob)

      do k=1,mV    
         i1=ndofV*k-2    
         i2=ndofV*k-1    
         i3=ndofV*k    
         Bmat(1,i1)=dNdx(k)
         Bmat(2,i2)=dNdy(k)
         Bmat(3,i3)=dNdz(k)
         Bmat(4,i1)=dNdy(k) ; Bmat(4,i2)=dNdx(k)
         Bmat(5,i1)=dNdz(k) ; Bmat(5,i3)=dNdx(k)
         Bmat(6,i2)=dNdz(k) ; Bmat(6,i3)=dNdy(k)
      end do 

   end if

   K_el=K_el+matmul(transpose(Bmat(1:ndim2,1:mV*ndofV)),&
                    matmul(Kmat,Bmat(1:ndim2,1:mV*ndofV)))&
                    *penalty*etaq*jcob*weightq

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
