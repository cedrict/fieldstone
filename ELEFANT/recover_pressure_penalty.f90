!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine recover_pressure_penalty

use global_parameters
use structures
use timing
use matrices, only : csrM
use global_arrays, only : solP,solQ

implicit none

integer iq,i,j,k,k1,k2,inode,jnode
real(8) Mel(mV,mV),f_el(mV),rq,sq,tq,NNNV(mV),jcob,etaq
real(8) dNdx(mV),dNdy(mV),dNdz(mV),div_v,weightq,rhs(NV),diag(NV)

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{recover\_pressure\_penalty}
!@@ This is scheme 6 in Section~\ref{XXX}. 
!@@ The viscosity at the reduced quadrature location 
!@@ is obtained by taking the maximum viscosity value carried by the quadrature points of 
!@@ the element. 
!==================================================================================================!

rhs=0d0
diag=0d0
csrM%mat=0d0
solQ=0

do iel=1,nel

   !compute mass matrix with velocity
   !basis functions on 2**ndim quadrature points
   Mel=0d0
   do iq=1,nqel
      rq=mesh(iel)%rq(iq)
      sq=mesh(iel)%sq(iq)
      tq=mesh(iel)%tq(iq)
      call NNV(rq,sq,tq,NNNV(1:mV),mV,ndim,pair)
      do i=1,mV   
      do j=1,mV   
         Mel(i,j)=Mel(i,j)+NNNV(i)*NNNV(j)*mesh(iel)%JxWq(iq)
      end do   
      end do   
   end do   

   !compute rhs on 1 quad point
   f_el=0d0
   rq=0d0
   sq=0d0
   tq=0d0
   weightq=2**ndim
   etaq=maxval(mesh(iel)%etaq(:))


   if (ndim==2) then
      call compute_dNdx_dNdy(rq,sq,dNdx,dNdy,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))
   else
      call compute_dNdx_dNdy(rq,sq,tq,dNdx,dNdy,dNdz,jcob)
      div_v=sum(dNdx(1:mV)*mesh(iel)%u(1:mV))&
           +sum(dNdy(1:mV)*mesh(iel)%v(1:mV))&
           +sum(dNdz(1:mV)*mesh(iel)%w(1:mV))
   end if 
   do k=1,mV    
      f_el(k)=f_el(k)+NNNV(k)*div_v*jcob*weightq*etaq*penalty
   end do
   solP(iel)=-penalty*etaq*div_v

   do k1=1,mV
      inode=mesh(iel)%iconV(k1)
      do k2=1,mV
         jnode=mesh(iel)%iconV(k2)
         do k=csrM%ia(inode),csrM%ia(inode+1)-1
            if (csrM%ja(k)==jnode) then
               csrM%mat(k)=csrM%mat(k)+Mel(k1,k2)
               exit
            end if
         end do
      end do
      rhs(inode)=rhs(inode)+f_el(k1)
      diag(inode)=diag(inode)+Mel(k1,k1)
   end do

end do

diag=1
!print *,'diag (m/M):',minval(diag),maxval(diag)
!print *,'rhs  (m/M):',minval(rhs),maxval(rhs)
!print *,'mat  (m/M):',minval(csrM%mat),maxval(csrM%mat)
!print *,csrM%mat   
!print *,diag
!print *,rhs


call pcg_solver_csr(csrM,solQ,rhs,diag)

print *,shape(solQ),nel,shape(diag)

do iel=1,nel
   write(777,*) mesh(iel)%xc,mesh(iel)%yc,solP(iel)
end do

do iel=1,nel
   do k=1,mV
      write(888,*) mesh(iel)%xV(k),mesh(iel)%yV(k),solQ(mesh(iel)%iconV(k)) 
      mesh(iel)%q(k)=solQ(mesh(iel)%iconV(k))
   end do
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
