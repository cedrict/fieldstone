!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine impose_boundary_conditions_stokes(K_el,G_el,f_el,h_el)

use module_parameters, only: mU,mV,mW,mVel,mP,iel
use module_mesh 

implicit none

real(8),intent(inout) :: K_el(mVel,mVel)
real(8),intent(inout) :: f_el(mVel)
real(8),intent(inout) :: G_el(mVel,mP)
real(8),intent(inout) :: h_el(mP)

integer k,i,j
real(8) bcvalue,Kref

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{impose\_boundary\_conditions\_stokes}
!@@ This subroutine modifies the elemental $\K$, $\G$ and $\C$ matrices as well as the 
!@@ elemental rhs $f_{el}$ and $h_{el}$ and returns them modified after imposing
!@@ velocity Dirichlet boundary conditions.
!==================================================================================================!

do k=1,mU
   if (mesh(iel)%fix_u(k)) then
      bcvalue=mesh(iel)%u(k)
      i=k
      Kref=K_el(i,i)    
      do j=1,mVel
         f_el(j)=f_el(j)-K_el(j,i)*bcvalue    
         K_el(i,j)=0d0 
         K_el(j,i)=0d0  
      enddo  
      do j=1,mP
         h_el(j)=h_el(j)-G_el(i,j)*bcvalue
         G_el(i,j)=0d0
      enddo  
      K_el(i,i)=Kref    
      f_el(i)=Kref*bcvalue   
   end if
end do

do k=1,mV
   if (mesh(iel)%fix_v(k)) then
      bcvalue=mesh(iel)%v(k)
      i=mU+k
      Kref=K_el(i,i)    
      do j=1,mVel
         f_el(j)=f_el(j)-K_el(j,i)*bcvalue    
         K_el(i,j)=0d0 
         K_el(j,i)=0d0  
      enddo  
      do j=1,mP
         h_el(j)=h_el(j)-G_el(i,j)*bcvalue
         G_el(i,j)=0d0
      enddo  
      K_el(i,i)=Kref    
      f_el(i)=Kref*bcvalue   
   end if
end do

do k=1,mW
   if (mesh(iel)%fix_w(k)) then
      bcvalue=mesh(iel)%w(k)
      i=mU+mV+k
      Kref=K_el(i,i)    
      do j=1,mVel
         f_el(j)=f_el(j)-K_el(j,i)*bcvalue    
         K_el(i,j)=0d0 
         K_el(j,i)=0d0  
      enddo  
      do j=1,mP
         h_el(j)=h_el(j)-G_el(i,j)*bcvalue
         G_el(i,j)=0d0
      enddo  
      K_el(i,i)=Kref    
      f_el(i)=Kref*bcvalue   
   end if
end do

end subroutine

!==================================================================================================!
!==================================================================================================!
