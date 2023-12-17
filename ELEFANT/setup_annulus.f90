!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_annulus

use module_parameters, only: iproc,debug,spaceV,nelphi,nelr,inner_radius,outer_radius,NV,mV
use module_mesh 
use module_constants, only: pi
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none

integer ielr,ielphi,counter,nnx,k
real(8) hr,hphi

!==================================================================================================!
!==================================================================================================!
!@@ \subsubsection{setup\_annulus}
!@@ The annulus is placed in the equatorial plane (x,y).
!@@ We then use the cylindrical coordinates $r,\phi$ which correspond
!@@ to the spherical coordinates with $\theta=\pi/2$ (equator).
!@@ The mesh is built in the $(r,\phi)$ space.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (abs(inner_radius)<1e-6) stop 'setup_annulus: inner_radius=0'
if (abs(outer_radius)<1e-6) stop 'setup_annulus: outer_radius=0'
if (outer_radius<inner_radius) stop 'setup_annulus: outer_radius<inner_radius'

hr=(outer_radius-inner_radius)/nelr
hphi=2*pi/nelphi

select case(spaceV)

!------------------
case('__Q1','_Q1+')

   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%iconV(1)=ielr+(ielphi-1)*(nelr+1)    
         mesh(counter)%iconV(2)=ielr+1+(ielphi-1)*(nelr+1)    
         mesh(counter)%iconV(3)=ielr+1+ielphi*(nelr+1)    
         mesh(counter)%iconV(4)=ielr+ielphi*(nelr+1)    
         if (mesh(counter)%iconV(3)>NV) mesh(counter)%iconV(3)=mesh(counter)%iconV(3)-NV
         if (mesh(counter)%iconV(4)>NV) mesh(counter)%iconV(4)=mesh(counter)%iconV(4)-NV

         mesh(counter)%rV(1)=inner_radius+(ielr-1)*hr
         mesh(counter)%rV(2)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rV(3)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rV(4)=inner_radius+(ielr-1)*hr

         mesh(counter)%phiV(1)=(ielphi-1)*hphi
         mesh(counter)%phiV(2)=(ielphi-1)*hphi
         mesh(counter)%phiV(3)=(ielphi-1)*hphi+hphi
         mesh(counter)%phiV(4)=(ielphi-1)*hphi+hphi

         do k=1,mV
            mesh(counter)%xV(k)=mesh(counter)%rV(k)*cos(mesh(counter)%phiV(k))
            mesh(counter)%yV(k)=mesh(counter)%rV(k)*sin(mesh(counter)%phiV(k))
         end do

         mesh(counter)%xc=(mesh(counter)%rV(1)+hr/2)*cos(mesh(counter)%phiV(1)+hphi/2)
         mesh(counter)%yc=(mesh(counter)%rV(1)+hr/2)*sin(mesh(counter)%phiV(1)+hphi/2)

         mesh(counter)%inner_elt=(ielr==1)
         mesh(counter)%outer_elt=(ielr==nelr)

         if (mesh(counter)%inner_elt) then
            mesh(counter)%inner_node(1)=.true.
            mesh(counter)%outer_node(4)=.true.
         end if
         if (mesh(counter)%outer_elt) then
            mesh(counter)%outer_node(2)=.true.
            mesh(counter)%outer_node(3)=.true.
         end if

      end do
   end do

!------------
case('__Q2')

   nnx=2*nelr+1
   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%iconV(1)=(ielr-1)*2+1+(ielphi-1)*2*nnx
         mesh(counter)%iconV(2)=(ielr-1)*2+2+(ielphi-1)*2*nnx
         mesh(counter)%iconV(3)=(ielr-1)*2+3+(ielphi-1)*2*nnx
         mesh(counter)%iconV(4)=(ielr-1)*2+1+(ielphi-1)*2*nnx+nnx
         mesh(counter)%iconV(5)=(ielr-1)*2+2+(ielphi-1)*2*nnx+nnx
         mesh(counter)%iconV(6)=(ielr-1)*2+3+(ielphi-1)*2*nnx+nnx
         mesh(counter)%iconV(7)=(ielr-1)*2+1+(ielphi-1)*2*nnx+nnx*2
         mesh(counter)%iconV(8)=(ielr-1)*2+2+(ielphi-1)*2*nnx+nnx*2
         mesh(counter)%iconV(9)=(ielr-1)*2+3+(ielphi-1)*2*nnx+nnx*2
         if (mesh(counter)%iconV(7)>NV) mesh(counter)%iconV(7)=mesh(counter)%iconV(7)-NV
         if (mesh(counter)%iconV(8)>NV) mesh(counter)%iconV(8)=mesh(counter)%iconV(8)-NV
         if (mesh(counter)%iconV(9)>NV) mesh(counter)%iconV(9)=mesh(counter)%iconV(9)-NV

         mesh(counter)%rV(1)=inner_radius+(ielr-1)*hr
         mesh(counter)%rV(2)=inner_radius+(ielr-1)*hr+hr/2
         mesh(counter)%rV(3)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rV(4)=inner_radius+(ielr-1)*hr
         mesh(counter)%rV(5)=inner_radius+(ielr-1)*hr+hr/2
         mesh(counter)%rV(6)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rV(7)=inner_radius+(ielr-1)*hr
         mesh(counter)%rV(8)=inner_radius+(ielr-1)*hr+hr/2
         mesh(counter)%rV(9)=inner_radius+(ielr-1)*hr+hr

         mesh(counter)%phiV(1)=(ielphi-1)*hphi
         mesh(counter)%phiV(2)=(ielphi-1)*hphi
         mesh(counter)%phiV(3)=(ielphi-1)*hphi
         mesh(counter)%phiV(4)=(ielphi-1)*hphi+hphi/2
         mesh(counter)%phiV(5)=(ielphi-1)*hphi+hphi/2
         mesh(counter)%phiV(6)=(ielphi-1)*hphi+hphi/2
         mesh(counter)%phiV(7)=(ielphi-1)*hphi+hphi
         mesh(counter)%phiV(8)=(ielphi-1)*hphi+hphi
         mesh(counter)%phiV(9)=(ielphi-1)*hphi+hphi

         do k=1,mV
            mesh(counter)%xV(k)=mesh(counter)%rV(k)*cos(mesh(counter)%phiV(k))
            mesh(counter)%yV(k)=mesh(counter)%rV(k)*sin(mesh(counter)%phiV(k))
         end do

         mesh(counter)%xc=mesh(counter)%xV(5)
         mesh(counter)%yc=mesh(counter)%yV(5)

         mesh(counter)%inner_elt=(ielr==1)
         mesh(counter)%outer_elt=(ielr==nelr)

         if (mesh(counter)%inner_elt) then
            mesh(counter)%inner_node(1)=.true.
            mesh(counter)%outer_node(4)=.true.
            mesh(counter)%outer_node(7)=.true.
         end if
         if (mesh(counter)%outer_elt) then
            mesh(counter)%outer_node(3)=.true.
            mesh(counter)%outer_node(6)=.true.
            mesh(counter)%outer_node(9)=.true.
         end if

      end do
   end do

case default

   stop 'setup_annulus: geometry unknown'

end select

!=====================

if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_annulus (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
