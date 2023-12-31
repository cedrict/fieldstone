!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_annulus

use module_parameters, only: iproc,debug,spaceVelocity,nelphi,nelr,inner_radius,outer_radius,NV,mV,iel,&
                             nel,mP,spacePressure,use_T,NP
use module_mesh 
use module_constants, only: pi
use module_timing

implicit none

integer ielr,ielphi,counter,nnx,k
real(8) hr,hphi

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_annulus}
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

select case(spaceVelocity)

!------------------
case('__Q1')

   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%hr=hr
         mesh(counter)%hphi=hphi
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
            mesh(counter)%inner_Unode(1)=.true.
            mesh(counter)%outer_Unode(4)=.true.
            mesh(counter)%inner_Vnode(1)=.true.
            mesh(counter)%outer_Vnode(4)=.true.
         end if
         if (mesh(counter)%outer_elt) then
            mesh(counter)%outer_Unode(2)=.true.
            mesh(counter)%outer_Unode(3)=.true.
            mesh(counter)%outer_Vnode(2)=.true.
            mesh(counter)%outer_Vnode(3)=.true.
         end if

         mesh(counter)%iconU=mesh(counter)%iconV
         mesh(counter)%xU=mesh(counter)%xV
         mesh(counter)%yU=mesh(counter)%yV

      end do
   end do

!------------
case('__Q2')

   nnx=2*nelr+1
   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%hr=hr
         mesh(counter)%hphi=hphi
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
            mesh(counter)%inner_Unode(1)=.true.
            mesh(counter)%outer_Unode(4)=.true.
            mesh(counter)%outer_Unode(7)=.true.
            mesh(counter)%inner_Vnode(1)=.true.
            mesh(counter)%outer_Vnode(4)=.true.
            mesh(counter)%outer_Vnode(7)=.true.
         end if
         if (mesh(counter)%outer_elt) then
            mesh(counter)%outer_Unode(3)=.true.
            mesh(counter)%outer_Unode(6)=.true.
            mesh(counter)%outer_Unode(9)=.true.
            mesh(counter)%outer_Vnode(3)=.true.
            mesh(counter)%outer_Vnode(6)=.true.
            mesh(counter)%outer_Vnode(9)=.true.
         end if

         mesh(counter)%iconU=mesh(counter)%iconV
         mesh(counter)%xU=mesh(counter)%xV
         mesh(counter)%yU=mesh(counter)%yV

      end do
   end do

case default

   stop 'setup_annulus: geometry unknown'

end select

!----------------------------------------------------------
! pressure 
!----------------------------------------------------------

select case(spacePressure)
case('__Q0','__P0')
   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%iconP(1)=counter
         mesh(counter)%xP(1)=mesh(counter)%xC
         mesh(counter)%yP(1)=mesh(counter)%yC
      end do    
   end do    

case('__Q1')
   counter=0    
   do ielphi=1,nelphi    
      do ielr=1,nelr    
         counter=counter+1    
         mesh(counter)%iconP(1)=ielr+(ielphi-1)*(nelr+1)    
         mesh(counter)%iconP(2)=ielr+1+(ielphi-1)*(nelr+1)    
         mesh(counter)%iconP(3)=ielr+1+ielphi*(nelr+1)    
         mesh(counter)%iconP(4)=ielr+ielphi*(nelr+1)    
         if (mesh(counter)%iconP(3)>NP) mesh(counter)%iconP(3)=mesh(counter)%iconP(3)-NP
         if (mesh(counter)%iconP(4)>NP) mesh(counter)%iconP(4)=mesh(counter)%iconP(4)-NP

         mesh(counter)%rP(1)=inner_radius+(ielr-1)*hr
         mesh(counter)%rP(2)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rP(3)=inner_radius+(ielr-1)*hr+hr
         mesh(counter)%rP(4)=inner_radius+(ielr-1)*hr

         mesh(counter)%phiP(1)=(ielphi-1)*hphi
         mesh(counter)%phiP(2)=(ielphi-1)*hphi
         mesh(counter)%phiP(3)=(ielphi-1)*hphi+hphi
         mesh(counter)%phiP(4)=(ielphi-1)*hphi+hphi

         do k=1,mP
            mesh(counter)%xP(k)=mesh(counter)%rP(k)*cos(mesh(counter)%phiP(k))
            mesh(counter)%yP(k)=mesh(counter)%rP(k)*sin(mesh(counter)%phiP(k))
         end do

      end do
   end do

case default
   stop 'setup_annulus: unknwon spacePressure'
end select 

!----------------------------------------------------------
! temperature (assumption: spaceT~spaceVelocity) 
!----------------------------------------------------------

if (use_T) then
select case(spaceVelocity)
case('__Q1','__Q2','__Q3','__P1','__P2','__P3')
   do iel=1,nel
      mesh(iel)%xT=mesh(iel)%xV
      mesh(iel)%yT=mesh(iel)%yV
      mesh(iel)%zT=mesh(iel)%zV
      mesh(iel)%iconT=mesh(iel)%iconV
   end do
case('_Q1+')
   do iel=1,nel
      mesh(iel)%xT=mesh(iel)%xV(1:4)
      mesh(iel)%yT=mesh(iel)%yV(1:4)
      mesh(iel)%zT=mesh(iel)%zV(1:4)
      mesh(iel)%iconT=mesh(iel)%iconV(1:4)
   end do
case default
   stop 'setup_annulus: spaceT/spaceVelocity problem'
end select

end if

!----------------------------------------------------------
! initialise boundary arrays

do iel=1,nel
   mesh(iel)%fix_u=.false.
   mesh(iel)%fix_v=.false.
   mesh(iel)%fix_w=.false.
   mesh(iel)%fix_T=.false.
end do

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'setup_annulus'//limit
do iel=1,nel
write(2345,*) 'elt:',iel,' | iconV',mesh(iel)%iconV(1:mV),'iconP',mesh(iel)%iconP(1:mP)
do k=1,mV
write(2345,*) mesh(iel)%xV(k),mesh(iel)%yV(k),mesh(iel)%zV(k)
end do
end do
do iel=1,nel
write(2345,*) 'iel,hr,hphi,',iel,mesh(iel)%hr,mesh(iel)%hphi
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_annulus (',elapsed,' s)'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
