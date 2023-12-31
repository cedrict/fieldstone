!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine process_bc

use module_parameters, only: iel,nel,iproc,geometry,bnd1_bcV_type,bnd2_bcV_type,bnd3_bcV_type,&
                             bnd4_bcV_type,bnd5_bcV_type,bnd6_bcV_type,mU,mV,mW,debug,ndim
use module_mesh 
use module_timing

implicit none

integer :: k

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{process\_bc}
!@@ This subroutine 'translates' the bnd1\_bcV\_type and other variables into booleans in 
!@@ arrays fix\_u and others.
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (ndim==2) then

   select case(geometry)

   case('cartesian','john')

      do iel=1,nel
         mesh(iel)%fix_u(:)=.false. 
         mesh(iel)%fix_v(:)=.false. 

         !-------------
         !left boundary

         do k=1,mU
            if (mesh(iel)%bnd1_Unode(k) .and. bnd1_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd1_Vnode(k) .and. bnd1_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         !-------------
         !right boundary

         do k=1,mU
            if (mesh(iel)%bnd2_Unode(k) .and. bnd2_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd2_Vnode(k) .and. bnd2_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         !-------------
         !bottom boundary

         do k=1,mU
            if (mesh(iel)%bnd3_Unode(k) .and. bnd3_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd3_Vnode(k) .and. bnd3_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         !-------------
         !top boundary

         do k=1,mU
            if (mesh(iel)%bnd4_Unode(k) .and. bnd4_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd4_Vnode(k) .and. bnd4_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

      end do

   case default
      stop 'process_bc: geometry not supported yet'

   end select

else

   select case(geometry)

   case('cartesian')

      do iel=1,nel
         mesh(iel)%fix_u(:)=.false. 
         mesh(iel)%fix_v(:)=.false. 
         mesh(iel)%fix_w(:)=.false. 

         !-------------
         !left boundary

         do k=1,mU
            if (mesh(iel)%bnd1_Unode(k) .and. bnd1_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd1_Vnode(k) .and. bnd1_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd1_Wnode(k) .and. bnd1_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do

         !-------------
         !right boundary

         do k=1,mU
            if (mesh(iel)%bnd2_Unode(k) .and. bnd2_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd2_Vnode(k) .and. bnd2_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd2_Wnode(k) .and. bnd2_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do

         !-------------
         !back boundary

         do k=1,mU
            if (mesh(iel)%bnd3_Unode(k) .and. bnd3_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd3_Vnode(k) .and. bnd3_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd3_Wnode(k) .and. bnd3_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do

         !-------------
         !front boundary

         do k=1,mU
            if (mesh(iel)%bnd4_Unode(k) .and. bnd4_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd4_Vnode(k) .and. bnd4_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd4_Wnode(k) .and. bnd4_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do

         !-------------
         !bottom boundary

         do k=1,mU
            if (mesh(iel)%bnd5_Unode(k) .and. bnd5_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd5_Vnode(k) .and. bnd5_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd5_Wnode(k) .and. bnd5_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do

         !------------
         !top boundary

         do k=1,mU
            if (mesh(iel)%bnd6_Unode(k) .and. bnd6_bcV_type=='noslip') then
               mesh(iel)%fix_u(k)=.true. ; mesh(iel)%u(k)=0.d0
            end if
         end do

         do k=1,mV
            if (mesh(iel)%bnd6_Vnode(k) .and. bnd6_bcV_type=='noslip') then
               mesh(iel)%fix_v(k)=.true. ; mesh(iel)%v(k)=0.d0
            end if
         end do

         do k=1,mW
            if (mesh(iel)%bnd6_Wnode(k) .and. bnd6_bcV_type=='noslip') then
               mesh(iel)%fix_w(k)=.true. ; mesh(iel)%w(k)=0.d0
            end if
         end do


      end do

   case default
      stop 'process_bc: geometry not supported yet'

   end select

end if








if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'process_bc:',elapsed,' s                     |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
