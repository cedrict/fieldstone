!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine setup_mapping

use module_parameters, only: iel,spaceVelocity,mapping,nel,iproc,debug,isoparametric_mapping,ndim,&
                             nelx,nely,geometry,mapping,iel,mmapping
use module_constants, only: frac23,frac13,frac12
use module_mesh 
use module_timing

implicit none

integer :: counter,ielx,iely,nnx,nny

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{setup\_mapping}
!@@ This subroutine computes the coordinates {\tt xM,yM,zM} of the mapping nodes for each element, 
!@@ as well as the corresponding connectivity array {\tt iconM}.
!@@ If the mapping polynomial space is identical to the velocity polynomial space, then 
!@@ the mapping nodes are the velocity nodes (same for connectivity array).
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

write(*,'(a)') shift//'mapping='//mapping

if (isoparametric_mapping) then

   if (spaceVelocity=='_Q1F') stop 'setup_mapping: isoparametric not possible'
   !if (spaceVelocity=='_Q1+') stop 'setup_mapping: isoparametric not possible'
   if (spaceVelocity=='Q1++') stop 'setup_mapping: isoparametric not possible'
   
   do iel=1,nel
      mesh(iel)%xM=mesh(iel)%xV   
      mesh(iel)%yM=mesh(iel)%yV   
      mesh(iel)%zM=mesh(iel)%zV   
      mesh(iel)%iconM=mesh(iel)%iconV 
   end do

else

   if (ndim==2) then

      select case(geometry)
      case('cartesian')
         select case(mapping)
         !-----------
         case('__Q1')
            counter=0    
            do iely=1,nely    
               do ielx=1,nelx    
                  counter=counter+1    
                  mesh(counter)%iconM(1)=ielx+(iely-1)*(nelx+1)    
                  mesh(counter)%iconM(2)=ielx+1+(iely-1)*(nelx+1)    
                  mesh(counter)%iconM(3)=ielx+1+iely*(nelx+1)    
                  mesh(counter)%iconM(4)=ielx+iely*(nelx+1)    
                  mesh(counter)%xM(1)=(ielx-1)*mesh(counter)%hx
                  mesh(counter)%xM(2)=(ielx  )*mesh(counter)%hx
                  mesh(counter)%xM(3)=(ielx  )*mesh(counter)%hx
                  mesh(counter)%xM(4)=(ielx-1)*mesh(counter)%hx
                  mesh(counter)%yM(1)=(iely-1)*mesh(counter)%hy
                  mesh(counter)%yM(2)=(iely-1)*mesh(counter)%hy
                  mesh(counter)%yM(3)=(iely  )*mesh(counter)%hy
                  mesh(counter)%yM(4)=(iely  )*mesh(counter)%hy
               end do
            end do
            mesh(counter)%zM(:)=0
         !-----------
         case('__Q2')
            nnx=2*nelx+1
            nny=2*nely+1
            counter=0    
            do iely=1,nely    
               do ielx=1,nelx    
                  counter=counter+1    
                  mesh(counter)%iconM(1)=(ielx-1)*2+1+(iely-1)*2*nnx
                  mesh(counter)%iconM(2)=(ielx-1)*2+2+(iely-1)*2*nnx
                  mesh(counter)%iconM(3)=(ielx-1)*2+3+(iely-1)*2*nnx
                  mesh(counter)%iconM(4)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx
                  mesh(counter)%iconM(5)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx
                  mesh(counter)%iconM(6)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx
                  mesh(counter)%iconM(7)=(ielx-1)*2+1+(iely-1)*2*nnx+nnx*2
                  mesh(counter)%iconM(8)=(ielx-1)*2+2+(iely-1)*2*nnx+nnx*2
                  mesh(counter)%iconM(9)=(ielx-1)*2+3+(iely-1)*2*nnx+nnx*2
                  mesh(counter)%xM(1)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(2)=(ielx-frac12)*mesh(counter)%hx
                  mesh(counter)%xM(3)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%xM(4)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(5)=(ielx-frac12)*mesh(counter)%hx
                  mesh(counter)%xM(6)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%xM(7)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(8)=(ielx-frac12)*mesh(counter)%hx
                  mesh(counter)%xM(9)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%yM(1)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(2)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(3)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(4)=(iely-frac12)*mesh(counter)%hy
                  mesh(counter)%yM(5)=(iely-frac12)*mesh(counter)%hy
                  mesh(counter)%yM(6)=(iely-frac12)*mesh(counter)%hy
                  mesh(counter)%yM(7)=(iely       )*mesh(counter)%hy
                  mesh(counter)%yM(8)=(iely       )*mesh(counter)%hy
                  mesh(counter)%yM(9)=(iely       )*mesh(counter)%hy
               end do
            end do
            mesh(counter)%zM(:)=0
         !-----------
         case('__Q3')
            nnx=3*nelx+1
            nny=3*nely+1
            counter=0    
            do iely=1,nely    
               do ielx=1,nelx    
                  counter=counter+1    
                  mesh(counter)%iconM(01)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*0
                  mesh(counter)%iconM(02)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*0
                  mesh(counter)%iconM(03)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*0
                  mesh(counter)%iconM(04)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*0
                  mesh(counter)%iconM(05)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*1
                  mesh(counter)%iconM(06)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*1
                  mesh(counter)%iconM(07)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*1
                  mesh(counter)%iconM(08)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*1
                  mesh(counter)%iconM(09)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*2
                  mesh(counter)%iconM(10)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*2
                  mesh(counter)%iconM(11)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*2
                  mesh(counter)%iconM(12)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*2
                  mesh(counter)%iconM(13)=(ielx-1)*3+1+(iely-1)*3*nnx+nnx*3
                  mesh(counter)%iconm(14)=(ielx-1)*3+2+(iely-1)*3*nnx+nnx*3
                  mesh(counter)%iconM(15)=(ielx-1)*3+3+(iely-1)*3*nnx+nnx*3
                  mesh(counter)%iconM(16)=(ielx-1)*3+4+(iely-1)*3*nnx+nnx*3
                  mesh(counter)%xM(01)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(02)=(ielx-frac23)*mesh(counter)%hx
                  mesh(counter)%xM(03)=(ielx-frac13)*mesh(counter)%hx
                  mesh(counter)%xM(04)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%xM(05)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(06)=(ielx-frac23)*mesh(counter)%hx
                  mesh(counter)%xM(07)=(ielx-frac13)*mesh(counter)%hx
                  mesh(counter)%xM(08)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%xm(09)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(10)=(ielx-frac23)*mesh(counter)%hx
                  mesh(counter)%xM(11)=(ielx-frac13)*mesh(counter)%hx
                  mesh(counter)%xm(12)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%xM(13)=(ielx-1     )*mesh(counter)%hx
                  mesh(counter)%xM(14)=(ielx-frac23)*mesh(counter)%hx
                  mesh(counter)%xM(15)=(ielx-frac13)*mesh(counter)%hx
                  mesh(counter)%xM(16)=(ielx       )*mesh(counter)%hx
                  mesh(counter)%yM(01)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(02)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(03)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(04)=(iely-1     )*mesh(counter)%hy
                  mesh(counter)%yM(05)=(iely-frac23)*mesh(counter)%hy
                  mesh(counter)%yM(06)=(iely-frac23)*mesh(counter)%hy
                  mesh(counter)%yM(07)=(iely-frac23)*mesh(counter)%hy
                  mesh(counter)%yM(08)=(iely-frac23)*mesh(counter)%hy
                  mesh(counter)%yM(09)=(iely-frac13)*mesh(counter)%hy
                  mesh(counter)%yM(10)=(iely-frac13)*mesh(counter)%hy
                  mesh(counter)%yM(11)=(iely-frac13)*mesh(counter)%hy
                  mesh(counter)%yM(12)=(iely-frac13)*mesh(counter)%hy
                  mesh(counter)%yM(13)=(iely       )*mesh(counter)%hy
                  mesh(counter)%yM(14)=(iely       )*mesh(counter)%hy
                  mesh(counter)%ym(15)=(iely       )*mesh(counter)%hy
                  mesh(counter)%yM(16)=(iely       )*mesh(counter)%hy
               end do    
            end do    
            mesh(counter)%zM(:)=0
         !-----------
         case default
            stop 'setup_mapping: mapping not supported yet'
         end select
      case default
         stop 'setup_mapping: 2D geometry not supported yet'
      end select

   else

      select case(geometry)
      case('cartesian')
      case default
         stop 'setup_mapping: 3D geometry not supported yet'
      end select

   end if

end if

!----------------------------------------------------------

if (debug) then
write(2345,*) limit//'setup_mapping'//limit
do iel=1,nel
write(2345,*) 'elt:',iel,' | iconM',mesh(iel)%iconM(1:mmapping)
end do
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'setup_mapping:',elapsed,' s                  |'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
